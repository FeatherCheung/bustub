//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// transaction_manager.cpp
//
// Identification: src/concurrency/transaction_manager.cpp
//
// Copyright (c) 2015-2019, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "concurrency/transaction_manager.h"

#include <cstddef>
#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include "catalog/catalog.h"
#include "catalog/column.h"
#include "catalog/schema.h"
#include "common/config.h"
#include "common/exception.h"
#include "common/macros.h"
#include "concurrency/transaction.h"
#include "execution/execution_common.h"
#include "fmt/format.h"
#include "storage/table/table_heap.h"
#include "storage/table/tuple.h"
#include "type/type_id.h"
#include "type/value.h"
#include "type/value_factory.h"

namespace bustub {

// begin: mod by zhangyu for p4t1 at 2025/12/8
auto TransactionManager::Begin(IsolationLevel isolation_level) -> Transaction * {
  std::unique_lock<std::shared_mutex> l(txn_map_mutex_);
  auto txn_id = next_txn_id_++;
  auto txn = std::make_unique<Transaction>(txn_id, isolation_level);
  auto *txn_ref = txn.get();
  txn_map_.insert(std::make_pair(txn_id, std::move(txn)));

  /*
   * set the timestamps here. Watermark updated below.
   * 写入最近的时间，并将水印加入进去
   */
  txn_ref->read_ts_.store(last_commit_ts_);
  running_txns_.AddTxn(txn_ref->read_ts_);
  return txn_ref;
}

auto TransactionManager::VerifyTxn(Transaction *txn) -> bool { return true; }

auto TransactionManager::Commit(Transaction *txn) -> bool {
  std::unique_lock<std::mutex> commit_lck(commit_mutex_);

  /*
   * P4T3 at 2026/1/11
   * 获取全局时间，并将临时时间改为全局时间，再更改事务的状态为committed
   */
  auto commit_ts = last_commit_ts_.load() + 1;
  auto write_set = txn->GetWriteSets();

  auto iter = write_set.begin();
  while (iter != write_set.end()) {
    auto table_oid = iter->first;
    auto rids = iter->second;
    auto table_info = catalog_->GetTable(table_oid);
    auto iter_rid = rids.begin();
    while (iter_rid != rids.end()) {
      auto rid = *iter_rid;
      auto meta = table_info->table_->GetTupleMeta(rid);
      table_info->table_->UpdateTupleMeta({commit_ts, meta.is_deleted_}, rid);
      ++iter_rid;
    }
    ++iter;
  }

  if (txn->state_ != TransactionState::RUNNING) {
    throw Exception("txn not in running state");
  }

  if (txn->GetIsolationLevel() == IsolationLevel::SERIALIZABLE) {
    if (!VerifyTxn(txn)) {
      commit_lck.unlock();
      Abort(txn);
      return false;
    }
  }

  // TODO(fall2023): Implement the commit logic!

  std::unique_lock<std::shared_mutex> lck(txn_map_mutex_);

  // TODO(fall2023): set commit timestamp + update last committed timestamp here.
  txn->commit_ts_.store(++last_commit_ts_);

  txn->state_ = TransactionState::COMMITTED;
  running_txns_.UpdateCommitTs(txn->commit_ts_);
  running_txns_.RemoveTxn(txn->read_ts_);

  return true;
}

void TransactionManager::Abort(Transaction *txn) {
  if (txn->state_ != TransactionState::RUNNING && txn->state_ != TransactionState::TAINTED) {
    throw Exception("txn not in running / tainted state");
  }

  // TODO(fall2023): Implement the abort logic!

  std::unique_lock<std::shared_mutex> lck(txn_map_mutex_);
  txn->state_ = TransactionState::ABORTED;
  running_txns_.RemoveTxn(txn->read_ts_);
}

// begin for P4T3.5 zhangyu at 2026/1/22
void TransactionManager::GarbageCollection() {
  // 1.获取最低读取时间戳事务
  auto watermark = GetWatermark();

  // 2.找到所有commit/abort 事务修改过的table和这个table中的rids
  std::unordered_map<table_oid_t, std::unordered_set<RID>> modified_rids;
  // 这个数据结构用来存储commit/abort 事务不可见的undolog的数量
  std::unordered_map<txn_id_t, size_t> txn_candidate_delete;
  for (auto [txn_id, txn] : txn_map_) {
    if (txn->state_ == TransactionState::COMMITTED || txn->state_ != TransactionState::ABORTED) {
      auto write_sets = txn->GetWriteSets();
      for (auto &[table_oid, rid_sets] : write_sets) {
        if (modified_rids.find(table_oid) != modified_rids.end()) {
          modified_rids[table_oid].insert(rid_sets.begin(), rid_sets.end());
        } else {
          modified_rids[table_oid] = rid_sets;
        }
      }
    }
  }

  // 3.判断可见性，遍历每张table中所有改动过的rids，遇到第一个 <= watermark的undolog，它之前的undolog都是不可见
  for (auto [table_oid, rid_sets] : modified_rids) {
    auto table = catalog_->GetTable(table_oid);
    for (auto rid : rid_sets) {
      auto [meta, _, cur_undolink_opt] = GetTupleAndUndoLink(this, table->table_.get(), rid);
      UndoLog cur_undolog;
      auto ts = meta.ts_;

      // ts_ 大于watermark，需要往下找
      auto cur_undolink = cur_undolink_opt.has_value() ? GetUndoLink(rid).value() : UndoLink{};
      while (ts > watermark && cur_undolink.IsValid()) {
        cur_undolog = GetUndoLog(cur_undolink);
        cur_undolink = cur_undolog.prev_version_;
        ts = cur_undolog.ts_;
      }
      while (cur_undolink.IsValid()) {
        auto cur_undolog_opt = GetUndoLogOptional(cur_undolink);
        if (cur_undolog_opt.has_value()) {
          cur_undolog = cur_undolog_opt.value();
          txn_candidate_delete[cur_undolink.prev_txn_]++;
          cur_undolink = cur_undolog.prev_version_;
        } else {
          break;
        }
      }
    }
  }

  // 4.txn_candidate_delete.second == undolog.size  证明这个txn可以删除
  std::unique_lock<std::shared_mutex> l(txn_map_mutex_);
  auto txn_iter = txn_map_.begin();
  while (txn_iter != txn_map_.end()) {
    auto txn_id = txn_iter->first;
    auto txn = txn_iter->second.get();
    auto iter_cnt = txn_candidate_delete.find(txn_id);
    if ((txn->GetUndoLogNum() == 0 && txn->state_ == TransactionState::COMMITTED) ||
        (iter_cnt != txn_candidate_delete.end() && iter_cnt->second == txn->GetUndoLogNum())) {
      txn_map_.erase(txn_iter++);
    } else {
      ++txn_iter;
    }
  }
}
// end for P4T3.5 zhangyu at 2026/1/22
//  end: mod by zhangyu for p4t1 at 2025/12/8

}  // namespace bustub
