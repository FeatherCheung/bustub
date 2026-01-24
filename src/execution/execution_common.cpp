//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// execution_common.cpp
//
// Identification: src/execution/execution_common.cpp
//
// Copyright (c) 2024-2024, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/execution_common.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "catalog/catalog.h"
#include "catalog/column.h"
#include "catalog/schema.h"
#include "common/config.h"
#include "common/macros.h"
#include "concurrency/transaction.h"
#include "concurrency/transaction_manager.h"
#include "fmt/core.h"
#include "storage/table/table_heap.h"
#include "storage/table/tuple.h"
#include "type/value.h"

namespace bustub {

// begin: mod by zhangyu for p3t4 at 2025/12/6
TupleComparator::TupleComparator(std::vector<OrderBy> order_bys) : order_bys_(std::move(order_bys)) {}

/*
 * true a在b之前，false a == b 或者 b在a 之前
 */
auto TupleComparator::operator()(const SortEntry &entry_a, const SortEntry &entry_b) const -> bool {
  for (size_t i = 0; i < order_bys_.size(); i++) {
    const auto &v1 = entry_a.first[i];
    const auto &v2 = entry_b.first[i];

    if (v1.CompareEquals(v2) == CmpBool::CmpTrue) {
      continue;  // 继续比较下一列
    }

    auto comp = v1.CompareLessThan(v2);

    if (order_bys_[i].first == OrderByType::DESC) {
      return comp == CmpBool::CmpFalse;  // 反转
    }

    return comp == CmpBool::CmpTrue;  // ASC
  }
  return false;  // 完全相等
}
auto GenerateSortKey(const Tuple &tuple, const std::vector<OrderBy> &order_bys, const Schema &schema) -> SortKey {
  SortKey key;
  for (const auto &order : order_bys) {
    auto val = order.second->Evaluate(&tuple, schema);
    key.push_back(val);
  }
  return key;
}
// end: mod by zhangyu for p3t4 at 2025/12/6

/**
 * Above are all you need for P3.
 * You can ignore the remaining part of this file until P4.
 */

/**
 * @brief Reconstruct a tuple by applying the provided undo logs from the base tuple. All logs in the undo_logs are
 * applied regardless of the timestamp
 *
 * @param schema The schema of the base tuple and the returned tuple.
 * @param base_tuple The base tuple to start the reconstruction from.
 * @param base_meta The metadata of the base tuple.
 * @param undo_logs The list of undo logs to apply during the reconstruction, the front is applied first.
 * @return An optional tuple that represents the reconstructed tuple. If the tuple is deleted as the result, returns
 * std::nullopt.
 */
// begin: mod by zhangyu for p4t2 at 2025/12/10
auto ReconstructTuple(const Schema *schema, const Tuple &base_tuple, const TupleMeta &base_meta,
                      const std::vector<UndoLog> &undo_logs) -> std::optional<Tuple> {
  std::vector<Value> values;
  values.reserve(schema->GetColumnCount());
  for (uint32_t i = 0; i < schema->GetColumnCount(); i++) {
    values.emplace_back(base_tuple.GetValue(schema, i));
  }

  TupleMeta meta = base_meta;

  // 依次应用 undo_logs（前 → 后）
  for (const auto &log : undo_logs) {
    std::vector<Column> columns;
    std::vector<int> col_idxs;
    // 更新元信息（保持最新的 meta）
    meta.is_deleted_ = log.is_deleted_;
    meta.ts_ = log.ts_;
    if (log.modified_fields_.empty()) {
      continue;
    }
    for (uint32_t i = 0; i < schema->GetColumnCount(); i++) {
      if (log.modified_fields_[i]) {
        // 该列被修改过 → 覆盖
        auto column = schema->GetColumn(i);
        columns.emplace_back(column);
        col_idxs.emplace_back(i);
      }
    }
    Schema partial_schema(columns);
    int modified_col = 0;
    for (auto col_idx : col_idxs) {
      auto value = log.tuple_.GetValue(&partial_schema, modified_col++);
      values[col_idx] = value;
    }
  }

  // 如果最终版本是删除版本，返回 nullopt
  if (meta.is_deleted_) {
    return std::nullopt;
  }

  // 构造最终 tuple
  return Tuple{values, schema};
}
// end: mod by zhangyu for p4t2 at 2025/12/10

/**
 * @brief Collects the undo logs sufficient to reconstruct the tuple w.r.t. the txn.
 *
 * @param rid The RID of the tuple.
 * @param base_meta The metadata of the base tuple.
 * @param base_tuple The base tuple.
 * @param undo_link The undo link to the latest undo log.
 * @param txn The transaction.
 * @param txn_mgr The transaction manager.
 * @return An optional vector of undo logs to pass to ReconstructTuple(). std::nullopt if the tuple did not exist at the
 * time.
 */
// begin: mod by zhangyu for p4t2 at 2025/12/24
auto CollectUndoLogs(RID rid, const TupleMeta &base_meta, const Tuple &base_tuple, std::optional<UndoLink> undo_link,
                     Transaction *txn, TransactionManager *txn_mgr) -> std::optional<std::vector<UndoLog>> {
  /*
   * 收集undolog的三个情况
   * 元组被已提交的事务修改，被未提交且不是自身的事务修改，被未提交且是自身的事务修改
   * 第一种情况，元组时间戳 < TXN_START_ID，被已提交事务修改，根据事务read_ts 沿着undolink 历史性回退
   * 其他则是未提交，如果元组时间戳 = GetTransactionTempTs，证明当前版本是可见的，不需要回退，返回空
   * 未提交的且不是自己修改的，根据事务read_ts 沿着undolink 历史性回退
   */

  timestamp_t tup_timestamp = base_meta.ts_;
  const timestamp_t read_ts = txn->GetReadTs();
  std::vector<UndoLog> undologs;

  // 元组对于事务的read_ts 是最新可见的，不需要回退
  if (tup_timestamp < TXN_START_ID && tup_timestamp <= read_ts) {
    return undologs;
  }

  // 元组被未提交修改，且是当前事务所修改，那么当前版本可见，不需要回退
  if (tup_timestamp == txn->GetTransactionTempTs()) {
    return undologs;
  }

  // undo 日志为空，表示不可见了
  if (!undo_link.has_value()) {
    return std::nullopt;
  }

  auto cur_undolink = undo_link.value();
  // is_flag 用来表示 整个undolog中是否存在日志是 <= read_ts
  bool is_flag = false;

  while (cur_undolink.IsValid() && !is_flag) {
    auto pre_txn = txn_mgr->txn_map_.find(cur_undolink.prev_txn_)->second;
    auto cur_undolog = pre_txn->GetUndoLog(cur_undolink.prev_log_idx_);
    // if (tup_timestamp == pre_txn->GetTransactionTempTs()) {
    //   cur_undolink = cur_undolog.prev_version_;
    //   undologs.push_back(cur_undolog);
    //   continue;
    // }
    // 大于read_ts 那么需要回退
    if (cur_undolog.ts_ > read_ts) {
      undologs.push_back(cur_undolog);
      // 第一次碰到小于等于的情况，依然放入回退集合，但是此时不需要在循环了
    } else {
      undologs.push_back(cur_undolog);
      is_flag = true;
    }

    cur_undolink = cur_undolog.prev_version_;
  }

  return !undologs.empty() && is_flag ? std::make_optional(undologs) : std::nullopt;
}
// end: mod by zhangyu for p4t2 at 2025/12/24

/**
 * @brief Generates a new undo log as the transaction tries to modify this tuple at the first time.
 *
 * @param schema The schema of the table.
 * @param base_tuple The base tuple before the update, the one retrieved from the table heap. nullptr if the tuple is
 * deleted.
 * @param target_tuple The target tuple after the update. nullptr if this is a deletion.
 * @param ts The timestamp of the base tuple.
 * @param prev_version The undo link to the latest undo log of this tuple.
 * @return The generated undo log.
 */
auto GenerateNewUndoLog(const Schema *schema, const Tuple *base_tuple, const Tuple *target_tuple, timestamp_t ts,
                        UndoLink prev_version) -> UndoLog {
  /*
   * P4T3 at 2026/1/11 zhangyu
   * 首次生成undolog
   */
  // 初始元组为空，证明这是插入
  if (base_tuple == nullptr) {
    return UndoLog{true, {}, {}, ts, prev_version};
  }
  std::vector<bool> modified_fields;

  // 目标元组为空，证明是删除
  if (target_tuple == nullptr) {
    modified_fields.resize(schema->GetColumnCount(), true);
    return {false, modified_fields, *base_tuple, ts, prev_version};
  }
  std::vector<Value> values;
  std::vector<Column> columns;

  for (uint32_t column_idx = 0; column_idx < schema->GetColumnCount(); ++column_idx) {
    auto tar_val = target_tuple->GetValue(schema, column_idx);
    auto base_val = base_tuple->GetValue(schema, column_idx);
    if (!tar_val.CompareExactlyEquals(base_val)) {
      modified_fields.push_back(true);
      values.push_back(base_val);
      const auto &column = schema->GetColumn(column_idx);
      columns.push_back(column);
    } else {
      modified_fields.push_back(false);
    }
  }
  Schema partial_schema(columns);
  return {false, modified_fields, {values, &partial_schema}, ts, prev_version};
}

/**
 * @brief Generate the updated undo log to replace the old one, whereas the tuple is already modified by this txn once.
 *
 * @param schema The schema of the table.
 * @param base_tuple The base tuple before the update, the one retrieved from the table heap. nullptr if the tuple is
 * deleted.
 * @param target_tuple The target tuple after the update. nullptr if this is a deletion.
 * @param log The original undo log.
 * @return The updated undo log.
 */
auto GenerateUpdatedUndoLog(const Schema *schema, const Tuple *base_tuple, const Tuple *target_tuple,
                            const UndoLog &log) -> UndoLog {
  /*
   * P4T3 at 2026/1/11 zhangyu
   * 针对同一条tuple的多次修改，合并为一个undolog
   */

  // 如果undolog为空，表明之前的是插入
  if (log.is_deleted_) {
    return GenerateNewUndoLog(schema, {}, target_tuple, log.ts_, log.prev_version_);
  }

  // 1、先根据旧的日志，重构出原来的tuple
  std::vector<Value> old_values;
  old_values.reserve(schema->GetColumnCount());
  for (uint32_t i = 0; i < schema->GetColumnCount(); i++) {
    old_values.emplace_back(base_tuple->GetValue(schema, i));
  }

  std::vector<Column> old_columns;
  std::vector<int> col_idxs;

  for (uint32_t i = 0; i < schema->GetColumnCount(); i++) {
    if (log.modified_fields_[i]) {
      // 该列被修改过 → 覆盖
      auto column = schema->GetColumn(i);
      old_columns.emplace_back(column);
      col_idxs.emplace_back(i);
    }
  }
  Schema old_partial_schema(old_columns);
  int modified_col = 0;
  for (auto col_idx : col_idxs) {
    auto value = log.tuple_.GetValue(&old_partial_schema, modified_col++);
    old_values[col_idx] = value;
  }
  Tuple tup = Tuple{old_values, schema};

  std::vector<Value> new_values;
  std::vector<Column> new_columns;
  std::vector<bool> modified_fields;
  // 目标为nullptr，表示为删除
  if (target_tuple == nullptr) {
    modified_fields.resize(schema->GetColumnCount(), true);
    return {false, modified_fields, tup, log.ts_, log.prev_version_};
  }

  // 2、由base_tuple 和 target_tuple 构造新的log
  for (uint32_t column_idx = 0; column_idx < schema->GetColumnCount(); ++column_idx) {
    auto tar_val = target_tuple->GetValue(schema, column_idx);
    auto base_val = base_tuple->GetValue(schema, column_idx);
    // 两者不等，那么将这列存入构成undolog的values中
    if (log.modified_fields_[column_idx]) {
      // 两者相等，那么看原来的undolog是否有被修改
      modified_fields.push_back(true);
      new_values.push_back(tup.GetValue(schema, column_idx));
      const auto &column = schema->GetColumn(column_idx);
      new_columns.push_back(column);
    } else if (!tar_val.CompareExactlyEquals(base_val)) {
      modified_fields.push_back(true);
      new_values.push_back(base_val);
      const auto &column = schema->GetColumn(column_idx);
      new_columns.push_back(column);
    } else {
      modified_fields.push_back(false);
    }
  }
  Schema new_patrial_schema(new_columns);
  return {log.is_deleted_, modified_fields, {new_values, &new_patrial_schema}, log.ts_, log.prev_version_};
}

void TxnMgrDbg(const std::string &info, TransactionManager *txn_mgr, const TableInfo *table_info,
               TableHeap *table_heap) {
  // always use stderr for printing logs...
  // always use stderr for printing logs...
  fmt::println(stderr, "debug_hook: {}", info);
  for (auto &txn : txn_mgr->txn_map_) {
    fmt::println(stderr, "txn_id: {}, state: {}, read_ts: {}, commit_ts: {}", txn.first,
                 txn.second->GetTransactionState(), txn.second->GetReadTs(), txn.second->GetCommitTs());
  }
  fmt::println(stderr, "table_name: {}, table_schema: {}", table_info->name_, table_info->schema_.ToString());
  for (auto iter = table_heap->MakeIterator(); !iter.IsEnd(); ++iter) {
    auto rid = iter.GetRID();
    auto tuple = iter.GetTuple().second;
    auto tuple_meta = iter.GetTuple().first;
    auto pre_link = txn_mgr->GetUndoLink(rid);
    fmt::println(stderr, "tuple={}, tuple_ts={},{}", tuple.ToString(&table_info->schema_), tuple_meta.ts_,
                 tuple_meta.is_deleted_ ? "deleted" : "not deleted");
    if (!pre_link.has_value()) {
      continue;
    }
    UndoLink undo_link = pre_link.value();
    while (undo_link.IsValid()) {
      auto undo_log = txn_mgr->GetUndoLogOptional(undo_link);
      if (!undo_log.has_value()) {
        break;
      }
      auto old_tuple = ReconstructTuple(&table_info->schema_, tuple, tuple_meta, {*undo_log});
      if (old_tuple.has_value()) {
        tuple = old_tuple.value();
        tuple_meta = TupleMeta{undo_log->ts_, undo_log->is_deleted_};
        fmt::println(stderr, " => tuple={}, tuple_meta={},{}", tuple.ToString(&table_info->schema_), tuple_meta.ts_,
                     tuple_meta.is_deleted_ ? "deleted" : "not deleted");
        undo_link = undo_log->prev_version_;
      } else {
        fmt::println(stderr, " => tuple=deleted, tuple_meta={},deleted", undo_log->ts_);
        tuple_meta = TupleMeta{undo_log->ts_, true};
        undo_link = undo_log->prev_version_;
      }
    }
    std::cout << std::endl;
  }
  // We recommend implementing this function as traversing the table heap and print the version chain. An example output
  // of our reference solution:
  //
  // debug_hook: before verify scan
  // RID=0/0 ts=txn8 tuple=(1, <NULL>, <NULL>)
  //   txn8@0 (2, _, _) ts=1
  // RID=0/1 ts=3 tuple=(3, <NULL>, <NULL>)
  //   txn5@0 <del> ts=2
  //   txn3@0 (4, <NULL>, <NULL>) ts=1
  // RID=0/2 ts=4 <del marker> tuple=(<NULL>, <NULL>, <NULL>)
  //   txn7@0 (5, <NULL>, <NULL>) ts=3
  // RID=0/3 ts=txn6 <del marker> tuple=(<NULL>, <NULL>, <NULL>)
  //   txn6@0 (6, <NULL>, <NULL>) ts=2
  //   txn3@1 (7, _, _) ts=1
}

// p4t3 at 2026/1/22 zhangyu: check write-write conflict
/*
 * 两种情况的写-写冲突，
 * 1）元组被未提交的事务update/delete，其他事务如果修改或者删除这个元组，此时冲突
 * 2）事务B开始，事务a删除元组t，并commit，然后事务B再删除/修改这个元组并commit
 */
auto CheckWriteConflict(const TupleMeta *tupmeta, Transaction *txn) -> bool {
  auto tup_ts = tupmeta->ts_;
  // auto txn_commit_ts = txn->GetCommitTs();
  auto txn_temp_ts = txn->GetTransactionTempTs();
  auto txn_read_ts = txn->GetReadTs();

  // case 1: 被其他未提交事务修改过
  if (tup_ts >= TXN_START_ID && tup_ts != txn_temp_ts) {
    return true;  // 冲突)
  }
  // case 2: 被未来事务提交过
  if (tup_ts < TXN_START_ID && tup_ts > txn_read_ts) {
    return true;  // 冲突)
  }
  return false;
}

}  // namespace bustub
