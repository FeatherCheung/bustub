//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// insert_executor.cpp
//
// Identification: src/execution/insert_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <string>

#include "common/config.h"
#include "concurrency/transaction_manager.h"
#include "execution/execution_common.h"
#include "execution/executors/insert_executor.h"
#include "fmt/core.h"
#include "storage/table/tuple.h"

namespace bustub {

// begin：mod by zhangyu at 2025/10/27 for p3t1
InsertExecutor::InsertExecutor(ExecutorContext *exec_ctx, const InsertPlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan) {
  child_executor_ = std::move(child_executor);
  executed_ = false;
}

void InsertExecutor::Init() { child_executor_->Init(); }

auto InsertExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  if (executed_) {
    return false;
  }
  auto catalog = exec_ctx_->GetCatalog();
  auto txn = exec_ctx_->GetTransaction();
  auto txn_mgr = exec_ctx_->GetTransactionManager();
  auto table_info = catalog->GetTable(plan_->GetTableOid());
  auto indexes_info = catalog->GetTableIndexes(table_info->name_);
  // begin added by zhangyu at 2026/3/17 for P4T4
  IndexInfo *index_info = nullptr;
  if (!indexes_info.empty()) {
    index_info = indexes_info[0].get();
  }
  Tuple tup_res;
  RID rid_res;
  int insert_num = 0;
  while (child_executor_->Next(&tup_res, &rid_res)) {
    // insert index
    // p4t3 at 2026/1/22 zhangyu: generate undolink for insertion
    if (index_info != nullptr) {
      auto index = index_info->index_.get();
      auto b_plus_tree_index = dynamic_cast<BPlusTreeIndexForTwoIntegerColumn *>(index);
      // 尝试查找这个key是否存在
      std::vector<bustub::RID> rids;
      b_plus_tree_index->ScanKey(
          tup_res.KeyFromTuple(table_info->schema_, *index->GetKeySchema(), index->GetKeyAttrs()), &rids, txn);

      // 如果插入了key值，则需要判断这个tuple是否被delete
      if (!rids.empty()) {
        auto rid = rids[0];
        auto [tup_meta, tuple, undolink_opt] = GetTupleAndUndoLink(txn_mgr, table_info->table_.get(), rid);
        // 此时需要判断是否生成uodolog
        TupleMeta tupmeta{};
        tupmeta.is_deleted_ = false;
        tupmeta.ts_ = txn->GetTransactionTempTs();
        auto new_undolink_opt = GenerateUndoLink(txn_mgr, txn, undolink_opt.value(), &tuple, &tup_res, tup_meta,
                                                 tupmeta, &table_info->schema_);

        auto success =
            UpdateTupleAndUndoLink(txn_mgr, rid, new_undolink_opt.has_value() ? new_undolink_opt : UndoLink{},
                                   table_info->table_.get(), txn, tupmeta, tup_res,
                                   [txn](const TupleMeta &tup_meta, const Tuple &tuple, RID rid,
                                         std::optional<UndoLink> undolink_opt) -> bool {
                                     // check 函数进行write-write 判断，并验证是否可以update
                                     if (CheckWriteConflict(&tup_meta, txn)) {
                                       txn->SetTainted();
                                       throw ExecutionException("write -write conflict in delete!");
                                     }
                                     if (!tup_meta.is_deleted_) {
                                       txn->SetTainted();
                                       throw ExecutionException("write -write conflict in delete 2!");
                                     }
                                     return true;
                                   });
        if (!success) {
          txn->SetTainted();
          throw ExecutionException("this index-tuple isn't deleted for insertion ");
        }
        ++insert_num;
        rid_res = rid;
        txn->AppendWriteSet(table_info->oid_, rid_res);
      } else {
        // 如果索引为空，那么插入tuple和index
        TupleMeta tupmeta{};
        tupmeta.is_deleted_ = false;
        tupmeta.ts_ = txn->GetTransactionTempTs();
        auto rid_opt = table_info->table_->InsertTuple(tupmeta, tup_res);
        if (rid_opt.has_value()) {
          ++insert_num;
          rid_res = rid_opt.value();
          txn->AppendWriteSet(table_info->oid_, rid_res);
          UndoLink link;
          txn_mgr->UpdateUndoLink(rid_res, link, nullptr);
        }
        auto index = index_info->index_.get();
        auto b_plus_tree_index = dynamic_cast<BPlusTreeIndexForTwoIntegerColumn *>(index);
        auto inserted = b_plus_tree_index->InsertEntry(
            tup_res.KeyFromTuple(table_info->schema_, *index->GetKeySchema(), index->GetKeyAttrs()), rid_res, nullptr);
        if (!inserted) {
          txn->SetTainted();
          throw ExecutionException("this index-tuple has been inserted for insertion ");
        }
      }
    } else {
      // 没有index 直接插入即可
      TupleMeta tupmeta{};
      tupmeta.is_deleted_ = false;
      tupmeta.ts_ = txn->GetTransactionTempTs();
      UndoLink link;
      auto rid_opt = table_info->table_->InsertTuple(tupmeta, tup_res);
      if (rid_opt.has_value()) {
        ++insert_num;
        rid_res = rid_opt.value();
        txn->AppendWriteSet(table_info->oid_, rid_res);
        txn_mgr->UpdateUndoLink(rid_res, link, nullptr);
      }
    }
    // end added by zhangyu at 2026/3/17 for P4T4
  }
  Tuple tup(*rid, reinterpret_cast<const char *>(&insert_num), sizeof(insert_num));
  *tuple = tup;
  executed_ = true;
  return true;
}
// end by zhangyu at 2025/10/27 for p3t1

}  // namespace bustub
