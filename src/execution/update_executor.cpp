//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// update_executor.cpp
//
// Identification: src/execution/update_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "concurrency/transaction_manager.h"
#include "execution/execution_common.h"
#include "execution/executor_factory.h"
#include "execution/executors/seq_scan_executor.h"
#include "execution/executors/update_executor.h"
#include "execution/executors/values_executor.h"
#include "execution/plans/seq_scan_plan.h"
#include "storage/table/tuple.h"

namespace bustub {

// begin added by zhangyu at 2025/10/27 for p3t1
UpdateExecutor::UpdateExecutor(ExecutorContext *exec_ctx, const UpdatePlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan) {
  // As of Fall 2022, you DON'T need to implement update executor to have perfect score in project 3 / project 4.
  child_executor_ = std::move(child_executor);
}

void UpdateExecutor::Init() {
  auto catalog = exec_ctx_->GetCatalog();
  table_info_ = catalog->GetTable(plan_->GetTableOid()).get();
  child_executor_->Init();
}

auto UpdateExecutor::Next([[maybe_unused]] Tuple *tuple, RID *rid) -> bool {
  if (executed_) {
    return false;
  }
  Tuple old_tup;
  RID old_rid;
  int update_num = 0;
  auto catalog = exec_ctx_->GetCatalog();
  auto indexes_info = catalog->GetTableIndexes(table_info_->name_);
  while (child_executor_->Next(&old_tup, &old_rid)) {
    std::vector<Value> values{};
    // 新的tuple应该和原表的schema一致
    values.reserve(table_info_->schema_.GetColumnCount());
    for (const auto &expr : plan_->target_expressions_) {
      values.push_back(expr->Evaluate(&old_tup, table_info_->schema_));
    }

    // old tuple 标记删除
    // new tuple 插入表中
    // begin: p4t3 at 2026/1/22 zhangyu: generate undolog for update
    Tuple new_tup(values, &table_info_->schema_);

    // 伪更新，直接continue
    if (IsTupleContentEqual(old_tup, new_tup)) {
      continue;
    }

    auto txn = exec_ctx_->GetTransaction();
    auto txnmrg = exec_ctx_->GetTransactionManager();
    auto [old_tupmeta, _, undo_link_opt] = GetTupleAndUndoLink(txnmrg, table_info_->table_.get(), old_rid);
    /* write-write check */

    if (CheckWriteConflict(&old_tupmeta, txn)) {
      txn->SetTainted();
      throw ExecutionException("write -write conflict in delete!");
    }

    auto new_tupmeta = old_tupmeta;
    auto cur_undo_link = undo_link_opt.value();
    UndoLink new_undo_link;

    // 判断是否生成 UndoLog（或合并UndoLog）
    if (cur_undo_link.IsValid()) {
      auto undo_log_opt = txnmrg->GetUndoLogOptional(cur_undo_link);
      if (undo_log_opt.has_value()) {
        auto old_undo_log = undo_log_opt.value();
        if (cur_undo_link.prev_txn_ == txn->GetTransactionTempTs()) {
          auto new_uodo_log = GenerateUpdatedUndoLog(&table_info_->schema_, &old_tup, &new_tup, old_undo_log);
          txn->ModifyUndoLog(cur_undo_link.prev_log_idx_, new_uodo_log);
          new_undo_link = cur_undo_link;
          new_undo_link.prev_txn_ = txn->GetTransactionId();
          // 更新元组和它的undolink
          new_tupmeta.ts_ = txn->GetTransactionTempTs();
          new_tup.SetRid(old_tup.GetRid());
          UpdateTupleAndUndoLink(exec_ctx_->GetTransactionManager(), old_rid, new_undo_link, table_info_->table_.get(),
                                 txn, new_tupmeta, new_tup);
          break;
        }
      }
    }

    if (old_tupmeta.ts_ != txn->GetTransactionTempTs()) {
      auto undo_log = GenerateNewUndoLog(&table_info_->schema_, &old_tup, &new_tup, old_tupmeta.ts_, cur_undo_link);
      new_undo_link = txn->AppendUndoLog(undo_log);
      // 更新元组和它的undolink
      new_tupmeta.ts_ = txn->GetTransactionTempTs();
      new_tup.SetRid(old_tup.GetRid());
      UpdateTupleAndUndoLink(exec_ctx_->GetTransactionManager(), old_rid, new_undo_link, table_info_->table_.get(), txn,
                             new_tupmeta, new_tup);

    } else {
      // 直接插入即可
      new_tupmeta.ts_ = txn->GetTransactionTempTs();
      new_tup.SetRid(old_tup.GetRid());
      table_info_->table_->UpdateTupleInPlace(new_tupmeta, new_tup, old_rid);
    }

    // 记录写集（commit 用）
    txn->AppendWriteSet(table_info_->oid_, old_rid);
    // end: p4t3 at 2026/1/22 zhangyu: generate undolog for update

    // 删除旧的索引，插入新的索引
    for (const auto &index_info : indexes_info) {
      auto index = index_info->index_.get();
      auto b_plus_tree_index = dynamic_cast<BPlusTreeIndexForTwoIntegerColumn *>(index);
      b_plus_tree_index->DeleteEntry(
          old_tup.KeyFromTuple(table_info_->schema_, *index->GetKeySchema(), index->GetKeyAttrs()), old_rid, nullptr);
      b_plus_tree_index->InsertEntry(
          new_tup.KeyFromTuple(table_info_->schema_, *index->GetKeySchema(), index->GetKeyAttrs()), old_rid, nullptr);
    }

    ++update_num;
  }
  // 没有需要更新的了，结束即可
  Tuple tup(*rid, reinterpret_cast<const char *>(&update_num), sizeof(update_num));
  *tuple = tup;
  executed_ = true;
  return true;
}
// end added by zhangyu at 2025/10/27 for p3t1

}  // namespace bustub
