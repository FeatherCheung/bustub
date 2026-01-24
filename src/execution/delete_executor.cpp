//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// delete_executor.cpp
//
// Identification: src/execution/delete_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "common/rid.h"
#include "concurrency/transaction.h"
#include "concurrency/transaction_manager.h"
#include "execution/execution_common.h"
#include "execution/executors/delete_executor.h"

namespace bustub {

// added by zhangyu at 2025/10/27 for p3t1
DeleteExecutor::DeleteExecutor(ExecutorContext *exec_ctx, const DeletePlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan) {
  child_executor_ = std::move(child_executor);
}

void DeleteExecutor::Init() {
  auto catalog = exec_ctx_->GetCatalog();
  table_info_ = catalog->GetTable(plan_->GetTableOid()).get();
  child_executor_->Init();
}

auto DeleteExecutor::Next([[maybe_unused]] Tuple *tuple, RID *rid) -> bool {
  if (executed_) {
    return false;
  };
  auto catalog = exec_ctx_->GetCatalog();
  auto indexes_info = catalog->GetTableIndexes(table_info_->name_);
  Tuple tup_delete{};
  RID rid_delete{};
  int delete_num = 0;
  while (child_executor_->Next(&tup_delete, &rid_delete)) {
    // begin p4t3 at 2026/1/22 generate undologs for deletion
    auto [tup_meta, tup_delete, undo_link_opt] =
        GetTupleAndUndoLink(exec_ctx_->GetTransactionManager(), table_info_->table_.get(), rid_delete);
    /* write-write check */
    auto txn = exec_ctx_->GetTransaction();
    auto txnmrg = exec_ctx_->GetTransactionManager();
    if (CheckWriteConflict(&tup_meta, txn)) {
      txn->SetTainted();
      throw ExecutionException("write -write conflict in delete!");
    }

    // 生成 undolog
    TupleMeta new_tupmeta = {txn->GetTransactionTempTs(), true};
    auto cur_undo_link = undo_link_opt.value();
    UndoLink new_undo_link;
    if (cur_undo_link.IsValid()) {
      // 如果这个元组有undolink，并且是同一个事务删除，undolog合并即可
      auto undo_log_opt = txnmrg->GetUndoLogOptional(cur_undo_link);
      if (undo_log_opt.has_value()) {
        auto old_undo_log = undo_log_opt.value();
        if (cur_undo_link.prev_txn_ == txn->GetTransactionTempTs()) {
          auto new_uodo_log = GenerateUpdatedUndoLog(&table_info_->schema_, &tup_delete, nullptr, old_undo_log);
          txn->ModifyUndoLog(cur_undo_link.prev_log_idx_, new_uodo_log);
          new_undo_link = cur_undo_link;
          new_undo_link.prev_txn_ = txn->GetTransactionId();
          // 更新元组和它的undolink
          UpdateTupleAndUndoLink(exec_ctx_->GetTransactionManager(), rid_delete, new_undo_link,
                                 table_info_->table_.get(), txn, new_tupmeta, tup_delete);
          break;
        }
      }
    }

    // 这个元组没有undolink，那么要么新生成undolog，要么直接删除
    if (tup_meta.ts_ != txn->GetTransactionTempTs()) {
      auto undo_log = GenerateNewUndoLog(&table_info_->schema_, &tup_delete, nullptr, tup_meta.ts_, cur_undo_link);
      new_undo_link = txn->AppendUndoLog(undo_log);
      // 更新元组和它的undolink
      UpdateTupleAndUndoLink(exec_ctx_->GetTransactionManager(), rid_delete, new_undo_link, table_info_->table_.get(),
                             txn, new_tupmeta, tup_delete);
    } else {
      // 删除即可
      table_info_->table_->UpdateTupleInPlace(new_tupmeta, tup_delete, rid_delete);
    }

    // 记录写集（commit 用）
    txn->AppendWriteSet(table_info_->oid_, rid_delete);
    // end p4t3 at 2026/1/22 by zhangyu: generate undologs for deletion
    // 删除旧的索引
    for (const auto &index_info : indexes_info) {
      auto index = index_info->index_.get();
      auto b_plus_tree_index = dynamic_cast<BPlusTreeIndexForTwoIntegerColumn *>(index);
      b_plus_tree_index->DeleteEntry(
          tup_delete.KeyFromTuple(table_info_->schema_, *index->GetKeySchema(), index->GetKeyAttrs()), rid_delete,
          nullptr);
    }
    ++delete_num;
  }
  // 没有需要删除的了，结束即可
  Tuple tup(*rid, reinterpret_cast<const char *>(&delete_num), sizeof(delete_num));
  *tuple = tup;
  executed_ = true;
  return true;
}
// end by zhangyu at 2025/10/27 for p3t1

}  // namespace bustub
