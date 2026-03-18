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

// begin added by zhangyu at 2026/3/17 for P4T4 为mvcc调整删除操作
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
    auto txn_mgr = exec_ctx_->GetTransactionManager();
    if (CheckWriteConflict(&tup_meta, txn)) {
      txn->SetTainted();
      throw ExecutionException("write -write conflict in delete!");
    }

    TupleMeta new_tupmeta = {txn->GetTransactionTempTs(), true};
    auto cur_undolink = undo_link_opt.value();
    auto new_undolink_opt = GenerateUndoLink(txn_mgr, txn, cur_undolink, &tup_delete, nullptr, tup_meta, new_tupmeta,
                                             &table_info_->schema_);

    new_tupmeta.ts_ = txn->GetTransactionTempTs();
    if (new_undolink_opt.has_value()) {
      auto ret = UpdateTupleAndUndoLink(
          exec_ctx_->GetTransactionManager(), rid_delete, new_undolink_opt.value(), table_info_->table_.get(), txn,
          new_tupmeta, tup_delete,
          [txn](const TupleMeta &tup_meta, const Tuple &tuple, RID rid, std::optional<UndoLink> undolink_opt) -> bool {
            // check 函数进行write-write 判断，并验证是否可以delete
            if (CheckWriteConflict(&tup_meta, txn)) {
              txn->SetTainted();
              throw ExecutionException("write -write conflict in delete!");
            }
            return true;
          });
      if (!ret) {
        txn->SetTainted();
        throw ExecutionException("deleted failed!");
      }
    } else {
      table_info_->table_->UpdateTupleInPlace(new_tupmeta, tup_delete, rid_delete);
    }

    // 记录写集（commit 用）
    txn->AppendWriteSet(table_info_->oid_, rid_delete);
    // end p4t3 at 2026/1/22 by zhangyu: generate undologs for deletion
    // P4T4不再删除index entry，只标记删除tuuple
    ++delete_num;
  }
  // end added by zhangyu at 2026/3/17 for P4T4 为mvcc调整删除操作
  // 没有需要删除的了，结束即可
  Tuple tup(*rid, reinterpret_cast<const char *>(&delete_num), sizeof(delete_num));
  *tuple = tup;
  executed_ = true;
  return true;
}
// end by zhangyu at 2025/10/27 for p3t1

}  // namespace bustub
