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
#include "execution/executors/insert_executor.h"
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
  Tuple tup_res;
  RID rid_res;
  int insert_num = 0;
  while (child_executor_->Next(&tup_res, &rid_res)) {
    // insert index
    // p4t3 at 2026/1/22 zhangyu: generate undolink for insertion
    TupleMeta tupmeta{};
    tupmeta.is_deleted_ = false;
    tupmeta.ts_ = txn->GetTransactionTempTs();
    auto value_opt = table_info->table_->InsertTuple(tupmeta, tup_res);
    if (value_opt.has_value()) {
      ++insert_num;
      rid_res = value_opt.value();
      txn->AppendWriteSet(table_info->oid_, rid_res);
      UndoLink link;
      txn_mgr->UpdateUndoLink(rid_res, link, nullptr);
    }
    for (const auto &index_info : indexes_info) {
      auto index = index_info->index_.get();
      auto b_plus_tree_index = dynamic_cast<BPlusTreeIndexForTwoIntegerColumn *>(index);
      b_plus_tree_index->InsertEntry(
          tup_res.KeyFromTuple(table_info->schema_, *index->GetKeySchema(), index->GetKeyAttrs()), rid_res, nullptr);
    }
  }
  Tuple tup(*rid, reinterpret_cast<const char *>(&insert_num), sizeof(insert_num));
  *tuple = tup;
  executed_ = true;
  return true;
}
// end by zhangyu at 2025/10/27 for p3t1

}  // namespace bustub
