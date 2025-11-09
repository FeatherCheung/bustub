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
    TupleMeta tup_meta = table_info_->table_->GetTupleMeta(rid_delete);
    tup_meta.is_deleted_ = true;
    table_info_->table_->UpdateTupleMeta(tup_meta, rid_delete);

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
