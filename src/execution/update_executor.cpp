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

#include "execution/executors/seq_scan_executor.h"
#include "execution/executors/update_executor.h"
#include "execution/executors/values_executor.h"
#include "execution/plans/seq_scan_plan.h"
#include "execution/executor_factory.h"
#include "storage/table/tuple.h"

namespace bustub {

//begin added by zhangyu at 2025/10/27 for p3t1
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
  if(executed_){
    return false;
  }
  Tuple old_tup;
  RID old_rid;
  int update_num = 0;
  while (child_executor_->Next(&old_tup, &old_rid)) {
    std::vector<Value> values{};
    // 新的tuple应该和原表的schema一致
    values.reserve(table_info_->schema_.GetColumnCount());
    for (const auto &expr : plan_->target_expressions_) {
      values.push_back(expr->Evaluate(&old_tup, table_info_->schema_));
    }

    // old tuple 标记删除
    Tuple new_tup(values, &table_info_->schema_);
    TupleMeta tup_meta = table_info_->table_->GetTupleMeta(old_rid);
    tup_meta.is_deleted_ = true;
    table_info_->table_->UpdateTupleMeta(tup_meta, old_rid);

    // 新tuple 插入
    table_info_->table_->InsertTuple({0, false}, new_tup);
    ++update_num;
  }
  // 没有需要更新的了，结束即可
  Tuple tup(*rid, reinterpret_cast<const char *>(&update_num), sizeof(update_num));
  *tuple = tup;
  executed_ = true;
  return true;
}
//end added by zhangyu at 2025/10/27 for p3t1

}  // namespace bustub
