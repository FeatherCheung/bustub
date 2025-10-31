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

void InsertExecutor::Init() {
    auto catalog = exec_ctx_->GetCatalog();
    tableinfo_ = catalog->GetTable(plan_->GetTableOid());
    child_executor_->Init();
}

auto InsertExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  if(executed_) {
    return false;
  }
  Tuple tup_res;
  RID rid_res;
  int insert_num = 0;
  while (child_executor_->Next(&tup_res, &rid_res)) {
    auto value_opt = tableinfo_->table_->InsertTuple(TupleMeta{0, false}, tup_res);
    if (value_opt.has_value()) {
      ++insert_num;
    }
  }
  Tuple tup(*rid, reinterpret_cast<const char*>(&insert_num), sizeof(insert_num));
  *tuple = tup;
  executed_ = true;
  return true;
}
// end by zhangyu at 2025/10/27 for p3t1

}  // namespace bustub
