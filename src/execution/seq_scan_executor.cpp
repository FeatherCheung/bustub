//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// seq_scan_executor.cpp
//
// Identification: src/execution/seq_scan_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/seq_scan_executor.h"
#include "common/config.h"
#include "storage/page/table_page.h"
#include "storage/table/table_iterator.h"
#include "storage/table/tuple.h"

namespace bustub {

// begin：added by zhangyu at 2025/10/27 for p3t1
SeqScanExecutor::SeqScanExecutor(ExecutorContext *exec_ctx, const SeqScanPlanNode *plan)
    : AbstractExecutor(exec_ctx), plan_(plan) {
        if(plan_->filter_predicate_ != nullptr){
            filter_expr_ = plan_->filter_predicate_;
        }
    }

SeqScanExecutor::~SeqScanExecutor(){
    if(iter_ != nullptr){
        delete iter_;
        iter_ = nullptr;
    }
};
void SeqScanExecutor::Init() {
    iter_ = nullptr;
 }

auto SeqScanExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  auto catalog = exec_ctx_->GetCatalog();
  auto tableinfo = catalog->GetTable(plan_->GetTableOid());
  if (iter_ == nullptr) {
    iter_ = new TableIterator(tableinfo->table_->MakeIterator());
  }
  while (!iter_->IsEnd()) {
    auto [tuplemeta, tuple_res] = iter_->GetTuple();
    ++(*iter_);  // 提前自增迭代器，避免多处写

    if (tuplemeta.is_deleted_) {
      continue;
    }

    if (filter_expr_ != nullptr) {
      auto value = filter_expr_->Evaluate(&tuple_res, GetOutputSchema());
      if (!value.IsNull() && !value.GetAs<bool>()) {
        continue;
      }
    }

    *tuple = tuple_res;
    *rid = tuple_res.GetRid();
    return true;
  }
  return false;
}
  // end: added by zhangyu at 2025/10/27 for p3t1
}  // namespace bustub
