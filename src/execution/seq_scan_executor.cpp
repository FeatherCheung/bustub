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
#include <optional>
#include "common/config.h"
#include "concurrency/transaction_manager.h"
#include "execution/execution_common.h"
#include "storage/page/table_page.h"
#include "storage/table/table_iterator.h"
#include "storage/table/tuple.h"

namespace bustub {

// begin：added by zhangyu at 2025/10/27 for p3t1
SeqScanExecutor::SeqScanExecutor(ExecutorContext *exec_ctx, const SeqScanPlanNode *plan)
    : AbstractExecutor(exec_ctx), plan_(plan) {
  if (plan_->filter_predicate_ != nullptr) {
    filter_expr_ = plan_->filter_predicate_;
  }
}

SeqScanExecutor::~SeqScanExecutor() {
  if (iter_ != nullptr) {
    delete iter_;
    iter_ = nullptr;
  }
};
void SeqScanExecutor::Init() {
  auto catalog = exec_ctx_->GetCatalog();
  table_info_ = catalog->GetTable(plan_->GetTableOid()).get();
  if (iter_ != nullptr) {
    delete iter_;
    iter_ = nullptr;
  }
}

auto SeqScanExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  if (iter_ == nullptr) {
    iter_ = new TableIterator(table_info_->table_->MakeIterator());
  }
  while (!iter_->IsEnd()) {
    auto [tuplemeta, tuple_res] = iter_->GetTuple();
    ++(*iter_);  // 提前自增迭代器，避免多处写

    /* P4T2 实现元组的重构 */
    auto undo_logs_opt = CollectUndoLogs(tuple_res.GetRid(), tuplemeta, tuple_res,
                                         exec_ctx_->GetTransactionManager()->GetUndoLink(tuple_res.GetRid()),
                                         exec_ctx_->GetTransaction(), exec_ctx_->GetTransactionManager());

    /*
     * 如果undo_logs_opt无值表示这个元组不可见了
     * 如果undo_logs_opt有值，且undolog不为空，重构日志，如果undolog为空，表示不需要回退了
     */
    if (undo_logs_opt.has_value()) {
      auto undo_logs = undo_logs_opt.value();
      if (!undo_logs.empty()) {
        auto tuple_opt = ReconstructTuple(&GetOutputSchema(), tuple_res, tuplemeta, undo_logs_opt.value());
        /* 如果没有返回tuple，则表明这个元组被删除了 */
        if (!tuple_opt.has_value()) {
          continue;
        }
        tuple_res = tuple_opt.value();
      } else {
        if (tuplemeta.is_deleted_) {
          continue;
        }
      }
    } else {
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
