//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// index_scan_executor.cpp
//
// Identification: src/execution/index_scan_executor.cpp
//
// Copyright (c) 2015-19, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//
#include "execution/executors/index_scan_executor.h"
#include <cstddef>
#include "catalog/schema.h"
#include "concurrency/transaction_manager.h"
#include "execution/execution_common.h"
#include "execution/expressions/abstract_expression.h"
#include "execution/expressions/comparison_expression.h"
#include "execution/expressions/constant_value_expression.h"
#include "execution/expressions/logic_expression.h"

// begin: Mod by zhangyu for p3t1 at 2025/11/3
namespace bustub {
IndexScanExecutor::IndexScanExecutor(ExecutorContext *exec_ctx, const IndexScanPlanNode *plan)
    : AbstractExecutor(exec_ctx) {
  plan_ = plan;
  filter_predicate_ = plan_->filter_predicate_;
  pred_keys_ = plan_->pred_keys_;
  key_index_ = 0;
}

IndexScanExecutor::~IndexScanExecutor() {
  if (iter_ != nullptr) {
    delete iter_;
    iter_ = nullptr;
  }
}

void IndexScanExecutor::Init() {
  auto catalog = exec_ctx_->GetCatalog();
  auto index_info = catalog->GetIndex(plan_->index_oid_);
  const auto &index = index_info->index_;
  b_plus_tree_index_ = dynamic_cast<BPlusTreeIndexForTwoIntegerColumn *>(index.get());
  table_info_ = catalog->GetTable(plan_->table_oid_).get();
  key_index_ = 0;
  delete iter_;
  iter_ = nullptr;
  executed_ = false;
}

auto IndexScanExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  auto filter_expr = filter_predicate_.get();
  return GetIndexScanTuple(tuple, rid, filter_expr);
}

auto IndexScanExecutor::GetIndexScanTuple(Tuple *tuple, RID *rid, AbstractExpression *expr) -> bool {
  // where 为空
  auto txn_mgr = exec_ctx_->GetTransactionManager();
  auto txn = exec_ctx_->GetTransaction();
  if (expr == nullptr) {
    if (iter_ == nullptr) {
      iter_ = new BPlusTreeIndexIteratorForTwoIntegerColumn(b_plus_tree_index_->GetBeginIterator());
    }
    while (!iter_->IsEnd()) {
      // 获取到key和rid，通过rid获取对应tuple
      auto [key_new, rid_new] = **iter_;
      auto [tupleMeta, tup_new] = table_info_->table_->GetTuple(rid_new);
      if (tupleMeta.is_deleted_) {
        ++(*iter_);
        continue;
      }
      *tuple = tup_new;
      *rid = rid_new;
      ++(*iter_);
      return true;
    }
    return false;
  }

  // 等于,点查找
  if (executed_) {
    return false;
  }

  // begin added by zhangyu at 2026/3/17 for P4T4
  while (key_index_ < static_cast<int>(pred_keys_.size())) {
    auto value = dynamic_cast<const ConstantValueExpression *>(pred_keys_[key_index_].get())->val_;
    std::vector<RID> result{};
    std::vector<Value> values{value};
    Tuple tup_key(values, b_plus_tree_index_->GetKeySchema());
    // fmt::println("the tuple in index scan: {}", tup_key.ToString(b_plus_tree_index_->GetKeySchema()));
    b_plus_tree_index_->ScanKey(tup_key, &result, nullptr);

    // 没找到，如果还有pred_key,继续执行
    if (result.empty()) {
      ++key_index_;
      continue;
    }

    auto [tupleMeta, tup_new] = table_info_->table_->GetTuple(result[0]);
    /* P4T4 at 2026/1/27 zhangyu: index scan判断是否需要undo */
    auto undo_logs_opt =
        CollectUndoLogs(result[0], tupleMeta, tup_new, txn_mgr->GetUndoLink(tup_new.GetRid()), txn, txn_mgr);

    /*
     * 如果undo_logs_opt无值表示这个元组不可见了，直接continue
     * 如果undo_logs_opt有值
     * 1. undolog不为空，重构日志
     *   1) 重构后有tuple，使用重构后的tuple
     *   2) 重构后无tuple，证明这个元组被删除了，continue
     * 2. undolog为空，直接使用当前元组即可
     */
    if (!undo_logs_opt.has_value()) {
      ++key_index_;
      continue;
    }
    auto undo_logs = undo_logs_opt.value();

    if (!undo_logs.empty()) {
      auto tuple_opt = ReconstructTuple(&GetOutputSchema(), tup_new, tupleMeta, undo_logs_opt.value());
      /* 如果没有返回tuple，则表明这个元组被删除了 */
      if (!tuple_opt.has_value()) {
        ++key_index_;
        continue;
      }
      tup_new = tuple_opt.value();
    } else {
      if (tupleMeta.is_deleted_) {
        ++key_index_;
        continue;
      }
    }
    // end added by zhangyu at 2026/3/17 for P4T4
    *tuple = tup_new;
    *rid = result[0];
    ++key_index_;
    return true;
  }

  executed_ = true;
  return false;
}
// end: Mod by zhangyu for p3t1 at 2025/11/3

}  // namespace bustub
