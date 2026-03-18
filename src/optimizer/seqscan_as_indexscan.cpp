#include <memory>
#include <vector>
#include "concurrency/transaction.h"
#include "execution/expressions/abstract_expression.h"
#include "execution/expressions/column_value_expression.h"
#include "execution/expressions/comparison_expression.h"
#include "execution/expressions/constant_value_expression.h"
#include "execution/expressions/logic_expression.h"
#include "execution/plans/index_scan_plan.h"
#include "execution/plans/seq_scan_plan.h"
#include "optimizer/optimizer.h"

namespace bustub {

// begin: mod by zhangyu for p3t1 at 2025/11/3
auto Optimizer::OptimizeGetColValue(const AbstractExpression *expr, std::vector<AbstractExpressionRef> &pred_keys)
    -> std::vector<const ColumnValueExpression *> {
  std::vector<const ColumnValueExpression *> column_exprs;
  if (const auto *logic_expr = dynamic_cast<const LogicExpression *>(expr);
      logic_expr != nullptr && logic_expr->logic_type_ == LogicType::Or) {
    const auto *left_expr = logic_expr->GetChildAt(0).get();
    const auto *right_expr = logic_expr->GetChildAt(1).get();

    auto left_values = OptimizeGetColValue(left_expr, pred_keys);
    auto right_values = OptimizeGetColValue(right_expr, pred_keys);

    if (!left_values.empty() && !right_values.empty()) {
      column_exprs.insert(column_exprs.end(), left_values.begin(), left_values.end());
      column_exprs.insert(column_exprs.end(), right_values.begin(), right_values.end());
    }
  } else if (const auto *comp_expr = dynamic_cast<const ComparisonExpression *>(expr);
             comp_expr != nullptr &&
             (comp_expr->comp_type_ == ComparisonType::Equal || comp_expr->comp_type_ == ComparisonType::NotEqual)) {
    // check if this is a single equality comparison (column = constant or constant = column)
    const auto left_expr = comp_expr->GetChildAt(0);
    const auto right_expr = comp_expr->GetChildAt(1);

    if (const auto *left_col = dynamic_cast<const ColumnValueExpression *>(left_expr.get()); left_col != nullptr) {
      if (const auto *right_const = dynamic_cast<const ConstantValueExpression *>(right_expr.get());
          right_const != nullptr) {
        column_exprs.push_back(left_col);
        pred_keys.push_back(right_expr);
      }
    } else if (const auto *right_col = dynamic_cast<const ColumnValueExpression *>(right_expr.get());
               right_col != nullptr) {
      if (const auto *left_const = dynamic_cast<const ConstantValueExpression *>(left_expr.get());
          left_const != nullptr) {
        column_exprs.push_back(right_col);
        pred_keys.push_back(left_expr);
      }
    }
  }
  return column_exprs;
}
auto Optimizer::OptimizeSeqScanAsIndexScan(const bustub::AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  // TODO(student): implement seq scan with predicate -> index scan optimizer rule
  // The Filter Predicate Pushdown has been enabled for you in optimizer.cpp when forcing starter rule
  // 递归的优化子节点
  std::vector<AbstractPlanNodeRef> children;
  for (const auto &child : plan->GetChildren()) {
    children.emplace_back(OptimizeSeqScanAsIndexScan(child));
  }

  auto optimized_plan = plan->CloneWithChildren(std::move(children));
  if (optimized_plan->GetType() == PlanType::SeqScan) {
    // 此时
    const auto &seq_plan = dynamic_cast<const SeqScanPlanNode &>(*optimized_plan);
    // BUSTUB_ASSERT(optimized_plan->children_.size() == 1, "must have exactly one children");
    // const auto &child_plan = *optimized_plan->children_[0];
    if (seq_plan.filter_predicate_ != nullptr) {
      // const auto &seq_scan_plan = dynamic_cast<const SeqScanPlanNode &>(child_plan);
      auto filter_expr = seq_plan.filter_predicate_.get();
      std::vector<AbstractExpressionRef> pred_keys;
      auto column_exprs = OptimizeGetColValue(filter_expr, pred_keys);

      // 为空，则没有
      if (column_exprs.empty()) {
        return optimized_plan;
      }
      std::string col_name;
      for (auto expr : column_exprs) {
        auto col_val = dynamic_cast<const ColumnValueExpression *>(expr);
        if (col_name.empty()) {
          col_name = col_val->GetReturnType().GetName();
          continue;
        }
        // 存在多个索引列，则依然采用seqscan
        if (col_name != col_val->GetReturnType().GetName()) {
          return optimized_plan;
        }
      }
      // 通过matchIndex判断某列是否有索引
      auto expr = column_exprs[0];
      if (auto index = MatchIndex(seq_plan.table_name_, expr->GetColIdx()); index != std::nullopt) {
        auto [index_oid, index_name] = *index;
        std::cout << index_name << std::endl;
        return std::make_shared<IndexScanPlanNode>(seq_plan.output_schema_, seq_plan.table_oid_, index_oid,
                                                   seq_plan.filter_predicate_, pred_keys);
      }
    }
  }
  return optimized_plan;
}
// end: mod by zhangyu for p3t1 at 2025/11/3

}  // namespace bustub
