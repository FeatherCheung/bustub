#include <algorithm>
#include <memory>
#include "binder/table_ref/bound_join_ref.h"
#include "catalog/column.h"
#include "catalog/schema.h"
#include "common/exception.h"
#include "common/macros.h"
#include "execution/expressions/column_value_expression.h"
#include "execution/expressions/comparison_expression.h"
#include "execution/expressions/constant_value_expression.h"
#include "execution/expressions/logic_expression.h"
#include "execution/plans/abstract_plan.h"
#include "execution/plans/filter_plan.h"
#include "execution/plans/hash_join_plan.h"
#include "execution/plans/nested_loop_join_plan.h"
#include "execution/plans/projection_plan.h"
#include "optimizer/optimizer.h"
#include "type/type_id.h"

namespace bustub {

// begin: mod by zhangyu for p3t3 at 2025/12/4
auto IsExprAnd(const AbstractExpression *expr) -> bool {
  const auto *logic_expr = dynamic_cast<const LogicExpression *>(expr);
  return logic_expr != nullptr && logic_expr->logic_type_ == LogicType::And;
}

auto IsExprEqual(const AbstractExpression *expr) -> bool {
  const auto *comp_expr = dynamic_cast<const ComparisonExpression *>(expr);
  return comp_expr != nullptr && comp_expr->comp_type_ == ComparisonType::Equal;
}

auto GetHashKeys(const AbstractExpression *expr, std::vector<AbstractExpressionRef> &left_key_expressions,
                 std::vector<AbstractExpressionRef> &right_key_expressions) -> bool {
  // 继续迭代下去
  if (IsExprAnd(expr)) {
    const auto *left_expr = expr->GetChildAt(0).get();
    const auto *right_expr = expr->GetChildAt(1).get();
    return GetHashKeys(left_expr, left_key_expressions, right_key_expressions) &&
           GetHashKeys(right_expr, left_key_expressions, right_key_expressions);
  }
  // 如果是等号，收集key
  if (IsExprEqual(expr)) {
    auto left_expr = dynamic_cast<const ColumnValueExpression *>(expr->GetChildAt(0).get());
    auto right_expr = dynamic_cast<const ColumnValueExpression *>(expr->GetChildAt(1).get());
    // tuple_idx {tuple index 0 = left side of join, tuple index 1 = right side of join}
    if (left_expr->GetTupleIdx() == 0 && right_expr->GetTupleIdx() == 1) {
      left_key_expressions.push_back(expr->GetChildAt(0));
      right_key_expressions.push_back(expr->GetChildAt(1));
    }
    if (left_expr->GetTupleIdx() == 1 && right_expr->GetTupleIdx() == 0) {
      left_key_expressions.push_back(expr->GetChildAt(1));
      right_key_expressions.push_back(expr->GetChildAt(0));
    }
    return true;
  }
  return false;
}

auto Optimizer::OptimizeNLJAsHashJoin(const AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  // TODO(student): implement NestedLoopJoin -> HashJoin optimizer rule
  // Note for 2023 Fall: You should support join keys of any number of conjunction of equi-conditions:
  // E.g. <column expr> = <column expr> AND <column expr> = <column expr> AND ...
  std::vector<AbstractPlanNodeRef> children;
  for (const auto &child : plan->GetChildren()) {
    children.emplace_back(OptimizeNLJAsHashJoin(child));
  }
  auto optimized_plan = plan->CloneWithChildren(std::move(children));

  if (optimized_plan->GetType() == PlanType::NestedLoopJoin) {
    const auto &nlj_plan = dynamic_cast<const NestedLoopJoinPlanNode &>(*optimized_plan);
    // Has exactly two children
    BUSTUB_ENSURE(nlj_plan.children_.size() == 2, "NLJ should have exactly 2 children.");

    // 只适用于左连接和内连接
    if (nlj_plan.GetJoinType() != JoinType::INNER && nlj_plan.GetJoinType() != JoinType::LEFT) {
      return optimized_plan;
    }
    // 判断是不是AND
    auto expr = nlj_plan.Predicate().get();

    std::vector<AbstractExpressionRef> left_key_expressions;
    std::vector<AbstractExpressionRef> right_key_expressions;

    if (!(GetHashKeys(expr, left_key_expressions, right_key_expressions))) {
      return optimized_plan;
    }

    return std::make_shared<HashJoinPlanNode>(nlj_plan.output_schema_, nlj_plan.GetLeftPlan(), nlj_plan.GetRightPlan(),
                                              left_key_expressions, right_key_expressions, nlj_plan.GetJoinType());
  }

  return optimized_plan;
}
// end: mod by zhangyu for p3t3 at 2025/12/4

}  // namespace bustub
