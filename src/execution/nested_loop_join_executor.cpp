//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// nested_loop_join_executor.cpp
//
// Identification: src/execution/nested_loop_join_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/nested_loop_join_executor.h"
#include "binder/table_ref/bound_join_ref.h"
#include "common/exception.h"
#include "common/rid.h"
#include "type/value_factory.h"

namespace bustub {

// begin: mod by zhangyu for p3t2 at 2025/11/13
NestedLoopJoinExecutor::NestedLoopJoinExecutor(ExecutorContext *exec_ctx, const NestedLoopJoinPlanNode *plan,
                                               std::unique_ptr<AbstractExecutor> &&left_executor,
                                               std::unique_ptr<AbstractExecutor> &&right_executor)
    : AbstractExecutor(exec_ctx) {
  if (!(plan->GetJoinType() == JoinType::LEFT || plan->GetJoinType() == JoinType::INNER)) {
    // Note for 2023 Fall: You ONLY need to implement left join and inner join.
    throw bustub::NotImplementedException(fmt::format("join type {} not supported", plan->GetJoinType()));
  }
  plan_ = plan;
  left_executor_ = std::move(left_executor);
  right_executor_ = std::move(right_executor);
}

void NestedLoopJoinExecutor::Init() {
  predicate_ = plan_->Predicate();
  left_plan_ = plan_->GetLeftPlan();
  right_plan_ = plan_->GetRightPlan();

  left_executor_->Init();
  right_executor_->Init();
  has_left_tuple_ = false;
  matched_ = false;
}

auto NestedLoopJoinExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  while (true) {
    // 如果当前没有左 tuple，就从左执行器获取一个
    if (!has_left_tuple_) {
      if (!left_executor_->Next(&left_tuple_, &left_rid_)) {
        return false;  // 左表也结束，整个 JOIN 完成
      }
      // 新左 tuple 需要右执行器重新 Init
      right_executor_->Init();
      matched_ = false;
      has_left_tuple_ = true;
    }

    // 遍历右表
    Tuple right_tuple;
    RID right_rid;
    while (right_executor_->Next(&right_tuple, &right_rid)) {
      auto value =
          predicate_->EvaluateJoin(&left_tuple_, left_plan_->OutputSchema(), &right_tuple, right_plan_->OutputSchema());
      if (!value.IsNull() && value.GetAs<bool>()) {
        matched_ = true;

        std::vector<Value> output;
        // 左
        for (size_t i = 0; i < left_plan_->OutputSchema().GetColumnCount(); i++) {
          output.push_back(left_tuple_.GetValue(&left_plan_->OutputSchema(), i));
        }
        // 右
        for (size_t i = 0; i < right_plan_->OutputSchema().GetColumnCount(); i++) {
          output.push_back(right_tuple.GetValue(&right_plan_->OutputSchema(), i));
        }

        *tuple = Tuple(output, &GetOutputSchema());
        return true;  // 注意：右表仍然停在当前位置，Next() 会接着来！
      }
    }

    // 右表遍历完，还没匹配
    if (!matched_ && plan_->GetJoinType() == JoinType::LEFT) {
      std::vector<Value> output;
      for (size_t i = 0; i < left_plan_->OutputSchema().GetColumnCount(); i++) {
        output.push_back(left_tuple_.GetValue(&left_plan_->OutputSchema(), i));
      }
      for (size_t i = 0; i < right_plan_->OutputSchema().GetColumnCount(); i++) {
        output.push_back(ValueFactory::GetNullValueByType(right_plan_->OutputSchema().GetColumn(i).GetType()));
      }
      has_left_tuple_ = false;  // 准备处理下一行左表
      matched_ = false;
      *tuple = Tuple(output, &GetOutputSchema());
      return true;
    }

    // 当前左 tuple 已经处理完，进入下一个
    has_left_tuple_ = false;
    matched_ = false;
  }
}

// end: mod by zhangyu for p3t2 at 2025/11/13

}  // namespace bustub
