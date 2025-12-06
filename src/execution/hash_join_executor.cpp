//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// hash_join_executor.cpp
//
// Identification: src/execution/hash_join_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/hash_join_executor.h"
#include "binder/table_ref/bound_join_ref.h"

namespace bustub {

// begin: mod by zhangyu for p3t3 at 2025/12/5
HashJoinExecutor::HashJoinExecutor(ExecutorContext *exec_ctx, const HashJoinPlanNode *plan,
                                   std::unique_ptr<AbstractExecutor> &&left_child,
                                   std::unique_ptr<AbstractExecutor> &&right_child)
    : AbstractExecutor(exec_ctx) {
  if (!(plan->GetJoinType() == JoinType::LEFT || plan->GetJoinType() == JoinType::INNER)) {
    // Note for Fall 2024: You ONLY need to implement left join and inner join.
    throw bustub::NotImplementedException(fmt::format("join type {} not supported", plan->GetJoinType()));
  }
  plan_ = plan;
  left_executor_ = std::move(left_child);
  right_executor_ = std::move(right_child);
  left_plan_ = plan->GetLeftPlan();
  right_plan_ = plan->GetRightPlan();
  left_key_expressions_ = plan->LeftJoinKeyExpressions();
  right_key_expressions_ = plan->RightJoinKeyExpressions();
}

HashJoinExecutor::~HashJoinExecutor() { delete jht_; }

void HashJoinExecutor::Init() {
  delete jht_;
  jht_ = new SimpleHashJoinHashTable();
  Tuple tuple{};
  RID rid;

  right_executor_->Init();
  left_executor_->Init();
  // 使用右表构造
  while (right_executor_->Next(&tuple, &rid)) {
    jht_->InsertKey(GetRightJoinKey(&tuple), tuple);
  }
}

auto HashJoinExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  while (true) {
    // 如果当前没有左tuple，就从左执行器获取一个
    if (!has_left_tuple_) {
      if (!left_executor_->Next(&left_tuple_, &left_rid_)) {
        return false;  // 左表也结束
      }
      has_left_tuple_ = true;
      match_idx_ = 0;

      auto hash_join_key = GetLeftJoinKey(&left_tuple_);
      cur_match_list_ = jht_->GetValues(hash_join_key);

      if ((cur_match_list_ == nullptr || cur_match_list_->empty()) && plan_->GetJoinType() == JoinType::LEFT) {
        std::vector<Value> output;
        /* 左边 */
        for (size_t i = 0; i < left_plan_->OutputSchema().GetColumnCount(); i++) {
          output.push_back(left_tuple_.GetValue(&left_plan_->OutputSchema(), i));
        }
        /* 右边补 null */
        for (size_t i = 0; i < right_plan_->OutputSchema().GetColumnCount(); i++) {
          output.push_back(ValueFactory::GetNullValueByType(right_plan_->OutputSchema().GetColumn(i).GetType()));
        }
        /* 输出 LEFT JOIN 无匹配行 */
        *tuple = Tuple(output, &GetOutputSchema());

        /* 准备下一个 left tuple */
        has_left_tuple_ = false;
        return true;
      }
      /* 若是 inner join，无匹配则继续下一 tuple */
      continue;
    }

    /*如果当前左 tuple 的匹配还没输出完 */
    if (cur_match_list_ != nullptr && match_idx_ < cur_match_list_->size()) {
      std::vector<Value> output;

      /* 左边 */
      for (size_t i = 0; i < left_plan_->OutputSchema().GetColumnCount(); i++) {
        output.push_back(left_tuple_.GetValue(&left_plan_->OutputSchema(), i));
      }
      /* 右边 */
      const Tuple &right_tuple = (*cur_match_list_)[match_idx_++];
      for (size_t i = 0; i < right_plan_->OutputSchema().GetColumnCount(); i++) {
        output.push_back(right_tuple.GetValue(&right_plan_->OutputSchema(), i));
      }

      *tuple = Tuple(output, &GetOutputSchema());
      return true;
    }

    /* 匹配全部输出完，准备下一个左 tuple */
    has_left_tuple_ = false;
  }
}
// end: mod by zhangyu for p3t3 at 2025/12/5

}  // namespace bustub
