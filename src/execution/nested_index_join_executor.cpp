//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// nested_index_join_executor.cpp
//
// Identification: src/execution/nested_index_join_executor.cpp
//
// Copyright (c) 2015-19, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/nested_index_join_executor.h"
#include "execution/expressions/constant_value_expression.h"
#include "type/value_factory.h"

namespace bustub {

// begin: mod by zhangyu for p3t2 at 2025/12/3
NestIndexJoinExecutor::NestIndexJoinExecutor(ExecutorContext *exec_ctx, const NestedIndexJoinPlanNode *plan,
                                             std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx) {
  if (!(plan->GetJoinType() == JoinType::LEFT || plan->GetJoinType() == JoinType::INNER)) {
    // Note for 2023 Spring: You ONLY need to implement left join and inner join.
    throw bustub::NotImplementedException(fmt::format("join type {} not supported", plan->GetJoinType()));
  }
  plan_ = plan;
  child_executor_ = std::move(child_executor);
}

void NestIndexJoinExecutor::Init() {
  auto catalog = exec_ctx_->GetCatalog();
  auto index_info = catalog->GetIndex(plan_->index_oid_);
  const auto &index = index_info->index_;
  b_plus_tree_index_ = dynamic_cast<BPlusTreeIndexForTwoIntegerColumn *>(index.get());
  table_info_ = catalog->GetTable(plan_->inner_table_oid_).get();
  // The predicate to be used to extract the join key from the child
  key_predicate_ = plan_->KeyPredicate();
  child_executor_->Init();
}

auto NestIndexJoinExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  Tuple child_tuple{};
  RID child_rid{};
  while (child_executor_->Next(&child_tuple, &child_rid)) {
    auto value = key_predicate_->Evaluate(&child_tuple, child_executor_->GetOutputSchema());
    std::vector<RID> result{};
    std::vector<Value> values{value};
    Tuple tup_key(values, b_plus_tree_index_->GetKeySchema());
    b_plus_tree_index_->ScanKey(tup_key, &result, nullptr);

    // 先收集左边的tuple的values
    values.clear();
    for (size_t i = 0; i < child_executor_->GetOutputSchema().GetColumnCount(); ++i) {
      values.push_back(child_tuple.GetValue(&child_executor_->GetOutputSchema(), i));
    }
    // 没找到，则拼接为null
    if (result.empty()) {
      if (plan_->GetJoinType() == JoinType::LEFT) {
        for (size_t i = 0; i < plan_->InnerTableSchema().GetColumnCount(); ++i) {
          values.emplace_back(ValueFactory::GetNullValueByType(TypeId::INTEGER));
        }
        Tuple res(values, &GetOutputSchema());
        *tuple = res;
        return true;
      }
      continue;
    }

    auto [tupleMeta, tup_index] = table_info_->table_->GetTuple(result[0]);
    for (size_t i = 0; i < plan_->InnerTableSchema().GetColumnCount(); ++i) {
      values.push_back(tup_index.GetValue(&plan_->InnerTableSchema(), i));
    }

    Tuple res(values, &GetOutputSchema());
    *tuple = res;
    return true;
  }
  return false;
}

// end: mod by zhangyu for p3t2 at 2025/12/3
}  // namespace bustub
