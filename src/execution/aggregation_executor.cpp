//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// aggregation_executor.cpp
//
// Identification: src/execution/aggregation_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "execution/executors/aggregation_executor.h"
#include "execution/executors/delete_executor.h"
#include "execution/plans/aggregation_plan.h"
#include "storage/table/tuple.h"

namespace bustub {

// begin: mod by zhangyu for p3t2 at 2025/11/11
AggregationExecutor::AggregationExecutor(ExecutorContext *exec_ctx, const AggregationPlanNode *plan,
                                         std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx) {
  plan_ = plan;
  child_executor_ = std::move(child_executor);
}

AggregationExecutor::~AggregationExecutor() {
  delete aht_;
  delete aht_iterator_;
}

void AggregationExecutor::Init() {
  delete aht_;
  aht_ = new SimpleAggregationHashTable(plan_->GetAggregates(), plan_->GetAggregateTypes());
  Tuple tuple{};
  RID rid;

  child_executor_->Init();
  while (child_executor_->Next(&tuple, &rid)) {
    aht_->InsertCombine(MakeAggregateKey(&tuple), MakeAggregateValue(&tuple));
    ++tuple_size_;
  }
  if (tuple_size_ == 0) {
    aht_->Initial({{}});
  }
  delete aht_iterator_;
  aht_iterator_ = new SimpleAggregationHashTable::Iterator(aht_->Begin());
}

auto AggregationExecutor::Next(Tuple *tuple, RID *rid) -> bool {
  while (*aht_iterator_ != aht_->End()) {
    auto values = aht_iterator_->Val().aggregates_;
    auto key_values = aht_iterator_->Key().group_bys_;

    // 有group by 且表为空时，返回空
    if (!plan_->group_bys_.empty() && tuple_size_ == 0) {
      return false;
    }
    values.insert(values.begin(), key_values.begin(), key_values.end());
    Tuple tup_res(values, &GetOutputSchema());
    *tuple = tup_res;
    ++(*aht_iterator_);
    return true;
  }
  return false;
}
// end: mod by zhangyu for p3t2 at 2025/11/11

auto AggregationExecutor::GetChildExecutor() const -> const AbstractExecutor * { return child_executor_.get(); }

}  // namespace bustub
