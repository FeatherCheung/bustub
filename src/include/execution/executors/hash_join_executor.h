//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// hash_join_executor.h
//
// Identification: src/include/execution/executors/hash_join_executor.h
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <utility>

#include "execution/executor_context.h"
#include "execution/executors/abstract_executor.h"
#include "execution/executors/aggregation_executor.h"
#include "execution/plans/hash_join_plan.h"
#include "storage/table/tuple.h"

// begin: mod by zhangyu for p3t3 at 2025/12/5
namespace bustub {
/* HashJoinKey 结构定义，仿照AggregateKey实现 */
struct HashJoinKey {
  /** The join key values */
  std::vector<Value> joinkey_values_;

  auto operator==(const HashJoinKey &other) const -> bool {
    for (uint32_t i = 0; i < other.joinkey_values_.size(); i++) {
      if (joinkey_values_[i].CompareEquals(other.joinkey_values_[i]) != CmpBool::CmpTrue) {
        return false;
      }
    }
    return true;
  }
};
};  // namespace bustub

namespace std {
/** 为HashJoinKey计算hash值 */
template <>
struct std::hash<bustub::HashJoinKey> {
  auto operator()(const bustub::HashJoinKey &hashjoin_key) const -> std::size_t {
    size_t curr_hash = 0;
    for (const auto &key : hashjoin_key.joinkey_values_) {
      if (!key.IsNull()) {
        curr_hash = bustub::HashUtil::CombineHashes(curr_hash, bustub::HashUtil::HashValue(&key));
      }
    }
    return curr_hash;
  }
};
};  // namespace std

namespace bustub {

class SimpleHashJoinHashTable {
 public:
  /* 插入key和tuple */
  void InsertKey(const HashJoinKey &join_key, const Tuple &tuple) {
    if (ht_.count(join_key) == 0) {
      std::vector<Tuple> tuples;
      tuples.push_back(tuple);
      ht_.insert({join_key, tuples});
    } else {
      ht_.at(join_key).push_back(tuple);
    }
  }

  auto GetValues(const HashJoinKey &join_key) -> std::vector<Tuple> * {
    if (ht_.find(join_key) == ht_.end()) {
      return nullptr;
    }
    return &(ht_.find(join_key)->second);
  }

  void Clear() { ht_.clear(); }

 private:
  /** hashjoin hash table */
  std::unordered_map<HashJoinKey, std::vector<Tuple>> ht_{};
};

/**
 * HashJoinExecutor executes a nested-loop JOIN on two tables.
 */
class HashJoinExecutor : public AbstractExecutor {
 public:
  /**
   * Construct a new HashJoinExecutor instance.
   * @param exec_ctx The executor context
   * @param plan The HashJoin join plan to be executed
   * @param left_child The child executor that produces tuples for the left side of join
   * @param right_child The child executor that produces tuples for the right side of join
   */
  HashJoinExecutor(ExecutorContext *exec_ctx, const HashJoinPlanNode *plan,
                   std::unique_ptr<AbstractExecutor> &&left_child, std::unique_ptr<AbstractExecutor> &&right_child);

  ~HashJoinExecutor() override;
  /** Initialize the join */
  void Init() override;

  /**
   * Yield the next tuple from the join.
   * @param[out] tuple The next tuple produced by the join.
   * @param[out] rid The next tuple RID, not used by hash join.
   * @return `true` if a tuple was produced, `false` if there are no more tuples.
   */
  auto Next(Tuple *tuple, RID *rid) -> bool override;

  /** @return The output schema for the join */
  auto GetOutputSchema() const -> const Schema & override { return plan_->OutputSchema(); };

  auto GetLeftJoinKey(const Tuple *tuple) -> HashJoinKey {
    std::vector<Value> joinkey_values;
    for (const auto &expr : left_key_expressions_) {
      joinkey_values.emplace_back(expr->Evaluate(tuple, left_executor_->GetOutputSchema()));
    }
    return HashJoinKey{joinkey_values};
  }

  auto GetRightJoinKey(const Tuple *tuple) -> HashJoinKey {
    std::vector<Value> joinkey_values;
    for (const auto &expr : right_key_expressions_) {
      joinkey_values.emplace_back(expr->Evaluate(tuple, right_executor_->GetOutputSchema()));
    }
    return HashJoinKey{joinkey_values};
  }

 private:
  /** The HashJoin plan node to be executed. */
  const HashJoinPlanNode *plan_;
  std::unique_ptr<AbstractExecutor> left_executor_;
  std::unique_ptr<AbstractExecutor> right_executor_;
  AbstractPlanNodeRef left_plan_;
  AbstractPlanNodeRef right_plan_;
  std::vector<AbstractExpressionRef> left_key_expressions_;
  std::vector<AbstractExpressionRef> right_key_expressions_;
  SimpleHashJoinHashTable *jht_{nullptr};
  bool has_left_tuple_{false};
  Tuple left_tuple_{};
  RID left_rid_{};
  std::vector<Tuple> *cur_match_list_ = nullptr;
  size_t match_idx_ = 0;
};

}  // namespace bustub
