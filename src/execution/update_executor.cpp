//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// update_executor.cpp
//
// Identification: src/execution/update_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//
#include <memory>
#include <optional>
#include <unordered_map>

#include "concurrency/transaction_manager.h"
#include "execution/execution_common.h"
#include "execution/executor_factory.h"
#include "execution/executors/seq_scan_executor.h"
#include "execution/executors/update_executor.h"
#include "execution/executors/values_executor.h"
#include "execution/plans/seq_scan_plan.h"
#include "storage/table/tuple.h"

namespace bustub {

// begin added by zhangyu at 2025/10/27 for p3t1
UpdateExecutor::UpdateExecutor(ExecutorContext *exec_ctx, const UpdatePlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), plan_(plan) {
  // As of Fall 2022, you DON'T need to implement update executor to have perfect score in project 3 / project 4.
  child_executor_ = std::move(child_executor);
}

void UpdateExecutor::Init() {
  auto catalog = exec_ctx_->GetCatalog();
  table_info_ = catalog->GetTable(plan_->GetTableOid()).get();
  child_executor_->Init();
}

// begin added by zhangyu at 2026/3/17 for P4T4
auto UpdateExecutor::UpdateTuple(Tuple &new_tup, TupleMeta &new_tupmeta, Tuple &old_tup, TupleMeta &old_tupmeta,
                                 std::optional<UndoLink> undo_link_opt, RID &old_rid) -> bool {
  auto txn = exec_ctx_->GetTransaction();
  auto txn_mgr = exec_ctx_->GetTransactionManager();

  /* write-write confliction check */
  if (CheckWriteConflict(&old_tupmeta, txn)) {
    txn->SetTainted();
    throw ExecutionException("write -write conflict in update 1!");
  }

  auto cur_undolink = undo_link_opt.value();

  auto new_undolink_opt =
      GenerateUndoLink(txn_mgr, txn, cur_undolink, &old_tup, &new_tup, old_tupmeta, new_tupmeta, &table_info_->schema_);

  new_tupmeta.ts_ = txn->GetTransactionTempTs();
  new_tup.SetRid(old_tup.GetRid());
  if (new_undolink_opt.has_value()) {
    UpdateTupleAndUndoLink(exec_ctx_->GetTransactionManager(), old_rid, new_undolink_opt.value(),
                           table_info_->table_.get(), txn, new_tupmeta, new_tup,
                           [new_tup, txn](const TupleMeta &tup_meta, const Tuple &tuple, RID rid,
                                          std::optional<UndoLink> undolink_opt) -> bool {
                             // check 函数进行write-write 判断，并验证是否可以delete
                             if (CheckWriteConflict(&tup_meta, txn)) {
                               // fmt::println("failed update tuple 2: {}", new_tup.ToString(schema));
                               txn->SetTainted();
                               throw ExecutionException("write -write conflict in update 2!");
                               return false;
                             }
                             return true;
                           });
  } else {
    auto ret = table_info_->table_->UpdateTupleInPlace(new_tupmeta, new_tup, old_rid);
    if (!ret) {
      txn->SetTainted();
      throw ExecutionException("update failed! 2");
    }
  }

  // 记录写集（commit 用）
  txn->AppendWriteSet(table_info_->oid_, old_rid);
  return true;
}

// 判断是否两个tuple的primary key相同，相同则返回true，不同则返回false
auto UpdateExecutor::PkCompare(const Tuple &new_tup, const Tuple &old_tup) -> bool {
  auto catalog = exec_ctx_->GetCatalog();
  auto indexes_info = catalog->GetTableIndexes(table_info_->name_);
  auto index_info = indexes_info.empty() ? nullptr : indexes_info[0];
  if (index_info == nullptr || !index_info->is_primary_key_) {
    return true;
  }
  auto index = index_info->index_.get();
  auto old_key = old_tup.KeyFromTuple(table_info_->schema_, *index->GetKeySchema(), index->GetKeyAttrs());
  auto new_key = new_tup.KeyFromTuple(table_info_->schema_, *index->GetKeySchema(), index->GetKeyAttrs());
  return old_key.GetValue(index->GetKeySchema(), 0).CompareEquals(new_key.GetValue(index->GetKeySchema(), 0)) ==
         CmpBool::CmpTrue;
}
// end added by zhangyu at 2026/3/17 for P4T4

auto UpdateExecutor::Next([[maybe_unused]] Tuple *tuple, RID *rid) -> bool {
  if (executed_) {
    return false;
  }
  Tuple old_tup;
  RID old_rid;
  int update_num = 0;
  auto catalog = exec_ctx_->GetCatalog();
  auto indexes_info = catalog->GetTableIndexes(table_info_->name_);
  // begin modified by zhangyu at 2026/3/17 for P4T4
  auto txn = exec_ctx_->GetTransaction();
  auto index_info = indexes_info.empty() ? nullptr : indexes_info[0];
  std::vector<Tuple> old_tups;
  std::vector<Tuple> new_tups;
  while (child_executor_->Next(&old_tup, &old_rid)) {
    std::vector<Value> values{};
    // 新的tuple应该和原表的schema一致
    values.reserve(table_info_->schema_.GetColumnCount());
    for (uint32_t i = 0; i < table_info_->schema_.GetColumnCount(); i++) {
      // 如果该列有 update 表达式，使用新值，否则用原值
      auto expr = plan_->target_expressions_[i];
      if (expr != nullptr) {
        values.push_back(expr->Evaluate(&old_tup, table_info_->schema_));
      } else {
        values.push_back(old_tup.GetValue(&table_info_->schema_, i));
      }
    }
    Tuple new_tup = Tuple(values, &table_info_->schema_);
    auto [old_tupmeta, old_tup, undo_link_opt] =
        GetTupleAndUndoLink(exec_ctx_->GetTransactionManager(), table_info_->table_.get(), old_rid);

    // 判断是否是主键更新
    if (!PkCompare(new_tup, old_tup)) {
      if (!new_tups.empty()) {
        // 防止{1, 1, 1}的情况发生，违背主键唯一约束
        for (auto &tup : new_tups) {
          if (PkCompare(new_tup, tup)) {
            txn->SetTainted();
            throw ExecutionException("update failed in checking unique primary key!");
          }
        }
      }

      new_tups.push_back(new_tup);
      old_tups.push_back(old_tup);
      continue;
    }
    // 不是主键更新，直接更新即可
    new_tups.erase(new_tups.begin(), new_tups.end());
    auto new_tupmeta = old_tupmeta;
    UpdateTuple(new_tup, new_tupmeta, old_tup, old_tupmeta, undo_link_opt, old_rid);
    ++update_num;
  }
  if (!new_tups.empty()) {
    std::vector<RID> new_rids;
    auto index = index_info->index_.get();
    auto b_plus_tree_index = dynamic_cast<BPlusTreeIndexForTwoIntegerColumn *>(index);

    // 标记删除所有需要更新的old_tups
    for (auto &old_tup : old_tups) {
      old_rid = old_tup.GetRid();
      auto [old_tupmeta, _, undo_link_opt] =
          GetTupleAndUndoLink(exec_ctx_->GetTransactionManager(), table_info_->table_.get(), old_rid);
      Tuple new_tup = old_tup;
      TupleMeta new_tupmeta = {txn->GetTransactionTempTs(), true};

      auto old_key =
          old_tup.KeyFromTuple(table_info_->schema_, *index->GetKeySchema(), index_info->index_->GetKeyAttrs());

      /*
       * 如果new_pk中存在和old_pk重合的，那么说明这个key被占用，此时old不应该被删除，
       * 否则不同事务看到的index会出问题
       */
      bool is_old_pk = false;
      for (auto &new_tup : new_tups) {
        if (PkCompare(new_tup, old_tup)) {
          is_old_pk = true;
          break;
        }
      }

      if (!is_old_pk) {
        // 只标记删除tup，不删除index entry
        UpdateTuple(new_tup, new_tupmeta, old_tup, old_tupmeta, undo_link_opt, old_rid);
        continue;
      }
    }
    /*
     * 插入新的new_tup，如果new_pk已经存在，那么说明这个key已经被占用
     * 根据bustub的设计，key一旦被插入，key -> rid就被绑定rid不可以被修改，而是执行update的操作
     */
    for (auto &new_tup : new_tups) {
      auto new_pk = new_tup.KeyFromTuple(table_info_->schema_, *index->GetKeySchema(), index->GetKeyAttrs());
      std::vector<RID> result{};
      b_plus_tree_index->ScanKey(new_pk, &result, nullptr);

      // 没找到，直接可以插入
      if (result.empty()) {
        TupleMeta tupmeta{};
        tupmeta.is_deleted_ = false;
        tupmeta.ts_ = txn->GetTransactionTempTs();
        RID new_rid = table_info_->table_->InsertTuple(tupmeta, new_tup).value();
        UndoLink link;
        exec_ctx_->GetTransactionManager()->UpdateUndoLink(new_rid, link, nullptr);
        b_plus_tree_index->InsertEntry(new_pk, new_rid, nullptr);
        txn->AppendWriteSet(table_info_->oid_, new_rid);
        continue;
      }
      // 如果这个key已经存在，做一个版本控制
      TupleMeta tupmeta{};
      tupmeta.is_deleted_ = false;
      tupmeta.ts_ = txn->GetTransactionTempTs();
      auto overlap_old_rid = result[0];
      auto [overlap_old_tupmeta, ovlap_old_tup, undo_link_opt] =
          GetTupleAndUndoLink(exec_ctx_->GetTransactionManager(), table_info_->table_.get(), overlap_old_rid);
      UpdateTuple(new_tup, tupmeta, ovlap_old_tup, overlap_old_tupmeta, undo_link_opt, overlap_old_rid);
      ++update_num;
      // end modified by zhangyu at 2026/3/17 for P4T4
    }
  }
  // 没有需要更新的了，结束即可
  Tuple tup(*rid, reinterpret_cast<const char *>(&update_num), sizeof(update_num));
  *tuple = tup;
  executed_ = true;
  return true;
}
// end added by zhangyu at 2025/10/27 for p3t1

}  // namespace bustub
