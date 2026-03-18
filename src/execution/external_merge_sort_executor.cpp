//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// external_merge_sort_executor.cpp
//
// Identification: src/execution/external_merge_sort_executor.cpp
//
// Copyright (c) 2015-2024, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/external_merge_sort_executor.h"
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>
#include "buffer/buffer_pool_manager.h"
#include "common/config.h"
#include "execution/execution_common.h"
#include "execution/plans/sort_plan.h"
#include "storage/page/page.h"
#include "storage/page/page_guard.h"

namespace bustub {

// begin: mod by zhangyu for p3t4 at 2025/12/6
template <size_t K>
ExternalMergeSortExecutor<K>::ExternalMergeSortExecutor(ExecutorContext *exec_ctx, const SortPlanNode *plan,
                                                        std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx), cmp_(plan->GetOrderBy()) {
  plan_ = plan;
  child_executor_ = std::move(child_executor);
  bpm_ = exec_ctx->GetBufferPoolManager();
}

template <size_t K>
auto ExternalMergeSortExecutor<K>::WriteSortedBufferToNewSortPage(std::vector<SortEntry> &buffer)
    -> std::vector<page_id_t> {
  /* 先排序 */
  std::sort(buffer.begin(), buffer.end(), cmp_);
  std::vector<page_id_t> pages;
  page_id_t current_pid = bpm_->NewPage();
  WritePageGuard write_guard = bpm_->WritePage(current_pid);
  auto sortpage = write_guard.AsMut<SortPage>();
  sortpage->Init(tuple_size_);
  for (const auto &iter : buffer) {
    sortpage->InsertTuple(iter.second);
  }

  // 添加最后一个页面
  pages.emplace_back(current_pid);
  return pages;
}

template <size_t K>
auto ExternalMergeSortExecutor<K>::MergeTwoRuns(MergeSortRun &run1, MergeSortRun &run2) -> MergeSortRun {
  std::vector<page_id_t> pages{};
  MergeSortRun::Iterator iter1 = run1.Begin();
  MergeSortRun::Iterator iter2 = run2.Begin();
  page_id_t new_pid = bpm_->NewPage();
  WritePageGuard write_gurad = bpm_->WritePage(new_pid);
  auto sortpage = write_gurad.AsMut<SortPage>();
  sortpage->Init(tuple_size_);
  while (iter1 != run1.End() && iter2 != run2.End()) {
    SortKey t1 = GenerateSortKey(*iter1, plan_->GetOrderBy(), child_executor_->GetOutputSchema());
    SortKey t2 = GenerateSortKey(*iter2, plan_->GetOrderBy(), child_executor_->GetOutputSchema());
    if (cmp_({t1, *iter1}, {t2, *iter2})) {
      if (!sortpage->InsertTuple(*iter1)) {
        /* 如果page已经满了，那么需要保存这个pid并申请新的pid */
        pages.emplace_back(new_pid);
        new_pid = bpm_->NewPage();
        write_gurad = bpm_->WritePage(new_pid);
        sortpage = write_gurad.AsMut<SortPage>();
        sortpage->Init(tuple_size_);
        // 重新插入当前元组到新页面
        if (!sortpage->InsertTuple(*iter1)) {
          // 这不应该发生，因为新页面是空的
        }
      }
      ++iter1;
    } else {
      if (!sortpage->InsertTuple(*iter2)) {
        /* 如果page已经满了，那么需要保存这个pid并申请新的pid */
        pages.emplace_back(new_pid);
        new_pid = bpm_->NewPage();
        write_gurad = bpm_->WritePage(new_pid);
        sortpage = write_gurad.AsMut<SortPage>();
        sortpage->Init(tuple_size_);
        // 重新插入当前元组到新页面
        if (!sortpage->InsertTuple(*iter2)) {
          // 这不应该发生，因为新页面是空的
        }
      }
      ++iter2;
    }
  }
  while (iter1 != run1.End()) {
    if (!sortpage->InsertTuple(*iter1)) {
      /* 如果page已经满了，那么需要保存这个pid并申请新的pid */
      pages.emplace_back(new_pid);
      new_pid = bpm_->NewPage();
      write_gurad = bpm_->WritePage(new_pid);
      sortpage = write_gurad.AsMut<SortPage>();
      sortpage->Init(tuple_size_);
      // 重新插入当前元组到新页面
      if (!sortpage->InsertTuple(*iter1)) {
        // 这不应该发生，因为新页面是空的
      }
    }
    ++iter1;
  }
  while (iter2 != run2.End()) {
    if (!sortpage->InsertTuple(*iter2)) {
      /* 如果page已经满了，那么需要保存这个pid并申请新的pid */
      pages.emplace_back(new_pid);
      new_pid = bpm_->NewPage();
      write_gurad = bpm_->WritePage(new_pid);
      sortpage = write_gurad.AsMut<SortPage>();
      sortpage->Init(tuple_size_);
      // 重新插入当前元组到新页面
      if (!sortpage->InsertTuple(*iter2)) {
        // 这不应该发生，因为新页面是空的
      }
    }
    ++iter2;
  }
  // 添加最后一个页面
  pages.emplace_back(new_pid);
  run1.Clear();
  run2.Clear();
  return {pages, bpm_};
}

template <size_t K>
void ExternalMergeSortExecutor<K>::Init() {
  child_executor_->Init();
  free_size_ = BUSTUB_PAGE_SIZE - sizeof(int32_t) * 3;
  tuple_size_ = {0};
  sortrun_.clear();
  Tuple tuple{};
  RID rid{};
  std::vector<SortEntry> buffer{};

  while (child_executor_->Next(&tuple, &rid)) {
    if (tuple_size_ == 0) {
      tuple_size_ = tuple.GetLength();
    }
    if (free_size_ >= tuple_size_) {
      buffer.emplace_back(GenerateSortKey(tuple, plan_->GetOrderBy(), child_executor_->GetOutputSchema()), tuple);
      free_size_ -= tuple_size_;
      continue;
    }
    auto pages = WriteSortedBufferToNewSortPage(buffer);
    sortrun_.emplace_back(MergeSortRun{pages, bpm_});
    buffer.clear();
    free_size_ = BUSTUB_PAGE_SIZE - sizeof(int32_t) * 3;
    // 将当前元组添加到新缓冲区
    buffer.emplace_back(GenerateSortKey(tuple, plan_->GetOrderBy(), child_executor_->GetOutputSchema()), tuple);
    free_size_ -= tuple_size_;
  }

  if (!buffer.empty()) {
    auto pages = WriteSortedBufferToNewSortPage(buffer);
    sortrun_.emplace_back(MergeSortRun{pages, bpm_});
    buffer.clear();
    free_size_ = BUSTUB_PAGE_SIZE - sizeof(int32_t) * 3;
  }

  /* 多次执行二路归并，直到只剩一个run */
  while (sortrun_.size() > 1) {
    std::vector<MergeSortRun> next_sortrun;
    for (size_t i = 0; i + 1 < sortrun_.size(); i += K) {
      /* 二路归并两个 run */
      MergeSortRun &run1 = sortrun_[i];
      MergeSortRun &run2 = sortrun_[i + 1];

      MergeSortRun merged = MergeTwoRuns(run1, run2);

      next_sortrun.emplace_back(std::move(merged));
    }

    // 如果 run 数量是奇数，把最后一个直接加进去
    if (sortrun_.size() % 2 == 1) {
      next_sortrun.emplace_back(std::move(sortrun_.back()));
    }
    for (auto &run : sortrun_) {
      run.Clear();
    }
    sortrun_ = std::move(next_sortrun);
  }
  if (!sortrun_.empty()) {
    cur_iter_ = sortrun_[0].Begin();
    end_iter_ = sortrun_[0].End();
  }
}

template <size_t K>
auto ExternalMergeSortExecutor<K>::Next(Tuple *tuple, RID *rid) -> bool {
  if (sortrun_.empty()) {
    return false;
  }

  while (cur_iter_ != end_iter_) {
    *tuple = *cur_iter_;
    ++cur_iter_;
    return true;
  }
  return false;
}

template class ExternalMergeSortExecutor<2>;

// end: mod by zhangyu for p3t4 at 2025/12/6
}  // namespace bustub
