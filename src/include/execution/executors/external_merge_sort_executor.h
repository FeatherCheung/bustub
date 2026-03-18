//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// external_merge_sort_executor.h
//
// Identification: src/include/execution/executors/external_merge_sort_executor.h
//
// Copyright (c) 2015-2024, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include "buffer/buffer_pool_manager.h"
#include "common/config.h"
#include "common/macros.h"
#include "execution/execution_common.h"
#include "execution/executors/abstract_executor.h"
#include "execution/plans/sort_plan.h"
#include "storage/table/tuple.h"

namespace bustub {

// begin: mod by zhangyu for p3t4 at 2025/12/6
/**
 * Page to hold the intermediate data for external merge sort.
 *
 * Only fixed-length data will be supported in Fall 2024.
 */
/* 无序列的tuple写入内存的sortpage，再排序，然后写回磁盘 */
/*
 * ----------------------------------------------
 * | Header |  tuple(1)  | ... ...|  tuple(n)  |
 *  ---------------------------------------------
 *  Header format (size in byte, 16 bytes in total):
 *  -----------------------------------------------
 * | free_size (4) | count_ (4) | tuple_size_ (4) |
 *  -----------------------------------------------
 */

class SortPage {
 public:
  /* bpm申请到内存后，只是将这块原始内存映射为sortpage，所以需要自己初始化！*/
  void Init(int tuple_size) {
    free_size_ = BUSTUB_PAGE_SIZE - sizeof(int32_t) * 3;
    count_ = 0;
    tuple_size_ = tuple_size;
    memset(tuples_, 0, BUSTUB_PAGE_SIZE - sizeof(int32_t) * 3);
  }
  /* 把 tuple 写进这个 SortPage, 如果放不下，返回 false。如果成功更新 meta 信息 */
  auto InsertTuple(const Tuple &tuple) -> bool {
    if (free_size_ >= tuple_size_) {
      memcpy(tuples_ + count_ * tuple_size_, tuple.GetData(), tuple_size_);
      count_++;
      free_size_ -= tuple_size_;
      return true;
    }
    return false;
  }

  /* 按 index 取 tuple，用于 Merge 阶段读取单页记录。*/
  auto GetTuple(int index) const -> Tuple {
    Tuple res({}, tuples_ + index * tuple_size_, tuple_size_);
    return res;
  }

  /* 返回tuple数目 */
  auto GetTupleCount() const -> size_t { return count_; }

 private:
  int32_t free_size_;
  int32_t count_;
  int32_t tuple_size_;
  char tuples_[BUSTUB_PAGE_SIZE - sizeof(int32_t) * 3];
};

/**
 * A data structure that holds the sorted tuples as a run during external merge sort.
 * Tuples might be stored in multiple pages, and tuples are ordered both within one page
 * and across pages.
 */
class MergeSortRun {
 public:
  MergeSortRun() = default;
  MergeSortRun(std::vector<page_id_t> pages, BufferPoolManager *bpm) : pages_(std::move(pages)), bpm_(bpm) {}

  auto GetPageCount() -> size_t { return pages_.size(); }

  void Clear() {
    for (auto pid : pages_) {
      bpm_->DeletePage(pid);
    }
    pages_.clear();
  }

  /** Iterator for iterating on the sorted tuples in one run. */
  class Iterator {
    friend class MergeSortRun;

   public:
    Iterator() = default;

    /**
     * Advance the iterator to the next tuple. If the current sort page is exhausted, move to the
     * next sort page.
     *
     * 迭代器指针下移
     */
    auto operator++() -> Iterator & {
      tuple_idx_++;
      if (tuple_idx_ < current_page_->GetTupleCount()) {
        return *this;
      }
      page_idx_++;
      tuple_idx_ = 0;
      /* 再下一个块中，需要加载下个page的内容到内存中 */
      if (page_idx_ < run_->pages_.size()) {
        auto next_page_id = run_->pages_[page_idx_];
        read_guard_ = run_->bpm_->ReadPage(next_page_id);
        this->current_page_ = read_guard_.As<SortPage>();
        current_pid_ = next_page_id;
      }
      return *this;
    }

    /**
     * Dereference the iterator to get the current tuple in the sorted run that the iterator is
     * pointing to.
     *
     * 获取tuple
     */
    auto operator*() -> Tuple { return current_page_->GetTuple(tuple_idx_); }

    /**
     * Checks whether two iterators are pointing to the same tuple in the same sorted run.
     * 指向同一个sortpage的同一个指针
     */
    auto operator==(const Iterator &other) const -> bool {
      return this->run_ == other.run_ && this->page_idx_ == other.page_idx_ && this->tuple_idx_ == other.tuple_idx_;
    }

    /**
     * Checks whether two iterators are pointing to different tuples in a sorted run or iterating
     * on different sorted runs.
     *
     * 比较两个迭代器是否相同
     */
    auto operator!=(const Iterator &other) const -> bool {
      return this->run_ != other.run_ || this->page_idx_ != other.page_idx_ || this->tuple_idx_ != other.tuple_idx_;
    }

   private:
    explicit Iterator(const MergeSortRun *run) : run_(run) {}

    /** The sorted run that the iterator is iterating on. */
    [[maybe_unused]] const MergeSortRun *run_;
    /* run中的page的索引 */
    size_t page_idx_{0};
    /* 当前页中tuple的索引 */
    size_t tuple_idx_{0};
    page_id_t current_pid_{0};
    const SortPage *current_page_{nullptr};
    ReadPageGuard read_guard_{};
  };

  /**
   * Get an iterator pointing to the beginning of the sorted run, i.e. the first tuple.
   *
   * 构造Iterator，并指向第一条tuple
   */
  auto Begin() -> Iterator {
    Iterator iter(this);
    if (!this->pages_.empty()) {
      auto first_page_id = iter.run_->pages_[0];
      iter.read_guard_ = iter.run_->bpm_->ReadPage(first_page_id);
      iter.current_page_ = iter.read_guard_.As<SortPage>();
    }
    iter.page_idx_ = 0;
    iter.tuple_idx_ = 0;
    return iter;
  }

  /**
   * Get an iterator pointing to the end of the sorted run, i.e. the position after the last tuple.
   *
   * 指向末尾的迭代器
   */
  auto End() -> Iterator {
    Iterator last_iter(this);
    last_iter.page_idx_ = this->pages_.size();
    last_iter.tuple_idx_ = 0;
    return last_iter;
  }

 private:
  /** The page IDs of the sort pages that store the sorted tuples. */
  std::vector<page_id_t> pages_;
  /**
   * The buffer pool manager used to read sort pages. The buffer pool manager is responsible for
   * deleting the sort pages when they are no longer needed.
   */
  [[maybe_unused]] BufferPoolManager *bpm_;
};

/**
 * ExternalMergeSortExecutor executes an external merge sort.
 *
 * In Fall 2024, only 2-way external merge sort is required.
 */
template <size_t K>
class ExternalMergeSortExecutor : public AbstractExecutor {
 public:
  ExternalMergeSortExecutor(ExecutorContext *exec_ctx, const SortPlanNode *plan,
                            std::unique_ptr<AbstractExecutor> &&child_executor);

  /* 执行结束后的中间page 也要删除 */
  ~ExternalMergeSortExecutor() override {
    for (auto &run : sortrun_) {
      run.Clear();
    }
  }

  /** Initialize the external merge sort */
  void Init() override;

  /**
   * Yield the next tuple from the external merge sort.
   * @param[out] tuple The next tuple produced by the external merge sort.
   * @param[out] rid The next tuple RID produced by the external merge sort.
   * @return `true` if a tuple was produced, `false` if there are no more tuples
   */
  auto Next(Tuple *tuple, RID *rid) -> bool override;

  /** @return The output schema for the external merge sort */
  auto GetOutputSchema() const -> const Schema & override { return plan_->OutputSchema(); }

 private:
  /** The sort plan node to be executed */
  const SortPlanNode *plan_;

  /** Compares tuples based on the order-bys */
  TupleComparator cmp_;

  std::unique_ptr<AbstractExecutor> child_executor_;

  BufferPoolManager *bpm_;

  auto WriteSortedBufferToNewSortPage(std::vector<SortEntry> &buffer) -> std::vector<page_id_t>;
  auto MergeTwoRuns(MergeSortRun &run1, MergeSortRun &run2) -> MergeSortRun;

  int32_t free_size_ = BUSTUB_PAGE_SIZE - sizeof(int32_t) * 3;
  int32_t tuple_size_ = {0};
  MergeSortRun::Iterator cur_iter_;
  MergeSortRun::Iterator end_iter_;
  std::vector<MergeSortRun> sortrun_{};
  /** TODO: You will want to add your own private members here. */
};

// end: mod by zhangyu for p3t4 at 2025/12/6
}  // namespace bustub
