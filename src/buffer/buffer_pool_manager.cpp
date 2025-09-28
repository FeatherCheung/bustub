//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// buffer_pool_manager.cpp
//
// Identification: src/buffer/buffer_pool_manager.cpp
//
// Copyright (c) 2015-2024, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "buffer/buffer_pool_manager.h"
#include <memory>
#include <mutex>
#include <optional>
#include "common/config.h"
#include "common/macros.h"
#include "storage/disk/disk_scheduler.h"
#include "storage/page/page_guard.h"

namespace bustub {

/**
 * @brief The constructor for a `FrameHeader` that initializes all fields to default values.
 *
 * See the documentation for `FrameHeader` in "buffer/buffer_pool_manager.h" for more information.
 *
 * @param frame_id The frame ID / index of the frame we are creating a header for.
 */
FrameHeader::FrameHeader(frame_id_t frame_id) : frame_id_(frame_id), data_(BUSTUB_PAGE_SIZE, 0) { Reset(); }

/**
 * @brief Get a raw const pointer to the frame's data.
 *
 * @return const char* A pointer to immutable data that the frame stores.
 */
auto FrameHeader::GetData() const -> const char * { return data_.data(); }

/**
 * @brief Get a raw mutable pointer to the frame's data.
 *
 * @return char* A pointer to mutable data that the frame stores.
 */
auto FrameHeader::GetDataMut() -> char * { return data_.data(); }

/**
 * @brief Resets a `FrameHeader`'s member fields.
 */
void FrameHeader::Reset() {
  std::fill(data_.begin(), data_.end(), 0);
  pin_count_.store(0);
  is_dirty_ = false;
  page_id_ = INVALID_PAGE_ID;
}

/**
 * @brief Creates a new `BufferPoolManager` instance and initializes all fields.
 *
 * See the documentation for `BufferPoolManager` in "buffer/buffer_pool_manager.h" for more information.
 *
 * ### Implementation
 *
 * We have implemented the constructor for you in a way that makes sense with our reference solution. You are free to
 * change anything you would like here if it doesn't fit with you implementation.
 *
 * Be warned, though! If you stray too far away from our guidance, it will be much harder for us to help you. Our
 * recommendation would be to first implement the buffer pool manager using the stepping stones we have provided.
 *
 * Once you have a fully working solution (all Gradescope test cases pass), then you can try more interesting things!
 *
 * @param num_frames The size of the buffer pool.
 * @param disk_manager The disk manager.
 * @param k_dist The backward k-distance for the LRU-K replacer.
 * @param log_manager The log manager. Please ignore this for P1.
 */
BufferPoolManager::BufferPoolManager(size_t num_frames, DiskManager *disk_manager, size_t k_dist,
                                     LogManager *log_manager)
    : num_frames_(num_frames),
      next_page_id_(0),
      bpm_latch_(std::make_shared<std::mutex>()),
      replacer_(std::make_shared<LRUKReplacer>(num_frames, k_dist)),
      disk_scheduler_(std::make_unique<DiskScheduler>(disk_manager)),
      log_manager_(log_manager) {
  // Not strictly necessary...
  std::scoped_lock latch(*bpm_latch_);

  // Initialize the monotonically increasing counter at 0.
  next_page_id_.store(0);

  // Allocate all of the in-memory frames up front.
  frames_.reserve(num_frames_);

  // The page table should have exactly `num_frames_` slots, corresponding to exactly `num_frames_` frames.
  page_table_.reserve(num_frames_);

  // Initialize all of the frame headers, and fill the free frame list with all possible frame IDs (since all frames are
  // initially free).
  for (size_t i = 0; i < num_frames_; i++) {
    frames_.push_back(std::make_shared<FrameHeader>(i));
    free_frames_.push_back(static_cast<int>(i));
  }
}

/**
 * @brief Destroys the `BufferPoolManager`, freeing up all memory that the buffer pool was using.
 */
BufferPoolManager::~BufferPoolManager() = default;

/**
 * @brief Returns the number of frames that this buffer pool manages.
 */
auto BufferPoolManager::Size() const -> size_t { return num_frames_; }

// begin: Modified by zhangyu at 2025/9/18
/**
 * @brief Allocates a new page on disk.
 *
 * ### Implementation
 *
 * You will maintain a thread-safe, monotonically increasing counter in the form of a `std::atomic<page_id_t>`.
 * See the documentation on [atomics](https://en.cppreference.com/w/cpp/atomic/atomic) for more information.
 *
 * Also, make sure to read the documentation for `DeletePage`! You can assume that you will never run out of disk
 * space (via `DiskScheduler::IncreaseDiskSpace`), so this function _cannot_ fail.
 *
 * Once you have allocated the new page via the counter, make sure to call `DiskScheduler::IncreaseDiskSpace` so you
 * have enough space on disk!
 *
 * TODO(P1): Add implementation.
 *
 * @return The page ID of the newly allocated page.
 */
auto BufferPoolManager::NewPage() -> page_id_t {
  // 从全局的原子计数器里分配新的 Page ID
  // relaxed 只保证原子操作本身不被打断（原子性），不保证和其它内存操作的顺序，用于计数器、自增器等场景
  page_id_t new_page_id = next_page_id_.fetch_add(1);

  // 磁盘分配实际的物理存储空间，即总的page数目
  // IncreaseDiskSpace 的作用
  disk_scheduler_->IncreaseDiskSpace(new_page_id + 1);

  return new_page_id;
}

/**
 * @brief Removes a page from the database, both on disk and in memory.
 *
 * If the page is pinned in the buffer pool, this function does nothing and returns `false`. Otherwise, this function
 * removes the page from both disk and memory (if it is still in the buffer pool), returning `true`.
 *
 * ### Implementation
 *
 * Think about all of the places a page or a page's metadata could be, and use that to guide you on implementing this
 * function. You will probably want to implement this function _after_ you have implemented `CheckedReadPage` and
 * `CheckedWritePage`.
 *
 * Ideally, we would want to ensure that all space on disk is used efficiently. That would mean the space that deleted
 * pages on disk used to occupy should somehow be made available to new pages allocated by `NewPage`.
 *
 * If you would like to attempt this, you are free to do so. However, for this implementation, you are allowed to
 * assume you will not run out of disk space and simply keep allocating disk space upwards in `NewPage`.
 *
 * For (nonexistent) style points, you can still call `DeallocatePage` in case you want to implement something slightly
 * more space-efficient in the future.
 *
 * TODO(P1): Add implementation.
 *
 * @param page_id The page ID of the page we want to delete.
 * @return `false` if the page exists but could not be deleted, `true` if the page didn't exist or deletion succeeded.
 */
auto BufferPoolManager::DeletePage(page_id_t page_id) -> bool {
  std::unique_lock<std::mutex> latch_lock(*bpm_latch_);
  // 先查找这个page是不是在缓冲池中
  auto it = page_table_.find(page_id);

  if (it == page_table_.end()) {
    return true;
  }

  frame_id_t frame_id = it->second;
  auto frame = frames_[frame_id];

  // 如果 page 仍然 pinned（被使用），不能删除
  if (frame->pin_count_.load() > 0) {
    return false;
  }

  // 内存中清除
  replacer_->Remove(frame_id);
  page_table_.erase(page_id);
  free_frames_.push_back(frame_id);
  frame->Reset();
  // 磁盘上清除
  disk_scheduler_->DeallocatePage(page_id);

  return true;
}

/**
 * @brief Acquires an optional write-locked guard over a page of data. The user can specify an `AccessType` if needed.
 *
 * If it is not possible to bring the page of data into memory, this function will return a `std::nullopt`.
 *
 * Page data can _only_ be accessed via page guards. Users of this `BufferPoolManager` are expected to acquire either a
 * `ReadPageGuard` or a `WritePageGuard` depending on the mode in which they would like to access the data, which
 * ensures that any access of data is thread-safe.
 *
 * There can only be 1 `WritePageGuard` reading/writing a page at a time. This allows data access to be both immutable
 * and mutable, meaning the thread that owns the `WritePageGuard` is allowed to manipulate the page's data however they
 * want. If a user wants to have multiple threads reading the page at the same time, they must acquire a `ReadPageGuard`
 * with `CheckedReadPage` instead.
 *
 * ### Implementation
 *
 * There are 3 main cases that you will have to implement. The first two are relatively simple: one is when there is
 * plenty of available memory, and the other is when we don't actually need to perform any additional I/O. Think about
 * what exactly these two cases entail.
 *
 * The third case is the trickiest, and it is when we do not have any _easily_ available memory at our disposal. The
 * buffer pool is tasked with finding memory that it can use to bring in a page of memory, using the replacement
 * algorithm you implemented previously to find candidate frames for eviction.
 *
 * Once the buffer pool has identified a frame for eviction, several I/O operations may be necessary to bring in the
 * page of data we want into the frame.
 *
 * There is likely going to be a lot of shared code with `CheckedReadPage`, so you may find creating helper functions
 * useful.
 *
 * These two functions are the crux of this project, so we won't give you more hints than this. Good luck!
 *
 * TODO(P1): Add implementation.
 *
 * @param page_id The ID of the page we want to write to.
 * @param access_type The type of page access.
 * @return std::optional<WritePageGuard> An optional latch guard where if there are no more free frames (out of memory)
 * returns `std::nullopt`, otherwise returns a `WritePageGuard` ensuring exclusive and mutable access to a page's data.
 */
auto BufferPoolManager::CheckedWritePage(page_id_t page_id, AccessType access_type) -> std::optional<WritePageGuard> {
  std::unique_lock<std::mutex> bpm_lock(*bpm_latch_);
  // 先查找这个page是不是在缓冲池中
  if (auto it = page_table_.find(page_id); it != page_table_.end()) {
    //在内存中，则直接构造返回
    auto frame_id = it->second;
    replacer_->RecordAccess(frame_id);
    {
      std::lock_guard<std::mutex> frame_lock(frames_[frame_id]->latch_);
      ++frames_[frame_id]->pin_count_;
      replacer_->SetEvictable(frame_id, false);
    }
    bpm_lock.unlock();
    return WritePageGuard(page_id, frames_[frame_id], replacer_, bpm_latch_);
  }

  // 不在内存中，需要载入到内存中
  frame_id_t victim_frame_id;
  bool bflag_dirty;
  // 如果还有空余帧，直接使用，并构造请求
  if (!free_frames_.empty()) {
    victim_frame_id = free_frames_.front();
    free_frames_.pop_front();
    bflag_dirty = false;
  } else {
    /* 如果没有空余帧，则选择一个帧驱逐，没有可驱逐的返回nullopt */
    auto victim_opt = replacer_->Evict();
    if (victim_opt.has_value()) {
      victim_frame_id = victim_opt.value();
      bflag_dirty = frames_[victim_frame_id]->is_dirty_;
    } else {
      return std::nullopt;
    }
  }

  auto victim_frame = frames_[victim_frame_id];
  auto victim_page_id = victim_frame->page_id_;
  /* 该页是脏页，需要先刷盘，然后将frame内容清理，然后将pageid和frameid的关系清理 */
  if (bflag_dirty) {
    bool ret = FlushPage(victim_page_id);
    BUSTUB_ASSERT(ret, "flush error");
  }
  victim_frame->Reset();
  /* 将frame内容清理，然后将pageid和frameid的关系清理 */
  page_table_.erase(victim_page_id);
  // 此时，帧已经可用，上锁，构造读取请求
  victim_frame->page_id_ = page_id;

  DiskRequest req;
  req.is_write_ = false;
  req.data_ = victim_frame->GetDataMut();
  req.page_id_ = page_id;
  std::future<bool> future = req.callback_.get_future();
  disk_scheduler_->Schedule(std::move(req));
  bool bflag = future.get();
  BUSTUB_ASSERT(bflag, "read error");
  // 异步处理读请求
  /* 建立 pageid和frameid的关系，然后记录访问，方便后续替换 */
  replacer_->RecordAccess(victim_frame_id);
  page_table_[page_id] = victim_frame_id;
  // replacer_->SetEvictable(victim_frame_id, false);
  {
    std::lock_guard<std::mutex> frame_lock(victim_frame->latch_);
    ++victim_frame->pin_count_;
    replacer_->SetEvictable(victim_frame_id, false);
  }
  bpm_lock.unlock();
  return WritePageGuard(page_id, victim_frame, replacer_, bpm_latch_);
}

/**
 * @brief Acquires an optional read-locked guard over a page of data. The user can specify an `AccessType` if needed.
 *
 * If it is not possible to bring the page of data into memory, this function will return a `std::nullopt`.
 *
 * Page data can _only_ be accessed via page guards. Users of this `BufferPoolManager` are expected to acquire either a
 * `ReadPageGuard` or a `WritePageGuard` depending on the mode in which they would like to access the data, which
 * ensures that any access of data is thread-safe.
 *
 * There can be any number of `ReadPageGuard`s reading the same page of data at a time across different threads.
 * However, all data access must be immutable. If a user wants to mutate the page's data, they must acquire a
 * `WritePageGuard` with `CheckedWritePage` instead.
 *
 * ### Implementation
 *
 * See the implementation details of `CheckedWritePage`.
 *
 * TODO(P1): Add implementation.
 *
 * @param page_id The ID of the page we want to read.
 * @param access_type The type of page access.
 * @return std::optional<ReadPageGuard> An optional latch guard where if there are no more free frames (out of memory)
 * returns `std::nullopt`, otherwise returns a `ReadPageGuard` ensuring shared and read-only access to a page's data.
 */
auto BufferPoolManager::CheckedReadPage(page_id_t page_id, AccessType access_type) -> std::optional<ReadPageGuard> {
  std::unique_lock<std::mutex> bpm_lock(*bpm_latch_);
  // 先查找这个page是不是在缓冲池中
  if (auto it = page_table_.find(page_id); it != page_table_.end()) {
    //在内存中，则直接构造返回
    auto frame_id = it->second;
    replacer_->RecordAccess(frame_id);
    {
      std::lock_guard<std::mutex> frame_lock(frames_[frame_id]->latch_);
      ++frames_[frame_id]->pin_count_;
      replacer_->SetEvictable(frame_id, false);
    }
    bpm_lock.unlock();
    return ReadPageGuard(page_id, frames_[frame_id], replacer_, bpm_latch_);
  }

  // 不在内存中，需要载入到内存中
  frame_id_t victim_frame_id;
  bool bflag_dirty;
  // 如果还有空余帧，直接使用，并构造请求
  if (!free_frames_.empty()) {
    victim_frame_id = free_frames_.front();
    free_frames_.pop_front();
    bflag_dirty = false;
  } else {
    /* 如果没有空余帧，则选择一个帧驱逐，没有可驱逐的返回nullopt */
    auto victim_opt = replacer_->Evict();
    if (victim_opt.has_value()) {
      victim_frame_id = victim_opt.value();
      bflag_dirty = frames_[victim_frame_id]->is_dirty_;
    } else {
      return std::nullopt;
    }
  }

  auto victim_frame = frames_[victim_frame_id];
  auto victim_page_id = victim_frame->page_id_;
  /* 该页是脏页，需要先刷盘 */
  if (bflag_dirty) {
    bool ret = FlushPage(victim_page_id);
    BUSTUB_ASSERT(ret, "flush error");
  }
  victim_frame->Reset();
  /* 将frame内容清理，然后将pageid和frameid的关系清理 */
  page_table_.erase(victim_page_id);

  // 此时，帧已经可用，构造读取的请求
  victim_frame->page_id_ = page_id;
  DiskRequest req;
  req.is_write_ = false;
  req.data_ = victim_frame->GetDataMut();
  req.page_id_ = page_id;
  std::future<bool> future = req.callback_.get_future();
  disk_scheduler_->Schedule(std::move(req));
  bool bflag = future.get();
  BUSTUB_ASSERT(bflag, "read error");
  /* 建立 pageid和frameid的关系，然后记录访问，方便后续替换 */
  replacer_->RecordAccess(victim_frame_id);
  page_table_[page_id] = victim_frame_id;
  {
    std::lock_guard<std::mutex> frame_lock(victim_frame->latch_);
    ++victim_frame->pin_count_;
    replacer_->SetEvictable(victim_frame_id, false);
  }
  bpm_lock.unlock();
  return ReadPageGuard(page_id, victim_frame, replacer_, bpm_latch_);
}

/**
 * @brief A wrapper around `CheckedWritePage` that unwraps the inner value if it exists.
 *
 * If `CheckedWritePage` returns a `std::nullopt`, **this function aborts the entire process.**
 *
 * This function should **only** be used for testing and ergonomic's sake. If it is at all possible that the buffer pool
 * manager might run out of memory, then use `CheckedPageWrite` to allow you to handle that case.
 *
 * See the documentation for `CheckedPageWrite` for more information about implementation.
 *
 * @param page_id The ID of the page we want to read.
 * @param access_type The type of page access.
 * @return WritePageGuard A page guard ensuring exclusive and mutable access to a page's data.
 */
auto BufferPoolManager::WritePage(page_id_t page_id, AccessType access_type) -> WritePageGuard {
  auto guard_opt = CheckedWritePage(page_id, access_type);

  if (!guard_opt.has_value()) {
    fmt::println(stderr, "\n`CheckedWritePage` failed to bring in page {}\n", page_id);
    std::abort();
  }

  return std::move(guard_opt).value();
}

/**
 * @brief A wrapper around `CheckedReadPage` that unwraps the inner value if it exists.
 *
 * If `CheckedReadPage` returns a `std::nullopt`, **this function aborts the entire process.**
 *
 * This function should **only** be used for testing and ergonomic's sake. If it is at all possible that the buffer pool
 * manager might run out of memory, then use `CheckedPageWrite` to allow you to handle that case.
 *
 * See the documentation for `CheckedPageRead` for more information about implementation.
 *
 * @param page_id The ID of the page we want to read.
 * @param access_type The type of page access.
 * @return ReadPageGuard A page guard ensuring shared and read-only access to a page's data.
 */
auto BufferPoolManager::ReadPage(page_id_t page_id, AccessType access_type) -> ReadPageGuard {
  auto guard_opt = CheckedReadPage(page_id, access_type);

  if (!guard_opt.has_value()) {
    fmt::println(stderr, "\n`CheckedReadPage` failed to bring in page {}\n", page_id);
    std::abort();
  }

  return std::move(guard_opt).value();
}

/**
 * @brief Flushes a page's data out to disk.
 *
 * This function will write out a page's data to disk if it has been modified. If the given page is not in memory, this
 * function will return `false`.
 *
 * ### Implementation
 *
 * You should probably leave implementing this function until after you have completed `CheckedReadPage` and
 * `CheckedWritePage`, as it will likely be much easier to understand what to do.
 *
 * TODO(P1): Add implementation
 *
 * @param page_id The page ID of the page to be flushed.
 * @return `false` if the page could not be found in the page table, otherwise `true`.
 */
auto BufferPoolManager::FlushPage(page_id_t page_id) -> bool {
  // 不在内存中，直接返回false
  if (page_id == INVALID_PAGE_ID || page_table_.find(page_id) == page_table_.end()) {
    return false;
  }
  auto frame_id = page_table_[page_id];
  auto frame = frames_[frame_id];
  DiskRequest req;
  req.is_write_ = true;
  req.data_ = frame->GetDataMut();
  req.page_id_ = page_id;
  std::future<bool> future = req.callback_.get_future();
  disk_scheduler_->Schedule(std::move(req));

  return future.get();
}

/**
 * @brief Flushes all page data that is in memory to disk.
 *
 * ### Implementation
 *
 * You should probably leave implementing this function until after you have completed `CheckedReadPage`,
 * `CheckedWritePage`, and `FlushPage`, as it will likely be much easier to understand what to do.
 *
 * TODO(P1): Add implementation
 */
void BufferPoolManager::FlushAllPages() {
  for (auto &it : page_table_) {
    auto page_id = it.first;
    auto frame_id = it.second;
    auto frame = frames_[frame_id];
    DiskRequest req;
    req.is_write_ = true;
    req.data_ = frame->GetDataMut();
    req.page_id_ = page_id;
    std::future<bool> future = req.callback_.get_future();
    disk_scheduler_->Schedule(std::move(req));
  }
}

/**
 * @brief Retrieves the pin count of a page. If the page does not exist in memory, return `std::nullopt`.
 *
 * This function is thread safe. Callers may invoke this function in a multi-threaded environment where multiple threads
 * access the same page.
 *
 * This function is intended for testing purposes. If this function is implemented incorrectly, it will definitely cause
 * problems with the test suite and autograder.
 *
 * # Implementation
 *
 * We will use this function to test if your buffer pool manager is managing pin counts correctly. Since the
 * `pin_count_` field in `FrameHeader` is an atomic type, you do not need to take the latch on the frame that holds the
 * page we want to look at. Instead, you can simply use an atomic `load` to safely load the value stored. You will still
 * need to take the buffer pool latch, however.
 *
 * Again, if you are unfamiliar with atomic types, see the official C++ docs
 * [here](https://en.cppreference.com/w/cpp/atomic/atomic).
 *
 * TODO(P1): Add implementation
 *
 * @param page_id The page ID of the page we want to get the pin count of.
 * @return std::optional<size_t> The pin count if the page exists, otherwise `std::nullopt`.
 */
auto BufferPoolManager::GetPinCount(page_id_t page_id) -> std::optional<size_t> {
  std::lock_guard<std::mutex> latch_lock(*bpm_latch_);
  auto it = page_table_.find(page_id);
  if (it == page_table_.end()) {
    return std::nullopt;
  }
  auto frame_id = it->second;
  auto frame = frames_[frame_id];
  return frame->pin_count_.load();
}

// end: Modified by zhangyu at 2025/9/18
}  // namespace bustub
