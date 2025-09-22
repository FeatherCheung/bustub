//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// lru_k_replacer.cpp
//
// Identification: src/buffer/lru_k_replacer.cpp
//
// Copyright (c) 2015-2022, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "buffer/lru_k_replacer.h"
#include "common/exception.h"

namespace bustub {

// begin: added by zhangyu at 2025/9/14
auto LRUKNode::Gethistory() -> std::list<size_t> { return history_; }

auto LRUKNode::GetKvalue() -> size_t { return k_; }

auto LRUKNode::Getframeid() -> frame_id_t { return fid_; }

auto LRUKNode::Isevictable() -> bool { return is_evictable_; }

// 记录访问
void LRUKNode::Sethistory(size_t current_timestamp_) {
  // 因为只需要前k个元素，所以大于k的话，第一个元素可以删除了
  history_.push_back(current_timestamp_);
  if (history_.size() > k_) {
    history_.pop_front();  // 保证最多存 k 个
  }
}

void LRUKNode::SetKvalue(size_t K) { k_ = K; }

void LRUKNode::Setframeid(frame_id_t frame_id) { fid_ = frame_id; }
void LRUKNode::Setisevictable(bool evictable) { is_evictable_ = evictable; }

// end: added by zhangyu at 2025/9/14

// begin: modified by zhangyu at 2025/9/14
LRUKReplacer::LRUKReplacer(size_t num_frames, size_t k) : replacer_size_(num_frames), k_(k) {
  if (k == 0) {
    BUSTUB_ASSERT(k, "LRU-K parameter k must be positive (got k = 0)");
  }

  if (num_frames == 0) {
    BUSTUB_ASSERT(num_frames, "Number of frames must be positive (got num_frames = 0)");
  }
  node_store_.reserve(num_frames);
}

auto LRUKReplacer::Evict() -> std::optional<frame_id_t> {
  // latch_.lock();
  std::lock_guard<std::mutex> lock(latch_);
  // 如果没有可驱逐的frame，直接返回
  if (curr_size_.load() == 0) {
    return std::nullopt;
  }

  frame_id_t victim_frame = -1;
  size_t max_k_distance = 0;
  size_t earliest_timestamp = std::numeric_limits<size_t>::max();
  bool found_candidate = false;

  // 遍历所有frame，寻找最佳驱逐候选
  for (auto &it : node_store_) {
    // 如果是可淘汰的
    auto &node = it.second;
    if (!node.Isevictable()) {
      continue;
    }

    // 获取记录
    const auto &history = node.Gethistory();

    size_t k_distance;
    size_t first_access_time;

    if (history.size() < k_) {
      // K次向后的距离设置为无穷大
      k_distance = std::numeric_limits<size_t>::max();
      // 对于无穷大距离的frame，使用最早的访问时间进行LRU比较
      first_access_time = history.empty() ? 0 : history.front();
    } else {
      // 计算实际的后向k-距离
      // 后向k-距离 = 当前时间戳 - 第k次最近访问的时间戳
      // history设置为k个，所以返回队首即可
      size_t kth_recent_access = history.front();
      k_distance = current_timestamp_ - kth_recent_access;
      first_access_time = history.front();
    }

    // 选择驱逐候选的决策逻辑：
    // 1. 优先选择后向k-距离最大的frame
    // 2. 如果k-距离相同（特别是都为inf），选择最早访问时间的frame (LRU)

    bool should_select = !found_candidate || k_distance > max_k_distance ||
                         (k_distance == max_k_distance && first_access_time < earliest_timestamp);

    if (should_select) {
      victim_frame = node.Getframeid();
      max_k_distance = k_distance;
      earliest_timestamp = first_access_time;
      found_candidate = true;
    }
  }
  if (found_candidate && victim_frame != -1) {
    node_store_.erase(victim_frame);
    curr_size_.fetch_sub(1);
    // latch_.unlock();
    return victim_frame;
  }
  // latch_.unlock();
  return std::nullopt;
}

void LRUKReplacer::RecordAccess(frame_id_t frame_id, [[maybe_unused]] AccessType access_type) {
  // 加锁
  std::lock_guard<std::mutex> lock(latch_);
  if (static_cast<size_t>(frame_id) > replacer_size_) {
    BUSTUB_ASSERT(frame_id, "invalid frame_id in RecordAccess");
  }
  current_timestamp_++;
  auto it = node_store_.find(frame_id);

  if (it != node_store_.end()) {
    // 已经存在，则将记录存放
    auto &node = it->second;
    node.Sethistory(current_timestamp_);
  } else {
    // 不存在则新创建
    LRUKNode node;
    node.Setframeid(frame_id);
    node.SetKvalue(k_);
    node.Sethistory(current_timestamp_);
    node_store_.insert({frame_id, node});
  }
}

void LRUKReplacer::SetEvictable(frame_id_t frame_id, bool set_evictable) {
  std::lock_guard<std::mutex> lock(latch_);
  if (static_cast<size_t>(frame_id) >= replacer_size_) {
    BUSTUB_ASSERT(frame_id, "invalid frame_id in SetEvictable");
  }
  auto it = node_store_.find(frame_id);

  if (it != node_store_.end()) {
    auto &node = it->second;
    // 之前为不可淘汰，现在改为可淘汰，
    if (set_evictable && !node.Isevictable()) {
      node.Setisevictable(set_evictable);
      curr_size_.fetch_add(1);
    } else if (!set_evictable && node.Isevictable()) {
      node.Setisevictable(set_evictable);
      curr_size_.fetch_sub(1);
    }
  }
}

void LRUKReplacer::Remove(frame_id_t frame_id) {
  std::lock_guard<std::mutex> lock(latch_);
  if (static_cast<size_t>(frame_id) > replacer_size_) {
    BUSTUB_ASSERT(frame_id, "invalid frame_id in Remove");
  }
  auto it = node_store_.find(frame_id);

  if (it != node_store_.end()) {
    auto node = it->second;
    // 如果是可淘汰的
    if (node.Isevictable()) {
      node_store_.erase(frame_id);
      curr_size_.fetch_sub(1);
    } else {
      BUSTUB_ASSERT(frame_id, "This frame_id can't be removed");
    }
  }
}

auto LRUKReplacer::Size() -> size_t { return curr_size_; }
// end: modified by zhangyu at 2025/9/14
}  // namespace bustub
