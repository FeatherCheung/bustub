//===----------------------------------------------------------------------===//
//
//                         CMU-DB Project (15-445/645)
//                         ***DO NO SHARE PUBLICLY***
//
// Identification: src/page/b_plus_tree_page.cpp
//
// Copyright (c) 2018, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "storage/page/b_plus_tree_page.h"
#include "common/macros.h"

namespace bustub {

// begin: added by zhangyu at 2025/9/28 for P2:Task1
/*
 * Helper methods to get/set page type
 * Page type enum class is defined in b_plus_tree_page.h
 */
/* 判断节点类型 */
auto BPlusTreePage::IsLeafPage() const -> bool { return page_type_ == IndexPageType::LEAF_PAGE; }
/* 设置节点类型 */
void BPlusTreePage::SetPageType(IndexPageType page_type) { page_type_ = page_type; }

/*
 * Helper methods to get/set size (number of key/value pairs stored in that
 * page)
 */
/*
 * size 的定义：
 * 叶子节点 LeafPage
    size = 当前叶子里的 key-value 对数量
    非根叶子的最小允许值 = ceil(max_size / 2)
 * 内部节点 InternalPage
    size = 当前 internal 里的 child pointer 数量
    有效 key 数 = size - 1
    key[0] 不参与正常分隔
    非根 internal 的最小允许值 = ceil(max_size / 2)
 */
auto BPlusTreePage::GetSize() const -> int { return size_; }
void BPlusTreePage::SetSize(int size) { size_ = size; }
void BPlusTreePage::ChangeSizeBy(int amount) {
  auto new_size = size_ + amount;

  BUSTUB_ASSERT(new_size >= 0, "the current size of B plus tree can't be negative");
  BUSTUB_ASSERT(new_size <= max_size_, "the current size of B plus tree can't bypass max_size");

  size_ = new_size;
}

/*
 * Helper methods to get/set max size (capacity) of the page
 */
auto BPlusTreePage::GetMaxSize() const -> int { return max_size_; }
void BPlusTreePage::SetMaxSize(int size) { max_size_ = size; }

/*
 * Helper method to get min page size
 * Generally, min page size == max page size / 2
 * But whether you will take ceil() or floor() depends on your implementation
 */
auto BPlusTreePage::GetMinSize() const -> int {
  // 非根节点：最小允许值 = ceil(max_size / 2)
  return ceil(max_size_ / 2.0);
}
}  // namespace bustub
// end: added by zhangyu at 2025/9/28 for P2:Task1
