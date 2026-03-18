/**
 * index_iterator.cpp
 */
#include <cassert>
#include <utility>

#include "common/config.h"
#include "storage/index/index_iterator.h"
#include "storage/page/b_plus_tree_page.h"
#include "storage/page/page_guard.h"

namespace bustub {

/*
 * NOTE: you can change the destructor/constructor method here
 * set your own input parameters
 */
INDEX_TEMPLATE_ARGUMENTS
INDEXITERATOR_TYPE::IndexIterator() = default;

// Mod by zhangyu at 2025/10/14 for P2:Task3
INDEX_TEMPLATE_ARGUMENTS
INDEXITERATOR_TYPE::IndexIterator(page_id_t current_page_id, int index, BufferPoolManager *buffer_pool_manager)
    : current_index_(index), current_page_id_(current_page_id), bpm_(buffer_pool_manager) {
  if (current_page_id != INVALID_PAGE_ID) {
    ReadPageGuard read_guard = bpm_->ReadPage(current_page_id);
    current_page_ = read_guard.As<LeafPage>();
  } else {
    current_page_ = nullptr;
  }
}

INDEX_TEMPLATE_ARGUMENTS
INDEXITERATOR_TYPE::~IndexIterator() = default;  // NOLINT

INDEX_TEMPLATE_ARGUMENTS
auto INDEXITERATOR_TYPE::IsEnd() -> bool {
  return current_page_ == nullptr && current_index_ == -1 && current_page_id_ == INVALID_PAGE_ID;
}

INDEX_TEMPLATE_ARGUMENTS
auto INDEXITERATOR_TYPE::operator*() -> std::pair<const KeyType &, const ValueType &> {
  return std::make_pair(current_page_->KeyAt(current_index_), current_page_->ValueAt(current_index_));
}

INDEX_TEMPLATE_ARGUMENTS
auto INDEXITERATOR_TYPE::operator++() -> INDEXITERATOR_TYPE & {
  // C++ STL的迭代器不做指针越界检查，需要注意
  ++current_index_;
  if (current_index_ < current_page_->GetSize()) {
    return *this;
  }
  auto next_page_id = current_page_->GetNextPageId();
  if (next_page_id != INVALID_PAGE_ID) {
    current_page_id_ = next_page_id;
    ReadPageGuard read_guard = bpm_->ReadPage(current_page_id_);
    current_page_ = read_guard.As<LeafPage>();
    current_index_ = 0;
    return *this;
  }
  //说明已经没有下一个元素，到达末尾了
  current_page_ = nullptr;
  current_index_ = -1;
  current_page_id_ = INVALID_PAGE_ID;
  return *this;
}

template class IndexIterator<GenericKey<4>, RID, GenericComparator<4>>;

template class IndexIterator<GenericKey<8>, RID, GenericComparator<8>>;

template class IndexIterator<GenericKey<16>, RID, GenericComparator<16>>;

template class IndexIterator<GenericKey<32>, RID, GenericComparator<32>>;

template class IndexIterator<GenericKey<64>, RID, GenericComparator<64>>;

}  // namespace bustub
