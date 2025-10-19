//===----------------------------------------------------------------------===//
//
//                         CMU-DB Project (15-445/645)
//                         ***DO NO SHARE PUBLICLY***
//
// Identification: src/page/b_plus_tree_leaf_page.cpp
//
// Copyright (c) 2018-2024, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <sstream>

#include "common/config.h"
#include "common/exception.h"
#include "common/macros.h"
#include "common/rid.h"
#include "storage/page/b_plus_tree_leaf_page.h"

// begin: added by zhangyu at 2025/9/28 for P2:Task1
namespace bustub {

/*****************************************************************************
 * HELPER METHODS AND UTILITIES
 *****************************************************************************/

/**
 * Init method after creating a new leaf page
 * Including set page type, set current size to zero, set next page id and set max size
 */
INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_LEAF_PAGE_TYPE::Init(int max_size) {
  SetPageType(IndexPageType::LEAF_PAGE);
  SetSize(0);
  SetMaxSize(max_size);
  SetNextPageId(INVALID_PAGE_ID);
}

/**
 * Helper methods to set/get next page id
 */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::GetNextPageId() const -> page_id_t { return next_page_id_; }

INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_LEAF_PAGE_TYPE::SetNextPageId(page_id_t next_page_id) { next_page_id_ = next_page_id; }

/*
 * Helper method to find and return the key associated with input "index" (a.k.a
 * array offset)
 */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::KeyAt(int index) const -> KeyType {
  BUSTUB_ASSERT(index < GetMaxSize(), "invalid index in leaf keyAt()");
  return key_array_[index];
}

/* 根据index 获取value值 */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::ValueAt(int index) const -> ValueType {
  BUSTUB_ASSERT(index < GetMaxSize(), "invalid index in leaf ValueAt()");
  return rid_array_[index];
}

/* 根据index 设置key值 */
INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_LEAF_PAGE_TYPE::SetKeyAt(int index, const KeyType &key) {
  BUSTUB_ASSERT(index < GetMaxSize(), "invalid index in leaf SetkeyAt()");
  key_array_[index] = key;
}

/* 根据index 设置value值 */
INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_LEAF_PAGE_TYPE::SetValueAt(int index, const ValueType &value) {
  BUSTUB_ASSERT(index < GetMaxSize(), "invalid index in leaf SetValueAt()");
  rid_array_[index] = value;
}

/* insert和delete函数，index统一为下标 */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Insert(int index, const KeyType &key, const ValueType &value) -> bool {
  BUSTUB_ASSERT(0 <= index && index <= GetSize(), "invalid index in leaf insert()");
  for (int i = GetSize(); i > index; --i) {
    SetKeyAt(i, KeyAt(i - 1));
    SetValueAt(i, ValueAt(i - 1));
  }
  SetKeyAt(index, key);
  SetValueAt(index, value);
  ChangeSizeBy(1);
  return true;
}
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_LEAF_PAGE_TYPE::Delete(int index) -> bool {
  BUSTUB_ASSERT(0 <= index && index < GetSize(), "invalid index in leaf Delete()");
  for (int i = index + 1; i < GetSize(); ++i) {
    SetKeyAt(i - 1, KeyAt(i));
    SetValueAt(i - 1, ValueAt(i));
  }
  ChangeSizeBy(-1);
  return true;
}

template class BPlusTreeLeafPage<GenericKey<4>, RID, GenericComparator<4>>;
template class BPlusTreeLeafPage<GenericKey<8>, RID, GenericComparator<8>>;
template class BPlusTreeLeafPage<GenericKey<16>, RID, GenericComparator<16>>;
template class BPlusTreeLeafPage<GenericKey<32>, RID, GenericComparator<32>>;
template class BPlusTreeLeafPage<GenericKey<64>, RID, GenericComparator<64>>;
}  // namespace bustub
// end: added by zhangyu at 2025/9/28 for P2:Task1
