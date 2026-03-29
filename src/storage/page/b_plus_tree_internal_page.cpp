//===----------------------------------------------------------------------===//
//
//                         CMU-DB Project (15-445/645)
//                         ***DO NO SHARE PUBLICLY***
//
// Identification: src/page/b_plus_tree_internal_page.cpp
//
// Copyright (c) 2018-2024, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sstream>

#include "common/exception.h"
#include "common/logger.h"
#include "common/macros.h"
#include "storage/page/b_plus_tree_internal_page.h"
#include "storage/page/b_plus_tree_page.h"

// begin: added by zhangyu at 2025/9/28 for P2:Task1
namespace bustub {
/*****************************************************************************
 * HELPER METHODS AND UTILITIES
 *****************************************************************************/
/*
 * Init method after creating a new internal page
 * Including set page type, set current size, and set max page size
 */
/* 设置page头 */
INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_INTERNAL_PAGE_TYPE::Init(int max_size) {
  SetPageType(IndexPageType::INTERNAL_PAGE);
  SetSize(0);
  SetMaxSize(max_size);
}
/*
 * Helper method to get/set the key associated with input "index" (a.k.a
 * array offset)
 */
/* 根据索引获取key */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::KeyAt(int index) const -> KeyType {
  BUSTUB_ASSERT(index < GetMaxSize(), "invalid index in Internal keyAt()");
  return key_array_[index];
}

/* 根据索引设置key */

INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_INTERNAL_PAGE_TYPE::SetKeyAt(int index, const KeyType &key) {
  // LOG_DEBUG("LOG_DEBUG, index: %u, maxsize: %u", index, GetMaxSize());
  BUSTUB_ASSERT(index < GetMaxSize(), "invalid index in Internal SetKeyAt()");
  key_array_[index] = key;
}

/* 根据索引设置value */
INDEX_TEMPLATE_ARGUMENTS
void B_PLUS_TREE_INTERNAL_PAGE_TYPE::SetValueAt(int index, const ValueType &value) {
  BUSTUB_ASSERT(index < GetMaxSize(), "invalid index in Internal SetValueAt()");
  page_id_array_[index] = value;
}

/*
 * Helper method to get the value associated with input "index" (a.k.a array
 * offset)
 */
/* 根据索引获取value */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::ValueAt(int index) const -> ValueType {
  BUSTUB_ASSERT(index < GetMaxSize(), "invalid index in ValueAt()");
  return page_id_array_[index];
}

/* Insert
 * index在 1- size之间，正常插入; index为0需要做特殊处理
 */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::Insert(int index, const KeyType &key, const ValueType &value) -> bool {
  BUSTUB_ASSERT(0 <= index && index <= GetSize(), "invalid index in internal insert()");
  if (index >= 1 && index <= GetSize() - 1) {
    for (int i = GetSize(); i > index; --i) {
      SetKeyAt(i, KeyAt(i - 1));
      SetValueAt(i, ValueAt(i - 1));
    }
    SetKeyAt(index, key);
    SetValueAt(index, value);
    ChangeSizeBy(1);
    return true;
  }

  if (index == 0) {
    SetKeyAt(1, key);
    SetValueAt(1, ValueAt(0));
    SetValueAt(0, value);
    ChangeSizeBy(1);
    return true;
  }

  SetKeyAt(index, key);
  SetValueAt(index, value);
  ChangeSizeBy(1);
  return true;
}

/*
 * internal node delete
 * 这个删除函数只对internal node的key和value进行删除，index为1-size-1之间正常删除
 */
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::Delete(int index) -> bool {
  BUSTUB_ASSERT(1 <= index && index < GetSize(), "invalid index internal Delete()");

  for (int i = index + 1; i < GetSize(); ++i) {
    SetKeyAt(i - 1, KeyAt(i));
    SetValueAt(i - 1, ValueAt(i));
  }

  ChangeSizeBy(-1);
  return true;
}
// end: added by zhangyu at 2025/9/28 for P2:Task1

// begin: added by zhangyu at 2025/10/19 for P2:Task4
INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::IsSafeInternalForDelete(int deletenum) -> bool {
  return GetSize() - deletenum >= GetMinSize();
}

INDEX_TEMPLATE_ARGUMENTS
auto B_PLUS_TREE_INTERNAL_PAGE_TYPE::IsSafeInternalForInsert(int insertnum) -> bool {
  return GetSize() + insertnum <= GetMaxSize();
}
// end: added by zhangyu at 2025/10/19 for P2:Task4

// valuetype for internalNode should be page id_t
template class BPlusTreeInternalPage<GenericKey<4>, page_id_t, GenericComparator<4>>;
template class BPlusTreeInternalPage<GenericKey<8>, page_id_t, GenericComparator<8>>;
template class BPlusTreeInternalPage<GenericKey<16>, page_id_t, GenericComparator<16>>;
template class BPlusTreeInternalPage<GenericKey<32>, page_id_t, GenericComparator<32>>;
template class BPlusTreeInternalPage<GenericKey<64>, page_id_t, GenericComparator<64>>;
}  // namespace bustub
