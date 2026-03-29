//===----------------------------------------------------------------------===//
//
//                         CMU-DB Project (15-445/645)
//                         ***DO NO SHARE PUBLICLY***
//
// Identification: src/include/page/b_plus_tree_internal_page.h
//
// Copyright (c) 2018-2024, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//
#pragma once

#include <queue>
#include <string>

#include "storage/page/b_plus_tree_page.h"

namespace bustub {

#define B_PLUS_TREE_INTERNAL_PAGE_TYPE BPlusTreeInternalPage<KeyType, ValueType, KeyComparator>
#define INTERNAL_PAGE_HEADER_SIZE 12
#define INTERNAL_PAGE_SLOT_CNT \
  ((BUSTUB_PAGE_SIZE - INTERNAL_PAGE_HEADER_SIZE) / ((int)(sizeof(KeyType) + sizeof(ValueType))))  // NOLINT

/**
 * Store `n` indexed keys and `n + 1` child pointers (page_id) within internal page.
 * Pointer PAGE_ID(i) points to a subtree in which all keys K satisfy:
 * K(i) <= K < K(i+1).
 * NOTE: Since the number of keys does not equal to number of child pointers,
 * the first key in key_array_ always remains invalid. That is to say, any search / lookup
 * should ignore the first key.
 *
 * Internal page format (keys are stored in increasing order):
 *  ---------
 * | HEADER |
 *  ---------
 *  ------------------------------------------
 * | KEY(1)(INVALID) | KEY(2) | ... | KEY(n) |
 *  ------------------------------------------
 *  ---------------------------------------------
 * | PAGE_ID(1) | PAGE_ID(2) | ... | PAGE_ID(n) |
 *  ---------------------------------------------
 */
/*
 * internal node 的 size 不是“有效 key 的个数”，而是“child pointer 的个数”。
 * 当只有value[0] 时（即size == 1）
 * 如果它是根节点，那么树高减小，value[0]指向的新节点应该被提升为新的root
 * 如果它不是根节点，此时它不安全了，需要合并或者重分配
 *
 * 判断内部节点是否underflow
 * min_size = ceil(max_size / 2)
 * 删除时，size - 1 >= min_size 才安全，否则不安全，触发合并/重分配
 * 插入时，size + 1 <= max_size 才安全，否则不安全，触发分裂
 */
INDEX_TEMPLATE_ARGUMENTS
class BPlusTreeInternalPage : public BPlusTreePage {
 public:
  // Delete all constructor / destructor to ensure memory safety
  BPlusTreeInternalPage() = delete;
  BPlusTreeInternalPage(const BPlusTreeInternalPage &other) = delete;

  /**
   * Writes the necessary header information to a newly created page, must be called after
   * the creation of a new page to make a valid `BPlusTreeInternalPage`
   * @param max_size Maximal size of the page
   */
  void Init(int max_size = INTERNAL_PAGE_SLOT_CNT);

  /**
   * @param index The index of the key to get. Index must be non-zero.
   * @return Key at index
   */
  auto KeyAt(int index) const -> KeyType;

  /**
   * @param index The index of the key to set. Index must be non-zero.
   * @param key The new value for key
   */
  void SetKeyAt(int index, const KeyType &key);
  // begin added by zhangyu at 2025/10/10 for P2:Task2
  void SetValueAt(int index, const ValueType &value);
  /* insert和delete函数，index对应的是key的数组下标 */
  auto Insert(int index, const KeyType &key, const ValueType &value) -> bool;
  auto Delete(int index) -> bool;
  auto IsSafeInternalForDelete(int deletenum) -> bool;
  auto IsSafeInternalForInsert(int insertnum) -> bool;
  // end added by zhangyu at 2025/10/10 for P2:Task2

  /**
   * @param value The value to search for
   * @return The index that corresponds to the specified value
   */
  //  auto ValueIndex(const ValueType &value) const -> int;

  /**
   * @param index The index to search for
   * @return The value at the index
   */
  auto ValueAt(int index) const -> ValueType;

  /**
   * @brief For test only, return a string representing all keys in
   * this internal page, formatted as "(key1,key2,key3,...)"
   *
   * @return The string representation of all keys in the current internal page
   */
  auto ToString() const -> std::string {
    std::string kstr = "(";
    bool first = true;

    // First key of internal page is always invalid
    for (int i = 1; i < GetSize(); i++) {
      KeyType key = KeyAt(i);
      if (first) {
        first = false;
      } else {
        kstr.append(",");
      }

      kstr.append(std::to_string(key.ToString()));
    }
    kstr.append(")");

    return kstr;
  }

 private:
  // Array members for page data.
  KeyType key_array_[INTERNAL_PAGE_SLOT_CNT];
  ValueType page_id_array_[INTERNAL_PAGE_SLOT_CNT];
  // (Fall 2024) Feel free to add more fields and helper functions below if needed
};

}  // namespace bustub
