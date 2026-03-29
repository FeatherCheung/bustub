#include "storage/index/b_plus_tree.h"
#include <cstddef>
#include <cstring>
#include <utility>
#include "common/config.h"
#include "common/logger.h"
#include "common/macros.h"
#include "storage/index/b_plus_tree_debug.h"
#include "storage/page/b_plus_tree_header_page.h"
#include "storage/page/b_plus_tree_internal_page.h"
#include "storage/page/b_plus_tree_page.h"
#include "storage/page/page_guard.h"

namespace bustub {

INDEX_TEMPLATE_ARGUMENTS
BPLUSTREE_TYPE::BPlusTree(std::string name, page_id_t header_page_id, BufferPoolManager *buffer_pool_manager,
                          const KeyComparator &comparator, int leaf_max_size, int internal_max_size)
    : index_name_(std::move(name)),
      bpm_(buffer_pool_manager),
      comparator_(std::move(comparator)),
      leaf_max_size_(leaf_max_size),
      internal_max_size_(internal_max_size),
      header_page_id_(header_page_id) {
  WritePageGuard guard = bpm_->WritePage(header_page_id_);
  auto header_page = guard.AsMut<BPlusTreeHeaderPage>();
  header_page->root_page_id_ = INVALID_PAGE_ID;
  leftmost_leaf_page_id_ = INVALID_PAGE_ID;
}

// begin: added by zhangyu at 2025/9/28 for P2:Task2

/*
 * Find the leaf page that contains the input key
 * @return : 返回的是key在这个internal_page的第几个child point里面
 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::InternalBinarySearch(const InternalPage *internal_page, const KeyType &key) -> int {
  int left = 1;
  /* 根据haoze的测试，key0也被计数了的 */
  int right = internal_page->GetSize();
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (comparator_(key, internal_page->KeyAt(mid)) >= 0) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

/*
 * Find the leaf page that contains the input key
 * @return : page_id of the leaf page
 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::LeafBinarySearch(const LeafPage *leaf_page, const KeyType &key) -> int {
  int left = 0;
  int right = leaf_page->GetSize();
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (comparator_(key, leaf_page->KeyAt(mid)) > 0) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

/*
 * Helper function to decide whether current b+tree is empty
 */
/* 如果根节点的root pageid 是invalid 证明初始化后无操作，所以为空 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::IsEmpty() const -> bool {
  ReadPageGuard guard = bpm_->ReadPage(header_page_id_);
  auto header_page = guard.As<BPlusTreeHeaderPage>();
  return header_page->root_page_id_ == INVALID_PAGE_ID;
}

/*****************************************************************************
 * SEARCH
 *****************************************************************************/
/*
 * Return the only value that associated with input key
 * This method is used for point query
 * @return : true means key exists
 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::GetValue(const KeyType &key, std::vector<ValueType> *result) -> bool {
  // Declaration of context instance.
  /* 为空，则返回false */
  // GetValue不进行修改，所以readpageguard可以drop掉
  Context ctx;
  ReadPageGuard head_guard = bpm_->ReadPage(header_page_id_);
  auto header_page = head_guard.As<BPlusTreeHeaderPage>();
  if (header_page->root_page_id_ == INVALID_PAGE_ID) {
    return false;
  }
  ReadPageGuard read_guard = bpm_->ReadPage(header_page->root_page_id_);
  auto page = read_guard.As<BPlusTreePage>();
  /* if not leaf_node, keep searching */
  while (!page->IsLeafPage()) {
    auto internal_page = read_guard.As<InternalPage>();
    {
      /* 顺序查找，直到 K(i) <= key < K(i+1)*/
      auto page_index = InternalBinarySearch(internal_page, key) - 1;
      read_guard.Drop();
      auto read_page_id = internal_page->ValueAt(page_index);
      read_guard = bpm_->ReadPage(read_page_id);
      page = read_guard.As<BPlusTreePage>();
    }
  }
  /* it's leaf_node */
  auto leaf_page = read_guard.As<LeafPage>();
  auto key_index = LeafBinarySearch(leaf_page, key);
  /* key_index 可能是应该插入的位置，也可能是该节点的位置，所以需要判断
   * 如果不在叶子节点内，那么说明这个key不存在，直接返回
   */
  // begin added by zhangyu at 2026/1/27 for P4T4 for debugging index scan
  if (key_index >= leaf_page->GetSize() || comparator_(key, leaf_page->KeyAt(key_index)) != 0) {
    // fmt::println("GetValue key 不存在");
    // fmt::println("leafpage: {}", leaf_page->ToString());
    return false;
  }
  auto rid_value = leaf_page->ValueAt(key_index);
  result->push_back(static_cast<RID>(rid_value));
  // fmt::println("check key in leaf page {}", leaf_page->ToString());
  // end added by zhangyu at 2026/1/27 for P4T4 for debugging index scan
  return true;
}

/*****************************************************************************
 * Split
 *****************************************************************************/
/*
 * If the insert operation causes the leaf node to become full,
 * this function is called to split the leaf node and adjust the tree structure..
 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Split(Context &ctx, LeafPage *current_page, page_id_t current_pid, const KeyType &key,
                           const ValueType &value, size_t key_index) -> bool {
  //申请分配空间
  std::vector<std::pair<KeyType, ValueType>> leaf_vec;
  leaf_vec.reserve(current_page->GetSize() + 1);
  for (int index = 0; index < current_page->GetSize(); ++index) {
    leaf_vec.push_back(std::make_pair(current_page->KeyAt(index), current_page->ValueAt(index)));
  }
  leaf_vec.insert(leaf_vec.begin() + key_index, std::make_pair(key, value));

  // 申请新结点
  page_id_t right_page_id = bpm_->NewPage();
  WritePageGuard write_guard = bpm_->WritePage(right_page_id);
  auto right_page = write_guard.AsMut<LeafPage>();
  right_page->Init(leaf_max_size_);
  right_page->SetPageType(IndexPageType::LEAF_PAGE);

  auto left_page = current_page;
  auto left_page_id = current_pid;

  auto mid = leaf_vec.size() / 2;
  auto mid_key = leaf_vec[mid].first;
  auto new_size = leaf_vec.size() - mid;

  for (size_t i = 0; i < mid; ++i) {
    left_page->SetKeyAt(i, leaf_vec[i].first);
    left_page->SetValueAt(i, leaf_vec[i].second);
  }
  left_page->SetSize(mid);

  for (size_t i = 0; i < new_size; ++i) {
    right_page->SetKeyAt(i, leaf_vec[i + mid].first);
    right_page->SetValueAt(i, leaf_vec[i + mid].second);
  }
  right_page->SetSize(new_size);

  right_page->SetNextPageId(left_page->GetNextPageId());
  left_page->SetNextPageId(right_page_id);

  auto header_page = ctx.header_page_->AsMut<BPlusTreeHeaderPage>();
  if (current_pid == leftmost_leaf_page_id_.load()) {
    leftmost_leaf_page_id_.store(left_page_id);
  }

  while (!ctx.write_set_.empty()) {
    //队尾的是parent node
    auto parent_guard = std::move(ctx.write_set_.back());
    ctx.write_set_.pop_back();
    auto parent_page = parent_guard.AsMut<InternalPage>();
    auto index = InternalBinarySearch(parent_page, mid_key);
    //如果内部节点没有满，直接将分裂的新叶子插入到新位置
    if (parent_page->GetSize() < parent_page->GetMaxSize()) {
      parent_page->Insert(index, mid_key, right_page_id);
      return true;
    }
    // 内部节点满了，需要分裂父节点
    auto new_internal_page_id = bpm_->NewPage();
    auto new_internal_guard = bpm_->WritePage(new_internal_page_id);
    auto new_internal_page = new_internal_guard.AsMut<InternalPage>();
    new_internal_page->Init(internal_max_size_);
    new_internal_page->SetPageType(IndexPageType::INTERNAL_PAGE);

    //此处复制时，把key[0]也复制进去了，所以后续需要明确，internal节点的kv数组不是对齐的
    std::vector<std::pair<KeyType, page_id_t>> internal_vec;
    internal_vec.reserve(parent_page->GetSize() + 1);
    for (int index = 0; index < parent_page->GetSize(); ++index) {
      internal_vec.push_back(std::make_pair(parent_page->KeyAt(index), parent_page->ValueAt(index)));
    }
    internal_vec.insert(internal_vec.begin() + index, std::make_pair(mid_key, right_page_id));

    // mid_key用于提升
    mid = internal_vec.size() / 2;
    mid_key = internal_vec[mid].first;
    new_size = internal_vec.size() - mid;

    // parent_page 存储[0... mid - 1]
    for (size_t i = 1; i < internal_vec.size() - mid; ++i) {
      parent_page->SetKeyAt(i, internal_vec[i].first);
      parent_page->SetValueAt(i, internal_vec[i].second);
    }
    parent_page->SetSize(mid);

    // new_internal_page 存储[mid + 1 ... internal_vec.size() - 1
    for (size_t i = 1; i < new_size; ++i) {
      new_internal_page->SetKeyAt(i, internal_vec[mid + i].first);
      new_internal_page->SetValueAt(i, internal_vec[mid + i].second);
    }
    // page[0]需要再设置一下
    new_internal_page->SetValueAt(0, internal_vec[mid].second);
    new_internal_page->SetSize(new_size);

    left_page_id = parent_guard.GetPageId();
    right_page_id = new_internal_page_id;
  }

  auto new_root_page_id = bpm_->NewPage();
  auto new_root_page_guard = bpm_->WritePage(new_root_page_id);
  auto new_root_page = new_root_page_guard.AsMut<InternalPage>();
  new_root_page->Init(internal_max_size_);
  new_root_page->SetPageType(IndexPageType::INTERNAL_PAGE);
  //最左边的不计数，只是做占位使用
  new_root_page->SetKeyAt(1, mid_key);
  new_root_page->SetValueAt(0, left_page_id);
  new_root_page->SetValueAt(1, right_page_id);
  new_root_page->SetSize(2);
  ctx.root_page_id_ = new_root_page_id;
  header_page->root_page_id_ = new_root_page_id;
  return true;
}

/*****************************************************************************
 * INSERTION
 *****************************************************************************/
/*
 * Insert constant key & value pair into b+ tree
 * if current tree is empty, start new tree, update root page id and insert
 * entry, otherwise insert into leaf page.
 * @return: since we only support unique key, if user try to insert duplicate
 * keys return false, otherwise return true.
 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Insert(const KeyType &key, const ValueType &value) -> bool {
  // Declaration of context instance.
  Context ctx;
  (void)ctx;
  ctx.header_page_ = bpm_->WritePage(header_page_id_);
  auto header_page = ctx.header_page_->AsMut<BPlusTreeHeaderPage>();
  if (header_page->root_page_id_ == INVALID_PAGE_ID) {
    page_id_t root_page_id = bpm_->NewPage();
    WritePageGuard write_guard = bpm_->WritePage(root_page_id);
    auto root_page = write_guard.AsMut<LeafPage>();
    //插入键值对
    root_page->Init(leaf_max_size_);
    root_page->Insert(0, key, value);
    root_page->SetNextPageId(INVALID_PAGE_ID);
    header_page->root_page_id_ = root_page_id;
    leftmost_leaf_page_id_.store(root_page_id);
    return true;
  }

  {
    page_id_t write_page_id = header_page->root_page_id_;
    WritePageGuard write_guard = bpm_->WritePage(write_page_id);
    auto current_page = write_guard.AsMut<BPlusTreePage>();
    // 找到应该插入的位置
    while (!current_page->IsLeafPage()) {
      auto internal_page = write_guard.AsMut<InternalPage>();
      //先判断这个结点是否是安全的，是的话则把之前的结点释放了
      if (internal_page->IsSafeInternalForInsert(1)) {
        for (auto &parent_guard : ctx.write_set_) {
          parent_guard.Drop();
        }
        ctx.write_set_.clear();
      }
      // 找到应该插入的位置
      ctx.write_set_.push_back(std::move(write_guard));
      auto page_index = InternalBinarySearch(internal_page, key) - 1;
      write_page_id = internal_page->ValueAt(page_index);
      write_guard = bpm_->WritePage(write_page_id);
      current_page = write_guard.AsMut<BPlusTreePage>();
    }

    // 找到插入的叶子节点
    auto leaf_page = write_guard.AsMut<LeafPage>();
    auto key_index = LeafBinarySearch(leaf_page, key);

    // 只支持唯一key
    if (key_index != leaf_page->GetSize() && comparator_(key, leaf_page->KeyAt(key_index)) == 0) {
      return false;
    }

    //节点已满，需要分裂
    if (leaf_page->GetSize() == leaf_page->GetMaxSize()) {
      Split(ctx, leaf_page, write_page_id, key, value, key_index);
      goto block1;
    }
    // 节点未满，直接插入即可
    leaf_page->Insert(key_index, key, value);
  }
block1:
  //   for (auto &guard : ctx.write_set_) {
  //         guard.Drop();
  //   }
  //   std::cout << "insert key " << key <<std::endl;
  //   std::ostringstream out_buf;
  //   PrintableBPlusTree p_root = ToPrintableBPlusTree(header_page->root_page_id_);
  //   p_root.Print(out_buf);
  //   std::cout << out_buf.str() << std::endl;
  return true;
}

// begin: added by zhangyu at 2025/10/15 for P2:Task2
INDEX_TEMPLATE_ARGUMENTS
void BPLUSTREE_TYPE::RedistributionLeaf(LeafPage *current_page, LeafPage *sibling_page, InternalPage *parent_page,
                                        bool isright, int current_index, int sibling_index) {
  //针对向右借1个
  if (isright) {
    current_page->Insert(current_page->GetSize(), sibling_page->KeyAt(0), sibling_page->ValueAt(0));
    // sibling_page 删除第一个
    sibling_page->Delete(0);

    //修改父结点的separator key为sibling_page的第一个key
    parent_page->SetKeyAt(sibling_index, sibling_page->KeyAt(0));
    return;
  }
  //针对向左借
  // current_page第一个需要插入左兄弟的尾巴
  current_page->Insert(0, sibling_page->KeyAt(sibling_page->GetSize() - 1),
                       sibling_page->ValueAt(sibling_page->GetSize() - 1));

  // sibling 删除末尾
  sibling_page->Delete(sibling_page->GetSize() - 1);
  //修改父节点
  parent_page->SetKeyAt(current_index, current_page->KeyAt(0));
}

INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::MergeLeaf(LeafPage *left_page, LeafPage *right_page, int left_index, int right_index,
                               InternalPage *parent_page) -> InternalPage * {
  // 将右节点的全部挪到左节点
  int left_size = left_page->GetSize();
  for (int i = 0; i < right_page->GetSize(); ++i) {
    left_page->Insert(i + left_size, right_page->KeyAt(i), right_page->ValueAt(i));
  }
  // parent_page->SetKeyAt(left_index, left_page->KeyAt(0));
  left_page->SetNextPageId(right_page->GetNextPageId());
  //删除右边结点
  // free(right_page);

  //父结点删除k-v对
  parent_page->Delete(right_index);
  return parent_page;
}

/*
 * internal 的借用和leaf的借用不同。把父节点的key下移给当前页，再把兄弟一侧的第一个child借过来
 * 然后把兄弟相应的新边界 key 上提到父节点，替换原 separator
 */
INDEX_TEMPLATE_ARGUMENTS
void BPLUSTREE_TYPE::RedistributionInternal(InternalPage *current_page, InternalPage *sibling_page,
                                            InternalPage *parent_page, bool isright, int current_index,
                                            int sibling_index) {
  //针对向右借1个
  if (isright) {
    //父节点下移动key给current, 兄弟的value[0]借过来给current的value[current_size]
    auto kp = parent_page->KeyAt(sibling_index);
    current_page->Insert(current_page->GetSize(), kp, sibling_page->ValueAt(0));

    // sibling的key[1]上提到父节点替换原separator
    parent_page->SetKeyAt(sibling_index, sibling_page->KeyAt(1));

    // sibling删除第一个元素
    // 注意此时需要先将value[0]用value[1]覆盖掉，后面删除操作就统一了
    sibling_page->SetValueAt(0, sibling_page->ValueAt(1));
    sibling_page->Delete(1);
    return;
  }

  // 如果current_page已经是最右边的，那么需要向左借
  //父节点下移动key给current
  auto kp = parent_page->KeyAt(current_index);

  // current_page头节点插入
  // current_page->Insert(1, kp, sibling_page->ValueAt(sibling_page->GetSize() - 1));
  for (int i = current_page->GetSize(); i > 0; --i) {
    current_page->SetKeyAt(i, current_page->KeyAt(i - 1));
    current_page->SetValueAt(i, current_page->ValueAt(i - 1));
  }
  current_page->SetKeyAt(1, kp);
  current_page->SetValueAt(1, current_page->ValueAt(0));
  current_page->SetValueAt(0, sibling_page->ValueAt(sibling_page->GetSize() - 1));
  current_page->ChangeSizeBy(1);
  // LOG_DEBUG("RedistributionInternal current_page");
  //原来kp被替换
  parent_page->SetKeyAt(current_index, sibling_page->KeyAt(sibling_page->GetSize() - 1));
  sibling_page->Delete(sibling_page->GetSize() - 1);
}

/*
 * 把父节点中间分割他们的key下移动到左边，然后再把右边内容接到左页后面
 * left + parent_separator + right
 * 父节点删除移动的key和指向右的指针
 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::MergeInternal(InternalPage *left_page, InternalPage *right_page, int current_index,
                                   int sibling_index, InternalPage *parent_page, bool isright) -> InternalPage * {
  KeyType kp;
  if (isright) {
    // c -> kp -> s
    kp = parent_page->KeyAt(sibling_index);
  } else {
    // s-> kp -> c
    kp = parent_page->KeyAt(current_index);
  }

  // kp下移，且右边追加到左边
  // LOG_DEBUG("MergeInternal 这个函数里执行left_page->Insert 1");
  left_page->Insert(left_page->GetSize(), kp, right_page->ValueAt(0));

  // int offset = left_page->GetSize() - 1;

  for (int i = 1; i < right_page->GetSize(); ++i) {
    // LOG_DEBUG("MergeInternal 这个函数里执行left_page->Insert 2");
    left_page->Insert(left_page->GetSize(), right_page->KeyAt(i), right_page->ValueAt(i));
  }
  //删除右边结点
  // free(right_page);
  // 父结点删除kp
  if (isright) {
    // c -> kp -> s
    // LOG_DEBUG("MergeInternal 这个函数里执行parent_page->Delete 1");
    parent_page->Delete(sibling_index);
  } else {
    // s-> kp -> c
    // LOG_DEBUG("MergeInternal 这个函数里执行parent_page->Delete 2");
    parent_page->Delete(current_index);
  }

  return parent_page;
}
// end: added by zhangyu at 2025/10/15 for P2:Task2
/*****************************************************************************
 * REMOVE
 *****************************************************************************/
/*
 * Delete key & value pair associated with input key
 * If current tree is empty, return immediately.
 * If not, User needs to first find the right leaf page as deletion target, then
 * delete entry from leaf page. Remember to deal with redistribute or merge if
 * necessary.
 */
//  B+树删除的整体流程
// 1、找到目标 key 所在的叶子节点。
// 2、如果该叶子节点同时也是根节点：
// 	直接删除 key；
// 	若删后为空，则整棵树置空；
// 	结束。
// 3、如果该叶子不是根节点：
// 	删除该 key；
// 	若叶子仍满足最小占用要求，则必要时更新父节点分隔键，结束。
// 4、如果叶子 underflow：
// 	若兄弟可借，则做叶子重分配，并更新父节点分隔键，结束；
// 	若兄弟不可借，则做叶子合并，并从父节点删除对应 separator 和 child pointer。
// 5、父节点若因此 underflow，则继续向上处理内部节点：
// 	若兄弟可借，则做内部节点重分配：父 separator 下移、兄弟 child 借入、新 separator 上提，并更新 child 的 parent
// 指针，结束；
//      若兄弟不可借，则做内部节点合并：父 separator 下移后与左右 internal 合并，并从父节点删除对应 separator 和 child
//      pointer。
// 6、不断向上回溯，直到某一层恢复合法，或者处理到根节点。
// 7、若根节点最终只剩一个 child，则缩高，让该 child 成为新根；若根为空，则树为空。
INDEX_TEMPLATE_ARGUMENTS
void BPLUSTREE_TYPE::Remove(const KeyType &key) {
  // Declaration of context instance.
  //   LOG_DEBUG("Remove 开始执行  key");
  //   std::cout << "delete key: " << key << std::endl;
  //   std::cout << DrawBPlusTree() << std::endl;
  Context ctx;
  (void)ctx;
  ctx.header_page_ = bpm_->WritePage(header_page_id_);
  auto header_page = ctx.header_page_->AsMut<BPlusTreeHeaderPage>();
  if (header_page->root_page_id_ == INVALID_PAGE_ID) {
    return;
  }
  {
    // BLOCK1: 先找到key所在的叶子节点和leaf_index
    page_id_t current_page_id = header_page->root_page_id_;
    WritePageGuard current_guard = bpm_->WritePage(current_page_id);
    auto check_page = current_guard.AsMut<BPlusTreePage>();
    // 找到
    while (!check_page->IsLeafPage()) {
      auto internal_page = current_guard.AsMut<InternalPage>();
      // 先判断这个内部节点，即使删除了1个位置也不会underflow，那么它不用调整，从写集中删除
      if (internal_page->IsSafeInternalForDelete(1)) {
        for (auto &parent_guard : ctx.write_set_) {
          parent_guard.Drop();
        }
        ctx.write_set_.clear();
      }
      // key所在的叶子节点的父节点，和这个叶子节点在父节点中的index
      ctx.write_set_.push_back(std::move(current_guard));
      auto page_index = InternalBinarySearch(internal_page, key) - 1;
      current_page_id = internal_page->ValueAt(page_index);
      current_guard = bpm_->WritePage(current_page_id);
      check_page = current_guard.AsMut<BPlusTreePage>();
    }

    // 找到key所在叶子节点和leaf_index
    auto leaf_page = current_guard.AsMut<LeafPage>();
    auto leaf_index = LeafBinarySearch(leaf_page, key);

    // 如果不在叶子节点内，那么说明这个key不存在，直接返回
    if (leaf_index >= leaf_page->GetSize() || comparator_(key, leaf_page->KeyAt(leaf_index)) != 0) {
      return;
    }

    // BLOCK2: 当前节点即为根结点又为叶节点, 直接删除key，如果删除后为空了，整颗树为空
    if (current_page_id == header_page->root_page_id_) {
      leaf_page->Delete(leaf_index);
      if (leaf_page->GetSize() == 0) {
        //此时b+树为空了
        header_page->root_page_id_ = INVALID_PAGE_ID;
        current_guard.Drop();
      }
      return;
    }

    // BLOCK3: 该叶子节点不是根节点，先删除key
    leaf_page->Delete(leaf_index);

    InternalPage *current_page;
    auto parent_guard = std::move(ctx.write_set_.back());
    ctx.write_set_.pop_back();
    // 获取到父亲结点parent_page, 当前结点在父亲节点中对应的指针的下标curpage_index
    auto parent_page = parent_guard.AsMut<InternalPage>();
    auto current_index = InternalBinarySearch(parent_page, key) - 1;

    // 当前叶子如果安全，那么直接调整父节点的separator key后就可以结束了
    if (leaf_page->IsSafeLeafForDelete(0)) {
      parent_page->SetKeyAt(current_index, leaf_page->KeyAt(0));
      goto block1;
    }

    // BLOCK4:叶子结点underflow，需要重新分配或者合并
    // 先根据parent_page找到其兄弟sibling_page
    int sibling_index;
    page_id_t sibling_page_id;
    bool isright;
    // 当前节点有右兄弟节点
    if (current_index < parent_page->GetSize() - 1) {
      sibling_index = current_index + 1;
      isright = true;  //证明需要向右借，或者合并
    } else {
      sibling_index = current_index - 1;
      isright = false;  //证明需要向左借，或者合并
    }
    sibling_page_id = parent_page->ValueAt(sibling_index);

    WritePageGuard sibling_guard = bpm_->WritePage(sibling_page_id);
    auto sibling_page = sibling_guard.AsMut<LeafPage>();

    // sibling可以借一个结点，重新分配即可
    if (sibling_page->IsSafeLeafForDelete(1)) {
      RedistributionLeaf(leaf_page, sibling_page, parent_page, isright, current_index, sibling_index);
      goto block1;
    }
    // 兄弟借出后不安全，那么需要合并，统一向左边那个结点合并
    LeafPage *left_page;
    LeafPage *right_page;
    int left_index;
    int right_index;
    if (isright) {
      left_page = leaf_page;
      right_page = sibling_page;
      left_index = current_index;
      right_index = sibling_index;
    } else {
      left_page = sibling_page;
      right_page = leaf_page;
      left_index = sibling_index;
      right_index = current_index;
      if (current_guard.GetPageId() == leftmost_leaf_page_id_.load()) {
        leftmost_leaf_page_id_.store(sibling_guard.GetPageId());
      }
    }
    current_page = MergeLeaf(left_page, right_page, left_index, right_index, parent_page);
    current_guard = std::move(parent_guard);

    // BLOCK5: MergeLeaf之后, 父节点(internal
    // node)的key和value都会被删除一个，所以需要判断父节点是否安全，如果不安全则继续向上调整
    while (!ctx.write_set_.empty()) {
      auto parent_guard = std::move(ctx.write_set_.back());
      ctx.write_set_.pop_back();
      auto parent_page = parent_guard.AsMut<InternalPage>();
      // 如果当前结点安全，则直接释放父节点
      if (current_page->IsSafeInternalForDelete(1)) {
        for (auto &guard : ctx.write_set_) {
          guard.Drop();
        }
        ctx.write_set_.clear();
      }

      // parent_page是current_page的父节点，current_index是current_page在parent_page中的index
      auto current_index = InternalBinarySearch(parent_page, key) - 1;

      // current_page如果删除一个仍然安全，则不需要调整
      if (current_page->IsSafeInternalForDelete(1)) {
        goto block1;
      }

      // 先根据parent_page找到其兄弟sibling_page
      int sibling_index;
      page_id_t sibling_page_id;
      bool isright;
      if (current_index < parent_page->GetSize() - 1) {
        sibling_index = current_index + 1;
        isright = true;  //证明需要向右借，或者合并
      } else {
        sibling_index = current_index - 1;
        isright = false;  //证明需要向左借，或者合并
      }
      sibling_page_id = parent_page->ValueAt(sibling_index);

      WritePageGuard sibling_guard = bpm_->WritePage(sibling_page_id);
      auto sibling_page = sibling_guard.AsMut<InternalPage>();

      // 兄弟节点可以借出一个结点，重新分配即可
      if (sibling_page->IsSafeInternalForDelete(1)) {
        RedistributionInternal(current_page, sibling_page, parent_page, isright, current_index, sibling_index);
        goto block1;
      }
      // 兄弟也不足，那么需要合并
      if (isright) {
        // LOG_DEBUG("Remove 开始执行MergeInternal 1 current.size(): %u, sibling.size(): %u", current_page->GetSize(),
        //           sibling_page->GetSize());
        current_page = MergeInternal(current_page, sibling_page, current_index, sibling_index, parent_page, isright);
      } else {
        // LOG_DEBUG("Remove 开始执行MergeInternal 2");
        current_page = MergeInternal(sibling_page, current_page, current_index, sibling_index, parent_page, isright);
      }
      current_guard = std::move(parent_guard);
    }

    // 当前节点是根节点，并且只剩下一个孩子了，那么就缩高，让这个孩子成为新的根节点
    if (current_guard.GetPageId() == header_page->root_page_id_ && current_page->GetSize() == 1) {
      header_page->root_page_id_ = current_page->ValueAt(0);
    }
  }
block1:
  // for (auto &guard : ctx.write_set_) {
  //   guard.Drop();
  // }
  // std::ostringstream out_buf;
  // PrintableBPlusTree p_root = ToPrintableBPlusTree(header_page->root_page_id_);
  // p_root.Print(out_buf);
  // std::cout << "delete key" << key << std::endl;
  // std::cout << out_buf.str() << std::endl;
  return;
}
// end: added by zhangyu at 2025/9/28 for P2:Task2

/*****************************************************************************
 * INDEX ITERATOR
 *****************************************************************************/
/*
 * Input parameter is void, find the leftmost leaf page first, then construct
 * index iterator
 * @return : index iterator
 */
// begin: Mod by zhangyu at 2025/10/14 for P2:Task3
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Begin() -> INDEXITERATOR_TYPE {
  auto leftmost_leaf_page_id = leftmost_leaf_page_id_.load();
  BUSTUB_ASSERT(leftmost_leaf_page_id != INVALID_PAGE_ID, "Invalid page_id in B plus Begin()");
  return INDEXITERATOR_TYPE(leftmost_leaf_page_id, 0, bpm_);
}

/*
 * Input parameter is low key, find the leaf page that contains the input key
 * first, then construct index iterator
 * @return : index iterator
 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Begin(const KeyType &key) -> INDEXITERATOR_TYPE {
  auto root_page_id = GetRootPageId();
  if (root_page_id == INVALID_PAGE_ID) {
    return End();
  }
  ReadPageGuard read_guard = bpm_->ReadPage(root_page_id);
  auto page = read_guard.As<BPlusTreePage>();
  auto current_page_id = root_page_id;
  while (!page->IsLeafPage()) {
    auto internal_page = read_guard.As<InternalPage>();
    auto page_index = InternalBinarySearch(internal_page, key) - 1;
    current_page_id = internal_page->ValueAt(page_index);
    read_guard = bpm_->ReadPage(current_page_id);
    page = read_guard.As<BPlusTreePage>();
  }
  auto leaf_page = read_guard.As<LeafPage>();
  auto index = LeafBinarySearch(leaf_page, key);
  return INDEXITERATOR_TYPE(current_page_id, index, bpm_);
}

/*
 * Input parameter is void, construct an index iterator representing the end
 * of the key/value pair in the leaf node
 * @return : index iterator
 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::End() -> INDEXITERATOR_TYPE { return INDEXITERATOR_TYPE(INVALID_PAGE_ID, -1, bpm_); }
// end: Mod by zhangyu at 2025/10/14 for P2:Task3

/**
 * @return Page id of the root of this tree
 */
/* 不为空时，b+树的根节点的root pageid指向自己 */
INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::GetRootPageId() -> page_id_t {
  ReadPageGuard guard = bpm_->ReadPage(header_page_id_);
  auto root_page = guard.As<BPlusTreeHeaderPage>();
  return root_page->root_page_id_;
}

template class BPlusTree<GenericKey<4>, RID, GenericComparator<4>>;

template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>>;

template class BPlusTree<GenericKey<16>, RID, GenericComparator<16>>;

template class BPlusTree<GenericKey<32>, RID, GenericComparator<32>>;

template class BPlusTree<GenericKey<64>, RID, GenericComparator<64>>;

}  // namespace bustub
