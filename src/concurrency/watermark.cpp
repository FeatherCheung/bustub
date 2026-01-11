#include "concurrency/watermark.h"
#include <exception>
#include "common/exception.h"

namespace bustub {

// begin: mod by zhangyu for p4t2 at 2025/12/10
auto Watermark::AddTxn(timestamp_t read_ts) -> void {
  current_reads_.insert(read_ts);
  watermark_ = *current_reads_.begin();
}

auto Watermark::RemoveTxn(timestamp_t read_ts) -> void {
  auto it = current_reads_.find(read_ts);
  if (it != current_reads_.end()) {
    current_reads_.erase(it);
  }

  watermark_ = current_reads_.empty() ? commit_ts_ : *current_reads_.begin();
}
// end: mod by zhangyu for p4t2 at 2025/12/10

}  // namespace bustub
