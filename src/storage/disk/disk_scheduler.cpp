//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// disk_scheduler.cpp
//
// Identification: src/storage/disk/disk_scheduler.cpp
//
// Copyright (c) 2015-2023, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "storage/disk/disk_scheduler.h"
#include "common/exception.h"
#include "storage/disk/disk_manager.h"

namespace bustub {

DiskScheduler::DiskScheduler(DiskManager *disk_manager) : disk_manager_(disk_manager) {
  // TODO(P1): remove this line after you have implemented the disk scheduler API
  //   throw NotImplementedException(
  //       "DiskScheduler is not implemented yet. If you have finished implementing the disk scheduler, please remove
  //       the " "throw exception line in `disk_scheduler.cpp`.");

  // Spawn the background thread
  background_thread_.emplace([&] { StartWorkerThread(); });
}

DiskScheduler::~DiskScheduler() {
  // Put a `std::nullopt` in the queue to signal to exit the loop
  request_queue_.Put(std::nullopt);
  if (background_thread_.has_value()) {
    background_thread_->join();
  }
}

// begin: modified by zhangyu at 2025/9/14
void DiskScheduler::Schedule(DiskRequest r) {
  // 将请求放入线程安全队列
  // 这里的move是将r直接放到channel中而不是复制
  request_queue_.Put(std::move(r));
}

void DiskScheduler::StartWorkerThread() {
  while (true) {
    // 从队列取出一个请求，如果队列为空，这里会阻塞等待
    std::optional<DiskRequest> opt_request = request_queue_.Get();

    // 如果取出的值是 std::nullopt，说明析构函数希望线程退出
    if (!opt_request.has_value()) {
      break;
    }

    // 获取请求对象
    DiskRequest request = std::move(opt_request.value());

    // 执行磁盘操作
    if (request.is_write_) {
      disk_manager_->WritePage(request.page_id_, request.data_);
    } else {
      disk_manager_->ReadPage(request.page_id_, request.data_);
    }

    // 标记请求完成，通知调用者
    request.callback_.set_value(true);
  }
}

// end: modified by zhangyu at 2025/9/14
}  // namespace bustub
