// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

/**
 * Bounded, thread-safe data structures
 */

#include <condition_variable>
#include <mutex>
#include <queue>

namespace VideoStitch {

struct queue_tag {};
struct priority_queue_tag {};

template <typename Container>
struct container_traits;
template <typename T>
struct container_traits<std::queue<T>> {
  typedef queue_tag container_category;
};
template <typename T, typename Compare>
struct container_traits<std::priority_queue<T, std::vector<T>, Compare>> {
  typedef priority_queue_tag container_category;
};

template <typename Container>
class BoundedThreadSafeQueue {
 public:
  typedef typename Container::value_type T;

  explicit BoundedThreadSafeQueue(size_t boundary = std::numeric_limits<size_t>::max())
      : boundary(boundary), stp(false) {}

  ~BoundedThreadSafeQueue() {}

  void push(T& item) {
    {
      std::unique_lock<std::mutex> sl(mutex);
      condv.wait(sl, [&]() { return queue.size() < boundary || stp; });
      queue.push(std::move(item));
    }
    condv.notify_all();
  }

  template <typename Predicate>
  bool front(Predicate lambda) {
    std::unique_lock<std::mutex> sl(mutex);
    condv.wait(sl, [&]() { return !queue.empty() || stp; });
    typename container_traits<Container>::container_category category;
    return lambda(front_dispatch(category));
  }

  bool pop(T& item) {
    {
      std::unique_lock<std::mutex> sl(mutex);
      condv.wait(sl, [&]() { return !queue.empty() || stp; });
      if (stp && queue.empty()) {
        return false;
      }
      typename container_traits<Container>::container_category category;
      item = front_dispatch(category);
      queue.pop();
    }
    condv.notify_all();
    return true;
  }

  typename std::queue<T>::size_type size() {
    std::lock_guard<std::mutex> sl(mutex);
    return queue.size();
  }

  void stop() {
    {
      std::lock_guard<std::mutex> sl(mutex);
      stp = true;
    }
    condv.notify_all();
  }

 private:
  T front_dispatch(queue_tag) {
    return std::move(queue.front());  // invalidates the front effectively
  }
  T front_dispatch(priority_queue_tag) {
    return std::move(queue.top());  // invalidates the top effectively
  }

  Container queue;
  std::mutex mutex;
  std::condition_variable condv;
  size_t boundary;
  bool stp;
};

template <typename T, typename Compare>
using PriorityQueue = BoundedThreadSafeQueue<std::priority_queue<T, std::vector<T>, Compare>>;
template <typename T>
using Queue = BoundedThreadSafeQueue<std::queue<T>>;
}  // namespace VideoStitch
