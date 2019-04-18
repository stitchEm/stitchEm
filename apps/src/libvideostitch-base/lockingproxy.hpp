// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LOCKINGPROXY_HPP
#define LOCKINGPROXY_HPP

namespace VideoStitch {
namespace Helper {

/**
 * @brief A proxy class which locks the returned pointer. The proxified class must provide both member functions
 * lock() and unlock(). Re-entrant mutexes are recommended.
 *
 * Rationale: while you can lock accessors within class L, you cannot ensure thread-safeness when returning a pointer
 * (or reference) to a contained object T.
 */
template <class T, class L>
class LockingProxy {
 public:
  LockingProxy(T *ptr, L *locker) : ptr(ptr), locker(locker) {
    if (ptr) {
      locker->lock();
    }
  }

  virtual ~LockingProxy() {
    if (ptr) {
      locker->unlock();
    }
  }

  T *operator->() const { return ptr; }

  const T *get() const { return ptr; }

 private:
  friend L;  // C++11

  /**
   * @brief Forbids assignments. Wrap into a smart pointer if needed.
   */
  LockingProxy &operator=(const LockingProxy &);

  /**
   * @brief Forbids copy. We don't want the caller to create deadlocks inadvertently.
   */
  LockingProxy(const LockingProxy &);

  /**
   * @brief ptr The proxified object. Access it thru the -> operator.
   */
  T *ptr;

  /**
   * @brief locker The class holding the lock. Must provide lock() and unlock() member functions.
   */
  L *locker;
};

}  // namespace Helper
}  // namespace VideoStitch

#endif  // LOCKINGPROXY_HPP
