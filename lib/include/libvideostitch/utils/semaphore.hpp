// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SEMAPHORE_HPP_
#define SEMAPHORE_HPP_

#include "../config.hpp"

#include <mutex>
#include <condition_variable>

namespace VideoStitch {

/** \class Semaphore
 * \brief Custom Semaphore implementation.
 * On notify we increase internal counter.
 * On wait - we return if the counter is greater then 0 otherwise - we wait for that to happen.
 * On wait_for - we return if the counter is greater then 0 otherwise - we wait for that to happen or until timeout is
 * expired. In the first case we return true, in the latter case - false.
 */
class VS_EXPORT Semaphore {
  // NOTE: only the thread that locked the mutex can unlock it.
 public:
  /**
   * @brief Create Semaphore with given initial counter
   * @param initialCount - initial value of Semaphore counter.
   */
  explicit Semaphore(int initialCount = 1);

  /**
   * @brief Increase Semaphore counter.
   */
  void notify();

  /**
   * @brief Return if the counter is greater then 0 and decrease counter. Otherwise wait until counter is greater then
   * 0.
   */
  void wait();

  /**
   * @brief Return if the counter is greater then 0 and decrease counter. Otherwise wait until counter is greater then 0
   * or timeout is expired.
   * @return False if timeout expired. True otherwise.
   */
  bool wait_for(unsigned timeOutMs);

 private:
  volatile int count;
  std::mutex mutex;
  std::condition_variable condition;
};
}  // namespace VideoStitch

#endif
