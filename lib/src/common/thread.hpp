// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef COMMON_THREAD_HPP_
#define COMMON_THREAD_HPP_

#include <iosfwd>
#include <vector>
#include <thread>

namespace VideoStitch {
/**
 * @brief A simple thread class.
 *
 * Just implement run() to use.
 *
 * Typical use:
 * class MyThread: public Thread {
 * public:
 *   void run() { ... }
 * };
 * MyThread t;
 * t.start();
 * ...
 * t.join();
 */
class Thread {
 public:
  virtual ~Thread();

  /**
   * Thread worker function.
   */
  virtual void run() = 0;

  /**
   * Starts the thread. Must be called before join().
   * Returns immediately, the thread may or may not have started executing yet.
   * join() must be called between two calls to start().
   */
  void start();

  /**
   * Blocks until done. Must be called exactly one after each call to start().
   */
  void join();

 protected:
  Thread();

 private:
  static void runPrivate(Thread* thread);
  std::thread* thread;
};

/**
 * @brief A very simple thread pool class.
 *
 * Intended for small pools (i.e. a few threads).
 * Take care if your tasks depend on one another.
 */
class ThreadPool {
 public:
  /**
   * @brief A task for the thread pool.
   */
  class Task {
   public:
    virtual ~Task();

    /**
     * @brief Run the task.
     */
    virtual void run() = 0;

   protected:
    Task();
  };

  /**
   * Creates a thread pool.
   * @param numThreads The number of concurrent threads.
   */
  explicit ThreadPool(int numThreads);

  /**
   * Waits for all tasks to finish.
   */
  ~ThreadPool();

  /**
   * Run a task. The task is deleted after completion.
   * @return false if there are no available workers.
   * @note NOT thread safe.
   */
  bool tryRun(Task* task);

  /**
   * Waits for all tasks to finish.
   */
  void waitAll();

 private:
  std::vector<Thread*> workers;
};

/**
 * @brief Gets the number of available CPU cores.
 */
int getNumCores();

}  // namespace VideoStitch

#endif
