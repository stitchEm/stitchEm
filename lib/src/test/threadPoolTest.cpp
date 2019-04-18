// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <common/thread.hpp>
#include "libvideostitch/utils/semaphore.hpp"

namespace VideoStitch {
namespace Testing {
/**
 * A task that blocks until someone unblocks it.
 */
class BlockingTask : public ThreadPool::Task {
 public:
  BlockingTask(int id, Semaphore* done) : id(id), sem(0), done(done) {}

  void run() {
    sem.wait();
    printf("task %d running\n", id);
    done->notify();
  }

  void unblock() { sem.notify(); }

 private:
  int id;
  Semaphore sem;
  Semaphore* done;
};

void threadPoolTest() {
  ThreadPool pool(3);

  Semaphore task1Done(0), task2Done(0), task3Done(0), task4Done(0);
  BlockingTask* task1 = new BlockingTask(1, &task1Done);
  BlockingTask* task2 = new BlockingTask(2, &task2Done);
  BlockingTask* task3 = new BlockingTask(3, &task3Done);
  BlockingTask* task4 = new BlockingTask(4, &task4Done);

  ENSURE(pool.tryRun(task1));
  ENSURE(pool.tryRun(task2));
  ENSURE(pool.tryRun(task3));
  ENSURE(!pool.tryRun(task4));

  task1->unblock();
  task1Done.wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ENSURE(pool.tryRun(task4));
  task2->unblock();
  task3->unblock();
  task4->unblock();
  task2Done.wait();
  pool.waitAll();
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::threadPoolTest();
  return 0;
}
