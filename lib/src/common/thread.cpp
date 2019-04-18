// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "thread.hpp"

#include "libvideostitch/utils/semaphore.hpp"

#ifdef _MSC_VER
#include <ws2tcpip.h>  //must be included before Windows.h
#include <iphlpapi.h>
#include <windows.h>
#else
#include <sys/param.h>
#if (!defined __ANDROID__)
#include <sys/sysctl.h>
#include <ifaddrs.h>
#endif /*	(!defined __ANDROID__)	*/
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <cassert>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>

namespace VideoStitch {

Thread::Thread() : thread(NULL) {}

Thread::~Thread() { assert(!thread); }

void Thread::start() {
  assert(!thread);
  thread = new std::thread(Thread::runPrivate, this);
}

void Thread::join() {
  thread->join();
  delete thread;
  thread = NULL;
}

void Thread::runPrivate(Thread* thread) { thread->run(); }

ThreadPool::Task::Task() {}
ThreadPool::Task::~Task() {}

namespace {
class ThreadPoolWorker : public Thread {
 public:
  ThreadPoolWorker() : busy(1), work(0), task(NULL), dead(false) {}

  ~ThreadPoolWorker() {
    // check that task has been deleted
    assert(task == NULL);
  }

  bool runTaskIfNotBusy(ThreadPool::Task* taskToRun) {
    if (busy.wait_for(0)) {
      assert(!task);
      assert(taskToRun);
      task = taskToRun;
      work.notify();
      return true;
    } else {
      return false;
    }
  }

  void run() {
    for (;;) {
      work.wait();
      if (dead) {
        return;
      }
      assert(task);
      task->run();
      delete task;
      task = NULL;
      busy.notify();
    }
  }

  void kill() {
    busy.wait();
    dead = true;
    work.notify();
    join();
  }

  void wait() {
    busy.wait();
    busy.notify();
  }

 private:
  Semaphore busy;
  Semaphore work;
  ThreadPool::Task* task;
  bool dead;
};
}  // namespace

ThreadPool::ThreadPool(int numThreads) {
  for (int i = 0; i < numThreads; ++i) {
    workers.push_back(new ThreadPoolWorker());
    assert(workers.back());  // TODO: should fail
    workers.back()->start();
  }
}

ThreadPool::~ThreadPool() {
  for (size_t i = 0; i < workers.size(); ++i) {
    static_cast<ThreadPoolWorker*>(workers[i])->kill();
    delete workers[i];
  }
}

bool ThreadPool::tryRun(Task* task) {
  for (size_t i = 0; i < workers.size(); ++i) {
    if (static_cast<ThreadPoolWorker*>(workers[i])->runTaskIfNotBusy(task)) {
      return true;
    }
  }
  return false;
}

void ThreadPool::waitAll() {
  for (size_t i = 0; i < workers.size(); ++i) {
    static_cast<ThreadPoolWorker*>(workers[i])->wait();
  }
}

int getNumCores() {
#ifdef WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;
#elif MACOS
  int nm[2];
  size_t len = 4;
  uint32_t count;

  nm[0] = CTL_HW;
  nm[1] = HW_AVAILCPU;
  sysctl(nm, 2, &count, &len, NULL, 0);

  if (count < 1) {
    nm[1] = HW_NCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);
    if (count < 1) {
      count = 1;
    }
  }
  return count;
#else
  return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

}  // namespace VideoStitch
