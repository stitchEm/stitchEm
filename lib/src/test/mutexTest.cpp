// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/utils/semaphore.hpp"
#include "libvideostitch/logging.hpp"

#include <mutex>
#include <thread>

namespace VideoStitch {
namespace Testing {
void basicMonoThreadTest() {
  std::mutex m;
  // not held, tryLock must succeed.
  ENSURE(m.try_lock(), "failed to lock unlocked mutex");
  // release
  m.unlock();
  // not held, tryLock must succeed.
  ENSURE(m.try_lock(), "failed to lock unlocked mutex");
  // release
  m.unlock();
  // not held, lock it.
  m.lock();
  // release
  m.unlock();
  // not held, tryLock must succeed.
  ENSURE(m.try_lock(), "failed to lock unlocked mutex");
  // release
  m.unlock();
}

/**
 * Testing semaphore
 */

void semWorker(Semaphore* s) {
  Logger::get(Logger::Info) << "worker starts." << std::endl;
  Logger::get(Logger::Info) << "worker wait()ing..." << std::endl;
  s->wait();
  Logger::get(Logger::Info) << "got notified." << std::endl;
  Logger::get(Logger::Info) << "worker wait()ing..." << std::endl;
  s->wait();
  Logger::get(Logger::Info) << "got notified." << std::endl;
  Logger::get(Logger::Info) << "worker done." << std::endl;
}

void basicSemaphoreTest() {
  Logger::get(Logger::Info) << "create sem with initial count of 1." << std::endl;
  Semaphore s(1);

  Logger::get(Logger::Info) << "create worker." << std::endl;
  std::thread thread(&semWorker, &s);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  Logger::get(Logger::Info) << "main notify" << std::endl;
  s.notify();
  Logger::get(Logger::Info) << "join()ing..." << std::endl;
  thread.join();
  Logger::get(Logger::Info) << "main done" << std::endl << std::endl;
}

/**
 * L means call lock()
 * U means unlock()
 * S means start
 * - means mutex owner
 * . means not mutex owner
 * z means sleep
 *
 * worker              S.T.L.........---zzzzzzzzzzzzz--U...........
 * main thread   S..L-----zzzzzzz--U..zzzzz..L............------...
 */

struct MSS {
  MSS(std::mutex* m, Semaphore* workerSem, Semaphore* mainSem) : m(m), workerSem(workerSem), mainSem(mainSem) {}
  std::mutex* m;
  Semaphore* workerSem;
  Semaphore* mainSem;
};

void worker(MSS* mss) {
  Logger::get(Logger::Info) << "worker starts" << std::endl;
  // The mutex is initially locked, try_lock shoudl fail.
  Logger::get(Logger::Info) << "worker trylock()" << std::endl;
  ENSURE(!mss->m->try_lock(), "locked an already locked mutex ?!");

  Logger::get(Logger::Info) << "worker notify() main thread" << std::endl;
  mss->mainSem->notify();

  Logger::get(Logger::Info) << "worker lock()" << std::endl;
  mss->m->lock();
  Logger::get(Logger::Info) << "worker got mutex" << std::endl;

  Logger::get(Logger::Info) << "worker notify() main thread" << std::endl;
  mss->mainSem->notify();

  Logger::get(Logger::Info) << "worker wait() for main" << std::endl;
  mss->workerSem->wait();

  Logger::get(Logger::Info) << "worker unlock()" << std::endl;
  mss->m->unlock();

  Logger::get(Logger::Info) << "worker done" << std::endl;
}

void basicMultiThreadTest() {
  std::mutex m;
  Semaphore workerSem(0);
  Semaphore mainSem(0);
  MSS mss(&m, &workerSem, &mainSem);

  Logger::get(Logger::Info) << "main lock()" << std::endl;
  m.lock();
  Logger::get(Logger::Info) << "main got mutex" << std::endl;

  std::thread thread(&worker, &mss);
  // wait to give the other thread the oportunity to try_lock.
  Logger::get(Logger::Info) << "main wait() for worker" << std::endl;
  mainSem.wait();
  // now unlock the mutex so that the other thread can lock it.
  Logger::get(Logger::Info) << "main unlock()" << std::endl;
  m.unlock();
  Logger::get(Logger::Info) << "main wait() for worker" << std::endl;
  mainSem.wait();
  // now we can't take it, the other thread has it.
  Logger::get(Logger::Info) << "main trylock()" << std::endl;
  ENSURE(!m.try_lock(), "locked an already locked mutex ?!");

  Logger::get(Logger::Info) << "main notify() worker" << std::endl;
  workerSem.notify();

  // wait for it
  Logger::get(Logger::Info) << "main lock()" << std::endl;
  m.lock();
  Logger::get(Logger::Info) << "main got mutex" << std::endl;

  Logger::get(Logger::Info) << "main join()" << std::endl;
  thread.join();
  Logger::get(Logger::Info) << "main unlock()" << std::endl << std::endl;
  m.unlock();
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::basicMonoThreadTest();
  VideoStitch::Testing::basicSemaphoreTest();
  VideoStitch::Testing::basicMultiThreadTest();

  return 0;
}
