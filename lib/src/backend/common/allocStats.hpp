// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/gpu_device.hpp"

#include <iostream>
#include <iomanip>
#include <mutex>
#include <algorithm>
#include <functional>

// log every allocation and delete in MB to std::cout
// #define LOG_ALLOCATIONS

#ifdef LOG_ALLOCATIONS

#include <cmath>

static std::string asMegaByteString(size_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1);
  out << bytes / 1000 / 1000 << " "
      << "MB";
  return out.str();
}

#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif  // NDEBUG
#endif  // LOG_ALLOCATIONS

namespace VideoStitch {
/**
 * A counter class.
 */
class AllocStats {
 public:
  AllocStats() : used_(0), max_(0) {
    /* Initialize the used_by_device_ vector */
    int numDevices = Discovery::getNumberOfDevices();
    used_by_devices_.assign(numDevices, 0);
  }

  ~AllocStats() { assert(used_ == 0); }

  void add(std::size_t size, int device_id) {
    used_ += size;
    if (used_ > max_) {
      max_ = used_;
    }
    if (device_id >= 0) {
      used_by_devices_[device_id] += size;
    }
  }

  void remove(std::size_t size, int device_id) {
    assert(size <= used_);
    used_ -= size;
    if (device_id >= 0) {
      used_by_devices_[device_id] -= size;
    }
  }

  std::size_t used() const { return used_; }

  std::vector<std::size_t> usedByDevices() const { return used_by_devices_; }

  std::size_t max() const { return max_; }

 private:
  std::size_t used_;
  std::size_t max_;
  std::vector<std::size_t> used_by_devices_;
};

/**
 * Info about a pointer.
 */
struct PtrInfo {
  PtrInfo(void* ptr, std::size_t size, int device_id, AllocStats& allocStats)
      : ptr(ptr), size(size), device_id(device_id), allocStats(allocStats) {}
  const void* ptr;
  const std::size_t size;
  const int device_id;
  AllocStats& allocStats;
};

/**
 * A locked map type to hold AllocStats.
 * Not reentrant.
 */
class AllocStatsMap {
 public:
  /**
   * Adds a pointer to a named pool.
   */
  void addPtr(const char* name, void* ptr, std::size_t size) {
    std::unique_lock<std::mutex> lock(mutex);
    AllocStats& allocStats = stats[std::string(name)];
    if (ptr) {
      int device_id;
      if (!GPU::getDefaultBackendDevice(&device_id).ok()) {
        device_id = -1;
      }
      ptrs.insert(std::make_pair(ptr, PtrInfo(ptr, size, device_id, allocStats)));
      allocStats.add(size, device_id);

#ifdef LOG_ALLOCATIONS
      std::ostringstream msg;
      msg << "[ALLOCSTATS " << this->name << "] " << name << " (" << ptr << "): " << size << " ("
          << asMegaByteString(size) << ")";
      std::cout << msg.str() << std::endl;
#endif
    }
  }

  /**
   * Deletes a pointer.
   */
  void deletePtr(void* ptr) {
    if (!ptr) {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex);
    std::map<void*, PtrInfo>::iterator it = ptrs.find(ptr);
    if (it == ptrs.end()) {
      assert(false);
      return;
    }
    const PtrInfo& ptrInfo = it->second;
    ptrInfo.allocStats.remove(ptrInfo.size, ptrInfo.device_id);
    ptrs.erase(it);

#ifdef LOG_ALLOCATIONS
    std::ostringstream msg;
    msg << "[ALLOCSTATS " << this->name << "] DELETED: " << ptr;
    std::cout << msg.str() << std::endl;
#endif
  }

  /**
   * Gets the current allocated size
   */
  std::size_t bytesUsed() {
    std::unique_lock<std::mutex> lock(mutex);
    std::size_t total = 0;
    for (std::map<std::string, AllocStats>::const_iterator it = stats.begin(); it != stats.end(); ++it) {
      total += it->second.used();
    }
    return total;
  }

  /**
   * Gets the current allocated size
   */
  std::vector<std::size_t> bytesUsedByDevices() {
    std::unique_lock<std::mutex> lock(mutex);
    std::vector<std::size_t> total;
    for (std::map<std::string, AllocStats>::const_iterator it = stats.begin(); it != stats.end(); ++it) {
      std::vector<std::size_t> usedByDevices = it->second.usedByDevices();
      /* first iteration: initialize return vector */
      if (it == stats.begin()) {
        total = usedByDevices;
      }
      /* else accumulate */
      else {
        // add usedByDevice to total
        std::transform(total.begin(), total.end(), usedByDevices.begin(), total.begin(), std::plus<std::size_t>());
      }
    }
    /* when no data has been allocated yet, still try to return a vector filled with zeros */
    if (total.empty()) {
      int numberOfDevices = Discovery::getNumberOfDevices();
      total.assign(numberOfDevices, 0);
    }
    return total;
  }

  /**
   * Prints the stats
   */
  void print_(std::ostream& os) {
    std::cout << name << " allocated memory:" << std::endl;

    for (std::map<std::string, AllocStats>::const_iterator it = stats.begin(); it != stats.end(); ++it) {
      os << " ";
      os << it->first;
      os << (it->second.used() == 0 ? ": " : ": MEMORY LEAKING ");
      format(it->second.used(), os);
      os << " used (";
      format(it->second.max(), os);
      os << " max)" << std::endl;
    }
  }

  void print(std::ostream& os) {
    std::unique_lock<std::mutex> lock(mutex);
    print_(os);
  }

  static void format(std::size_t s, std::ostream& os) {
    if (s < 1024) {
      os << s;
    } else if (s < 1024 * 1024) {
      os << std::fixed << std::setprecision(2) << (double)s / 1024.0 << " KB";
    } else if (s < (std::size_t)1024 * (std::size_t)1024 * (std::size_t)1024) {
      os << std::fixed << std::setprecision(2) << (double)s / (1024.0 * 1024.0) << " MB";
    } else {
      os << std::fixed << std::setprecision(2) << (double)s / (1024.0 * 1024.0 * 1024.0) << " GB";
    }
  }

  explicit AllocStatsMap(const std::string& n) : name(n) {}

  ~AllocStatsMap() {
    print_(std::cout);
    std::map<std::string, AllocStats>::const_iterator it = stats.begin();
    while (it != stats.end()) {
      it = stats.erase(it);
    }
  }

 private:
  std::string name;
  std::mutex mutex;
  // Maps pool names to stats.
  std::map<std::string, AllocStats> stats;
  // Maps pointers to PointerStats.
  std::map<void*, PtrInfo> ptrs;
};

// Global allocation maps

extern AllocStatsMap deviceStats;

extern AllocStatsMap hostStats;

}  // namespace VideoStitch
