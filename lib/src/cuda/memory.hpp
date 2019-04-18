// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

#include <cassert>
#include <vector>

// TODO_OPENCL_IMPL
// TODO port Cuda::Malloc stuff to GPU::
// TODO handle alloc flags properly
const unsigned int HostAllocDefault = 0x0;

struct cudaArray;
struct cudaChannelFormatDesc;

namespace VideoStitch {
namespace Cuda {

/**
 * Allocates GPU memory and prints a message on error.
 * @param buf Forwarded to cudaMalloc.
 * @param size Forwarded to cudaMalloc.
 * @param name Name of the pool whose counter to increment.
 * @return false on error.
 */
Status __mallocVS(void** buf, size_t size, const char* name, unsigned flagsUnused = 0, const char* file = NULL,
                  int line = -1);
#ifdef NDEBUG
#define mallocVS(buf, size, name) __mallocVS(buf, size, name, (unsigned)0)
#else
#define mallocVS(buf, size, name) __mallocVS(buf, size, name, (unsigned)0, __FILE__, __LINE__)
#endif

/**
 * Frees GPU memory allocated with __mallocVS
 * @param buf Pointer to free. Can be NULL.
 */
Status freeVS(void* buf);

/**
 * Allocates Pinned CPU memory and prints a message on error.
 * @param buf Forwarded to cudaMallocHost.
 * @param size Forwarded to cudaMallocHost.
 * @param name Name of the pool whose counter to increment.
 * @return false on error.
 */
Status __mallocHostVS(void** buf, size_t size, const char* name, unsigned flags = HostAllocDefault,
                      const char* file = NULL, int line = -1);
#ifdef NDEBUG
#define mallocHostVS(buf, size, name) __mallocHostVS(buf, size, name)
#define mallocHostFVS(buf, size, name, flags) __mallocHostVS(buf, size, name, flags)
#else
#define mallocHostVS(buf, size, name) __mallocHostVS(buf, size, name, HostAllocDefault, __FILE__, __LINE__)
#define mallocHostFVS(buf, size, name, flags) __mallocHostVS(buf, size, name, flags, __FILE__, __LINE__)
#endif

/**
 * Frees Host memory allocated with __mallocHostVS
 * @param buf Pointer to free. Can be NULL.
 */
Status freeHostVS(void* buf);

/**
 * Allocates a CUDA array and prints a message on error.
 * @param array Forwarded to cudaMallocArray.
 * @param desc Forwarded to cudaMallocArray.
 * @param width Forwarded to cudaMallocArray.
 * @param height Forwarded to cudaMallocArray.
 * @param flags Forwarded to cudaMallocArray.
 * @param name Name of the pool whose counter to increment.
 * @return false on error.
 */
Status __mallocArrayVS(struct cudaArray** array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height,
                       unsigned int flags, const char* name, const char* file = NULL, int line = -1);
#ifdef NDEBUG
#define mallocArrayVS(array, desc, width, height, flags, name) __mallocArrayVS(array, desc, width, height, flags, name)
#else
#define mallocArrayVS(array, desc, width, height, flags, name) \
  __mallocArrayVS(array, desc, width, height, flags, name, __FILE__, __LINE__)
#endif

/**
 * Frees array memory allocated with __mallocArrayVS
 * @param buf Pointer to free. Can be NULL.
 */
Status freeArrayVS(struct cudaArray* array);

/**
 * Get the number of bytes allocated in the device pool.
 */
std::size_t getDevicePoolCurrentSize(void);

/**
 * Get the number of bytes allocated in the device pool, by devices.
 */
std::vector<std::size_t> getDevicePoolCurrentSizeByDevices(void);

/**
 * Get the number of bytes allocated in the host pool.
 */
std::size_t getHostPoolCurrentSize(void);

/**
 * Get the number of bytes allocated in the host pool, by devices.
 */
std::vector<std::size_t> getHostPoolCurrentSizeByDevices(void);

/**
 * Print the device pool.
 */
void printDevicePool();

/**
 * Print the host pool.
 */
void printHostPool();

/**
 * Base class for UniquePtrs
 */
template <typename T, Status (*allocer)(void**, size_t, const char*, unsigned, const char*, int),
          Status (*deleter)(void*)>
class BaseUniquePtr {
 public:
  /**
   *
   */
  operator bool() const { return ptr != NULL; }

  /**
   * Returns the pointer but keeps ownership.
   */
  T* get() const { return ptr; }

  /**
   * Releases ownership.
   */
  T* release() {
    T* res = ptr;
    ptr = NULL;
    numElements = 0;
    return res;
  }

  /**
   * Returns the allocated size, in bytes.
   */
  size_t byteSize() const { return numElements * sizeof(T); }

  /**
   * Returns the number of allocated elements.
   */
  size_t elemSize() const { return numElements; }

  /**
   * Allocates a pointer.
   * @param size Size (in elements of type T).
   * @param name pool name.
   */
  Status alloc(size_t size, const char* name, unsigned flags = 0) {
    if (ptr) {
      deleter(ptr);
      ptr = NULL;
      numElements = 0;
    }
    Status status = allocer((void**)&ptr, size * sizeof(T), name, flags, NULL, -1);
    if (status.ok()) {
      numElements = size;
    }
    return status;
  }

 protected:
  /**
   * Creates a NULL pointer.
   */
  BaseUniquePtr() : ptr(NULL), numElements(0) {}

  /**
   * Takes ownership of ptr.
   * @param ptr Pointer.
   */
  explicit BaseUniquePtr(T* ptr) : ptr(ptr), numElements(0) {}

  ~BaseUniquePtr() { deleter(ptr); }

 private:
  T* ptr;
  size_t numElements;
};

/**
 * Device equivalent of std::unique_ptr.
 */
template <typename T>
class DeviceUniquePtr : public BaseUniquePtr<T, __mallocVS, freeVS> {
 public:
  typedef BaseUniquePtr<T, __mallocVS, freeVS> Base;

  /**
   * Creates a NULL pointer.
   */
  DeviceUniquePtr() : Base() {}

  /**
   * Takes ownership of ptr.
   * @param ptr Pointer.
   */
  explicit DeviceUniquePtr(T* ptr) : Base(ptr) {}
};

/**
 * Host equivalent of std::unique_ptr.
 */
template <typename T>
class HostUniquePtr : public BaseUniquePtr<T, __mallocHostVS, freeHostVS> {
 public:
  typedef BaseUniquePtr<T, __mallocHostVS, freeHostVS> Base;
  /**
   * Creates a NULL pointer.
   */
  HostUniquePtr() : Base() {}

  /**
   * Takes ownership of ptr.
   * @param ptr Pointer.
   */
  explicit HostUniquePtr(T* ptr) : Base(ptr) {}
};
}  // namespace Cuda
}  // namespace VideoStitch

#if (defined SHOW_CUDA_ALLOCS) && (!defined NDEBUG)
#define cudaMalloc(p, size) ::VideoStitch::Cuda::__mallocPrint(p, size, __FILE__, __LINE__)
/**
 * Forwards its arguments to cudaMalloc after printing where where an alloc has been made.
 * @param p Where to allocate memory.
 * @param size Number of bytes to allocate.
 * @param file Filename.
 * @param line Line number.
 * @note Never called directly.
 */
cudaError __mallocPrint(void** p, size_t size, const char* file, const int line);
#endif
