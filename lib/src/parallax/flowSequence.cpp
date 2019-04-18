// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "flowSequence.hpp"

#include "./imageFlow.hpp"

#include "cuda/error.hpp"
#include "cuda/util.hpp"
#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/uniqueBuffer.hpp"

#include <typeinfo>

namespace VideoStitch {
namespace Core {

template <typename T>
Status TypedCached<T>::update(const int index, const GPU::Buffer<const T> buffer, GPU::Stream gpuStream) {
  const size_t elems = size.x * size.y;
  const size_t offset = index * elems;
  if (offset + elems > cachedBuffer.borrow_const().numElements()) {
    return {Origin::Unspecified, ErrType::InvalidConfiguration, "Invalid size"};
  }
  FAIL_RETURN(GPU::memcpyAsync<T>(cachedBuffer.borrow().createSubBuffer(offset), buffer, elems * sizeof(T), gpuStream));
  return CUDA_STATUS;
}

template <typename T>
Status TypedCached<T>::init(const int leftOffset, const int rightOffset, const int2 size, const int2 offset) {
  this->size = size;
  this->offset = offset;
  FAIL_RETURN(cachedBuffer.alloc(size.x * size.y * (rightOffset - leftOffset + 1), "Type Cached"));
  return CUDA_STATUS;
}

template <typename T>
std::string TypedCached<T>::getTypeName() const {
  return typeid(T).name();
}

template <typename T>
GPU::Buffer<const T> TypedCached<T>::getBuffer() const {
  return cachedBuffer.borrow_const();
}

FlowSequence::FlowSequence(const int leftOffset, const int rightOffset)
    : keyFrame(-1), leftOffset(leftOffset), rightOffset(rightOffset) {
  frames.assign(rightOffset - leftOffset + 1, -1);
  framesBuffer.alloc(rightOffset - leftOffset + 1, "Flow Sequence");
}

frameid_t FlowSequence::getKeyFrame() const { return keyFrame; }

GPU::Buffer<const float> FlowSequence::getFrames() const { return framesBuffer.borrow_const(); }

std::shared_ptr<FlowCachedBuffer> FlowSequence::getFlowCachedBuffer(const std::string& name) const {
  if (cachedBuffers.find(name) == cachedBuffers.end()) {
    return std::shared_ptr<FlowCachedBuffer>(nullptr);
  } else {
    return cachedBuffers.find(name)->second;
  }
}

void FlowSequence::setKeyFrame(const frameid_t keyFrame) { this->keyFrame = keyFrame; }

int FlowSequence::getFrameCount() const { return (int)frames.size(); }

int FlowSequence::getFrameIndex(const frameid_t frame) const {
  for (size_t i = 0; i < frames.size(); i++) {
    if (frames[i] == frame) {
      return (int)i;
    }
  }
  return -1;
}

void FlowSequence::getFrameIndices(std::vector<float>& outFrames) { outFrames = frames; }

Status FlowSequence::checkForReset() {
  bool isCut = false;
  bool isDissolve = false;
  // TODO: Decide whether the new inserted frame is a cut or part of a dissolving process
  if (isCut || isDissolve) {
    frames.assign(frames.size(), -1);
  }
  return CUDA_STATUS;
}

template <typename T>
Status FlowSequence::cacheBuffer(const frameid_t frame, const std::string& name, const int2 size, const int2 offset,
                                 GPU::Buffer<const T> buffer, GPU::Stream gpuStream) {
  // First time to encounter this type of buffer, need to allocate memory
  TypedCached<T>* cached;

  if (cachedBuffers.find(name) == cachedBuffers.end()) {
    cached = new TypedCached<T>();
    Status initStatus = cached->init(leftOffset, rightOffset, size, offset);
    if (!initStatus.ok()) {
      delete cached;
      return initStatus;
    }
    std::shared_ptr<FlowCachedBuffer> sharedCached(dynamic_cast<FlowCachedBuffer*>(cached));
    auto p = std::make_pair(name, sharedCached);
    cachedBuffers.insert(p);
  } else {
    std::shared_ptr<FlowCachedBuffer>& flowCachedBuffer = cachedBuffers.find(name)->second;
    cached = dynamic_cast<TypedCached<T>*>(flowCachedBuffer.get());
  }
  bool updateCache = false;
  // If this frame is in the list, just update it at the right index
  for (size_t i = 0; i < frames.size(); i++) {
    if (frames[i] == frame) {
      FAIL_RETURN(cached->update((int)i, buffer, gpuStream));
      updateCache = true;
      break;
    }
  }

  if (!updateCache) {
    // Try to locate an invalid index
    for (size_t i = 0; i < frames.size(); i++) {
      if (frames[i] < keyFrame + leftOffset || frames[i] > keyFrame + rightOffset || frames[i] < 0) {
        FAIL_RETURN(cached->update((int)i, buffer, gpuStream));
        frames[i] = (float)frame;
        updateCache = true;
        break;
      }
    }
  }

  // Store frame id on gpu memory for later lookup
  GPU::memcpyAsync<float>(framesBuffer.borrow(), &frames[0], frames.size() * sizeof(float), gpuStream);
  if (!updateCache) {
    return {Origin::Unspecified, ErrType::InvalidConfiguration, "Cache was not updated"};
  }
  return gpuStream.synchronize();
}

template class TypedCached<float>;
template class TypedCached<float2>;
template class TypedCached<uint32_t>;
template Status FlowSequence::cacheBuffer(const frameid_t frame, const std::string& name, const int2 size,
                                          const int2 offset, GPU::Buffer<const float> cachedBuffer,
                                          GPU::Stream gpuStream);
template Status FlowSequence::cacheBuffer(const frameid_t frame, const std::string& name, const int2 size,
                                          const int2 offset, GPU::Buffer<const float2> cachedBuffer,
                                          GPU::Stream gpuStream);
template Status FlowSequence::cacheBuffer(const frameid_t frame, const std::string& name, const int2 size,
                                          const int2 offset, GPU::Buffer<const uint32_t> cachedBuffer,
                                          GPU::Stream gpuStream);

}  // namespace Core
}  // namespace VideoStitch
