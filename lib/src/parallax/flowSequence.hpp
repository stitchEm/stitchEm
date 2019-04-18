// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include "core/pyramid.hpp"
#include "core/rect.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"

#include <vector>
#include <map>
#include <unordered_map>

namespace VideoStitch {
namespace Core {

/**
 * @brief: These classes are used to cache data over time and regularizes flow temporally
 */

class FlowCachedBuffer {
 public:
  virtual ~FlowCachedBuffer() {}

  virtual std::string getTypeName() const { return ""; }
};

template <typename T>
class TypedCached : public FlowCachedBuffer {
 public:
  TypedCached() : size(make_int2(0, 0)), offset(make_int2(0, 0)) {}

  Status init(const int leftOffset, const int rightOffset, const int2 size, const int2 offset);

  Status update(const int index, const GPU::Buffer<const T> buffer, GPU::Stream gpuStream);

  virtual std::string getTypeName() const override;

  GPU::Buffer<const T> getBuffer() const;

 private:
  int2 size;
  int2 offset;
  GPU::UniqueBuffer<T> cachedBuffer;
};

class FlowSequence {
 public:
  FlowSequence(const int leftOffset, const int rightOffset);

  template <typename T>
  Status cacheBuffer(const frameid_t frame, const std::string& name, const int2 size, const int2 offset,
                     GPU::Buffer<const T> cachedBuffer, GPU::Stream gpuStream);

  GPU::Buffer<const float> getFrames() const;

  std::shared_ptr<FlowCachedBuffer> getFlowCachedBuffer(const std::string& name) const;

  Status regularizeFlowTemporally(const std::string& name, const frameid_t frame, const int2 size, const int2 offset,
                                  GPU::Buffer<float2> flow, GPU::Stream gpuStream);

  // @TODO: Need to call reset at video cut, do this heuristically later
  Status checkForReset();

  void getFrameIndices(std::vector<float>& outFrames);

  int getFrameCount() const;

  void setKeyFrame(const frameid_t keyFrame);

  int getFrameIndex(const frameid_t frame) const;

  frameid_t getKeyFrame() const;

 private:
  // To store image flow
  std::unordered_map<std::string, std::shared_ptr<FlowCachedBuffer>> cachedBuffers;
  std::vector<float> frames;

  GPU::UniqueBuffer<float> framesBuffer;
  frameid_t keyFrame;

  const int leftOffset;
  const int rightOffset;
};

}  // namespace Core
}  // namespace VideoStitch
