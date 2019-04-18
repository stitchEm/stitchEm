// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "./imageFlow.hpp"

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/imageFlowFactory.hpp"

#include <vector>

namespace VideoStitch {
namespace Core {

class NoFlow : public ImageFlow {
 public:
  class Factory : public ImageFlowFactory {
   public:
    static Potential<ImageFlowFactory> parse(const Ptv::Value& value);
    explicit Factory();
    virtual Potential<ImageFlow> create() const override;
    virtual bool needsInputPreProcessing() const override;
    virtual std::string getImageFlowName() const override;
    virtual Ptv::Value* serialize() const override;
    virtual std::string hash() const override;
    virtual ImageFlowFactory* clone() const override;
  };

  static std::string getName();

  virtual Status findSingleScaleImageFlow(const int2& offset0, const int2& size0,
                                          const GPU::Buffer<const uint32_t>& image0, const int2& offset1,
                                          const int2& size1, const GPU::Buffer<const uint32_t>& image1,
                                          GPU::Buffer<float2> finalFlow, GPU::Stream gpuStream) override;

  virtual Status upsampleFlow(const int2& size0, const int2& offset0, const GPU::Buffer<const uint32_t>& image0,
                              const int2& size1, const int2& offset1, const GPU::Buffer<const uint32_t>& image1,
                              const GPU::Buffer<const float2>& inputFlow, GPU::Buffer<float2> upsampledFlow,
                              GPU::Stream gpuStream) override;

  virtual ImageFlowAlgorithm getFlowAlgorithm() const override;

 private:
  friend class ImageFlow;
  explicit NoFlow(const std::map<std::string, float>& parameters);
};

}  // namespace Core
}  // namespace VideoStitch
