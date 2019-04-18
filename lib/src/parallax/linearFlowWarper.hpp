// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "./imageWarper.hpp"

#include "libvideostitch/imageWarperFactory.hpp"

#include <memory>

namespace VideoStitch {
namespace Core {

class LinearFlowWarper : public ImageWarper {
 public:
  class Factory : public ImageWarperFactory {
   public:
    static Potential<ImageWarperFactory> parse(const Ptv::Value& value);
    virtual std::string getImageWarperName() const override;
    explicit Factory(const float maxTransitionDistance, const double power);
    virtual ~Factory() {}

    virtual Potential<ImageWarper> create() const override;
    virtual bool needsInputPreProcessing() const override;
    virtual Ptv::Value* serialize() const override;
    virtual std::string hash() const override;
    virtual ImageWarperFactory* clone() const override;

   private:
    const float maxTransitionDistance;
    const double power;
  };

  static std::string getName();

  const GPU::Buffer<const unsigned char> getLinearMaskWeight() const;

  Rect getMaskRect() const;

  virtual Status warp(GPU::Buffer<uint32_t> warpedBuffer, const GPU::Buffer<const uint32_t> inputBuffer,
                      const Rect& flowRect, const GPU::Buffer<const float2> flow, const int lookupOffsetX,
                      const int lookupOffsetY, GPU::Buffer<float4> debug, GPU::Buffer<uint32_t> flowWarpedBuffer,
                      GPU::Stream gpuStream) override;

  virtual ImageWarperAlgorithm getWarperAlgorithm() const override;

  virtual bool needImageFlow() const override;

 private:
  friend class ImageWarper;

  explicit LinearFlowWarper(const std::map<std::string, float>& parameters);

  virtual Status setupCommon(GPU::Stream gpuStream) override;

  GPU::UniqueBuffer<unsigned char> linearMaskWeight;
};
}  // namespace Core
}  // namespace VideoStitch
