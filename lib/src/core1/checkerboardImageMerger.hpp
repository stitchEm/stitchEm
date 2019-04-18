// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "imageMerger.hpp"

#include "libvideostitch/imageMergerFactory.hpp"

namespace VideoStitch {
namespace Core {

/**
 * @brief Checkerboard image merger
 * A debug merger that stacks the inputs in a checkerboard pattern in overlapping areas
 * instead of blending. Useful to inspect calibration and exposure differences.
 */
class CheckerboardImageMerger : public ImageMerger {
 public:
  /**
   * @brief CheckerboardImageMerger factory
   */
  class Factory : public ImageMergerFactory {
   public:
    static Potential<ImageMergerFactory> parse(const Ptv::Value& value);
    explicit Factory(int checkerSize);
    virtual Potential<ImageMerger> create(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to,
                                          bool) const;
    virtual ~Factory() {}
    Ptv::Value* serialize() const;
    virtual CoreVersion version() const { return CoreVersion1; }
    virtual ImageMergerFactory* clone() const;
    virtual std::string hash() const;

   private:
    const int checkerSize;
  };

 public:
  ~CheckerboardImageMerger();

  Status mergeAsync(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t> panoDevOut,
                    GPU::UniqueBuffer<uint32_t>&, const ImageMapping&, bool isFirstMerger, GPU::Stream) const override;

 private:
  CheckerboardImageMerger(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to, int checkerSize);
  int checkerSize;
};
}  // namespace Core
}  // namespace VideoStitch
