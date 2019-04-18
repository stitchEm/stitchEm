// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "imageMerger.hpp"
#include "libvideostitch/imageMergerFactory.hpp"

namespace VideoStitch {
namespace Core {

/**
 * @brief no blend image merger.
 * A merger To know for each pano pixels what is the value of each input (up to 2)
 * For each input, the warped pixel is tranformed to grayscale.
 * The first input which maps to a given panorama pixel will be stored in the R component.
 * The second input which maps to the same panorama pixel will be stored in the G component.
 * The alpha bit is set to 1 if and only if two inputs map to a panorama pixel.
 * This is used to make computations on overlap regions in the panorama output space.
 */
class NoBlendImageMerger : public ImageMerger {
 public:
  /**
   * @brief NoBlendImageMerger factory.
   */
  class Factory : public ImageMergerFactory {
   public:
    virtual Potential<ImageMerger> create(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to,
                                          bool) const;
    virtual ~Factory(){};
    Ptv::Value* serialize() const;
    virtual CoreVersion version() const { return CoreVersion1; }
    virtual ImageMergerFactory* clone() const;
    virtual std::string hash() const;
  };

 public:
  ~NoBlendImageMerger();

  Status mergeAsync(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t> pbo, GPU::UniqueBuffer<uint32_t>&,
                    const ImageMapping&, bool isFirstMerger, GPU::Stream) const override;

 private:
  NoBlendImageMerger(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to);
};
}  // namespace Core
}  // namespace VideoStitch
