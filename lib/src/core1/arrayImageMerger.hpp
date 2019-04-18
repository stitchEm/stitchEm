// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "imageMerger.hpp"

#include "libvideostitch/imageMergerFactory.hpp"

namespace VideoStitch {
namespace Core {

/**
 * @brief Array image merger.
 * A merger that helps visualizing the camerra array.
 */
class ArrayImageMerger : public ImageMerger {
 public:
  /**
   * @brief ArrayImageMerger factory.
   */
  class Factory : public ImageMergerFactory {
   public:
    virtual Potential<ImageMerger> create(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to,
                                          bool) const;
    virtual ~Factory() {}
    Ptv::Value* serialize() const;
    virtual CoreVersion version() const { return CoreVersion1; }
    virtual ImageMergerFactory* clone() const;
    virtual std::string hash() const;
  };

 public:
  ~ArrayImageMerger();

  Status mergeAsync(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> panoDevOut,
                    GPU::UniqueBuffer<uint32_t>&, const ImageMapping& fromIm, bool isFirstMerger,
                    GPU::Stream stream) const override;

  Status reconstruct(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t> pbo, bool, GPU::Stream) const override;

 private:
  ArrayImageMerger(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to);
};
}  // namespace Core
}  // namespace VideoStitch
