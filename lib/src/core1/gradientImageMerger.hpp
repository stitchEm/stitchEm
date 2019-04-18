// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "imageMerger.hpp"
#include "parallax/imageWarper.hpp"
#include "gpu/buffer.hpp"

#include "libvideostitch/imageMergerFactory.hpp"

namespace VideoStitch {
namespace Core {
/**
 * See comments in imageMergerFactory.hpp
 */
class GradientImageMerger : public ImageMerger {
 public:
  class Factory : public ImageMergerFactory {
   public:
    static Potential<ImageMergerFactory> parse(const Ptv::Value& value);
    explicit Factory(int feather, MaskMerger::MaskMergerType maskMergerType);
    virtual Potential<ImageMerger> create(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to,
                                          bool) const;
    virtual ~Factory() {}

    Ptv::Value* serialize() const;
    virtual CoreVersion version() const { return CoreVersion1; }
    virtual ImageMergerFactory* clone() const;
    virtual std::string hash() const;

   private:
    const int feather;
    const MaskMerger::MaskMergerType maskMergerType;
  };

 public:
  GradientImageMerger(const PanoDefinition&, ImageMapping& fromIm, const ImageMerger* to, int feather,
                      MaskMerger::MaskMergerType maskMergerType);

  ~GradientImageMerger();

  Format warpMergeType() const override { return Format::Gradient; }

  Status mergeAsync(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t> pbo, GPU::UniqueBuffer<uint32_t>&,
                    const ImageMapping&, bool isFirstMerger, GPU::Stream) const override;

  Status setup(const PanoDefinition&, InputsMap&, const ImageMapping&, GPU::Stream) override;
  Status setupCubemap(const PanoDefinition&, InputsMap&, const ImageMapping&, GPU::Stream) override;
};
}  // namespace Core
}  // namespace VideoStitch
