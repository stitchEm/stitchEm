// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "imageMerger.hpp"

#include "libvideostitch/imageMergerFactory.hpp"

#include <memory>

namespace VideoStitch {
namespace Core {

template <class T>
class LaplacianPyramid;

/**
 * @brief Laplacian merger.
 * See comments in imageMergerFactory.hpp
 */
class LaplacianImageMerger : public ImageMerger {
 public:
  /**
   * @brief LaplacianImageMerger factory.
   */
  class Factory : public ImageMergerFactory {
   public:
    static Potential<ImageMergerFactory> parse(const Ptv::Value& value);
    /**
     * Creates a factory that creates LaplacianImageMergers with the given properties.
     */
    Factory(int feather, int levels, int64_t baseSize, int gaussianRadius, int filterPasses,
            MaskMerger::MaskMergerType maskMergerType);
    virtual Potential<ImageMerger> create(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to,
                                          bool progressive) const;
    virtual ~Factory() {}
    Ptv::Value* serialize() const;
    virtual CoreVersion version() const { return CoreVersion1; }
    virtual ImageMergerFactory* clone() const;
    virtual std::string hash() const;
    virtual uint32_t getBlockAlignment() const;

   private:
    /**
     * Compute the desired number of levels for a given panorama size.
     */
    int computeNumLevels(int64_t width, int64_t height) const;

    const int feather;
    const int levels;
    const int64_t baseSize;
    const int gaussianRadius;
    const int filterPasses;
    const MaskMerger::MaskMergerType maskMergerType;
  };

 public:
  ~LaplacianImageMerger();

  Status prepareMergeAsync(TextureTarget, const ImageMapping& fromIm, GPU::Stream stream) const override;

  Status mergeAsync(TextureTarget, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo,
                    GPU::UniqueBuffer<uint32_t>& progressivePbo, const ImageMapping& fromIm, bool isFirstMerger,
                    GPU::Stream stream) const override;

  Status reconstruct(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t> progressivePbo, bool progressive,
                     GPU::Stream) const override;

  Status setup(const PanoDefinition&, InputsMap&, const ImageMapping&, GPU::Stream) override;
  Status setupCubemap(const PanoDefinition&, InputsMap&, const ImageMapping&, GPU::Stream) override;

  bool isMultiScale() const override;

 private:
  LaplacianImageMerger(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to,
                       LaplacianPyramid<uint32_t>* const* globalPyramids, int feather, int gaussianRadius,
                       int filterPasses, MaskMerger::MaskMergerType maskMergerType);

  std::unique_ptr<LaplacianPyramid<uint32_t>> pyramids[7];
  LaplacianPyramid<uint32_t>* globalPyramids[7];  // Owned by the first merger.

  int gaussianRadius;
  int filterPasses;

  int64_t width, height;
};
}  // namespace Core
}  // namespace VideoStitch
