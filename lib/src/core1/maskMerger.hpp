// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "textureTarget.hpp"

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/ptv.hpp"

#include "core/pyramid.hpp"

#include <memory>
#include <vector>

namespace VideoStitch {
namespace Core {

class PanoDefinition;
class ImageMapping;
class ImageMerger;

class MaskMerger {
 public:
  enum class MaskMergerType {
    VoronoiMask = 0  // Voronoi Mask
  };
  static MaskMergerType getDefaultMaskMerger();
  MaskMerger();

  static MaskMerger* factor(const MaskMergerType maskMergerType);

  ~MaskMerger();

  virtual Status setParameters(const std::vector<double>& params) = 0;

  Status setupMask(const PanoDefinition& pano, GPU::Buffer<const uint32_t> panoDevOut, const ImageMapping& fromIm,
                   const ImageMerger* const to, GPU::Stream stream);
  Status setupMaskCubemap(const PanoDefinition& pano, GPU::Buffer<const uint32_t> panoDevOut,
                          const ImageMapping& fromIm, const ImageMerger* const to, GPU::Stream stream);

  /**
   * Called inside ImageMerger::prepareMergeAsync
   * This function is called for MaskMerger that would change overtime
   */
  virtual Status updateAsync() = 0;

  GPU::Buffer<unsigned char> getAlpha(TextureTarget) const;

  /**
   * Construct a pyramid of the generated mask, used for multi-scale based mergers.
   * This function should be called inside "ImageMerger::prepareMergeAsync"
   * and after "MaskMerger::Async" was called
   */
  Status buildPyramidMask(const ImageMapping&, std::string name, const int numLevels, const int gaussianRadius,
                          const int filterPasses, const bool warp, GPU::Stream stream);
  Status buildPyramidMaskCubemap(const PanoDefinition&, const ImageMapping&, std::string name, const int numLevels,
                                 const int gaussianRadius, const int filterPasses, const bool warp, GPU::Stream stream);

  LaplacianPyramid<unsigned char>* getAlphaPyramid(TextureTarget) const;

 protected:
  /**
   * Setup from the panorama setup mask. Asynchronous.
   * This function is "normally" called inside ImageMerger::setup
   * @return false on failure.
   */
  virtual Status setup(const PanoDefinition&, GPU::Buffer<const uint32_t> inputsMask, const ImageMapping& fromIm,
                       const ImageMerger* const to, GPU::Stream) = 0;

  std::array<GPU::UniqueBuffer<unsigned char>, 7> alpha;
  std::array<std::unique_ptr<LaplacianPyramid<unsigned char>>, 7> alphaPyramids;
};
}  // namespace Core
}  // namespace VideoStitch
