// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "maskMerger.hpp"
#include "textureTarget.hpp"
#include "parallax/imageWarper.hpp"

#include "gpu/uniqueBuffer.hpp"
#include "gpu/stream.hpp"

#include "libvideostitch/status.hpp"

#include <memory>

namespace VideoStitch {
namespace Core {

class ImageMapping;
class PanoDefinition;
class InputsMap;
class InputsMapCubemap;

class ImageMerger {
 public:
  static const int CudaBlockSize = 16;
  /**
   * Combined warp and merge format.
   */
  enum class Format { None = 0, Gradient = 1 };

 public:
  /**
   * Creates an image merger.
   * @param imId Input id.
   * @param to Points to the previous merger, NULL for the first one.
   */
  ImageMerger(videoreaderid_t imId, const ImageMerger* to);

  virtual ~ImageMerger();

  /**
   * Setup from the panorama setup image. Asynchronous.
   * @return false on failure.
   */
  virtual Status setup(const PanoDefinition&, InputsMap&, const ImageMapping&, GPU::Stream) { return Status::OK(); }

  /**
   * Setup from the cubemap setup image. Asynchronous.
   * @return false on failure.
   */
  virtual Status setupCubemap(const PanoDefinition&, InputsMap&, const ImageMapping&, GPU::Stream) {
    return Status::OK();
  }

  /**
   * Called just after mapping and before calling merge().
   * All asyncronous computations that do not depend on the result of merging of previous images should be done
   * in this function since they can be conducted in parallel with other operations to keep the GPU busy.
   */
  virtual Status prepareMergeAsync(TextureTarget, const ImageMapping&, GPU::Stream) const { return Status::OK(); }

  virtual Format warpMergeType() const { return Format::None; }

  /**
   * Merge the given remapped image @fromIm into the panorama.
   * Note that fromIm's buffer can be destroyed by this call.
   * All cuda operations must be done in the provided stream.
   */
  virtual Status mergeAsync(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t> pbo,
                            GPU::UniqueBuffer<uint32_t>& progressivePbo, const ImageMapping&, bool isFirstMerger,
                            GPU::Stream) const = 0;

  /**
   * Called after merge() has been completed.
   * @param progressive is true on all mergers except the last of the sequence.
   */
  virtual Status reconstruct(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t> pbo, bool progressive,
                             GPU::Stream stream) const;

  virtual bool isMultiScale() const;

  /**
   * get a mask with bits set for all images that are merged as a result of this ImageMerger
   */
  videoreaderid_t getIdMask() const { return idMask; }

  MaskMerger* getMaskMerger() const { return maskMerger.get(); }

 protected:
  const ImageMerger* const to;
  std::unique_ptr<MaskMerger> maskMerger;
  const videoreaderid_t idMask;
};
}  // namespace Core
}  // namespace VideoStitch
