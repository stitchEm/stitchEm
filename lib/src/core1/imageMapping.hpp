// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "textureTarget.hpp"

#include "core/rect.hpp"

#include "gpu/allocator.hpp"
#include "gpu/hostBuffer.hpp"
#include "gpu/uniqueBuffer.hpp"
#include "gpu/stream.hpp"
#include "gpu/surface.hpp"

namespace VideoStitch {
namespace Input {
class VideoReader;
struct PotentialFrame;
}  // namespace Input

namespace Core {

class InputDefinition;
class PanoDefinition;
class PreProcessor;
class Transform;
class ImageMerger;
class ImageWarper;
class ImageWarperFactory;
class ImageFlow;
class ImageFlowFactory;
class ImageMergerFactory;
class InputsMap;
class InputsMapCubemap;
class MergerPair;
class StereoRigDefinition;

class Buffer;

class ImageMapping {
 public:
  /**
   * Creates an ImageMapping for the @a imId th input of @a pano.
   * @param imId The input id.
   */
  explicit ImageMapping(videoreaderid_t imId);
  virtual ~ImageMapping();

  Status setup(ImageMapping* prevMapping, const PanoDefinition&, const ImageMergerFactory&, std::shared_ptr<InputsMap>,
               GPU::Stream, bool progressive = false);

  /**
   * Remap the next frame in @a reader asynchronously in stream @a stream.
   * @param frame Current frame id.
   * @param pano the pano definition
   * @param progressivePbo the progessive output panorama in case of other merger
   * @param panoSurf the output panorama to avoid memcopy in case of gradient merger
   * @param stream Where to do the computations.
   */
  virtual Status warp(frameid_t frame, const PanoDefinition& pano, GPU::Buffer<uint32_t> progressivePbo,
                      GPU::Surface& panoSurf, GPU::Stream& stream);
  Status warpCubemap(frameid_t frame, const PanoDefinition& pano, bool equiangular, GPU::Stream& stream);

  virtual Status reconstruct(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t>, bool final,
                             GPU::Stream&) const;

  /**
   * Precomputed the wrapped coordinate asynchronously in stream @a stream.
   * @param frame Current frame id.
   * @param pano the pano definition
   * @param stream Where to do the computations.
   * @param reader The input reader. If null, remap the contents of the current devArray.
   * @param preprocessor If not NULL, a preprocessor that will be applied before mapping.
   */
  Status precomputedCoord(frameid_t frame, const PanoDefinition& pano, GPU::Stream& stream);

  /**
   * Sets up the texture array from the given reader.
   * @param inputDef the input definition
   * @param stream Where to do the computations.
   * @param reader The input reader. If null, don't read.
   * @param preprocessor If not NULL, a preprocessor that will be applied before mapping.
   */
  Status setupTexArrayAsync(frameid_t frame1, const Input::PotentialFrame& inputFrame, const InputDefinition& inputDef,
                            GPU::Stream& stream, Input::VideoReader* reader, const PreProcessor* preprocessor);
  /**
   * Allocates all host and device buffers.
   */
  Status allocateUnpackBuffer(int64_t frameDataSize);
  Status allocateBuffers(TextureTarget, int64_t width, int64_t height);

  /**
   * Allocates device buffers, and use the passed-in buffer as host input buffer. This is used for re-setup.
   * The input buffer already contains data from the previous frame.
   * @param readerSpec Spec for the reader that this mapper reads from.
   * @param hostInputBuffer Input buffer. Destroyed.
   * @param allocPrecomputedCoordinate Whether to allocate the precomputed coordinate buffer.
   */
  Status allocateBuffersPartial(TextureTarget, int64_t width, int64_t height, SourceSurface* hostInputBuffer);

  void setHBounds(TextureTarget, int64_t l, int64_t r, int64_t panoCroppedWidth);

  void setVBounds(TextureTarget, int64_t t, int64_t b);

  const Rect& getOutputRect(TextureTarget t) const { return outputBounds[t]; }
  Rect& getOutputRect(TextureTarget t) { return outputBounds[t]; }

  frameid_t getFrameId() const { return frameId; }

  void setFrameId(const frameid_t frameId) { this->frameId = frameId; }

  /**
   * Give away our input buffers. This is used when re-setupping to keep the last input frame.
   * This invalidates *this, and must be the last operation before deletion.
   * @param buffers The buffers are given to this object.
   * @return The input buffer. The caller is responsible for releasing the result.
   */
  void releaseInputBuffers(SourceSurface** sourceSurf) {
    *sourceSurf = surface;
    surface = nullptr;
  }

  /**
   * returns the input buffer after unpacking (stored on the GPU)
   */
  GPU::Surface& getSurface() { return *surface->pimpl->surface; }

  /**
   * returns the precomputed buffer coordinate for mapping
   */
  GPU::Surface& getSurfaceCoord() { return *devCoord->pimpl->surface; }

  const ImageMerger& getMerger() const { return *merger; }
  ImageMerger& getMerger() { return *merger; }

  /**
   * returns a buffer of size inputArea() * 4,
   * correctly allocated for efficient CUDA transmission.
   */
  GPU::Buffer<const uint32_t> getDeviceOutputBuffer(TextureTarget t) const { return devWork[t].borrow_const(); }

  videoreaderid_t getImId() const { return imId; }

  bool wraps() const {
    assert(wrapsAround >= 0);  // <0 means not computed.
    return wrapsAround > 0 ? true : false;
  }

 protected:
  // Common part of allocateBuffers() and allocateBuffersPartial().
  virtual Status allocateOutputBuffers(TextureTarget, int64_t width, int64_t height);

  // bounding box
  frameid_t frameId;
  std::array<Rect, TEXTURE_TARGET_SIZE> outputBounds;
  int wrapsAround;

  std::array<GPU::UniqueBuffer<uint32_t>, TEXTURE_TARGET_SIZE>
      devWork;  // input buffer for no-flow, mapped output buffer

  const videoreaderid_t imId;
  SourceSurface* surface = nullptr;

  ImageMerger* merger = nullptr;
  Transform* transform = nullptr;

  GPU::UniqueBuffer<unsigned char> devUnpackTmp;  // array to hold intermediate data before unpacking
  Core::SourceSurface* devCoord = nullptr;        // array to hold precomputed coordinate for mapping
};

class ImageMappingFlow : public ImageMapping {
 public:
  explicit ImageMappingFlow(unsigned imId) : ImageMapping(imId), warper(nullptr), flow(nullptr) {}

  virtual ~ImageMappingFlow();

  Status setup(ImageMappingFlow* prevMapping, const PanoDefinition&, const StereoRigDefinition* rigDef,
               const ImageMergerFactory&, std::vector<readerid_t> alreadyWarped, std::shared_ptr<InputsMap>,
               const ImageWarperFactory&, const ImageFlowFactory&, GPU::Stream);

  /**
   * Remap the next frame in @a reader synchronously in stream @a stream using flow-based warper.
   * @param frame Current frame id.
   * @param pano the pano definition
   * @param progressivePbo the progessive output panorama in case of other merger
   * @param panoSurf the output panorama to avoid memcopy in case of gradient merger
   * @param stream Where to do the computations.
   */
  Status warp(frameid_t frame, const PanoDefinition& pano, GPU::Buffer<uint32_t> progressivePbo, GPU::Surface& panoSurf,
              GPU::Stream& stream) override;
  Status reconstruct(TextureTarget, const PanoDefinition&, GPU::Buffer<uint32_t> progressivePbo, bool final,
                     GPU::Stream&) const override;

  Status allocateOutputBuffers(TextureTarget, int64_t width, int64_t height) override;

 private:
  GPU::UniqueBuffer<uint32_t> devFlowIn;  // input image buffer when using flow based blending

  ImageWarper* warper;
  ImageFlow* flow;
  std::shared_ptr<MergerPair> mergerPair;

  ImageMappingFlow(const ImageMappingFlow&) = delete;
  ImageMappingFlow& operator=(const ImageMappingFlow&) = delete;
};

}  // namespace Core
}  // namespace VideoStitch
