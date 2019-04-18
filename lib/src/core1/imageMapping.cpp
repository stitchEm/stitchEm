// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "imageMapping.hpp"

#include "imageMerger.hpp"
#include "inputsMap.hpp"
#include "inputsMapCubemap.hpp"

#include "gpu/allocator.hpp"
#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "image/unpack.hpp"
#include "gpu/core1/transform.hpp"
#include "parallax/imageWarper.hpp"
#include "parallax/imageFlow.hpp"

#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/imageWarperFactory.hpp"
#include "libvideostitch/imageFlowFactory.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/preprocessor.hpp"
#include "libvideostitch/profile.hpp"
#include "input/inputFrame.hpp"
#include "gpu/image/imgInsert.hpp"

#include <cmath>
#include <cassert>
#include <sstream>

#undef max

//#define READBACK_INPUT_IMAGE
//#define READBACKMAPPEDIMAGE

#if defined(READBACK_INPUT_IMAGE) || defined(READBACKMAPPEDIMAGE)
#include "util/debugUtils.hpp"
#include "util/imageProcessingGPUUtils.hpp"

#include <sstream>
#endif

namespace VideoStitch {
namespace Core {

ImageMapping::ImageMapping(videoreaderid_t imId)
    : outputBounds(),
      wrapsAround(-1),  // invalid
      imId(imId) {}

ImageMapping::~ImageMapping() {
  delete surface;
  delete transform;
  delete merger;
  delete devCoord;
}

Status ImageMapping::setup(ImageMapping* prevMapping, const PanoDefinition& pano,
                           const ImageMergerFactory& mergerFactory, std::shared_ptr<InputsMap> inputsMap,
                           GPU::Stream stream, bool progressive) {
  // create merger
  ImageMerger* prev = nullptr;
  if (prevMapping) prev = prevMapping->merger;
  Potential<ImageMerger> cur = mergerFactory.create(pano, *this, prev, progressive);
  if (!cur.ok()) {
    return Status(Origin::Stitcher, ErrType::SetupFailure, "Could not setup merger for input " + std::to_string(imId),
                  cur.status());
  }
  merger = cur.release();

  // create transform
  const InputDefinition& inputDef = pano.getInput(imId);
  transform = Transform::create(inputDef, merger->warpMergeType());
  if (!transform) {
    return {Origin::Stitcher, ErrType::SetupFailure,
            "Cannot create v1 transformation for input " + std::to_string(imId)};
  }

  if (pano.getProjection() == PanoProjection::Cubemap || pano.getProjection() == PanoProjection::EquiangularCubemap) {
    const Status status = merger->setupCubemap(pano, *inputsMap, *this, stream);
    if (!status.ok()) {
      return Status(Origin::Stitcher, ErrType::SetupFailure, "Could not setup merger for input " + std::to_string(imId),
                    status);
    }
  } else {
    const Status status = merger->setup(pano, *inputsMap, *this, stream);
    if (!status.ok()) {
      return Status(Origin::Stitcher, ErrType::SetupFailure, "Could not setup merger for input " + std::to_string(imId),
                    status);
    }
  }

  // Precompute coordinate buffer
  FAIL_RETURN(precomputedCoord(0, pano, stream));

  return Status::OK();
}

Status ImageMapping::warp(frameid_t frame, const PanoDefinition& pano, GPU::Buffer<uint32_t> buffer, GPU::Surface& surf,
                          GPU::Stream& stream) {
  if (outputBounds[EQUIRECTANGULAR].empty()) {
    return Status::OK();  // nothing to do.
  }
  const InputDefinition& inputDef = pano.getInput(imId);

  GPU::Buffer<uint32_t> devBuffer =
      (getMerger().warpMergeType() == ImageMerger::Format::None) ? devWork[EQUIRECTANGULAR].borrow() : buffer;
  const unsigned char* mask = getMerger().getMaskMerger() && (getMerger().warpMergeType() != ImageMerger::Format::None)
                                  ? getMerger().getMaskMerger()->getAlpha(EQUIRECTANGULAR).devicePtr()
                                  : nullptr;
  // start mapping
  Status s = pano.getPrecomputedCoordinateBuffer()
                 ? transform->mapBufferLookup(frame, devBuffer, surf, mask, getSurfaceCoord(),
                                              (float)(1.0 / pano.getPrecomputedCoordinateShrinkFactor()),
                                              outputBounds[EQUIRECTANGULAR], pano, inputDef, getSurface(), stream)

                 : transform->mapBuffer(frame, devBuffer, surf, mask, outputBounds[EQUIRECTANGULAR], pano, inputDef,
                                        getSurface(), stream);
#ifdef READBACKMAPPEDIMAGE
  if (!getOutputRect(EQUIRECTANGULAR).empty()) {
    stream.synchronize();
    std::stringstream ss;
    ss << "warped-";
    ss << imId << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), devWork[EQUIRECTANGULAR].borrow(),
                                getOutputRect(EQUIRECTANGULAR).getWidth(), getOutputRect(EQUIRECTANGULAR).getHeight());
  }
#endif

  return s;
}

Status ImageMapping::warpCubemap(frameid_t frame, const PanoDefinition& pano, bool equiangular, GPU::Stream& stream) {
  const InputDefinition& inputDef = pano.getInput(imId);

  // start mapping
  Status s = transform->warpCubemap(
      frame, devWork[CUBE_MAP_POSITIVE_X].borrow(), outputBounds[CUBE_MAP_POSITIVE_X],
      devWork[CUBE_MAP_NEGATIVE_X].borrow(), outputBounds[CUBE_MAP_NEGATIVE_X], devWork[CUBE_MAP_POSITIVE_Y].borrow(),
      outputBounds[CUBE_MAP_POSITIVE_Y], devWork[CUBE_MAP_NEGATIVE_Y].borrow(), outputBounds[CUBE_MAP_NEGATIVE_Y],
      devWork[CUBE_MAP_POSITIVE_Z].borrow(), outputBounds[CUBE_MAP_POSITIVE_Z], devWork[CUBE_MAP_NEGATIVE_Z].borrow(),
      outputBounds[CUBE_MAP_NEGATIVE_Z], pano, inputDef, getSurface(), equiangular, stream);

#ifdef READBACKMAPPEDIMAGE
  for (int t = CUBE_MAP_POSITIVE_X; t <= CUBE_MAP_NEGATIVE_Z; ++t) {
    TextureTarget target = (TextureTarget)t;
    if (!getOutputRect(target).empty()) {
      stream.synchronize();
      std::stringstream ss;
      ss << "warped-";
      ss << imId << "-" << t << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), devWork[t].borrow(), getOutputRect(target).getWidth(),
                                  getOutputRect(target).getHeight());
    }
  }
#endif

  return s;
}

Status ImageMapping::reconstruct(TextureTarget target, const PanoDefinition& pano, GPU::Buffer<uint32_t> progressivePbo,
                                 bool final, GPU::Stream& stream) const {
  if (final) {
    FAIL_RETURN(merger->reconstruct(target, pano, progressivePbo, false, stream));
  }
  return Status::OK();
}

ImageMappingFlow::~ImageMappingFlow() {
  delete warper;
  delete flow;
}

Status ImageMappingFlow::setup(ImageMappingFlow* prevMapping, const PanoDefinition& pano,
                               const StereoRigDefinition* rigDef, const ImageMergerFactory& mergerFactory,
                               std::vector<readerid_t> id0s, std::shared_ptr<InputsMap> inputsMap,
                               const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory,
                               GPU::Stream stream) {
  FAIL_RETURN(ImageMapping::setup(prevMapping, pano, mergerFactory, inputsMap, stream, true));

  if (prevMapping) {
    Potential<ImageWarper> potWarper = warperFactory.create();
    FAIL_RETURN(potWarper.status());

    if (potWarper->needImageFlow()) {
      id0s.push_back((int)(prevMapping->imId));
      Potential<MergerPair> curPair =
          MergerPair::create(pano, rigDef, 1024, 175, id0s, imId, prevMapping->getOutputRect(EQUIRECTANGULAR),
                             getOutputRect(EQUIRECTANGULAR), stream);
      FAIL_RETURN(curPair.status());

      // It is a valid merger pair if and only if they are overlapping
      // Or else, there is nothing to be done here
      // TODO: Need to extend this concept later, in case they are not overlapping
      // but close to each other, finding the flow will potentially increase the
      // quality of blending
      if (curPair->doesOverlap()) {
        mergerPair = std::shared_ptr<MergerPair>(curPair.release());

        FAIL_RETURN(potWarper->init(mergerPair));
        FAIL_RETURN(potWarper->setupCommon(stream));
        warper = potWarper.release();

        Potential<ImageFlow> curFlow = flowFactory.create();
        FAIL_RETURN(curFlow.status());
        FAIL_RETURN(curFlow->init(mergerPair));
        flow = curFlow.release();
      }
    }
  }

  return Status::OK();
}

Status ImageMappingFlow::warp(frameid_t frame, const PanoDefinition& pano, GPU::Buffer<uint32_t> progressivePbo,
                              GPU::Surface& surf, GPU::Stream& stream) {
  // the first mapping is the reference image
  if (!mergerPair) {
    return ImageMapping::warp(frame, pano, progressivePbo, surf, stream);
  }

  // copy the input texture to a pixel buffer object
  GPU::memcpyAsync(devFlowIn.borrow(), getSurface(), stream);

  const int2 panoSize = make_int2((int)pano.getWidth(), (int)pano.getHeight());
  PROPAGATE_FAILURE_STATUS(flow->findMultiScaleImageFlow(
      frame, 0, panoSize, progressivePbo, make_int2(int(getSurface().width()), int(getSurface().height())),
      devFlowIn.borrow_const(), stream));

  int2 lookupOffset = flow->getLookupOffset(0);
  GPU::UniqueBuffer<float4> debug;
  PROPAGATE_FAILURE_STATUS(debug.alloc(outputBounds[0].getArea(), "Tmp Image Mapping"));
  GPU::UniqueBuffer<uint32_t> flowWarpedBuffer;
  PROPAGATE_FAILURE_STATUS(flowWarpedBuffer.alloc(outputBounds[0].getArea(), "Tmp Image Mapping"));

  // Need color remapping here as well, but this remain for later
  warper->warp(devWork[EQUIRECTANGULAR].borrow(), devFlowIn.borrow(), flow->getExtrapolatedFlowRect(0),
               flow->getFinalExtrapolatedFlowBuffer(), lookupOffset.x, lookupOffset.y, debug.borrow(),
               flowWarpedBuffer.borrow(), stream);
  PROPAGATE_FAILURE_STATUS(stream.synchronize());

#ifdef WARPED_INPUT_IMAGE
  {
    stream.synchronize();
    std::stringstream ss;
    ss.str("");
    ss << "debugOut-" << getImId() << ".png";
    Debug::dumpRGBADeviceBufferWithTransferFn<Debug::Float4ValueGetter<0, 1>, Debug::clamp0255>(
        ss.str().c_str(), debug.borrow_const(), outputBounds[EQUIRECTANGULAR].getWidth(),
        outputBounds[EQUIRECTANGULAR].getHeight());

    ss.str("");
    ss << "flowWarpedOut-" << getImId() << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), flowWarpedBuffer.borrow_const(),
                                outputBounds[EQUIRECTANGULAR].getWidth(), outputBounds[EQUIRECTANGULAR].getHeight());

    ss.str("");
    ss << "warpedOut-" << getImId() << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), devWork[EQUIRECTANGULAR].borrow_const(),
                                outputBounds[EQUIRECTANGULAR].getWidth(), outputBounds[EQUIRECTANGULAR].getHeight());

    GPU::UniqueBuffer<uint32_t> panoBuffer;
    PROPAGATE_FAILURE_STATUS(panoBuffer.alloc(pano.getWidth() * pano.getHeight(), "Tmp ImageMapping"));
    PROPAGATE_FAILURE_STATUS(Util::ImageProcessingGPU::packBuffer<uint32_t>(
        pano.getWidth(), 0, outputBounds[EQUIRECTANGULAR], devWork[EQUIRECTANGULAR].borrow_const(),
        Core::Rect(0, 0, pano.getHeight() - 1, pano.getWidth() - 1), panoBuffer.borrow(), stream));
    PROPAGATE_FAILURE_STATUS(stream.synchronize());
    ss.str("");
    ss << "warpedOutPanoSpace-" << getImId() << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), panoBuffer.borrow_const(), pano.getWidth(), pano.getHeight());

    ss.str("");
    ss << "panoDevOut-" << getImId() << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), panoDevOut, pano.getWidth(), pano.getHeight());
  }
#endif

  return Status::OK();
}

Status ImageMappingFlow::reconstruct(TextureTarget target, const PanoDefinition& pano,
                                     GPU::Buffer<uint32_t> progressivePbo, bool final, GPU::Stream& stream) const {
  if (merger->isMultiScale()) {
    FAIL_RETURN(merger->reconstruct(target, pano, progressivePbo, !final, stream));
  }
  return Status::OK();
}

Status ImageMapping::precomputedCoord(frameid_t frame, const PanoDefinition& pano, GPU::Stream& stream) {
  if (!pano.getPrecomputedCoordinateBuffer()) {
    return Status::OK();
  }
  setHBounds(EQUIRECTANGULAR_LOOKUP,
             int64_t(outputBounds[EQUIRECTANGULAR].left() / pano.getPrecomputedCoordinateShrinkFactor()),
             int64_t(outputBounds[EQUIRECTANGULAR].right() / pano.getPrecomputedCoordinateShrinkFactor()),
             int64_t(pano.getWidth() / pano.getPrecomputedCoordinateShrinkFactor()));
  setVBounds(EQUIRECTANGULAR_LOOKUP,
             int64_t(outputBounds[EQUIRECTANGULAR].top() / pano.getPrecomputedCoordinateShrinkFactor()),
             int64_t(outputBounds[EQUIRECTANGULAR].bottom() / pano.getPrecomputedCoordinateShrinkFactor()));

  // The coordinate buffer was not allocated, this is a valid call
  if (!devCoord || (devCoord->getWidth() != (size_t)outputBounds[EQUIRECTANGULAR_LOOKUP].getWidth()) ||
      (devCoord->getHeight() != (size_t)outputBounds[EQUIRECTANGULAR_LOOKUP].getHeight())) {
    delete devCoord;
    auto tex = Core::OffscreenAllocator::createCoordSurface(outputBounds[EQUIRECTANGULAR_LOOKUP].getWidth(),
                                                            outputBounds[EQUIRECTANGULAR_LOOKUP].getHeight(),
                                                            "Warped Coordinate Mapping");
    if (tex.ok()) {
      devCoord = tex.release();
    }
  }
  if (outputBounds.empty()) {
    return Status::OK();  // nothing to do.
  }
  const InputDefinition& inputDef = pano.getInput(imId);
  PROPAGATE_FAILURE_STATUS(transform->mapBufferCoord(frame, getSurfaceCoord(), outputBounds[EQUIRECTANGULAR_LOOKUP],
                                                     pano, inputDef, stream));
  return Status::OK();
}

Status ImageMapping::setupTexArrayAsync(frameid_t frame, const Input::PotentialFrame& inputFrame,
                                        const InputDefinition& inputDef, GPU::Stream& stream,
                                        Input::VideoReader* reader, const PreProcessor* preprocessor) {
  if (outputBounds.empty()) {
    return Status::OK();  // nothing to do.
  }

  GPU::Buffer<unsigned char> inputDevBuffer;
  if (inputFrame.status.ok()) {
    switch (inputFrame.frame.addressSpace()) {
      case Host:
        PROPAGATE_FAILURE_STATUS(GPU::memcpyAsync(devUnpackTmp.borrow(), inputFrame.frame.hostBuffer(),
                                                  (size_t)reader->getFrameDataSize(), stream));
        inputDevBuffer = devUnpackTmp.borrow();
        break;
      case Device:
        inputDevBuffer = inputFrame.frame.deviceBuffer();
        break;
    }
    PROPAGATE_FAILURE_STATUS(reader->unpackDevBuffer(getSurface(), inputDevBuffer, stream));
    if (preprocessor) {
      preprocessor->process(frame, getSurface(), inputDef.getWidth(), inputDef.getHeight(), imId, stream);
    }
  } else {
    // error policy : black frames in case of reader error/EOF
    // PROPAGATE_FAILURE_STATUS(GPU::memsetToZeroAsync(devOutBuf, inputDef.getWidth() * inputDef.getHeight() * 4,
    // stream));
  }

#ifdef READBACK_INPUT_IMAGE
  stream.synchronize();
  std::stringstream ss;
  ss << "inputdata-" << imId << ".png";
  Debug::dumpRGBATexture(ss.str().c_str(), getSurface(), inputDef.getWidth(), inputDef.getHeight());
#endif
  return Status::OK();
}

void ImageMapping::setHBounds(TextureTarget t, int64_t l, int64_t r, int64_t panoWidth) {
  assert(l <= panoWidth);
  if (t == EQUIRECTANGULAR) {
    if (l > r) {
      r += panoWidth;
      wrapsAround = 1;
    } else {
      wrapsAround = 0;
    }
  }
  outputBounds[t].setLeft(l);
  outputBounds[t].setRight(r);
}

void ImageMapping::setVBounds(TextureTarget t, int64_t top, int64_t bottom) {
  assert(top <= bottom);
  // see setHBounds
  outputBounds[t].setTop(top);
  outputBounds[t].setBottom(bottom);
}

Status ImageMapping::allocateUnpackBuffer(int64_t frameDataSize) {
  PROPAGATE_FAILURE_STATUS(devUnpackTmp.alloc((size_t)frameDataSize, "Unpacking Input Frame"));
  return Status::OK();
}

Status ImageMapping::allocateOutputBuffers(TextureTarget t, int64_t width, int64_t height) {
  if (!outputBounds[t].empty()) {
    PROPAGATE_FAILURE_STATUS(
        devWork[t].alloc((size_t)std::max(outputBounds[t].getArea(), width * height), "Warped Input Frame"));
  }
  return Status::OK();
}

Status ImageMappingFlow::allocateOutputBuffers(TextureTarget t, int64_t width, int64_t height) {
  ImageMapping::allocateOutputBuffers(t, width, height);
  if (!outputBounds[t].empty()) {
    PROPAGATE_FAILURE_STATUS(
        devFlowIn.alloc((size_t)std::max(outputBounds[t].getArea(), width * height), "Warped Input Frame"));
  }
  return Status::OK();
}

Status ImageMapping::allocateBuffers(TextureTarget t, int64_t width, int64_t height) {
  if (!surface) {
    auto tex = OffscreenAllocator::createSourceSurface(width, height, "ImageMapping");
    if (tex.ok()) {
      surface = tex.release();
    }
  }
  return allocateOutputBuffers(t, width, height);
}

Status ImageMapping::allocateBuffersPartial(TextureTarget t, int64_t width, int64_t height,
                                            SourceSurface* sourceSurface) {
  surface = sourceSurface;
  return allocateOutputBuffers(t, width, height);
}
}  // namespace Core
}  // namespace VideoStitch
