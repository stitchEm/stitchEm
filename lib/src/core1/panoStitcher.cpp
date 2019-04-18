// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoStitcher.hpp"

#include "bounds.hpp"
#include "imageMapping.hpp"
#include "imageMerger.hpp"
#include "panoRemapper.hpp"

#include "common/angles.hpp"
#include "common/container.hpp"
#include "core/geoTransform.hpp"
#include "core/photoTransform.hpp"
#include "core/stitchOutput/stitchOutput.hpp"
#include "gpu/core1/strip.hpp"
#include "gpu/core1/transform.hpp"
#include "gpu/image/imgInsert.hpp"
#include "image/unpack.hpp"
#include "input/maskedReader.hpp"
#include "processors/photoCorrProcessor.hpp"
#include "parallax/mergerPair.hpp"
#include "parallax/imageWarper.hpp"
#include "parallax/imageFlow.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/stereoRigDef.hpp"
#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/imageFlowFactory.hpp"
#include "libvideostitch/imageWarperFactory.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include "inputsMap.hpp"
#include <cassert>
#include <fstream>
#include <memory>
#include <sstream>
#include <iomanip>

//#define PROGRESSIVE_RESULT

#if defined(PROGRESSIVE_RESULT)
#include "util/pngutil.hpp"
#include "util/pnm.hpp"
#include "util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Core {

template <typename Output>
PanoStitcherImplV1<Output>::PanoStitcherImplV1(const std::string& name, const PanoDefinition& pano, Eye eye)
    : PanoStitcherImplBase<Output>(name, pano, eye),
      rigDef(nullptr),
      alignSize(ImageMerger::CudaBlockSize),
      merger(nullptr) {}

template <typename Output>
PanoStitcherImplV1<Output>::~PanoStitcherImplV1() {
  deleteAllValues(imageMappings);
}

//--------------------------------------- Runtime ---------------------------------------------

template <typename Output>
Status PanoStitcherImplV1<Output>::merge(frameid_t frame,
                                         const std::map<readerid_t, Input::PotentialFrame>& inputFrames,
                                         const std::map<readerid_t, Input::VideoReader*>& readers,
                                         const std::map<readerid_t, PreProcessor*>& preprocessors, PanoSurface& pano) {
  Status success = Status::OK();
  bool isFirstMerger = true;
  GPU::Event previousMergeFinishedEvent;
  std::vector<GPU::Event> readingFinishedEvents;

  pano.pimpl->reset(merger);

  FAIL_RETURN(adaptInputsMap(frame, readers));

  bool firstInput = true;
  for (auto mapping : imageMappings) {
    Input::VideoReader* reader = readers.at(mapping.first);
    const InputDefinition& inputDef = getPano().getInput(mapping.first);
    PreProcessor* preprocessor = preprocessors.at(mapping.first);
    GPU::Stream inputStream = getStreamForInput(mapping.first);

    FAIL_RETURN(mapping.second->setupTexArrayAsync(frame, inputFrames.find(mapping.first)->second, inputDef,
                                                   inputStream, reader, preprocessor));
    PotentialValue<GPU::Event> readingDone = inputStream.recordEvent();
    FAIL_RETURN(readingDone.status());
    readingFinishedEvents.push_back(readingDone.value());

    if (inputDef.getIsVideoEnabled()) {
      if (mapping.second->getMerger().warpMergeType() == ImageMerger::Format::None) {
        /* if merge is combined with wrap, stream synchro should be done before warp */
        FAIL_RETURN(pano.pimpl->warp(mapping.second, frame, getPano(), inputStream));
      }

      if (firstInput) {
        // setup may be done on merge stream, first stream waits for that
        FAIL_RETURN(inputStream.synchronizeOnStream(pano.pimpl->stream));
        firstInput = false;
      } else {
        // To launch merge for stream i, merging must be done for stream i-1.
        FAIL_RETURN(inputStream.waitOnEvent(previousMergeFinishedEvent));
      }

      if (mapping.second->getMerger().warpMergeType() != ImageMerger::Format::None) {
        FAIL_RETURN(pano.pimpl->warp(mapping.second, frame, getPano(), inputStream));
      }

      FAIL_RETURN(pano.pimpl->blend(getPano(), *mapping.second, isFirstMerger, inputStream));
      FAIL_RETURN(pano.pimpl->reconstruct(getPano(), *mapping.second, inputStream, false));
      isFirstMerger = false;
    }
    PotentialValue<GPU::Event> potEvent = inputStream.recordEvent();
    FAIL_RETURN(potEvent.status());
    previousMergeFinishedEvent = potEvent.value();

    // TODO_OPENCL_IMPL remove
    // workaround for VSA-5829
    inputStream.flush();

#ifdef PROGRESSIVE_RESULT
    std::stringstream ss;
    ss << "pano-output-" << mapping.first << ".png";
    if (warper != nullptr && warper->needImageFlow() && imageMergers[mapping.first]->isMultiScale()) {
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), tmpDevOut.borrow_const(), getPano().getWidth(),
                                  getPano().getHeight());
    } else {
      FAIL_RETURN(imageMergers[mapping.first]->finalizeToBuffer(pano.pimpl->buffer, inputStream));
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), pano.pimpl->buffer, getPano().getWidth(), getPano().getHeight());
    }
#endif
  }

  {
    // merge stream needs to wait for all operations to be finished on the input streams
    const Status status = pano.pimpl->stream.waitOnEvent(previousMergeFinishedEvent);
    if (!status.ok()) {
      Logger::get(Logger::Warning) << "Skipping output for frame " << frame << std::endl;
      if (success.ok()) {
        success = status;
      }
    }
  }

  if (success.ok()) {
    FAIL_RETURN(pano.pimpl->reconstruct(getPano(), *imageMappings.begin()->second, pano.pimpl->stream));

    SIMPLEPROFILE_MS("global pano remapping:");
    // apply global rotation
    Quaternion<double> stabilization = getPano().getStabilization().at(frame);
    Matrix33<double> perspective = stabilization.toRotationMatrix();
    Quaternion<double> orientation = getPano().getGlobalOrientation().at(frame);
    perspective *= orientation.toRotationMatrix();
    perspective *= getInteractivePersp();

    FAIL_RETURN(pano.pimpl->reproject(getPano(), perspective, merger));
  }

  // input frames are borrowed only for this call, make sure
  // all read operations have finished before returning
  for (auto& readFinished : readingFinishedEvents) {
    readFinished.synchronize();
  }

  return success;
}

//--------------------------------------- Setup ---------------------------------------------

template <typename Output>
Status PanoStitcherImplV1<Output>::computeSetupImage(const std::map<readerid_t, Input::VideoReader*>& readers,
                                                     const StereoRigDefinition* rigDef) {
  if (getPano().getProjection() == PanoProjection::Cubemap ||
      getPano().getProjection() == PanoProjection::EquiangularCubemap) {
    FAIL_RETURN(inputsMapCubemap->compute(readers, getPano()));
  }
  return inputsMap->compute(readers, getPano(), rigDef, this->getEye(), true);
}

template <typename Output>
Status PanoStitcherImplV1<Output>::setupImpl(const ImageMergerFactory& mergerFactory,
                                             const ImageWarperFactory& warperFactory,
                                             const ImageFlowFactory& flowFactory,
                                             const std::map<readerid_t, Input::VideoReader*>& readers,
                                             const StereoRigDefinition* rig) {
  Potential<InputsMap> potInputsMap = InputsMap::create(getPano());
  FAIL_RETURN(potInputsMap.status());
  inputsMap = std::shared_ptr<InputsMap>(potInputsMap.release());

  Potential<InputsMapCubemap> potInputsMapCubemap = InputsMapCubemap::create(getPano());
  FAIL_RETURN(potInputsMapCubemap.status());
  inputsMapCubemap = std::shared_ptr<InputsMapCubemap>(potInputsMapCubemap.release());

  PROPAGATE_FAILURE_CAUSE(setupCommon(mergerFactory, warperFactory, flowFactory, readers, rig), Origin::Stitcher,
                          ErrType::SetupFailure, "Could not setup stitcher");

  // Allocate buffers for mapping.
  // This must be done after merge setup since mergers can resize the mapping rectangles.
  {
    SIMPLEPROFILE_MS("Allocate buffers");
    for (auto reader : readers) {
      const Input::VideoReader::Spec& spec = reader.second->getSpec();
      const Status status = imageMappings[reader.second->id]->allocateUnpackBuffer(spec.frameDataSize);
      if (getPano().getProjection() == PanoProjection::Cubemap ||
          getPano().getProjection() == PanoProjection::EquiangularCubemap) {
        for (int target = CUBE_MAP_POSITIVE_X; target <= CUBE_MAP_NEGATIVE_Z; ++target) {
          const Status status =
              imageMappings[reader.second->id]->allocateBuffers((TextureTarget)target, spec.width, spec.height);
          if (!status.ok()) {
            return Status(Origin::Stitcher, ErrType::SetupFailure,
                          "Could not setup mapper for input " + std::to_string(reader.second->id), status);
          }
        }
      } else {
        const Status status =
            imageMappings[reader.second->id]->allocateBuffers(EQUIRECTANGULAR, spec.width, spec.height);
        if (!status.ok()) {
          return Status(Origin::Stitcher, ErrType::SetupFailure,
                        "Could not setup mapper for input " + std::to_string(reader.second->id), status);
        }
      }
    }
  }
  return Status::OK();
}

template <typename Output>
Status PanoStitcherImplV1<Output>::adaptInputsMap(const frameid_t frameId,
                                                  std::map<readerid_t, Input::VideoReader*> readers) {
  std::pair<int64_t, int64_t> boundedFrames = getPano().getBlendingMaskBoundedFrameIds(frameId);
  std::pair<int64_t, int64_t> inputsMapBoundedFrames = inputsMap->getBoundedFrameIds();
  const bool reloadBoundedFrame =
      boundedFrames.first != inputsMapBoundedFrames.first || boundedFrames.second != inputsMapBoundedFrames.second;
  const bool interpolateFrame = getPano().getBlendingMaskInterpolationEnabled() && frameId <= boundedFrames.second &&
                                frameId >= boundedFrames.first && boundedFrames.first != boundedFrames.second;

  if (reloadBoundedFrame || interpolateFrame) {
    bool loaded = false;
#ifndef VS_OPENCL
    FAIL_RETURN(inputsMap->loadPrecomputedMap(frameId, getPano(), readers, maskInterpolation, loaded));
#endif
    if (loaded) {
      FAIL_RETURN(prepareMappers(rigDef));

      // Update image merging
      for (auto mapping : imageMappings) {
        // Do each setup in a parallel stream, they are totally parallel (setupBuffer is const).
        FAIL_CAUSE(
            mapping.second->getMerger().setup(getPano(), *inputsMap, *mapping.second, getStreamForInput(mapping.first)),
            Origin::Stitcher, ErrType::SetupFailure,
            "Could not setup merger for input " + std::to_string(mapping.first));
      }

      // Precompute coordinate buffer
      for (auto mapping : imageMappings) {
        GPU::Stream inputStream = getStreamForInput(mapping.first);
        FAIL_RETURN(mapping.second->precomputedCoord(0, getPano(), inputStream));
      }

      // Synchronize all streams
      for (auto mapping : imageMappings) {
        FAIL_RETURN(getStreamForInput(mapping.first).synchronize());
      }
    }
  }
  return Status::OK();
}

template <typename Output>
Status PanoStitcherImplV1<Output>::setupCommon(const ImageMergerFactory& mergerFactory,
                                               const ImageWarperFactory& warperFactory,
                                               const ImageFlowFactory& flowFactory,
                                               const std::map<readerid_t, Input::VideoReader*>& readers,
                                               const StereoRigDefinition* rig) {
  this->rigDef = rig;

  FAIL_RETURN(computeSetupImage(readers, rig));

  const bool isFlowBasedBlending{warperFactory.needsInputPreProcessing() || flowFactory.needsInputPreProcessing()};

  // Create the mappers.
  std::map<readerid_t, ImageMappingFlow*> mappingsFlow;
  for (auto reader : readers) {
    if (isFlowBasedBlending) {
      mappingsFlow[reader.second->id] = new ImageMappingFlow(reader.second->id);
      imageMappings[reader.second->id] = mappingsFlow[reader.second->id];
    } else {
      imageMappings[reader.second->id] = new ImageMapping(reader.second->id);
    }
  }

  // Compute the bounding boxes.
  alignSize = mergerFactory.getBlockAlignment();
  FAIL_RETURN(prepareMappers(rig));

  // Setup the mappers
  if (isFlowBasedBlending) {
    std::vector<readerid_t> id0s;
    ImageMappingFlow* prevMapping = nullptr;
    for (auto mapping : mappingsFlow) {
      FAIL_RETURN(mapping.second->setup(prevMapping, getPano(), rig, mergerFactory, id0s, inputsMap, warperFactory,
                                        flowFactory, getStreamForInput(mapping.first)));
      id0s.push_back(mapping.first);
      prevMapping = mapping.second;
    }
    // OpenCL 1.2 does not support read_write images
    // We can not remove the internal memcopy at the beginning of the reproject()
    if (prevMapping && (GPU::getFramework() == Discovery::Framework::CUDA)) {
      merger = &prevMapping->getMerger();
    }
  } else {
    ImageMapping* prevMapping = nullptr;
    for (auto mapping : imageMappings) {
      FAIL_RETURN(
          mapping.second->setup(prevMapping, getPano(), mergerFactory, inputsMap, getStreamForInput(mapping.first)));
      prevMapping = mapping.second;
    }
    if (prevMapping && (GPU::getFramework() == Discovery::Framework::CUDA)) {
      merger = &prevMapping->getMerger();
    }
  }

  // Synchronize all streams.
  // for (auto order : maskOrders) {
  //  const int imId = order.second;
  for (auto mapping : imageMappings) {
    FAIL_RETURN(getStreamForInput(mapping.first).synchronize());
  }

#ifndef VS_OPENCL
  // Prepare mask interpolation
  Potential<MaskInterpolation::InputMaskInterpolation> potInputMaskInterpolation =
      MaskInterpolation::InputMaskInterpolation::create(getPano(), readers);
  if (potInputMaskInterpolation.status().ok()) {
    maskInterpolation.reset(potInputMaskInterpolation.release());
  }
#endif

  return Status::OK();
}

/**
 * Detect image boundaries
 */
template <typename Output>
Status PanoStitcherImplV1<Output>::prepareMappers(const StereoRigDefinition* rig) {
  int64_t maxDim = std::max(getPano().getWidth(), getPano().getHeight());
  maxDim = std::max(maxDim, getPano().getLength());

  auto tmpDevBuffer = GPU::Buffer<uint32_t>::allocate(maxDim, "Input Bounding boxes");
  FAIL_RETURN(tmpDevBuffer.status());

  auto tmpHostBuffer = GPU::HostBuffer<uint32_t>::allocate(maxDim, "Input Bounding boxes");
  FAIL_RETURN(tmpHostBuffer.status());

  SIMPLEPROFILE_MS("compute image bounding boxes");
  // const std::vector<size_t> maskOrders = getPano().getMasksOrder();
  GPU::Stream stream = getStreamForInput(imageMappings.begin()->first);
  for (int t = EQUIRECTANGULAR; t <= CUBE_MAP_NEGATIVE_Z; ++t) {
    TextureTarget target = (TextureTarget)t;
    GPU::Buffer<uint32_t> inputsMask;
    int64_t width, height;
    if (target == EQUIRECTANGULAR) {
      inputsMask = inputsMap->getMask();
      width = getPano().getWidth();
      height = getPano().getHeight();
    } else {
      inputsMask = inputsMapCubemap->getMask(target);
      width = getPano().getLength();
      height = getPano().getLength();
    }
    FAIL_RETURN(computeHBounds(target, width, height, imageMappings, rig, this->getEye(), inputsMask,
                               tmpHostBuffer.value(), tmpDevBuffer.value(), stream, true));
    FAIL_RETURN(computeVBounds(target, width, height, imageMappings, inputsMask, tmpHostBuffer.value(),
                               tmpDevBuffer.value(), stream));
  }

  for (auto mapping : imageMappings) {
    for (int t = EQUIRECTANGULAR; t <= CUBE_MAP_NEGATIVE_Z; ++t) {
      TextureTarget target = (TextureTarget)t;
      if (!mapping.second->getOutputRect(target).empty()) {
        // Make sure the left and top offsets are levels times divisible by two.
        // (see example in the Laplacian merger merge() for why).
        mapping.second->getOutputRect(target).growToAlignTo(alignSize, alignSize);
        mapping.second->getOutputRect(target).growToMultipleSizeOf(alignSize, alignSize);
      }
    }
  }

  FAIL_RETURN(tmpHostBuffer.value().release());
  FAIL_RETURN(tmpDevBuffer.value().release());
  return Status::OK();
}

template <typename Output>
Status PanoStitcherImplV1<Output>::redoSetupImpl(const ImageMergerFactory& mergerFactory,
                                                 const ImageWarperFactory& warperFactory,
                                                 const ImageFlowFactory& flowFactory,
                                                 const std::map<readerid_t, Input::VideoReader*>& readers,
                                                 const StereoRigDefinition* rig) {
  if (imageMappings.empty()) {
    return Status::OK();
  }

  // Delete all mappers, but before that steal all input buffers.
  // This will enable restitching directly because we will still have the reader data in the input buffer.
  std::map<videoreaderid_t, SourceSurface*> mapperHostInputBuffers;
  for (auto mapping : imageMappings) {
    SourceSurface* sourceSurf = nullptr;
    mapping.second->releaseInputBuffers(&sourceSurf);
    mapperHostInputBuffers[mapping.first] = sourceSurf;
  }
  deleteAllValues(imageMappings);

  FAIL_CAUSE(setupCommon(mergerFactory, warperFactory, flowFactory, readers, rig), Origin::Stitcher,
             ErrType::SetupFailure, "Could not set up stitcher");

  // Allocate buffers for mapping.
  // This must be done after merge setup since mergers can resize the mapping rectangles.
  {
    SIMPLEPROFILE_MS("Allocate buffers");
    for (auto reader : readers) {
      const Input::VideoReader::Spec& spec = reader.second->getSpec();
      const Status status = imageMappings[reader.second->id]->allocateUnpackBuffer(spec.frameDataSize);
      for (int t = EQUIRECTANGULAR; t <= CUBE_MAP_NEGATIVE_Z; ++t) {
        const Status status = imageMappings[reader.second->id]->allocateBuffersPartial(
            (TextureTarget)t, spec.width, spec.height, mapperHostInputBuffers[reader.second->id]);
        if (!status.ok()) {
          return Status(Origin::Stitcher, ErrType::SetupFailure,
                        "Could not setup mapper for input " + std::to_string(reader.second->id), status);
        }
      }
    }
  }
  return Status::OK();
}

template <typename Output>
ChangeCompatibility PanoStitcherImplV1<Output>::getCompatibility(const InputDefinition& im,
                                                                 const InputDefinition& newIm) {
  ChangeCompatibility compat = SetupCompatibleChanges;
#define DECLARE_INCOMPATIBLE(accessor)                                                              \
  static_assert(!std::is_pointer<decltype(im.accessor())>::value, "Are you comparing pointers ?!"); \
  if (im.accessor() != newIm.accessor()) {                                                          \
    return IncompatibleChanges;                                                                     \
  }
#define DECLARE_SETUPINCOMPATIBLE(accessor)                                                         \
  static_assert(!std::is_pointer<decltype(im.accessor())>::value, "Are you comparing pointers ?!"); \
  if (im.accessor() != newIm.accessor()) {                                                          \
    compat = worstCompatibility(compat, SetupIncompatibleChanges);                                  \
  }
#define DECLARE_SETUPCOMPATIBLE(accessor)
  DECLARE_INCOMPATIBLE(getReaderConfig);
  DECLARE_INCOMPATIBLE(getMaskData);
  DECLARE_SETUPINCOMPATIBLE(getWidth);
  DECLARE_SETUPINCOMPATIBLE(getHeight);
  DECLARE_SETUPINCOMPATIBLE(getCroppedWidth);
  DECLARE_SETUPINCOMPATIBLE(getCroppedHeight);
  DECLARE_SETUPINCOMPATIBLE(getCropLeft);
  DECLARE_SETUPINCOMPATIBLE(getCropRight);
  DECLARE_SETUPINCOMPATIBLE(getCropTop);
  DECLARE_SETUPINCOMPATIBLE(getCropBottom);
  DECLARE_SETUPINCOMPATIBLE(getFormat);
  DECLARE_SETUPCOMPATIBLE(getRedCB);
  DECLARE_SETUPCOMPATIBLE(getGreenCB);
  DECLARE_SETUPCOMPATIBLE(getBlueCB);
  DECLARE_SETUPCOMPATIBLE(getExposureValue);
  DECLARE_SETUPINCOMPATIBLE(getEmorA);
  DECLARE_SETUPINCOMPATIBLE(getEmorB);
  DECLARE_SETUPINCOMPATIBLE(getEmorC);
  DECLARE_SETUPINCOMPATIBLE(getEmorD);
  DECLARE_SETUPINCOMPATIBLE(getEmorE);
  DECLARE_SETUPINCOMPATIBLE(getGamma);
  DECLARE_SETUPINCOMPATIBLE(getVignettingCoeff0);
  DECLARE_SETUPINCOMPATIBLE(getVignettingCoeff1);
  DECLARE_SETUPINCOMPATIBLE(getVignettingCoeff2);
  DECLARE_SETUPINCOMPATIBLE(getVignettingCoeff3);
  DECLARE_SETUPINCOMPATIBLE(getVignettingCenterX);
  DECLARE_SETUPINCOMPATIBLE(getVignettingCenterY);
  DECLARE_SETUPINCOMPATIBLE(getPhotoResponse);
  DECLARE_SETUPINCOMPATIBLE(hasCroppedArea);
  DECLARE_SETUPCOMPATIBLE(getFrameOffset);
  DECLARE_SETUPCOMPATIBLE(getSynchroCost);
  DECLARE_SETUPCOMPATIBLE(getStack);  // ignored
  DECLARE_SETUPINCOMPATIBLE(getGeometries);
#undef DECLARE_INCOMPATIBLE
#undef DECLARE_SETUPINCOMPATIBLE
#undef DECLARE_SETUPCOMPATIBLE

  return compat;
}

template <typename Output>
ChangeCompatibility PanoStitcherImplV1<Output>::getCompatibility(const OverlayInputDefinition& im,
                                                                 const OverlayInputDefinition& newIm) {
  ChangeCompatibility compat = SetupCompatibleChanges;
#define DECLARE_INCOMPATIBLE(accessor)                                                              \
  static_assert(!std::is_pointer<decltype(im.accessor())>::value, "Are you comparing pointers ?!"); \
  if (im.accessor() != newIm.accessor()) {                                                          \
    return IncompatibleChanges;                                                                     \
  }
#define DECLARE_SETUPINCOMPATIBLE(accessor)                                                         \
  static_assert(!std::is_pointer<decltype(im.accessor())>::value, "Are you comparing pointers ?!"); \
  if (im.accessor() != newIm.accessor()) {                                                          \
    compat = worstCompatibility(compat, SetupIncompatibleChanges);                                  \
  }
#define DECLARE_SETUPCOMPATIBLE(accessor)
  DECLARE_INCOMPATIBLE(getReaderConfig);
  DECLARE_SETUPINCOMPATIBLE(getWidth);
  DECLARE_SETUPINCOMPATIBLE(getHeight);
  DECLARE_SETUPCOMPATIBLE(getFrameOffset);
  DECLARE_SETUPCOMPATIBLE(getScaleCurve);
  DECLARE_SETUPCOMPATIBLE(getAlphaCurve);
  DECLARE_SETUPCOMPATIBLE(getTransXCurve);
  DECLARE_SETUPCOMPATIBLE(getTransYCurve);
  DECLARE_SETUPCOMPATIBLE(getTransZCurve);
  DECLARE_SETUPCOMPATIBLE(getRotationCurve);
  DECLARE_SETUPCOMPATIBLE(getGlobalOrietationApplied);
#undef DECLARE_INCOMPATIBLE
#undef DECLARE_SETUPINCOMPATIBLE
#undef DECLARE_SETUPCOMPATIBLE

  return compat;
}

template <typename Output>
ChangeCompatibility PanoStitcherImplV1<Output>::getCompatibility(const PanoDefinition& pano,
                                                                 const PanoDefinition& newPano) const {
  if (pano.numInputs() != newPano.numInputs()) {
    return IncompatibleChanges;
  }
  ChangeCompatibility compat = SetupCompatibleChanges;
  // Inputs.
  for (readerid_t i = 0; i < pano.numInputs() && compat != IncompatibleChanges; ++i) {
    compat = worstCompatibility(compat, getCompatibility(pano.getInput(i), newPano.getInput(i)));
  }

  // Overlays.
  for (overlayreaderid_t i = 0; i < pano.numOverlays() && compat != IncompatibleChanges; ++i) {
    compat = worstCompatibility(compat, getCompatibility(pano.getOverlay(i), newPano.getOverlay(i)));
  }

#define DECLARE_INCOMPATIBLE(accessor)         \
  if (pano.accessor() != newPano.accessor()) { \
    return IncompatibleChanges;                \
  }
#define DECLARE_SETUPINCOMPATIBLE(accessor)                        \
  if (pano.accessor() != newPano.accessor()) {                     \
    compat = worstCompatibility(compat, SetupIncompatibleChanges); \
  }
#define DECLARE_SETUPCOMPATIBLE(accessor)
  DECLARE_INCOMPATIBLE(getWidth);
  DECLARE_INCOMPATIBLE(getHeight);
  DECLARE_SETUPINCOMPATIBLE(getLength);
  DECLARE_SETUPCOMPATIBLE(getExposureValue);
  DECLARE_SETUPINCOMPATIBLE(getProjection);
  DECLARE_SETUPINCOMPATIBLE(getHFOV);
  DECLARE_SETUPINCOMPATIBLE(getBlendingMaskEnabled);
  DECLARE_SETUPINCOMPATIBLE(getBlendingMaskWidth);
  DECLARE_SETUPINCOMPATIBLE(getBlendingMaskHeight);
  DECLARE_SETUPINCOMPATIBLE(getSphereScale);

#undef DECLARE_INCOMPATIBLE
#undef DECLARE_SETUPINCOMPATIBLE
#undef DECLARE_SETUPCOMPATIBLE
  return compat;
}

// explicit instantiations

template class PanoStitcherImplV1<StitchOutput>;
template class PanoStitcherImplV1<StereoOutput>;

}  // namespace Core
}  // namespace VideoStitch
