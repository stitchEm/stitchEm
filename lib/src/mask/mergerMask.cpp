// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mergerMask.hpp"
#include "seamFinder.hpp"

#include "core/controllerInputFrames.hpp"
#include "core1/bounds.hpp"
#include "core1/imageMapping.hpp"
#include "core1/imageMerger.hpp"
#include "core1/panoRemapper.hpp"
#include "core1/inputsMap.hpp"
#include "core1/imageMerger.hpp"
#include "core1/textureTarget.hpp"

#include "gpu/allocator.hpp"
#include "gpu/core1/strip.hpp"
#include "gpu/core1/transform.hpp"
#include "gpu/core1/voronoi.hpp"
#include "gpu/uniqueBuffer.hpp"
#include "gpu/stream.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/image/sampling.hpp"
#include "gpu/image/blur.hpp"
#include "util/registeredAlgo.hpp"
#include "util/compressionUtils.hpp"
#include "util/imageProcessingGPUUtils.hpp"

#include <unordered_set>
#include <vector>

//#define MERGERMASK_SETUP_DISTORTIONMAP
//#define MERGERMASK_SETUP_MAPPEDRECT
//#define MERGERMASK_SETUP_FRAMES
//#define MERGERMASK_GET_MERGERMASKS
//#define MERGERMASK_GET_MERGERMASKS_FULL
//#define MERGERMASK_GET_MERGERMASKS_FINAL
//#define MERGERMASK_INPUT_SPACE

//#define MERGERMASK_SEAM_INPUT
//#define MERGERMASK_SEAM_OPTIMIZATION
//#define MERGERMASK_SEAM_MASK

#if defined(MERGERMASK_SETUP_MAPPEDRECT) || defined(MERGERMASK_SEAM_OPTIMIZATION) || defined(MERGERMASK_SEAM_MASK) || \
    defined(MERGERMASK_SEAM_INPUT) || defined(MERGERMASK_GET_MERGERMASKS_FINAL) || defined(MERGERMASK_INPUT_SPACE)
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#endif

#if defined(MERGERMASK_GET_MERGERMASKS_FULL) || defined(MERGERMASK_SETUP_FRAMES) ||                             \
    defined(MERGERMASK_SETUP_DISTORTIONMAP) || defined(MERGERMASK_GET_MERGERMASKS) ||                           \
    defined(MERGERMASK_SEAM_OPTIMIZATION) || defined(MERGERMASK_SEAM_MASK) || defined(MERGERMASK_SEAM_INPUT) || \
    defined(MERGERMASK_GET_MERGERMASKS_FINAL) || defined(MERGERMASK_INPUT_SPACE)
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "util/pngutil.hpp"
#include "util/pnm.hpp"
#include "util/debugUtils.hpp"
#include "util/opticalFlowUtils.hpp"
#endif

namespace VideoStitch {
namespace MergerMask {

Potential<MergerMask> MergerMask::create(const std::map<readerid_t, Input::VideoReader*>& readers,
                                         const Core::PanoDefinition& pano, const Core::StereoRigDefinition* rigDef,
                                         const MergerMaskConfig& config, const MergerMaskProgress& progress) {
  std::unique_ptr<MergerMask> mergerMask;
  mergerMask.reset(new MergerMask(pano, rigDef, config, progress));
  FAIL_RETURN(mergerMask->setup(readers));
  return mergerMask.release();
}

MergerMask::MergerMask(const Core::PanoDefinition& pano, const Core::StereoRigDefinition* rigDef,
                       const MergerMaskConfig& config, const MergerMaskProgress& progress)
    : pano(pano),
      rigDef(rigDef),
      mergerMaskConfig(config),
      // downsamplingLevelCount(getDownLevel(pano, (int)config.getSizeThreshold())),
      downsamplingLevelCount(0),
      progress(progress) {}

MergerMask::~MergerMask() {
  for (size_t i = 0; i < imageMappings.size(); i++) {
    delete imageMappings[(int)i];
  }
  imageMappings.clear();
  for (size_t i = 0; i < transforms.size(); i++) {
    delete transforms[(int)i];
  }
  transforms.clear();
  for (size_t i = 0; i < distortionMaps.size(); i++) {
    distortionMaps[(int)i].release();
  }
  for (size_t i = 0; i < originalDistortionMaps.size(); i++) {
    originalDistortionMaps[(int)i].release();
  }

  distortionMaps.clear();
  originalDistortionMaps.clear();
}

int MergerMask::getDownSize(const int size) const {
  int level = downsamplingLevelCount;
  int downSize = size;
  while (level > 0) {
    downSize = (downSize + 1) / 2;
    level--;
  }
  return downSize;
}

int MergerMask::getDownCoord(const int coord) const {
  int level = downsamplingLevelCount;
  float downCoord = (float)coord;
  while (level > 0) {
    downCoord = (downCoord + 1.0f) / 2.0f;
    level--;
  }
  return (int)downCoord;
}

int MergerMask::getDownLevel(const Core::PanoDefinition& pano, const int sizeThreshold) {
  int width = (int)pano.getWidth();
  int height = (int)pano.getHeight();
  int level = 0;
  while (width > sizeThreshold || height > sizeThreshold) {
    width = (width + 1) / 2;
    height = (height + 1) / 2;
    level++;
  }
  return level;
}

Status MergerMask::setupMappings(const std::map<readerid_t, Input::VideoReader*>& readers) {
  // Prepare input maps
  GPU::Stream stream = GPU::Stream::getDefault();
  std::unique_ptr<Core::InputsMap> inputsMap;
  Potential<Core::InputsMap> potInputsMap = Core::InputsMap::create(pano);
  FAIL_RETURN(potInputsMap.status());
  inputsMap = std::unique_ptr<Core::InputsMap>(potInputsMap.release());
  FAIL_RETURN(inputsMap->compute(readers, pano, rigDef, LeftEye, false));

  // Create the mappers.
  imageMappings.clear();

  for (auto reader : readers) {
    imageMappings[reader.second->id] = new Core::ImageMapping(reader.second->id);
  }

  // Prepare mappings
  auto tmpDevBuffer =
      GPU::Buffer<uint32_t>::allocate(size_t(std::max(pano.getWidth(), pano.getHeight())), "Input Bounding boxes");
  FAIL_RETURN(tmpDevBuffer.status());

  auto tmpHostBuffer = GPU::HostBuffer<uint32_t>::allocate((unsigned)std::max(pano.getWidth(), pano.getHeight()),
                                                           "Input Bounding boxes");
  FAIL_RETURN(tmpHostBuffer.status());

  FAIL_RETURN(computeHBounds(Core::EQUIRECTANGULAR, pano.getWidth(), pano.getHeight(), imageMappings, rigDef, LeftEye,
                             inputsMap->getMask(), tmpHostBuffer.value(), tmpDevBuffer.value(), stream, true));
  FAIL_RETURN(computeVBounds(Core::EQUIRECTANGULAR, pano.getWidth(), pano.getHeight(), imageMappings,
                             inputsMap->getMask(), tmpHostBuffer.value(), tmpDevBuffer.value(), stream));

  int devWidth = (int)pano.getWidth();
  int devHeight = (int)pano.getHeight();
  // Duplicate the original inputs map
  FAIL_RETURN(inputsMapOriginalBuffer.alloc(devWidth * devHeight, "Merger Mask"));
  FAIL_RETURN(GPU::memcpyBlocking(inputsMapOriginalBuffer.borrow(), inputsMap->getMask(),
                                  devWidth * devHeight * sizeof(uint32_t)));

#ifdef MERGERMASK_GET_MERGERMASKS_FULL
  {
    std::stringstream ss;
    ss.str("");
    ss << "input-full-index.png";
    Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), inputsMapOriginalBuffer.borrow_const(), devWidth,
                                               devHeight);
    for (int i = 0; i < pano.numInputs(); i++) {
      std::stringstream ss;
      ss.str("");
      ss << "input-full-index" << i << ".png";
      Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), inputsMapOriginalBuffer.borrow_const(), devWidth,
                                                 devHeight, i);
    }
  }
#endif

  // Generate the down-sampled inputs map
  FAIL_RETURN(Util::ImageProcessingGPU::downSampleImages<uint32_t>(downsamplingLevelCount, devWidth, devHeight,
                                                                   inputsMap->getMask(), stream, true));
  FAIL_RETURN(inputsMapBuffer.alloc(devWidth * devHeight, "Merger Mask"));
  FAIL_RETURN(
      GPU::memcpyBlocking(inputsMapBuffer.borrow(), inputsMap->getMask(), devWidth * devHeight * sizeof(uint32_t)));
  FAIL_RETURN(tmpHostBuffer.value().release());
  FAIL_RETURN(tmpDevBuffer.value().release());
  return Status::OK();
}

Status MergerMask::setupTransform(const std::map<readerid_t, Input::VideoReader*>& readers) {
  // Prepare transform
  for (auto reader : readers) {
    const Core::InputDefinition& inputDef = pano.getInput(reader.second->id);
    Core::Transform* transform = Core::Transform::create(inputDef);
    if (!transform) {
      return {Origin::Stitcher, ErrType::SetupFailure,
              "Cannot create v1 transformation for input " + std::to_string(reader.second->id)};
    }
    transforms[reader.second->id] = transform;
  }
  return Status::OK();
}

Status MergerMask::setupDistortionMap() {
  // Prepare distortion maps
  GPU::Stream stream = GPU::Stream::getDefault();
  for (auto transform : transforms) {
    Core::ImageMapping* mapping = imageMappings[transform.first];
    auto potDevOut = GPU::uniqueBuffer<unsigned char>(mapping->getOutputRect(Core::EQUIRECTANGULAR).getArea(),
                                                      "Merger Mask Algorithm");
    FAIL_RETURN(potDevOut.status());

    const Core::InputDefinition& im = pano.getInput(transform.first);
    FAIL_RETURN(transform.second->mapDistortion(0, potDevOut.borrow(), mapping->getOutputRect(Core::EQUIRECTANGULAR),
                                                pano, im, stream));

    int devWidth = (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getWidth();
    int devHeight = (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getHeight();

    FAIL_RETURN(updateDistortionFromMask((int)transform.first, make_int2(devWidth, devHeight),
                                         make_int2((int)mapping->getOutputRect(Core::EQUIRECTANGULAR).left(),
                                                   (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).top()),
                                         potDevOut.borrow(), make_int2((int)pano.getWidth(), (int)pano.getHeight()),
                                         inputsMapOriginalBuffer.borrow_const(), stream));

#ifdef MERGERMASK_SETUP_DISTORTIONMAP
    {
      std::stringstream ss;
      ss.str("");
      ss << "distortion-ori" << i << ".png";
      Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), potDevOut.borrow_const(),
                                                       mapping->getOutputRect().getWidth(),
                                                       mapping->getOutputRect().getHeight());
    }
#endif

    // Down-sampling the image to the appropriate level
    auto originalCacheBuffer = GPU::uniqueBuffer<unsigned char>(devWidth * devHeight, "Merger Mask Algorithm");
    FAIL_RETURN(originalCacheBuffer.status());
    FAIL_RETURN(GPU::memcpyBlocking<unsigned char>(originalCacheBuffer.borrow(), potDevOut.borrow_const(),
                                                   devWidth * devHeight * sizeof(unsigned char)));
    FAIL_RETURN(Util::ImageProcessingGPU::downSampleImages<unsigned char>(downsamplingLevelCount, devWidth, devHeight,
                                                                          potDevOut.borrow(), stream, false));
    // Put the down-sampled image into the cached map
    auto cacheBuffer = GPU::uniqueBuffer<unsigned char>(devWidth * devHeight, "Merger Mask Algorithm");
    FAIL_RETURN(cacheBuffer.status());
    FAIL_RETURN(GPU::memcpyBlocking<unsigned char>(cacheBuffer.borrow(), potDevOut.borrow().as_const(),
                                                   devWidth * devHeight * sizeof(unsigned char)));
    FAIL_RETURN(transformDistortion(make_int2(devWidth, devHeight), cacheBuffer.borrow(), stream));
#ifdef MERGERMASK_SETUP_DISTORTIONMAP
    {
      std::stringstream ss;
      ss.str("");
      ss << "distortion-" << i << ".png";
      Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), cacheBuffer.borrow().as_const(), devWidth,
                                                       devHeight);
    }
#endif
    distortionMaps[transform.first] = cacheBuffer.releaseOwnership();
    originalDistortionMaps[transform.first] = originalCacheBuffer.releaseOwnership();
  }

  return Status::OK();
}

Status MergerMask::setupMappedRect() {
  // Precompute data for the low res
  cachedMappedRects.clear();
  std::vector<int2> rectOffset;
  std::vector<int2> rectSize;
  std::vector<uint32_t> frameOffsets;
  int offset = 0;

  // Precompute data for the original res
  cachedOriginalMappedRects.clear();
  std::vector<int2> originalRectOffset;
  std::vector<int2> originalRectSize;
  std::vector<uint32_t> originalFrameOffsets;
  int originalOffset = 0;

  for (auto transform : transforms) {
    Core::ImageMapping* mapping = imageMappings[transform.first];
    // Data for the original res
    cachedOriginalMappedRects.push_back(mapping->getOutputRect(Core::EQUIRECTANGULAR));
    originalRectOffset.push_back(make_int2((int)mapping->getOutputRect(Core::EQUIRECTANGULAR).left(),
                                           (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).top()));
    originalRectSize.push_back(make_int2((int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getWidth(),
                                         (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getHeight()));
    originalFrameOffsets.push_back(originalOffset);
    originalOffset += (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getArea();

    // Data for the low res
    int devWidth = (int)getDownSize((int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getWidth());
    int devHeight = (int)getDownSize((int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getHeight());
    int devLeft = (int)getDownCoord((int)mapping->getOutputRect(Core::EQUIRECTANGULAR).left());
    int devTop = (int)getDownCoord((int)mapping->getOutputRect(Core::EQUIRECTANGULAR).top());
    cachedMappedRects.push_back(
        Core::Rect::fromInclusiveTopLeftBottomRight(devTop, devLeft, devTop + devHeight - 1, devLeft + devWidth - 1));
    rectOffset.push_back(make_int2(devLeft, devTop));
    rectSize.push_back(make_int2(devWidth, devHeight));
    frameOffsets.push_back(offset);
    offset += devWidth * devHeight;
  }
  // For the original images
  FAIL_RETURN(originalMappedRectOffset.alloc(pano.numInputs(), "Merger Mask"));
  FAIL_RETURN(originalMappedRectSize.alloc(pano.numInputs(), "Merger Mask"));
  FAIL_RETURN(originalMappedOffset.alloc(pano.numInputs(), "Merger Mask"));

  FAIL_RETURN(
      GPU::memcpyBlocking(originalMappedRectOffset.borrow(), &originalRectOffset[0], transforms.size() * sizeof(int2)));
  FAIL_RETURN(
      GPU::memcpyBlocking(originalMappedRectSize.borrow(), &originalRectSize[0], transforms.size() * sizeof(int2)));
  FAIL_RETURN(GPU::memcpyBlocking(originalMappedOffset.borrow(), &originalFrameOffsets[0],
                                  transforms.size() * sizeof(uint32_t)));

  // For the resize caches
  FAIL_RETURN(mappedRectOffset.alloc(pano.numInputs(), "Merger Mask"));
  FAIL_RETURN(mappedRectSize.alloc(pano.numInputs(), "Merger Mask"));
  FAIL_RETURN(mappedOffset.alloc(pano.numInputs(), "Merger Mask"));

  FAIL_RETURN(GPU::memcpyBlocking(mappedRectOffset.borrow(), &rectOffset[0], transforms.size() * sizeof(int2)));
  FAIL_RETURN(GPU::memcpyBlocking(mappedRectSize.borrow(), &rectSize[0], transforms.size() * sizeof(int2)));
  FAIL_RETURN(GPU::memcpyBlocking(mappedOffset.borrow(), &frameOffsets[0], transforms.size() * sizeof(uint32_t)));

#ifdef MERGERMASK_SETUP_MAPPEDRECT
  std::vector<int2> debugRectOffset(transforms.size());
  std::vector<int2> debugRectSize(transforms.size());
  std::vector<uint32_t> debugFrameOffsets(transforms.size());
  cudaMemcpy(&debugRectOffset[0], mappedRectOffset.borrow_const().get(), transforms.size() * sizeof(int2),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&debugRectSize[0], mappedRectSize.borrow_const().get(), transforms.size() * sizeof(int2),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&debugFrameOffsets[0], mappedOffset.borrow_const().get(), transforms.size() * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
#endif
  return Status::OK();
}

Status MergerMask::setup(const std::map<readerid_t, Input::VideoReader*>& readers) {
  FAIL_RETURN(setupMappings(readers));

  FAIL_RETURN(setupTransform(readers));

  FAIL_RETURN(setupMappedRect());

  FAIL_RETURN(setupDistortionMap());

  FAIL_RETURN(setupOverlappingPair());

  FAIL_RETURN(setupFrames());

  FAIL_RETURN(progress.add("coarseMask_setup", "Setup blending mask"));
  return Status::OK();
}

Status MergerMask::retrieveImages(const std::vector<unsigned int>& frames, FrameBuffers& frameBuffers,
                                  const Core::PanoDefinition& pano) {
  frameBuffers.clear();
  auto container = Core::ControllerInputFrames<PixelFormat::RGBA, uint32_t>::create(&pano);
  FAIL_RETURN(container.status());

  for (auto& numFrame : frames) {
    FAIL_RETURN(container->seek((int)numFrame));

    std::map<readerid_t, PotentialValue<GPU::HostBuffer<uint32_t>>> loadedFrames;
    container->load(loadedFrames);

    for (auto& loadedFrame : loadedFrames) {
      readerid_t inputid = loadedFrame.first;

      if (inputid > (int)pano.numInputs()) {
        continue;
      }
      auto potLoadedFrame = loadedFrame.second;
      FAIL_RETURN(potLoadedFrame.status());

      GPU::HostBuffer<uint32_t> frame = potLoadedFrame.value();

      /* Get the size of the current image */
      const Core::InputDefinition& idef = pano.getInput(inputid);
      const int width = (int)idef.getWidth();
      const int height = (int)idef.getHeight();

      const size_t frameSize = (size_t)(width * height * sizeof(uint32_t));
      std::pair<size_t, size_t> id = std::make_pair(inputid, numFrame);
      auto frameBuffer = GPU::uniqueBuffer<uint32_t>(width * height, "Merger Mask Algorithm");
      FAIL_RETURN(frameBuffer.status());
      FAIL_RETURN(GPU::memcpyBlocking(frameBuffer.borrow(), frame.hostPtr(), frameSize));
      frameBuffers[id] = frameBuffer.releaseValue();
    }
  }
  return Status::OK();
}

Status MergerMask::setupFrames() {
  const std::vector<unsigned int> frames = mergerMaskConfig.getFrames();
  FrameBuffers frameBuffers;
  FAIL_RETURN(retrieveImages(mergerMaskConfig.getFrames(), frameBuffers, pano));

#ifdef MERGERMASK_SETUP_FRAMES
  for (FrameBuffers::iterator it = frameBuffers.begin(); it != frameBuffers.end(); ++it) {
    const int frameid = it->first.first;
    const int camid = it->first.second;
    const Core::InputDefinition& idef = pano.getInput(camid);
    const int width = (int)idef.getWidth();
    const int height = (int)idef.getHeight();
    std::stringstream ss;
    ss << "inputImage-" << frameid << " " << camid << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), (it->second).borrow_const(), width, height);
  }
#endif

  // Find the total number of element needed for a certain time frame
  int totalElementCount = 0;
  for (size_t i = 0; i < cachedMappedRects.size(); i++) {
    totalElementCount += (int)cachedMappedRects[i].getArea();
  }

  const int width = getDownSize((int)pano.getWidth());
  const int height = getDownSize((int)pano.getHeight());
  FAIL_RETURN(work1.alloc(pano.getWidth() * pano.getHeight(), "Merger Mask"));
  FAIL_RETURN(work2.alloc(pano.getWidth() * pano.getHeight(), "Merger Mask"));
  FAIL_RETURN(workMask.alloc(pano.getWidth() * pano.getHeight(), "Merger Mask"));
  FAIL_RETURN(workCost.alloc(width * height, "Merger Mask"));

  // Allocate temporal memory for processing the mapped buffer
  auto potDevOut = GPU::uniqueBuffer<uint32_t>(pano.getWidth() * pano.getHeight(), "Merger Mask Algorithm");
  FAIL_RETURN(potDevOut.status());

  cachedMappedFrames.clear();
  const int camCount = (int)pano.numInputs();
  GPU::Stream stream = GPU::Stream::getDefault();
  // For all frameid and camid, do the mapping
  for (size_t j = 0; j < frames.size(); j++) {
    auto cacheBuffer = GPU::uniqueBuffer<uint32_t>(totalElementCount, "Merger Mask");
    FAIL_RETURN(cacheBuffer.status());
    cachedMappedFrames.push_back(cacheBuffer.releaseValue());
    int elementOffset = 0;
    for (int i = 0; i < camCount; i++) {
      Core::Transform* transform = transforms[i];
      Core::ImageMapping* mapping = imageMappings[i];
      const Core::InputDefinition& im = pano.getInput(i);
      const std::pair<size_t, size_t> inPair = std::make_pair(i, frames[j]);
      FrameBuffers::const_iterator it = frameBuffers.find(inPair);
      if (it != frameBuffers.end()) {
        const GPU::Buffer<const uint32_t> inputBuffer = it->second.borrow_const();
        auto surface = Core::OffscreenAllocator::createSourceSurface(im.getWidth(), im.getHeight(), "Merger Mask");
        FAIL_RETURN(GPU::memcpyAsync(*surface->pimpl->surface, inputBuffer, stream));
        // Map the input to the output space
        transform->mapBuffer(frames[j], potDevOut.borrow(), *surface->pimpl->surface, nullptr,
                             mapping->getOutputRect(Core::EQUIRECTANGULAR), pano, im, *surface->pimpl->surface, stream);

        // Down-sampling the image to the appropriate level
        int devWidth = (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getWidth();
        int devHeight = (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getHeight();

#ifdef MERGERMASK_SETUP_FRAMES
        {
          std::stringstream ss;
          ss << "oriwarped-" << i << " " << frames[j] << ".png";
          Debug::dumpRGB210DeviceBuffer(ss.str().c_str(), devOut.borrow_const(), devWidth, devHeight);
        }
#endif
        Util::ImageProcessingGPU::downSampleImages(downsamplingLevelCount, devWidth, devHeight, potDevOut.borrow(),
                                                   stream, false);
        // Put the down-sampled image into the cached map
#ifdef MERGERMASK_SETUP_FRAMES
        {
          std::stringstream ss;
          ss << "warped-rgb210-" << i << " " << frames[j] << ".png";
          Debug::dumpRGB210DeviceBuffer(ss.str().c_str(), devOut.borrow_const(), devWidth, devHeight);
        }
#endif
        // Convert from rgb to lab
        GPU::Buffer<uint32_t> subBuffer = cachedMappedFrames.back().borrow().createSubBuffer(elementOffset);
        FAIL_RETURN(Util::ImageProcessingGPU::convertRGB210ToRGBandGradient(
            make_int2(devWidth, devHeight), potDevOut.borrow_const(), subBuffer, stream));
#ifdef MERGERMASK_SETUP_FRAMES
        {
          std::stringstream ss;
          ss << "warped-rgb-" << i << " " << frames[j] << ".png";
          Debug::dumpRGBADeviceBuffer(ss.str().c_str(), devOut.borrow_const(), devWidth, devHeight);
        }
#endif
        FAIL_RETURN(Util::ImageProcessingGPU::convertRGBandGradientToNormalizedLABandGradient(
            make_int2(devWidth, devHeight), subBuffer, stream));

#ifdef MERGERMASK_SETUP_FRAMES
        {
          std::stringstream ss;
          ss << "warped-" << i << " " << frames[j] << ".png";
          Debug::dumpRGB210DeviceBuffer(ss.str().c_str(), subBuffer.as_const(), devWidth, devHeight);
        }
        {
          std::stringstream ss;
          ss << "gradient-" << i << " " << frames[j] << ".png";
          GPU::UniqueBuffer<unsigned char> gradientBuffer;
          FAIL_RETURN(gradientBuffer.alloc(devWidth * devHeight, "Merger Mask"));
          FAIL_RETURN(
              extractChannel(make_int2(devWidth, devHeight), subBuffer.as_const(), 3, gradientBuffer.borrow(), stream));
          Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), gradientBuffer.borrow_const(), devWidth,
                                                           devHeight);
        }
#endif
        elementOffset += (int)cachedMappedRects[i].getArea();
      }
    }
  }
  return Status::OK();
}

Status MergerMask::setupOverlappingPair() {
  // Prepare overlapping pair
  GPU::Stream stream = GPU::Stream::getDefault();
  const GPU::Buffer<const uint32_t> mappingMask = inputsMapBuffer.borrow_const();
  // Copy data from GPU to CPU
  PotentialValue<GPU::HostBuffer<uint32_t>> potHostBuffer =
      GPU::HostBuffer<uint32_t>::allocate(mappingMask.numElements(), "Mask Merger Graph");
  FAIL_RETURN(potHostBuffer.status());
  GPU::HostBuffer<uint32_t> mask = potHostBuffer.value();
  FAIL_RETURN(GPU::memcpyBlocking(mask, mappingMask));

  // Find the overlapping bit set
  std::unordered_set<uint32_t> overlappingValueSet;
  const uint32_t* maskPtr = mask.hostPtr();
  for (size_t i = 0; i < mappingMask.numElements(); i++) {
    if (maskPtr[i] > 0) {
      if (overlappingValueSet.find(maskPtr[i]) == overlappingValueSet.end()) {
        overlappingValueSet.insert(maskPtr[i]);
      }
    }
  }

  // Find all pair mapping
  for (readerid_t i = 0; i < pano.numInputs(); i++) {
    isOverlapping.push_back(std::vector<int>(pano.numInputs(), 0));
  }
  for (auto i = overlappingValueSet.begin(); i != overlappingValueSet.end(); i++) {
    uint32_t v = *i;
    std::vector<int> indices;
    // Find the set of indices
    int t = 0;
    while (v > 0) {
      if ((v & 1) > 0) {
        indices.push_back(t);
      }
      t++;
      v = v >> 1;
    }
    for (size_t i = 0; i < indices.size(); i++)
      for (size_t j = i + 1; j < indices.size(); j++) {
        isOverlapping[indices[i]][indices[j]]++;
        isOverlapping[indices[j]][indices[i]]++;
      }
  }
  mask.release();
  return Status::OK();
}

// Step (3c)
Status MergerMask::getBlendingCost(const std::vector<int>& normalizedMaxOverlappingWidths, const int2& size,
                                   const std::vector<size_t>& allCam,
                                   const GPU::Buffer<const unsigned char>& inputDistortionBuffer,
                                   const GPU::Buffer<const uint32_t>& inputNonOverlappingIndexBuffer,
                                   GPU::Buffer<unsigned char> nextDistortionBuffer,
                                   GPU::Buffer<uint32_t> nextNonOverlappingIndexBuffer,
                                   GPU::Buffer<uint32_t> nextInputIndexBuffer, float& blendingCost) {
  GPU::Stream stream = GPU::Stream::getDefault();
  const std::vector<unsigned int> frames = mergerMaskConfig.getFrames();

  FAIL_RETURN(updateInputIndexByDistortionMap((int)allCam.back(), size, inputNonOverlappingIndexBuffer,
                                              inputDistortionBuffer, nextNonOverlappingIndexBuffer,
                                              nextDistortionBuffer, stream));

#ifdef MERGERMASK_GET_MERGERMASKS
  {
    std::stringstream ss;
    ss.str("");
    ss << "trial-distortion-";
    for (int i = 0; i < allCam.size(); i++) {
      ss << allCam[i] << "-";
    }
    ss << ".png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), nextDistortionBuffer.as_const(), size.x, size.y);
  }
#endif
  // To make sure the overlapping area is small (for efficiency), reduce the new area's mask to less than a threshold
  // - (C3) Make sure that the overlapping width of image pairs is less than a certain threshold

  FAIL_RETURN(updateOverlappingMap(normalizedMaxOverlappingWidths, size, allCam, nextNonOverlappingIndexBuffer,
                                   nextInputIndexBuffer, stream));

  // Thirdly, compute the image difference cost metric
  // Iterate over all frames
  const int kernelSize = mergerMaskConfig.getKernelSize();

  // First, initialize the distortion cost
  FAIL_RETURN(GPU::memsetToZeroBlocking(workCost.borrow(), sizeof(float) * size.x * size.y));
  // Now iterate over all frames and start accumulating the cost into buffers

  GPU::UniqueBuffer<uint32_t> debugBuffer0, debugBuffer1;
  FAIL_RETURN(debugBuffer0.alloc(size.x * size.y, "Merger Mask"));
  FAIL_RETURN(debugBuffer1.alloc(size.x * size.y, "Merger Mask"));

  for (size_t i = 0; i < frames.size(); i++) {
    FAIL_RETURN(updateStitchingCost(size, kernelSize, nextInputIndexBuffer.as_const(), mappedOffset.borrow_const(),
                                    mappedRectOffset.borrow_const(), mappedRectSize.borrow_const(),
                                    cachedMappedFrames[i].borrow_const(), workCost.borrow(), debugBuffer0.borrow(),
                                    debugBuffer1.borrow(), stream));
  }
#ifdef MERGERMASK_GET_MERGERMASKS
  {
    std::stringstream ss;
    /*ss.str("");
    ss << "cost-map-";
    for (int i = 0; i < prevCam.size(); i++) {
    ss << prevCam[i] << "-";
    }
    ss << camId << ".png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linearFloat>(ss.str().c_str(), workCost.borrow_const(), size.x, size.y);
    */
    ss.str("");
    ss << "cost-debug-map-zero-";
    for (int i = 0; i < allCam.size(); i++) {
      ss << allCam[i] << "-";
    }
    ss << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), debugBuffer0.borrow(), size.x, size.y);

    ss.str("");
    ss << "cost-debug-map-one-";
    for (int i = 0; i < allCam.size(); i++) {
      ss << allCam[i] << "-";
    }
    ss << ".png";
    Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), debugBuffer1.borrow(), size.x, size.y);

    ss.str("");
    ss << "cost-overlapping-mask-final-";
    for (int i = 0; i < allCam.size(); i++) {
      ss << allCam[i] << "-";
    }
    ss << ".png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), nextInputIndexBuffer.as_const(), size.x, size.y);
  }
#endif
  // Calculate cost now
  float output, mask;
  FAIL_RETURN(
      Util::ImageProcessingGPU::calculateSum(size.x * size.y, workCost.borrow_const(), 256, stream, output, mask));

  float cost = mask > 0 ? output / mask : 1;
  float maskRatio = mask / float(size.x * size.y);
  float preferBiggerMaskCost = expf(-maskRatio * maskRatio * 100);
  blendingCost = cost + 0.05f * preferBiggerMaskCost;
  return Status::OK();
}

Status MergerMask::updateOverlappingMap(const std::vector<int>& normalizedMaxOverlappingWidths, const int2& size,
                                        const std::vector<size_t>& allCam,
                                        const GPU::Buffer<const uint32_t>& inputNonOverlappingIndexBuffer,
                                        GPU::Buffer<uint32_t> inputIndexBuffer, GPU::Stream stream,
                                        const bool original) {
  assert(normalizedMaxOverlappingWidths.size() == allCam.size());
  assert(allCam.size() > 1);

  // Get all frame index as max value
  size_t maxCam = allCam[0];
  for (size_t i = 1; i < allCam.size(); i++) {
    if (allCam[i] > maxCam) {
      maxCam = allCam[i];
    }
  }
  std::vector<char> camIndex(maxCam + 1, 0);
  for (size_t i = 0; i < allCam.size(); i++) {
    camIndex[(int)allCam[i]] = (char)i;
  }

  GPU::UniqueBuffer<char> camIndexBuffer;
  FAIL_RETURN(camIndexBuffer.alloc(maxCam, "Merger Mask"));
  FAIL_RETURN(GPU::memcpyBlocking(camIndexBuffer.borrow(), &camIndex[0]));

  // Initialize the index buffer from the non-overlapping
  FAIL_RETURN(
      GPU::memcpyBlocking(inputIndexBuffer, inputNonOverlappingIndexBuffer, size.x * size.y * sizeof(uint32_t)));

  // Set up a mask for all camera previous to the current one to calculate the distance transform function
  for (size_t i = 0; i < allCam.size(); i++) {
    if (normalizedMaxOverlappingWidths[i] > 0) {
      // Compute the distance map for the current camera, make sure that it will have overlapping
      // area that is larger than a predefined threshold
      const int upSize = int(float(size.x) * float(normalizedMaxOverlappingWidths[i]) / float(getDownSize(size.x)));
      const int maxOverlappingWidth = original ? upSize : normalizedMaxOverlappingWidths[i];
      FAIL_RETURN(Core::computeEuclideanDistanceMap(workMask.borrow(), inputIndexBuffer, work1.borrow(), work2.borrow(),
                                                    size.x, size.y, 1 << allCam[i], true, (float)maxOverlappingWidth, 1,
                                                    stream));
    } else {
      FAIL_RETURN(GPU::memsetToZeroBlocking(workMask.borrow(), size.x * size.y));
    }
#ifdef MERGERMASK_GET_MERGERMASKS_FINAL
    if (original) {
      std::stringstream ss;
      ss.str("");
      ss << "work-mask-";
      for (int j = 0; j <= i; j++) {
        ss << allCam[j] << "-";
      }
      ss << ".png";
      Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), workMask.borrow_const(), size.x, size.y);
    }
#endif
    const int2 camSize = original ? make_int2((int)cachedOriginalMappedRects[allCam[i]].getWidth(),
                                              (int)cachedOriginalMappedRects[allCam[i]].getHeight())
                                  : make_int2((int)cachedMappedRects[allCam[i]].getWidth(),
                                              (int)cachedMappedRects[allCam[i]].getHeight());
    const int2 camOffset =
        original ? make_int2((int)cachedOriginalMappedRects[allCam[i]].left(),
                             (int)cachedOriginalMappedRects[allCam[i]].top())
                 : make_int2((int)cachedMappedRects[allCam[i]].left(), (int)cachedMappedRects[allCam[i]].top());

    FAIL_RETURN(
        updateIndexMask((int)allCam[i], original ? 10000 : 2, camIndexBuffer.borrow_const(), camSize, camOffset,
                        original ? originalDistortionMaps[(int)allCam[(int)i]] : distortionMaps[(int)allCam[(int)i]],
                        size, inputIndexBuffer, workMask.borrow(),
                        original ? inputsMapOriginalBuffer.borrow_const() : inputsMapBuffer.borrow_const(), stream));

#ifdef MERGERMASK_GET_MERGERMASKS_FINAL
    if (original) {
      {
        std::stringstream ss;
        ss.str("");
        ss << "index-buffer-final-";
        for (int j = 0; j <= i; j++) {
          ss << allCam[j] << "-";
        }
        ss << ".png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), inputIndexBuffer.as_const(), size.x, size.y);
      }
      /*{
        std::stringstream ss;
        ss.str("");
        ss << "distortion-buffer-final-";
        for (int j = 0; j <= i; j++) {
        ss << allCam[j] << "-";
        }
        ss << ".png";
        Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), originalDistortionMaps[allCam[i]].as_const(),
        size.x, size.y);
        }*/
    }
#endif
  }
  return Status::OK();
}

Status MergerMask::releaseMergerMaskMemory() {
  for (size_t i = 0; i < distortionMaps.size(); i++) {
    distortionMaps[(int)i].release();
  }
  for (size_t i = 0; i < originalDistortionMaps.size(); i++) {
    originalDistortionMaps[(int)i].release();
  }
  distortionMaps.clear();
  originalDistortionMaps.clear();
  workCost.releaseOwnership().release();
  inputsMapBuffer.releaseOwnership().release();
  inputsMapOriginalBuffer.releaseOwnership().release();
  cachedMappedFrames.clear();
  return Status::OK();
}

Status MergerMask::getMergerMasks(GPU::Buffer<uint32_t> inputIndexPixelBuffer, std::vector<size_t>& masksOrder) {
  const bool useBlendingOrder = mergerMaskConfig.useBlendingOrder();
  const int width = (int)getDownSize((int)pano.getWidth());
  const int height = (int)getDownSize((int)pano.getHeight());
  const int camCount = (int)pano.numInputs();

  GPU::UniqueBuffer<unsigned char> inputDistortionBuffer;
  FAIL_RETURN(inputDistortionBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<unsigned char> nextDistortionBuffer;
  FAIL_RETURN(nextDistortionBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<unsigned char> bestDistortionBuffer;
  FAIL_RETURN(bestDistortionBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<uint32_t> inputNonOverlappingIndexBuffer;
  FAIL_RETURN(inputNonOverlappingIndexBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<uint32_t> nextNonOverlappingIndexBuffer;
  FAIL_RETURN(nextNonOverlappingIndexBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<uint32_t> bestNonOverlappingIndexBuffer;
  FAIL_RETURN(bestNonOverlappingIndexBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<uint32_t> inputIndexBuffer;
  FAIL_RETURN(inputIndexBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<uint32_t> nextInputIndexBuffer;
  FAIL_RETURN(nextInputIndexBuffer.alloc(width * height, "Merger Mask"));

  GPU::Stream stream = GPU::Stream::getDefault();
  std::vector<size_t> bestCamOrder;
  float bestFinalCost = -1;
  std::vector<int> bestWidths;
  std::vector<int> normalizedMaxOverlappingWidths;
  if (mergerMaskConfig.getMaxOverlappingWidth() < -1) {
    normalizedMaxOverlappingWidths.push_back(-1);
    for (int i = 100; i <= 200; i += 50) {
      normalizedMaxOverlappingWidths.push_back(i);
    }
  } else {
    normalizedMaxOverlappingWidths.push_back(mergerMaskConfig.getMaxOverlappingWidth());
  }

  // Try it heuristically
  // For the first image, just try all of them
  for (int i0 = 0; i0 < camCount; i0++) {
    // Do put all image into frame
    std::vector<bool> processedCam(camCount, false);
    std::vector<size_t> camOrder;
    std::vector<int> widths;
    widths.push_back(-1);
    float finalCost = 0;
    processedCam[i0] = true;
    camOrder.push_back(i0);

    // Initialize the first map
    FAIL_RETURN(initializeMasks(make_int2(width, height), i0, inputNonOverlappingIndexBuffer.borrow(),
                                inputDistortionBuffer.borrow(), stream));

#ifdef MERGERMASK_GET_MERGERMASKS
    {
      std::stringstream ss;
      ss.str("");
      ss << "trial-distortion-" << i0 << ".png";
      Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), inputDistortionBuffer.borrow().as_const(),
                                                       width, height);
    }
#endif
    for (int t = 1; t < camCount; t++) {
      FAIL_RETURN(progress.add("coarseMask_lowRes", "Optimizing for the coarse masks"));

      // Find camera at position [t]
      float bestCost = 0;
      int bestNextCam = -1;
      std::vector<int> bestNextWidths;

      // Check whether there is at least one overlapping
      bool atLeastOneOverlapping = false;
      for (size_t i = 0; i < camOrder.size(); i++) {
        for (int j = 0; j < camCount; j++)
          if (!processedCam[j] && isOverlapping[j][camOrder[i]]) {
            atLeastOneOverlapping = true;
            break;
          }
      }

      for (int iNext = 0; iNext < camCount; iNext++)
        if (!processedCam[iNext]) {
          if (!useBlendingOrder) {
            iNext = t;
            atLeastOneOverlapping = false;
          }
          if (atLeastOneOverlapping) {
            bool isOverlapped = false;
            for (size_t i = 0; i < camOrder.size(); i++) {
              if (isOverlapping[iNext][camOrder[i]]) {
                isOverlapped = true;
                break;
              }
            }
            if (!isOverlapped) {
              continue;
            }
          }

          for (size_t k = 0; k < normalizedMaxOverlappingWidths.size(); k++) {
            std::vector<int> trialWidths = widths;
            trialWidths.push_back(normalizedMaxOverlappingWidths[k]);
            std::vector<size_t> allCam = camOrder;
            allCam.push_back(iNext);
            // Compute the cost of current configuration
            float cost = 100000000;
            FAIL_RETURN(getBlendingCost(trialWidths, make_int2(width, height), allCam,
                                        inputDistortionBuffer.borrow_const(),
                                        inputNonOverlappingIndexBuffer.borrow_const(), nextDistortionBuffer.borrow(),
                                        nextNonOverlappingIndexBuffer.borrow(), nextInputIndexBuffer.borrow(), cost));
            if (cost < bestCost || bestNextCam == -1) {
              bestCost = cost;
              bestNextCam = iNext;
              bestNextWidths = trialWidths;
              FAIL_RETURN(GPU::memcpyBlocking(bestNonOverlappingIndexBuffer.borrow(),
                                              nextNonOverlappingIndexBuffer.borrow_const(),
                                              width * height * sizeof(uint32_t)));
              FAIL_RETURN(GPU::memcpyBlocking(bestDistortionBuffer.borrow(), nextDistortionBuffer.borrow_const(),
                                              width * height * sizeof(unsigned char)));
            }
          }
          if (!useBlendingOrder) {
            break;
          }
        }

      // Perform the greedy step here
      // From all the trial in this step, pick the one that have the smallest cost and use it for the next iteration
      if (bestNextCam >= 0) {
        camOrder.push_back(bestNextCam);
        widths = bestNextWidths;
        finalCost = bestCost;
        processedCam[bestNextCam] = true;
        FAIL_RETURN(GPU::memcpyBlocking(inputNonOverlappingIndexBuffer.borrow(),
                                        bestNonOverlappingIndexBuffer.borrow_const(),
                                        width * height * sizeof(uint32_t)));
        FAIL_RETURN(GPU::memcpyBlocking(inputDistortionBuffer.borrow(), bestDistortionBuffer.borrow_const(),
                                        width * height * sizeof(unsigned char)));
      } else {
        finalCost = -1;
        break;
      }
    }
    // Find the best results from all initial camera
    if (camOrder.size() == (size_t)camCount) {
      if ((finalCost > 0) && (finalCost < bestFinalCost || bestFinalCost < 0)) {
        bestFinalCost = finalCost;
        bestCamOrder = camOrder;
        bestWidths = widths;
      }
    }
    if (!useBlendingOrder) {
      break;
    }
  }

  // Found the best order, now re-run the process in full resolution to generate the final map
  masksOrder = bestCamOrder;
  FAIL_RETURN(getOriginalMaskFromOrder(bestWidths, masksOrder, inputIndexPixelBuffer));
  if (mergerMaskConfig.useSeam()) {
    // Release all that are not needed
    FAIL_RETURN(releaseMergerMaskMemory());
    FAIL_RETURN(performSeamOptimization(masksOrder, inputIndexPixelBuffer));
  }
  return Status::OK();
}

Status MergerMask::getOriginalMaskFromOrder(const std::vector<int>& normalizedOverlappingWidths,
                                            const std::vector<size_t>& maskOrder,
                                            GPU::Buffer<uint32_t> inputIndexPixelBuffer) {
  const int width = (int)pano.getWidth();
  const int height = (int)pano.getHeight();
  GPU::Stream stream = GPU::Stream::getDefault();

  GPU::UniqueBuffer<unsigned char> inputDistortionBuffer;
  FAIL_RETURN(inputDistortionBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<unsigned char> nextDistortionBuffer;
  FAIL_RETURN(nextDistortionBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<uint32_t> inputNonOverlappingIndexBuffer;
  FAIL_RETURN(inputNonOverlappingIndexBuffer.alloc(width * height, "Merger Mask"));

  GPU::UniqueBuffer<uint32_t> nextNonOverlappingIndexBuffer;
  FAIL_RETURN(nextNonOverlappingIndexBuffer.alloc(width * height, "Merger Mask"));

  const int2 size = make_int2(width, height);

  FAIL_RETURN(initializeMasks(size, (int)maskOrder[0], inputNonOverlappingIndexBuffer.borrow(),
                              inputDistortionBuffer.borrow(), stream, true));

  for (size_t i = 1; i < maskOrder.size(); i++) {
    FAIL_RETURN(updateInputIndexByDistortionMap(
        (int)maskOrder[i], size, inputNonOverlappingIndexBuffer.borrow_const(), inputDistortionBuffer.borrow_const(),
        nextNonOverlappingIndexBuffer.borrow(), nextDistortionBuffer.borrow(), stream, true));
    FAIL_RETURN(GPU::memcpyBlocking(inputNonOverlappingIndexBuffer.borrow(),
                                    nextNonOverlappingIndexBuffer.borrow_const(), width * height * sizeof(uint32_t)));
    FAIL_RETURN(GPU::memcpyBlocking(inputDistortionBuffer.borrow(), nextDistortionBuffer.borrow_const(),
                                    width * height * sizeof(unsigned char)));
  }

#ifdef MERGERMASK_GET_MERGERMASKS_FULL
  {
    std::stringstream ss;
    ss.str("");
    ss << "final-non-overlapping-result-";
    for (int i = 0; i < normalizedOverlappingWidths.size(); i++) ss << normalizedOverlappingWidths[i] << " ";
    ss << "cam";
    for (int i = 0; i < maskOrder.size(); i++) {
      ss << maskOrder[i] << "-";
    }
    ss << ".png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), inputNonOverlappingIndexBuffer.borrow_const(), size.x, size.y);
  }

  {
    std::stringstream ss;
    ss.str("");
    ss << "final-distortion-map-";
    for (int i = 0; i < normalizedOverlappingWidths.size(); i++) ss << normalizedOverlappingWidths[i] << " ";
    ss << "cam";
    for (int i = 0; i < maskOrder.size(); i++) {
      ss << maskOrder[i] << "-";
    }
    ss << ".png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), inputDistortionBuffer.borrow_const(), size.x,
                                                     size.y);
  }
#endif

  FAIL_RETURN(updateOverlappingMap(normalizedOverlappingWidths, size, maskOrder,
                                   nextNonOverlappingIndexBuffer.borrow_const(), inputIndexPixelBuffer, stream, true));

#ifdef MERGERMASK_GET_MERGERMASKS_FULL
  {
    std::stringstream ss;
    ss.str("");
    ss << "final-full-result-";
    for (int i = 0; i < normalizedOverlappingWidths.size(); i++) ss << normalizedOverlappingWidths[i] << " ";
    ss << "cam";
    for (int i = 0; i < maskOrder.size(); i++) {
      ss << maskOrder[i] << "-";
    }
    ss << ".png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), inputIndexPixelBuffer.as_const(), size.x, size.y);
  }

  {
    std::stringstream ss;
    ss.str("");
    ss << "input-full-index.png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), inputsMapOriginalBuffer.borrow_const(), size.x, size.y);
  }
#endif
  FAIL_RETURN(progress.add("coarseMask_fulRes", "Coarse masks done"));
  return Status::OK();
}

Status MergerMask::performSeamOptimization(const std::vector<size_t>& maskOrder,
                                           GPU::Buffer<uint32_t> inputIndexBuffer) {
  // Copy input index buffer to a temporal buffer
  GPU::UniqueBuffer<uint32_t> tmpInputIndexBuffer;
  FAIL_RETURN(tmpInputIndexBuffer.alloc(inputIndexBuffer.numElements(), "Merger Mask"));
  FAIL_RETURN(GPU::memcpyBlocking(tmpInputIndexBuffer.borrow(), inputIndexBuffer));

  const int width = (int)pano.getWidth();
  const int height = (int)pano.getHeight();
  const int2 size = make_int2(width, height);

#ifdef MERGERMASK_SEAM_MASK
  {
    std::stringstream ss;
    ss.str("");
    ss << "before-seam-mask"
       << ".png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), inputIndexBuffer.as_const(), size.x, size.y);
  }
#endif

  GPU::UniqueBuffer<uint32_t> indexBuffer0;
  GPU::UniqueBuffer<uint32_t> indexBuffer1;
  FAIL_RETURN(indexBuffer0.alloc(inputIndexBuffer.numElements(), "Merger Mask"));
  FAIL_RETURN(indexBuffer1.alloc(inputIndexBuffer.numElements(), "Merger Mask"));
  GPU::Stream stream = GPU::Stream::getDefault();

  std::vector<unsigned int> frames = {mergerMaskConfig.getFrames()[0]};
  // Retrieve the first image from the selected set of frames
  FrameBuffers frameBuffers;
  FAIL_MSG(retrieveImages(frames, frameBuffers, pano),
           "Run out of memory while trying to read input frames for seam optimization");
  int totalSize = 0;
  for (auto mapping : imageMappings) {
    totalSize += (int)mapping.second->getOutputRect(Core::EQUIRECTANGULAR).getArea();
  }

  auto potMappedFrames = GPU::uniqueBuffer<uint32_t>(totalSize, "Merger Mask");
  FAIL_RETURN(potMappedFrames.status());

  // Transforms image from the input to the output space, store them in the mappedFrames
  int offset = 0;
  for (auto transformIterator : transforms) {
    const videoreaderid_t i = transformIterator.first;
    Core::Transform* transform = transformIterator.second;
    Core::ImageMapping* mapping = imageMappings[i];
    const Core::InputDefinition& im = pano.getInput(i);
    const std::pair<size_t, size_t> inPair = std::make_pair(i, frames[0]);
    FrameBuffers::const_iterator it = frameBuffers.find(inPair);
    if (it != frameBuffers.end()) {
      const GPU::Buffer<const uint32_t> inputBuffer = it->second.borrow_const();
      auto surface = Core::OffscreenAllocator::createSourceSurface(im.getWidth(), im.getHeight(), "Merger Mask");
      FAIL_RETURN(GPU::memcpyAsync(*surface->pimpl->surface, inputBuffer, stream));

      auto potDevOut =
          GPU::uniqueBuffer<uint32_t>(mapping->getOutputRect(Core::EQUIRECTANGULAR).getArea(), "Merger Mask");
      FAIL_RETURN(potDevOut.status());

      // Map the input to the output space
      FAIL_RETURN(transform->mapBuffer(frames[0], potDevOut.borrow(), *surface->pimpl->surface, nullptr,
                                       mapping->getOutputRect(Core::EQUIRECTANGULAR), pano, im,
                                       *surface->pimpl->surface, stream));

      // Down-sampling the image to the appropriate level
      int devWidth = (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getWidth();
      int devHeight = (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getHeight();

#ifdef MERGERMASK_SEAM_INPUT
      {
        std::stringstream ss;
        ss << "oriwarped-" << i << ".png";
        Debug::dumpRGB210DeviceBuffer(ss.str().c_str(), devOut.borrow_const(), devWidth, devHeight);
      }
#endif

      // Convert from rgb to lab
      GPU::Buffer<uint32_t> subBuffer = potMappedFrames.borrow().createSubBuffer(offset);
      FAIL_RETURN(Util::ImageProcessingGPU::convertRGB210ToRGBandGradient(make_int2(devWidth, devHeight),
                                                                          potDevOut.borrow_const(), subBuffer, stream));
#ifdef MERGERMASK_SEAM_INPUT
      {
        std::stringstream ss;
        ss << "warped-rgb-" << i << ".png";
        Debug::dumpRGBADeviceBuffer(ss.str().c_str(), devOut.borrow_const(), devWidth, devHeight);
      }
#endif
      FAIL_RETURN(Util::ImageProcessingGPU::convertRGBandGradientToNormalizedLABandGradient(
          make_int2(devWidth, devHeight), subBuffer, stream));
#ifdef MERGERMASK_SEAM_INPUT
      {
        std::stringstream ss;
        ss << "warped-" << i << ".png";
        Debug::dumpRGB210DeviceBuffer(ss.str().c_str(), subBuffer.as_const(), devWidth, devHeight);
      }
      {
        std::stringstream ss;
        ss << "gradient-" << i << ".png";
        GPU::UniqueBuffer<unsigned char> gradientBuffer;
        FAIL_RETURN(gradientBuffer.alloc(devWidth * devHeight, "Merger Mask"));
        FAIL_RETURN(
            extractChannel(make_int2(devWidth, devHeight), subBuffer.as_const(), 3, gradientBuffer.borrow(), stream));
        Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), gradientBuffer.borrow_const(), devWidth,
                                                         devHeight);
      }
#endif
    }
    offset += (int)mapping->getOutputRect(Core::EQUIRECTANGULAR).getArea();
  }

  FAIL_RETURN(progress.add("seam_setup", "Setup seam computation"));

  GPU::UniqueBuffer<uint32_t> input0Buffer;
  GPU::UniqueBuffer<uint32_t> input1Buffer;

  FAIL_RETURN(input0Buffer.alloc(size.x * size.y, "Merger Mask"));
  FAIL_RETURN(input1Buffer.alloc(size.x * size.y, "Merger Mask"));

  const int seamFeatheringSize = mergerMaskConfig.getSeamFeatheringSize();
  Core::Rect rect = Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, size.y - 1, size.x - 1);
  uint32_t prevIds = 1 << maskOrder[0];
  for (size_t i = 1; i < maskOrder.size(); i++) {
    // Optimize for the seam using the first image only, more will likely cause more harm than good
    const uint32_t currentId = 1 << maskOrder[i];
    std::vector<unsigned char> prevCams;
    for (size_t j = 0; j < i; j++) {
      prevCams.push_back((unsigned char)maskOrder[j]);
    }
    GPU::UniqueBuffer<unsigned char> prevCamsBuffer;
    FAIL_RETURN(prevCamsBuffer.alloc(prevCams.size(), "Merger Mask"));
    FAIL_RETURN(GPU::memcpyBlocking(prevCamsBuffer.borrow(), &prevCams[0]));

    std::vector<unsigned char> currentCam = {(unsigned char)maskOrder[i]};
    GPU::UniqueBuffer<unsigned char> currentCamBuffer;
    FAIL_RETURN(currentCamBuffer.alloc(currentCam.size(), "Merger Mask"));
    FAIL_RETURN(GPU::memcpyBlocking(currentCamBuffer.borrow(), &currentCam[0]));

    FAIL_RETURN(lookupColorBufferFromInputIndex(
        (int)rect.getWidth(), prevCamsBuffer.borrow_const(), originalMappedRectOffset.borrow_const(),
        originalMappedRectSize.borrow_const(), originalMappedOffset.borrow_const(), potMappedFrames.borrow_const(),
        size, inputIndexBuffer, input0Buffer.borrow(), stream));
    FAIL_RETURN(lookupColorBufferFromInputIndex(
        (int)rect.getWidth(), currentCamBuffer.borrow_const(), originalMappedRectOffset.borrow_const(),
        originalMappedRectSize.borrow_const(), originalMappedOffset.borrow_const(), potMappedFrames.borrow_const(),
        size, inputIndexBuffer, input1Buffer.borrow(), stream));

    auto potSeamFinder = SeamFinder::create(2, 1, (int)pano.getWidth(), rect, input0Buffer.borrow_const(), rect,
                                            input1Buffer.borrow_const(), stream, workMask, work1, work2);
    FAIL_MSG(potSeamFinder.status(), "Failed to initialize seam computation");
    std::unique_ptr<SeamFinder> seamFinder(potSeamFinder.release());

    FAIL_MSG(seamFinder->findSeam(), "Failed to find seam");
#ifdef MERGERMASK_SEAM_OPTIMIZATION
    // Convert input buffers into rgb for dumping
    {
      VideoStitch::MergerMask::MergerMask::convertNormalizedLABandGradientToRGBA(size, input0Buffer.borrow(), stream);
      VideoStitch::MergerMask::MergerMask::convertNormalizedLABandGradientToRGBA(size, input1Buffer.borrow(), stream);
    }
    {
      std::stringstream ss;
      ss << "ori-" << i << "-0.png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), input0Buffer.borrow_const(), size.x, size.y);
    }
    {
      std::stringstream ss;
      ss << "ori-" << i << "-1.png";
      work1 Debug::dumpRGBADeviceBuffer(ss.str().c_str(), input1Buffer.borrow_const(), size.x, size.y);
    }
    {
      std::stringstream ss;
      ss << "output-image" << i << "-with-seam.png";
      seamFinder.saveSeamImage(ss.str(), 2);
    }
    {
      std::stringstream ss;
      ss << "output-map-" << i << "-0.png";
      Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), seamFinder.getOutputsMap().as_const(), size.x, size.y);
    }
#endif
    // If there is no overlapping, just continue the process
    if (seamFinder->seamFound()) {
      FAIL_RETURN(
          updateIndexMaskAfterSeam(prevIds, currentId, size, seamFinder->getOutputsMap(), inputIndexBuffer, stream));

#ifdef MERGERMASK_SEAM_OPTIMIZATION
      {
        {
          std::stringstream ss;
          ss << "after-seam-" << i << "-index-map.png";
          Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), inputIndexBuffer.as_const(), size.x, size.y);
        }
      }
#endif
    }
    prevIds += currentId;
    FAIL_RETURN(progress.add("seam_fulRes", "Perform seam optimization"));
  }

#ifdef MERGERMASK_SEAM_MASK
  {
    std::stringstream ss;
    ss.str("");
    ss << "final-seam-mask-no-feathering"
       << ".png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), inputIndexBuffer.as_const(), size.x, size.y);
  }
#endif
  if (seamFeatheringSize > 0) {
    // Now perform feathering around borders
    GPU::UniqueBuffer<uint32_t> seamWithFeatheringBuffer;
    FAIL_RETURN(seamWithFeatheringBuffer.alloc(size.x * size.y, "Merger Mask"));
    FAIL_RETURN(GPU::memcpyBlocking(seamWithFeatheringBuffer.borrow(), inputIndexBuffer));

    for (size_t i = 0; i < maskOrder.size(); i++) {
      // Compute the distance map for the current camera, make sure that it will have overlapping
      // area that is larger than a predefined threshold
      FAIL_RETURN(Core::computeEuclideanDistanceMap(workMask.borrow(), inputIndexBuffer, work1.borrow(), work2.borrow(),
                                                    size.x, size.y, 1 << maskOrder[i], true, (float)seamFeatheringSize,
                                                    1, stream));

#ifdef MERGERMASK_SEAM_OPTIMIZATION
      {
        std::stringstream ss;
        ss.str("");
        ss << "work-mask-" << i << ".png";
        Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), workMask.borrow_const(), size.x, size.y);
      }
#endif

      FAIL_RETURN(updateSeamMask((int)maskOrder[i], size, tmpInputIndexBuffer.borrow_const(), workMask.borrow_const(),
                                 seamWithFeatheringBuffer.borrow(), stream));

#ifdef MERGERMASK_SEAM_OPTIMIZATION
      {
        std::stringstream ss;
        ss.str("");
        ss << "index-buffer-final-" << i << ".png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), seamWithFeatheringBuffer.borrow_const(), size.x, size.y);
      }
#endif
    }

    FAIL_RETURN(GPU::memcpyBlocking(inputIndexBuffer, seamWithFeatheringBuffer.borrow_const()));

#ifdef MERGERMASK_SEAM_MASK
    {
      std::stringstream ss;
      ss.str("");
      ss << "final-seam-mask.png";
      Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), inputIndexBuffer.as_const(), size.x, size.y);
    }
#endif
  }

#ifdef MERGERMASK_SEAM_MASK
  {
    std::stringstream ss;
    ss.str("");
    ss << "input-original-mask.png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), inputsMapOriginalBuffer.borrow_const(), size.x, size.y);
  }
#endif
  FAIL_RETURN(progress.add("seam_updateMask", "Update seam mask"));
  return Status::OK();
}

Status MergerMask::transformMasksFromOutputToEncodedInputSpace(
    const Core::PanoDefinition& pano, const std::map<readerid_t, Input::VideoReader*>& readers,
    const GPU::Buffer<const uint32_t>& maskOutputSpaces, std::map<videoreaderid_t, std::string>& maskInputSpaces) {
  // Prepare transform
  std::map<videoreaderid_t, std::unique_ptr<Core::Transform>> transforms;
  for (auto reader : readers) {
    const Core::InputDefinition& inputDef = pano.getInput(reader.second->id);
    Core::Transform* transform = Core::Transform::create(inputDef);
    if (!transform) {
      return {Origin::Stitcher, ErrType::SetupFailure,
              "Cannot create v1 transformation for input " + std::to_string(reader.second->id)};
    }
    transforms[reader.second->id] = std::unique_ptr<Core::Transform>(transform);
  }

  GPU::Stream stream = GPU::Stream::getDefault();
  Core::Rect outputBounds =
      Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, pano.getHeight() - 1, pano.getWidth() - 1);
  const int inputScaleFactor = pano.getBlendingMaskInputScaleFactor();

  maskInputSpaces.clear();

#ifdef MERGERMASK_INPUT_SPACE
  {
    std::stringstream ss;
    ss << "inputsMap-result.png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), maskOutputSpaces.as_const(), outputBounds.getWidth(),
                                     outputBounds.getHeight());
  }
#endif
  for (auto& transform : transforms) {
    const videoreaderid_t imId = transform.first;

    // Find the precomputed coordinate to map pixels from the output to the input space
    const Core::InputDefinition& inputDef = pano.getInput(imId);
    GPU::UniqueBuffer<float2> devCoord;
    FAIL_RETURN(devCoord.alloc(inputDef.getWidth() * inputDef.getHeight() * inputScaleFactor * inputScaleFactor,
                               "Merger Mask"));
    FAIL_RETURN(transform.second->mapCoordInput(0, inputScaleFactor, devCoord.borrow(), pano, inputDef, stream));

#ifdef MERGERMASK_INPUT_SPACE
    {
      std::stringstream ss;
      ss << "inputsMap-inputcoord-" << imId << ".png";
      GPU::UniqueBuffer<uint32_t> dst;
      FAIL_RETURN(
          dst.alloc(inputDef.getWidth() * inputDef.getHeight() * inputScaleFactor * inputScaleFactor, "Merger Mask"));
      Util::OpticalFlow::convertFlowToRGBA(
          make_int2((int)inputDef.getWidth() * inputScaleFactor, (int)inputDef.getHeight() * inputScaleFactor),
          devCoord.borrow_const(), make_int2((int)pano.getWidth(), (int)pano.getHeight()), dst.borrow(), stream);
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), dst.borrow_const(), inputDef.getWidth() * inputScaleFactor,
                                  inputDef.getHeight() * inputScaleFactor);
    }
#endif
    GPU::UniqueBuffer<unsigned char> inputMask;
    FAIL_RETURN(inputMask.alloc(inputDef.getWidth() * inputDef.getHeight() * inputScaleFactor * inputScaleFactor,
                                "Merger Mask"));
    GPU::UniqueBuffer<unsigned char> tmp;
    FAIL_RETURN(
        tmp.alloc(inputDef.getWidth() * inputDef.getHeight() * inputScaleFactor * inputScaleFactor, "Merger Mask"));
    FAIL_RETURN(GPU::memsetToZeroBlocking(inputMask.borrow(), inputMask.borrow().byteSize()));
    // Accumulate the pixels
    FAIL_RETURN(getInputMaskFromOutputIndices(
        imId, inputScaleFactor, make_int2((int)outputBounds.getWidth(), (int)outputBounds.getHeight()),
        maskOutputSpaces, make_int2((int)inputDef.getWidth(), (int)inputDef.getHeight()), devCoord.borrow(),
        inputMask.borrow(), stream));

    // Perform gaussian filter on the mask to remove spiky edges.
    // This step will improve polyline compression rate dramatically.
    FAIL_RETURN(Image::gaussianBlur2D(inputMask.borrow(), tmp.borrow(), (int)inputDef.getWidth() * inputScaleFactor,
                                      (int)inputDef.getHeight() * inputScaleFactor, 1, 6, 0, 16, stream));
    FAIL_RETURN(Util::ImageProcessingGPU::thresholdingBuffer<unsigned char>(
        make_int2((int)inputDef.getWidth() * inputScaleFactor, (int)inputDef.getHeight() * inputScaleFactor), 0, 0, 255,
        inputMask.borrow(), stream));
    std::vector<unsigned char> masks(inputDef.getWidth() * inputDef.getHeight() * inputScaleFactor * inputScaleFactor);
    FAIL_RETURN(
        GPU::memcpyBlocking<unsigned char>(&masks[0], inputMask.borrow_const(), masks.size() * sizeof(unsigned char)));
#ifdef MERGERMASK_INPUT_SPACE
    {
      std::stringstream ss;
      ss << "inputsMap-mask-" << imId << ".png";
      Debug::dumpRGBAIndexDeviceBuffer<unsigned char>(ss.str().c_str(), masks, inputDef.getWidth() * inputScaleFactor,
                                                      inputDef.getHeight() * inputScaleFactor);
    }
#endif
    std::string maskInputSpace;
    FAIL_RETURN(Util::Compression::polylineEncodeBinaryMask((int)inputDef.getWidth() * inputScaleFactor,
                                                            (int)inputDef.getHeight() * inputScaleFactor, masks,
                                                            maskInputSpace));
    maskInputSpaces[imId] = maskInputSpace;
  }

#ifdef MERGERMASK_INPUT_SPACE
  {
    GPU::UniqueBuffer<uint32_t> reconstructMaskOutput;
    FAIL_RETURN(reconstructMaskOutput.alloc(pano.getWidth() * pano.getHeight(), "Merger Mask"));
    FAIL_RETURN(
        transformMasksFromEncodedInputToOutputSpace(pano, readers, maskInputSpaces, reconstructMaskOutput.borrow()));

    std::stringstream ss;
    ss << "inputsMap-reconstruction.png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), reconstructMaskOutput.borrow_const(), pano.getWidth(),
                                     pano.getHeight());
  }
#endif

  return Status::OK();
}

Status MergerMask::transformMasksFromEncodedInputToOutputSpace(
    const Core::PanoDefinition& pano, const std::map<readerid_t, Input::VideoReader*>& readers,
    const std::map<videoreaderid_t, std::string>& maskInputSpaces, GPU::Buffer<uint32_t> maskOutputSpaces) {
  // Prepare transform
  std::map<videoreaderid_t, std::unique_ptr<Core::Transform>> transforms;
  for (auto reader : readers) {
    const Core::InputDefinition& inputDef = pano.getInput(reader.second->id);
    Core::Transform* transform = Core::Transform::create(inputDef);
    if (!transform) {
      return {Origin::Stitcher, ErrType::SetupFailure,
              "Cannot create v1 transformation for input " + std::to_string(reader.second->id)};
    }
    transforms[reader.second->id] = std::unique_ptr<Core::Transform>(transform);
  }

  const int inputScaleFactor = pano.getBlendingMaskInputScaleFactor();
  GPU::Stream stream = GPU::Stream::getDefault();
  Core::Rect outputBounds =
      Core::Rect::fromInclusiveTopLeftBottomRight(0, 0, pano.getHeight() - 1, pano.getWidth() - 1);
  auto tex =
      Core::OffscreenAllocator::createCoordSurface(outputBounds.getWidth(), outputBounds.getHeight(), "Merger Mask");
  if (!tex.ok()) {
    return tex.status();
  }
  std::unique_ptr<Core::SourceSurface> devCoord(tex.release());

  FAIL_RETURN(GPU::memsetToZeroBlocking<uint32_t>(maskOutputSpaces, maskOutputSpaces.byteSize()));
  for (auto& transform : transforms) {
    const videoreaderid_t imId = transform.first;
    const Core::InputDefinition& inputDef = pano.getInput(imId);
    FAIL_RETURN(
        transform.second->mapBufferCoord(0, *devCoord.get()->pimpl->surface, outputBounds, pano, inputDef, stream));

    GPU::UniqueBuffer<unsigned char> inputMask;
    FAIL_RETURN(inputMask.alloc(inputDef.getWidth() * inputDef.getHeight() * inputScaleFactor * inputScaleFactor,
                                "Merger Mask"));
    std::vector<unsigned char> masks;
    auto it = maskInputSpaces.find(imId);
    if (it == maskInputSpaces.end()) {
      return {Origin::BlendingMaskAlgorithm, ErrType::ImplementationError, "Data was not found"};
    }

    FAIL_RETURN(Util::Compression::polylineDecodeBinaryMask(
        (int)inputDef.getWidth() * inputScaleFactor, (int)inputDef.getHeight() * inputScaleFactor, it->second, masks));
    FAIL_RETURN(
        GPU::memcpyBlocking<unsigned char>(inputMask.borrow(), &masks[0], masks.size() * sizeof(unsigned char)));
    FAIL_RETURN(getOutputIndicesFromInputMask(
        imId, inputScaleFactor, make_int2((int)inputDef.getWidth(), (int)inputDef.getHeight()),
        inputMask.borrow_const(), make_int2((int)outputBounds.getWidth(), (int)outputBounds.getHeight()),
        *devCoord.get()->pimpl->surface, maskOutputSpaces, stream));
  }
  return Status::OK();
}

}  // namespace MergerMask
}  // namespace VideoStitch
