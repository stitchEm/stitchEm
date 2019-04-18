// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "./mergerPair.hpp"

#include "./flowConstant.hpp"
#ifndef VS_OPENCL
#include "./spaceTransform.hpp"
#endif

#include "core/geoTransform.hpp"
#include "core1/imageMerger.hpp"
#include "core1/imageMapping.hpp"
#include "gpu/image/imgInsert.hpp"
#include "gpu/image/imageOps.hpp"
#include "gpu/image/sampling.hpp"
#include "gpu/memcpy.hpp"
#define HOST_TRANSFORM
#include "backend/cpp/core/transformStack.hpp"
#undef HOST_TRANSFORM
#ifndef VS_OPENCL
#include "util/opticalFlowUtils.hpp"
#include "util/imageProcessingGPUUtils.hpp"
#endif

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/output.hpp"

//#define MERGER_PAIR_DEBUG
//#define MERGER_PAIR_MAPPING_DEBUG

#if defined(MERGER_PAIR_DEBUG) || defined(MERGER_PAIR_MAPPING_DEBUG)
#include "util/debugUtils.hpp"
#include <sstream>
#endif

namespace VideoStitch {
namespace Core {

MergerPair::MergerPair(const int boundingFirstLevelSize, const int boundingLastLevelSize, const int width0,
                       const int height0, const int offset0X, const int offset0Y,
                       const GPU::Buffer<const uint32_t>& buffer0, const int width1, const int height1,
                       const int offset1X, const int offset1Y, const GPU::Buffer<const uint32_t>& buffer1,
                       GPU::Stream stream)
    : id0s(1, -1),
      id1(-1),
      extendedRatio(0.0f),
      overlappedAreaOnly(false),
      boundingFirstLevelSize(boundingFirstLevelSize),
      boundingLastLevelSize(boundingLastLevelSize),
      wrapWidth(0),
      wrapHeight(0),
      input1Size(make_int2(width1, height1)),
      boundingPanoRect0(Rect::fromInclusiveTopLeftBottomRight(offset0Y, offset0X, height0 - 1, width0 - 1)),
      boundingPanoRect1(Rect::fromInclusiveTopLeftBottomRight(offset1Y, offset1X, height1 - 1, width1 - 1)),
      useInterToPano(false) {
#ifndef VS_OPENCL
  GPU::UniqueBuffer<float2> coord0Mapping;
  GPU::UniqueBuffer<uint32_t> weight0;
  Rect boundingRect0 =
      Rect::fromInclusiveTopLeftBottomRight(offset0Y, offset0X, offset0Y + height0 - 1, offset0X + width0 - 1);

  GPU::UniqueBuffer<float2> coord1Mapping;
  GPU::UniqueBuffer<uint32_t> weight1;
  Rect boundingRect1 =
      Rect::fromInclusiveTopLeftBottomRight(offset1Y, offset1X, offset1Y + height1 - 1, offset1X + width1 - 1);

  wrapWidth = std::max(boundingRect0.right(), boundingRect1.right()) + 1;
  wrapHeight = std::max(boundingRect0.bottom(), boundingRect1.bottom()) + 1;
  // Allocated memory for the cropped buffer
  if (!coord0Mapping.alloc(boundingRect0.getWidth() * boundingRect0.getHeight(), "Tmp Merger Pair").ok()) return;

  if (!weight0.alloc(boundingRect0.getWidth() * boundingRect0.getHeight(), "Tmp Merger Pair").ok()) return;

  if (!coord1Mapping.alloc(boundingRect1.getWidth() * boundingRect1.getHeight(), "Tmp Merger Pair").ok()) return;

  if (!weight1.alloc(boundingRect1.getWidth() * boundingRect1.getHeight(), "Tmp Merger Pair").ok()) return;

  Util::OpticalFlow::generateIdentityFlow(make_int2(width0, height0), coord0Mapping.borrow(), stream);
  Util::OpticalFlow::setAlphaToFlowBuffer(make_int2(width0, height0), buffer0, coord0Mapping.borrow(), stream);
  Util::ImageProcessingGPU::setConstantBuffer<uint32_t>(make_int2(width0, height0), weight0.borrow(), 1, stream);
  Util::OpticalFlow::generateIdentityFlow(input1Size, coord1Mapping.borrow(), stream);
  Util::OpticalFlow::setAlphaToFlowBuffer(input1Size, buffer1, coord1Mapping.borrow(), stream);
  Util::ImageProcessingGPU::setConstantBuffer<uint32_t>(input1Size, weight1.borrow(), 1, stream);

  if (!panoToInputSpaceCoordMapping0.alloc(width0 * height0, "Tmp Merger Pair").ok()) {
    return;
  }
  if (!panoToInputSpaceCoordMapping1.alloc(width1 * height1, "Tmp Merger Pair").ok()) {
    return;
  }
  Util::OpticalFlow::generateIdentityFlow(make_int2(width0, height0), panoToInputSpaceCoordMapping0.borrow(), stream);
  Util::OpticalFlow::setAlphaToFlowBuffer(make_int2(width0, height0), buffer0, panoToInputSpaceCoordMapping0.borrow(),
                                          stream);
  Util::OpticalFlow::generateIdentityFlow(make_int2(width1, height1), panoToInputSpaceCoordMapping1.borrow(), stream);
  Util::OpticalFlow::setAlphaToFlowBuffer(make_int2(width1, height1), buffer1, panoToInputSpaceCoordMapping1.borrow(),
                                          stream);

  if (!panoToInterSpaceCoordMapping0.alloc(width0 * height0, "Tmp Merger Pair").ok()) {
    return;
  }
  if (!panoToInterSpaceCoordMapping1.alloc(width1 * height1, "Tmp Merger Pair").ok()) {
    return;
  }
  Util::OpticalFlow::generateIdentityFlow(make_int2(width0, height0), panoToInterSpaceCoordMapping0.borrow(), stream);
  Util::OpticalFlow::setAlphaToFlowBuffer(make_int2(width0, height0), buffer0, panoToInterSpaceCoordMapping0.borrow(),
                                          stream);
  Util::OpticalFlow::generateIdentityFlow(make_int2(width1, height1), panoToInterSpaceCoordMapping1.borrow(), stream);
  Util::OpticalFlow::setAlphaToFlowBuffer(make_int2(width1, height1), buffer1, panoToInputSpaceCoordMapping1.borrow(),
                                          stream);

  float downRatio;
  buildLaplacianPyramids(PanoDefinition(), downRatio, coord0Mapping, weight0, boundingRect0, coord1Mapping, weight1,
                         boundingRect1, stream);
#else
  (void)buffer0;
  (void)buffer1;
#endif
}

MergerPair::MergerPair(const int boundingFirstLevelSize, const int boundingLastLevelSize,
                       const std::vector<videoreaderid_t>& id0s, const videoreaderid_t id1)
    : id0s(id0s),
      id1(id1),
      extendedRatio(0.25f),
      overlappedAreaOnly(true),
      boundingFirstLevelSize(boundingFirstLevelSize),
      boundingLastLevelSize(boundingLastLevelSize),
      boundingPanoRect0(Rect::fromInclusiveTopLeftBottomRight(0, 0, 0, 0)),
      boundingPanoRect1(Rect::fromInclusiveTopLeftBottomRight(0, 0, 0, 0)),
      useInterToPano(true) {}

Potential<MergerPair> MergerPair::create(
#ifndef VS_OPENCL
    const PanoDefinition& panoDef, const StereoRigDefinition* rigDef,
#else
    const PanoDefinition&, const StereoRigDefinition*,
#endif
    const int boundingFirstLevelSize, const int boundingLastLevelSize, const std::vector<videoreaderid_t>& id0s,
    const videoreaderid_t id1,
#ifndef VS_OPENCL
    const Rect& inBoundingPanoRect0, const Rect& inBoundingPanoRect1,
#else
    const Rect&, const Rect&,
#endif
    GPU::Stream stream) {
  std::unique_ptr<MergerPair> mergerPair(new MergerPair(boundingFirstLevelSize, boundingLastLevelSize, id0s, id1));
#ifndef VS_OPENCL
  FAIL_RETURN(mergerPair->init(panoDef, rigDef, inBoundingPanoRect0, inBoundingPanoRect1, stream));
#endif
  return Potential<MergerPair>(mergerPair.release());
}

std::vector<Rect> MergerPair::getBoundingInterRect1s() const { return boundingInterRect1s; }

int MergerPair::getWrapWidth() const { return (int)wrapWidth; }

int MergerPair::getWrapHeight() const { return (int)wrapHeight; }

int2 MergerPair::getInput1Size() const { return input1Size; }

const Rect MergerPair::getBoundingInterRect(const int index, const int level) const {
  if (index == 0 && level < (int)boundingInterRect0s.size()) {
    return boundingInterRect0s[level];
  } else if (index == 1 && level < (int)boundingInterRect1s.size()) {
    return boundingInterRect1s[level];
  }
  return Rect::fromInclusiveTopLeftBottomRight(0, 0, 0, 0);
}

GPU::Buffer<float2> MergerPair::getInterToLookupSpaceCoordMappingBufferLevel(const int index, const int level) const {
  if (index == 0) {
    if (!useInterToPano) {
      if (level < 0 || level > interToInputSpaceCoordMappingLaplacianPyramid0->numLevels()) {
        return GPU::Buffer<float2>();
      } else {
        return interToInputSpaceCoordMappingLaplacianPyramid0->getLevel(level).data();
      }
    } else {
      if (level < 0 || level > interToPanoSpaceCoordMappingLaplacianPyramid0->numLevels()) {
        return GPU::Buffer<float2>();
      } else {
        return interToPanoSpaceCoordMappingLaplacianPyramid0->getLevel(level).data();
      }
    }
  } else if (index == 1) {
    if (level < 0 || level > interToInputSpaceCoordMappingLaplacianPyramid1->numLevels()) {
      return GPU::Buffer<float2>();
    } else {
      return interToInputSpaceCoordMappingLaplacianPyramid1->getLevel(level).data();
    }
  }
  return GPU::Buffer<float2>();
}

GPU::Buffer<float2> MergerPair::getPanoToInputSpaceCoordMapping(const int index) const {
  if (index == 0) {
    return panoToInputSpaceCoordMapping0.borrow();
  } else if (index == 1) {
    return panoToInputSpaceCoordMapping1.borrow();
  }
  return GPU::Buffer<float2>();
}

GPU::Buffer<float2> MergerPair::getPanoToInterSpaceCoordMapping(const int index) const {
  if (index == 0) {
    return panoToInterSpaceCoordMapping0.borrow();
  } else if (index == 1) {
    return panoToInterSpaceCoordMapping1.borrow();
  }
  return GPU::Buffer<float2>();
}

bool MergerPair::doesOverlap() const { return boundingInterRect0s.size() > 0 && boundingInterRect1s.size() > 0; }

const LaplacianPyramid<float2>* MergerPair::getInterToInputSpaceCoordMappingLaplacianPyramid(const int index) const {
  if (index == 0) {
    if (!useInterToPano) {
      return interToInputSpaceCoordMappingLaplacianPyramid0.get();
    } else {
      return interToPanoSpaceCoordMappingLaplacianPyramid0.get();
    }
  } else if (index == 1) {
    return interToInputSpaceCoordMappingLaplacianPyramid1.get();
  }
  return nullptr;
}

#ifndef VS_OPENCL
Vector3<double> MergerPair::getAverageSphericalCoord(
    const PanoDefinition& panoDef, const std::vector<videoreaderid_t>& id0s, const std::vector<videoreaderid_t>& id1s,
    const Rect& boundingPanoRect0, const GPU::Buffer<const float2>& panoToInputSpaceCoordMapping0,
    const GPU::Buffer<const uint32_t>& maskBuffer0, const Rect& boundingPanoRect1,
    const GPU::Buffer<const float2>& panoToInputSpaceCoordMapping1, const GPU::Buffer<const uint32_t>& maskBuffer1) {
  const int wrapWidth = (int)panoDef.getWidth();
  // Download both buffer to cpu
  std::vector<float2> panoToInput0(boundingPanoRect0.getArea());
  std::vector<uint32_t> mask0(boundingPanoRect0.getArea());
  std::vector<float2> panoToInput1(boundingPanoRect1.getArea());
  std::vector<uint32_t> mask1(boundingPanoRect1.getArea());

  GPU::memcpyBlocking<float2>(&panoToInput0[0], panoToInputSpaceCoordMapping0);
  GPU::memcpyBlocking<float2>(&panoToInput1[0], panoToInputSpaceCoordMapping1);
  GPU::memcpyBlocking<uint32_t>(&mask0[0], maskBuffer0);
  GPU::memcpyBlocking<uint32_t>(&mask1[0], maskBuffer1);

  videoreaderid_t maxId = 0;
  for (size_t i = 0; i < id0s.size(); i++) {
    if (id0s[i] > maxId) {
      maxId = id0s[i];
    }
  }
  for (size_t i = 0; i < id1s.size(); i++) {
    if (id1s[i] > maxId) {
      maxId = id1s[i];
    }
  }
  maxId++;

  std::vector<std::unique_ptr<TransformStack::GeoTransform>> geoTransforms;
  for (int i = 0; i < maxId; i++) {
    geoTransforms.push_back(std::unique_ptr<TransformStack::GeoTransform>(nullptr));
  }
  std::vector<int> inWidth_div2, inHeight_div2;
  inWidth_div2.assign(maxId, 0);
  inHeight_div2.assign(maxId, 0);

  for (size_t i = 0; i < id0s.size(); i++) {
    if (geoTransforms[id0s[i]].get() == nullptr) {
      geoTransforms[id0s[i]].reset(TransformStack::GeoTransform::create(panoDef, panoDef.getInput(id0s[i])));
      inWidth_div2[id0s[i]] = (int)panoDef.getInput(id0s[i]).getWidth() / 2;
      inHeight_div2[id0s[i]] = (int)panoDef.getInput(id0s[i]).getHeight() / 2;
    }
  }
  for (size_t i = 0; i < id1s.size(); i++) {
    if (geoTransforms[id1s[i]].get() == nullptr) {
      geoTransforms[id1s[i]].reset(TransformStack::GeoTransform::create(panoDef, panoDef.getInput(id1s[i])));
      inWidth_div2[id1s[i]] = (int)panoDef.getInput(id1s[i]).getWidth() / 2;
      inHeight_div2[id1s[i]] = (int)panoDef.getInput(id1s[i]).getHeight() / 2;
    }
  }

  const int width0 = (int)boundingPanoRect0.getWidth();
  const int height0 = (int)boundingPanoRect0.getHeight();
  const int width1 = (int)boundingPanoRect1.getWidth();
  const int height1 = (int)boundingPanoRect1.getHeight();

  float3 avgCoord = make_float3(0, 0, 0);
  int sampleCount = 0;
  for (int x0 = 0; x0 < width0; x0++)
    for (int y0 = 0; y0 < height0; y0++) {
      const int index0 = (y0 * width0 + x0);
      const int x1 = (int)((x0 + boundingPanoRect0.left() - boundingPanoRect1.left()) % wrapWidth);
      const int y1 = (int)(y0 + boundingPanoRect0.top() - boundingPanoRect1.top());
      if (mask0[index0] > 0 && x1 >= 0 && y1 >= 0 && x1 < width1 && y1 < height1) {
        const int index1 = y1 * width1 + x1;
        if (mask1[index1] > 0) {
          // Only consider the overlapping area into the cost function
          // This will ensure after transforming to intermediate space,
          // the overlapping areas will be in the middle of the output projection
          const videoreaderid_t id0 = (int)log2f((float)mask0[index0]);
          const videoreaderid_t id1 = (int)log2f((float)mask1[index1]);
          const Core::CenterCoords2 uv0(panoToInput0[index0].x - inWidth_div2[id0],
                                        panoToInput0[index0].y - inHeight_div2[id0]);
          const Core::CenterCoords2 uv1(panoToInput1[index1].x - inWidth_div2[id1],
                                        panoToInput1[index1].y - inHeight_div2[id1]);
          avgCoord +=
              geoTransforms[id0]->mapInputToScaledCameraSphereInRigBase(panoDef.getInput(id0), uv0, 0).toFloat3();
          avgCoord +=
              geoTransforms[id1]->mapInputToScaledCameraSphereInRigBase(panoDef.getInput(id1), uv1, 0).toFloat3();
          sampleCount += 2;
        }
      }
    }
  if (sampleCount > 0) {
    avgCoord /= (float)sampleCount;
  }
  return Vector3<double>(avgCoord.x, avgCoord.y, avgCoord.z);
}

Status MergerPair::init(const PanoDefinition& panoDef, const StereoRigDefinition* rigDef,
                        const Rect& inBoundingPanoRect0, const Rect& inBoundingPanoRect1, GPU::Stream stream) {
  wrapWidth = panoDef.getWidth();
  wrapHeight = panoDef.getHeight();
  // bool usePassedPanoRect0 = !inBoundingPanoRect0.empty();
  boundingPanoRect0 = inBoundingPanoRect0;
  bool usePassedPanoRect1 = !inBoundingPanoRect1.empty();
  boundingPanoRect1 = inBoundingPanoRect1;

  // Find mapping from pano to input space
  GPU::UniqueBuffer<uint32_t> panoToInputSpaceMask0;
  GPU::UniqueBuffer<uint32_t> panoToInputSpaceMask1;
  FAIL_RETURN(findMappingToInputSpace(panoDef, rigDef, id0s, Vector3<double>(0, 0, 1), Vector3<double>(0, 0, 1),
                                      panoToInputSpaceCoordMapping0, panoToInputSpaceMask0, boundingPanoRect0, stream,
                                      false));
  FAIL_RETURN(findMappingToInputSpace(panoDef, rigDef, std::vector<videoreaderid_t>{id1}, Vector3<double>(0, 0, 1),
                                      Vector3<double>(0, 0, 1), panoToInputSpaceCoordMapping1, panoToInputSpaceMask1,
                                      boundingPanoRect1, stream, usePassedPanoRect1));

  Rect boundingPanoTightIRect;

  FAIL_RETURN(Util::ImageProcessingGPU::computeTightOverlappingRect(
      EQUIRECTANGULAR, getWrapWidth(), boundingPanoRect0, panoToInputSpaceMask0.borrow_const(), boundingPanoRect1,
      panoToInputSpaceMask1.borrow_const(), boundingPanoTightIRect, stream));

  // Check if the two rectangles are overlapping, if not, nothing need to be done
  if (boundingPanoTightIRect.empty()) {
    return Status::OK();
  }

  GPU::UniqueBuffer<float2> interToInputCoordMapping0;
  GPU::UniqueBuffer<uint32_t> interToInputMask0;
  Rect interBoundingRect0 = Rect::fromInclusiveTopLeftBottomRight(0, 0, 0, 0);

  GPU::UniqueBuffer<float2> interToInputCoordMapping1;
  GPU::UniqueBuffer<uint32_t> interToInputMask1;
  Rect interBoundingRect1 = Rect::fromInclusiveTopLeftBottomRight(0, 0, 0, 0);

  Vector3<double> avgCoord = getAverageSphericalCoord(
      panoDef, id0s, std::vector<videoreaderid_t>{id1}, boundingPanoRect0, panoToInputSpaceCoordMapping0.borrow_const(),
      panoToInputSpaceMask0.borrow_const(), boundingPanoRect1, panoToInputSpaceCoordMapping1.borrow_const(),
      panoToInputSpaceMask1.borrow_const());
  Vector3<double> newCoord = Vector3<double>(-4, -1, 4);
  if (panoDef.getProjection() == PanoProjection::Stereographic) {
    newCoord = Vector3<double>(0, 0, 1);
  }

  FAIL_RETURN(findMappingToInputSpace(panoDef, rigDef, id0s, avgCoord, newCoord, interToInputCoordMapping0,
                                      interToInputMask0, interBoundingRect0, stream));
  FAIL_RETURN(findMappingToInputSpace(panoDef, rigDef, std::vector<videoreaderid_t>{id1}, avgCoord, newCoord,
                                      interToInputCoordMapping1, interToInputMask1, interBoundingRect1, stream));

  input1Size = make_int2((int)panoDef.getInput(id1).getWidth(), (int)panoDef.getInput(id1).getHeight());
  float downRatio = 1;

  FAIL_RETURN(buildLaplacianPyramids(panoDef, downRatio, interToInputCoordMapping0, interToInputMask0,
                                     interBoundingRect0, interToInputCoordMapping1, interToInputMask1,
                                     interBoundingRect1, stream));
  if (doesOverlap()) {
    FAIL_RETURN(findMappingFromPanoToInterSpace(
        panoDef, downRatio, id0s, avgCoord, newCoord, panoToInputSpaceCoordMapping0.borrow_const(),
        panoToInputSpaceMask0.borrow_const(), boundingPanoRect0, panoToInterSpaceCoordMapping0, stream));

    FAIL_RETURN(findMappingFromPanoToInterSpace(panoDef, downRatio, std::vector<videoreaderid_t>{id1}, avgCoord,
                                                newCoord, panoToInputSpaceCoordMapping1.borrow_const(),
                                                panoToInputSpaceMask1.borrow_const(), boundingPanoRect1,
                                                panoToInterSpaceCoordMapping1, stream));
  }

#ifdef MERGER_PAIR_DEBUG
  const int input0Width = (int)panoDef.getInput(id0s[0]).getWidth();
  const int input0Height = (int)panoDef.getInput(id0s[0]).getHeight();

  GPU::UniqueBuffer<float2> inputFlowBuffer;
  GPU::UniqueBuffer<uint32_t> inputMaskBuffer;
  GPU::UniqueBuffer<uint32_t> panoToInputWeight;
  FAIL_RETURN(inputFlowBuffer.alloc(input0Width * input0Height, "Merger Pair"));
  FAIL_RETURN(inputMaskBuffer.alloc(input0Width * input0Height, "Merger Pair"));
  FAIL_RETURN(panoToInputCoordMapping0.alloc(panoDef.getWidth() * panoDef.getHeight(), "Merger Pair"));
  FAIL_RETURN(panoToInputWeight.alloc(panoDef.getWidth() * panoDef.getHeight(), "Merger Pair"));
  FAIL_RETURN(
      Util::OpticalFlow::generateIdentityFlow(make_int2(input0Width, input0Height), inputFlowBuffer.borrow(), stream));
  FAIL_RETURN(Util::ImageProcessingGPU::setConstantBuffer<uint32_t>(make_int2(input0Width, input0Height),
                                                                    inputMaskBuffer.borrow(), 1 << id0s[0], stream));

  FAIL_RETURN(inputToPanoCoordMapping0.alloc(input0Width * input0Height, "Merger Pair"));

  std::unique_ptr<VideoStitch::Core::SpaceTransform> inputToPanoTransform(VideoStitch::Core::SpaceTransform::create(
      panoDef.getInput(id0s[0]), Vector3<double>(0, 0, 1), Vector3<double>(0, 0, 1)));

  FAIL_RETURN(inputToPanoTransform->mapCoordInputToOutput(0, inputToPanoCoordMapping0.borrow(), input0Width,
                                                          input0Height, inputFlowBuffer.borrow(),
                                                          inputMaskBuffer.borrow(), panoDef, id0s[0], stream));

  FAIL_RETURN(inputToPanoTransform->mapCoordOutputToInput(0, 0, 0, (int)panoDef.getWidth(), (int)panoDef.getHeight(),
                                                          panoToInputCoordMapping0.borrow(), panoToInputWeight.borrow(),
                                                          panoDef, id0s[0], stream));
#endif

  return Status::OK();
}

Status MergerPair::packCoordBuffer(const int wrapWidth, const Core::Rect& inputRect,
                                   const GPU::Buffer<const float2>& inputBuffer,
                                   const GPU::Buffer<const uint32_t>& inputWeight, const Core::Rect& outputRect,
                                   GPU::Buffer<float2> outputBuffer, GPU::Buffer<uint32_t> outputWeight,
                                   GPU::Stream gpuStream) {
  FAIL_RETURN(
      Util::ImageProcessingGPU::packBuffer<float2>(wrapWidth, make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE),
                                                   inputRect, inputBuffer, outputRect, outputBuffer, gpuStream));
  return Util::ImageProcessingGPU::packBuffer<uint32_t>(wrapWidth, 0, inputRect, inputWeight, outputRect, outputWeight,
                                                        gpuStream);
}

Status MergerPair::calculateLaplacianPyramidsInfo(float& downRatio, GPU::UniqueBuffer<float2>& coord0Mapping,
                                                  GPU::UniqueBuffer<uint32_t>& weight0, Rect& boundingRect0,
                                                  GPU::UniqueBuffer<float2>& coord1Mapping,
                                                  GPU::UniqueBuffer<uint32_t>& weight1, Rect& boundingRect1,
                                                  GPU::Stream stream) {
  // Find the tight overlapping area of the two rects
  Rect iRect;
  FAIL_RETURN(Util::ImageProcessingGPU::computeTightOverlappingRect(EQUIRECTANGULAR, getWrapWidth(), boundingRect0,
                                                                    weight0.borrow_const(), boundingRect1,
                                                                    weight1.borrow_const(), iRect, stream));

  if (iRect.left() >= iRect.right() || iRect.top() >= iRect.bottom()) {
    // Image pair has no overlapped area, return as OK() but need to treat it differently later
    boundingRect0 = Rect{};
    boundingRect1 = Rect{};
    boundingInterRect0s.clear();
    boundingInterRect1s.clear();
    return Status::OK();
  }
  // If the 2 are not intersecting then

  if (overlappedAreaOnly) {
    // Extend Rect1 a bit to make sure the sure go to wider area
    const int extendedSize = (int)(extendedRatio * iRect.getWidth());
    // Shift the overlapped area so that the center would stay in the middle
    Rect shiftedRect0 = iRect;

    iRect.setBottom(iRect.bottom() + extendedSize);
    iRect.setTop(std::max(int(iRect.top()) - extendedSize, 0));
    iRect.setLeft(std::max(int(iRect.left()) - extendedSize, 0));
    iRect.setRight(std::min(int(iRect.right()) + extendedSize, getWrapWidth() - 1));
    Rect shiftedRect1 = iRect;

    GPU::UniqueBuffer<float2> coord0OverlappedMapping;
    FAIL_RETURN(coord0OverlappedMapping.alloc(shiftedRect0.getWidth() * shiftedRect0.getHeight(), "Merger Pair"));

    GPU::UniqueBuffer<uint32_t> overlappedWeight0;
    FAIL_RETURN(overlappedWeight0.alloc(shiftedRect0.getWidth() * shiftedRect0.getHeight(), "Merger Pair"));

    GPU::UniqueBuffer<float2> coord1OverlappedMapping;
    FAIL_RETURN(coord1OverlappedMapping.alloc(shiftedRect1.getWidth() * shiftedRect1.getHeight(), "Merger Pair"));

    GPU::UniqueBuffer<uint32_t> overlappedWeight1;
    FAIL_RETURN(overlappedWeight1.alloc(shiftedRect1.getWidth() * shiftedRect1.getHeight(), "Merger Pair"));

    FAIL_RETURN(packCoordBuffer((int)wrapWidth, boundingRect0, coord0Mapping.borrow_const(), weight0.borrow_const(),
                                shiftedRect0, coord0OverlappedMapping.borrow(), overlappedWeight0.borrow(), stream));
    FAIL_RETURN(packCoordBuffer((int)wrapWidth, boundingRect1, coord1Mapping.borrow_const(), weight1.borrow_const(),
                                shiftedRect1, coord1OverlappedMapping.borrow(), overlappedWeight1.borrow(), stream));

    FAIL_RETURN(coord0Mapping.releaseOwnership().release());
    FAIL_RETURN(coord0Mapping.alloc(shiftedRect0.getWidth() * shiftedRect0.getHeight(), "Merger Pair"));

    FAIL_RETURN(weight0.releaseOwnership().release());
    FAIL_RETURN(weight0.alloc(shiftedRect0.getWidth() * shiftedRect0.getHeight(), "Merger Pair"));

    FAIL_RETURN(GPU::memcpyBlocking<float2>(coord0Mapping.borrow(), coord0OverlappedMapping.borrow_const(),
                                            shiftedRect0.getWidth() * shiftedRect0.getHeight() * sizeof(float2)));
    FAIL_RETURN(GPU::memcpyBlocking<uint32_t>(weight0.borrow(), overlappedWeight0.borrow_const(),
                                              shiftedRect0.getWidth() * shiftedRect0.getHeight() * sizeof(uint32_t)));

    FAIL_RETURN(coord1Mapping.releaseOwnership().release());
    FAIL_RETURN(coord1Mapping.alloc(shiftedRect1.getWidth() * shiftedRect1.getHeight(), "Merger Pair"));

    FAIL_RETURN(weight1.releaseOwnership().release());
    FAIL_RETURN(weight1.alloc(shiftedRect1.getWidth() * shiftedRect1.getHeight(), "Merger Pair"));

    FAIL_RETURN(GPU::memcpyBlocking<float2>(coord1Mapping.borrow(), coord1OverlappedMapping.borrow_const(),
                                            shiftedRect1.getWidth() * shiftedRect1.getHeight() * sizeof(float2)));
    FAIL_RETURN(GPU::memcpyBlocking<uint32_t>(weight1.borrow(), overlappedWeight1.borrow_const(),
                                              shiftedRect1.getWidth() * shiftedRect1.getHeight() * sizeof(uint32_t)));
    // Re-calculate boundingRect0 and boundingRect1
    boundingRect0 = shiftedRect0;
    boundingRect1 = shiftedRect1;
  }

  // Resize image so that the first level is smaller than a boundingFirstLevelSize
  downRatio = 1;
  if (boundingFirstLevelSize > 0) {
    int width0 = (int)boundingRect0.getWidth();
    int height0 = (int)boundingRect0.getHeight();
    float l0 = (float)boundingRect0.left();
    float t0 = (float)boundingRect0.top();

    int width1 = (int)boundingRect1.getWidth();
    int height1 = (int)boundingRect1.getHeight();
    float l1 = (float)boundingRect1.left();
    float t1 = (float)boundingRect1.top();

    assert(boundingFirstLevelSize >= boundingLastLevelSize);

    int levelCount = 0;
    while (width0 > boundingFirstLevelSize || height0 > boundingFirstLevelSize) {
      const int dstWidth0 = (width0 + 1) / 2;
      const int dstHeight0 = (height0 + 1) / 2;
      const int dstWidth1 = (width1 + 1) / 2;
      const int dstHeight1 = (height1 + 1) / 2;
      downRatio *= float(width0) / dstWidth0;
      width0 = dstWidth0;
      height0 = dstHeight0;
      width1 = dstWidth1;
      height1 = dstHeight1;
      l0 = (l0 + 1) / 2;
      t0 = (t0 + 1) / 2;
      l1 = (l1 + 1) / 2;
      t1 = (t1 + 1) / 2;
      levelCount++;
    }

    FAIL_RETURN(Util::ImageProcessingGPU::downSampleCoordImage(
        (int)boundingRect0.getWidth(), (int)boundingRect0.getHeight(), levelCount, coord0Mapping, weight0, stream));
    FAIL_RETURN(Util::ImageProcessingGPU::downSampleCoordImage(
        (int)boundingRect1.getWidth(), (int)boundingRect1.getHeight(), levelCount, coord1Mapping, weight1, stream));

    boundingRect0 = Rect::fromInclusiveTopLeftBottomRight((int64_t)t0, (int64_t)l0, (int64_t)(t0 + height0 - 1),
                                                          (int64_t)(l0 + width0 - 1));
    boundingRect1 = Rect::fromInclusiveTopLeftBottomRight((int64_t)t1, (int64_t)l1, (int64_t)(t1 + height1 - 1),
                                                          (int64_t)(l1 + width1 - 1));
  }

  // Calculate level so the final level is bounded by a the boundingLastLevelSize
  if (boundingLastLevelSize > 0) {
    int width0 = (int)boundingRect0.getWidth();
    int height0 = (int)boundingRect0.getHeight();
    float l0 = (float)boundingRect0.left();
    float t0 = (float)boundingRect0.top();

    int width1 = (int)boundingRect1.getWidth();
    int height1 = (int)boundingRect1.getHeight();
    float l1 = (float)boundingRect1.left();
    float t1 = (float)boundingRect1.top();

    boundingInterRect0s.push_back(boundingRect0);
    boundingInterRect1s.push_back(boundingRect1);
    while (width0 > boundingLastLevelSize && height0 > boundingLastLevelSize) {
      const int dstWidth0 = (width0 + 1) / 2;
      const int dstHeight0 = (height0 + 1) / 2;
      const int dstWidth1 = (width1 + 1) / 2;
      const int dstHeight1 = (height1 + 1) / 2;
      width0 = dstWidth0;
      height0 = dstHeight0;
      width1 = dstWidth1;
      height1 = dstHeight1;
      l0 = (l0 + 1) / 2;
      t0 = (t0 + 1) / 2;
      l1 = (l1 + 1) / 2;
      t1 = (t1 + 1) / 2;
      boundingInterRect0s.push_back(Rect::fromInclusiveTopLeftBottomRight(
          (int64_t)t0, (int64_t)l0, (int64_t)(t0 + dstHeight0 - 1), (int64_t)(l0 + dstWidth0 - 1)));
      boundingInterRect1s.push_back(Rect::fromInclusiveTopLeftBottomRight(
          (int64_t)t1, (int64_t)l1, (int64_t)(t1 + dstHeight1 - 1), (int64_t)(l1 + dstWidth1 - 1)));
    }
  }
  FAIL_RETURN(stream.synchronize());
  return Status::OK();
}

Status MergerPair::findMappingFromInterToPanoSpace(const PanoDefinition& panoDef,
                                                   const std::vector<videoreaderid_t>& ids,
                                                   const GPU::Buffer<const float2>& interToInputSpaceCoordMapping,
                                                   const GPU::Buffer<const uint32_t>& interToInputSpaceMask,
                                                   const Rect& boundingInterRect,
                                                   GPU::UniqueBuffer<float2>& interToPanoSpaceCoordMapping,
                                                   GPU::Stream stream) {
  FAIL_RETURN(interToPanoSpaceCoordMapping.alloc(boundingInterRect.getArea(), "Merger Pair"));
  FAIL_RETURN(Util::ImageProcessingGPU::setConstantBuffer<float2>(
      make_int2((int)boundingInterRect.getWidth(), (int)boundingInterRect.getHeight()),
      interToPanoSpaceCoordMapping.borrow(), make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE), stream));

  for (size_t i = 0; i < ids.size(); i++) {
    const videoreaderid_t id = ids[i];
    // Find mapping from pano to intermediate space
    std::unique_ptr<VideoStitch::Core::SpaceTransform> inputToPanoTransform(VideoStitch::Core::SpaceTransform::create(
        panoDef.getInput(id), Vector3<double>(0, 0, 1), Vector3<double>(0, 0, 1)));

    FAIL_RETURN(inputToPanoTransform->mapCoordInputToOutput(
        0, interToPanoSpaceCoordMapping.borrow(), (int)boundingInterRect.getWidth(), (int)boundingInterRect.getHeight(),
        interToInputSpaceCoordMapping, interToInputSpaceMask, panoDef, id, stream));
  }
  return Status::OK();
}

Status MergerPair::findMappingFromPanoToInterSpace(const PanoDefinition& panoDef, const float downRatio,
                                                   const std::vector<videoreaderid_t>& ids,
                                                   const Vector3<double>& oldCoord, const Vector3<double>& newCoord,
                                                   const GPU::Buffer<const float2>& panoToInputSpaceCoordMapping,
                                                   const GPU::Buffer<const uint32_t>& panoToInputSpaceMask,
                                                   const Rect& boundingPanoRect,
                                                   GPU::UniqueBuffer<float2>& panoToInterSpaceCoordMapping,
                                                   GPU::Stream stream) {
  FAIL_RETURN(
      panoToInterSpaceCoordMapping.alloc(boundingPanoRect.getWidth() * boundingPanoRect.getHeight(), "Merger Pair"));
  for (size_t i = 0; i < ids.size(); i++) {
    const videoreaderid_t id = ids[i];
    // Find mapping from pano to intermediate space
    std::unique_ptr<VideoStitch::Core::SpaceTransform> inputToInterTransform(
        VideoStitch::Core::SpaceTransform::create(panoDef.getInput(id), oldCoord, newCoord));

    FAIL_RETURN(inputToInterTransform->mapCoordInputToOutput(
        0, panoToInterSpaceCoordMapping.borrow(), (int)boundingPanoRect.getWidth(), (int)boundingPanoRect.getHeight(),
        panoToInputSpaceCoordMapping, panoToInputSpaceMask, panoDef, id, stream));
    FAIL_RETURN(Util::OpticalFlow::mulFlowOperator(panoToInterSpaceCoordMapping.borrow(),
                                                   make_float2(1.0f / downRatio, 1.0f / downRatio),
                                                   boundingPanoRect.getArea(), stream));
  }

  FAIL_RETURN(stream.synchronize());
  return Status::OK();
}

std::string MergerPair::getImIdString(const int index) const {
  std::stringstream ss;
  if (index == 0) {
    ss.str("");
    for (size_t i = 0; i < id0s.size(); i++) {
      ss << id0s[i];
      if (i != id0s.size() - 1) {
        ss << "-";
      }
    }
  } else if (index == 1) {
    ss.str("");
    ss << id1;
  }
  return ss.str();
}

const Rect MergerPair::getBoundingPanoRect(const int index) const {
  if (index == 0) {
    return boundingPanoRect0;
  } else if (index == 1) {
    return boundingPanoRect1;
  }
  return Rect::fromInclusiveTopLeftBottomRight(0, 0, 0, 0);
}

Status MergerPair::buildLaplacianPyramids(const PanoDefinition& panoDef, float& downRatio,
                                          GPU::UniqueBuffer<float2>& interToInputSpaceCoordMapping0,
                                          GPU::UniqueBuffer<uint32_t>& interToInputSpaceMask0, Rect& boundingRect0,
                                          GPU::UniqueBuffer<float2>& interToInputSpaceCoordMapping1,
                                          GPU::UniqueBuffer<uint32_t>& interToInputSpaceMask1, Rect& boundingRect1,
                                          GPU::Stream stream) {
#ifdef MERGER_PAIR_MAPPING_DEBUG
  std::stringstream ss;
  ss.str("");
  ss << "interToInputMaskInit-";
  ss << getImIdString(0) << " - " << getImIdString(1) << "image index" << 0 << ".png";
  FAIL_RETURN(Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), interToInputSpaceMask0.borrow_const(),
                                                         boundingRect0.getWidth(), boundingRect0.getHeight()));
#endif
  // Compute all levels' image size
  FAIL_RETURN(calculateLaplacianPyramidsInfo(downRatio, interToInputSpaceCoordMapping0, interToInputSpaceMask0,
                                             boundingRect0, interToInputSpaceCoordMapping1, interToInputSpaceMask1,
                                             boundingRect1, stream));
#ifdef MERGER_PAIR_MAPPING_DEBUG
  ss.str("");
  ss << "interToInputMask-";
  ss << getImIdString(0) << " - " << getImIdString(1) << "image index" << 0 << ".png";
  FAIL_RETURN(Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), interToInputSpaceMask0.borrow_const(),
                                                         boundingRect0.getWidth(), boundingRect0.getHeight()));
#endif
  // Do not need to build the pyramid if the images does not overlapping
  if (!doesOverlap()) {
    return Status::OK();
  }
  if (!useInterToPano) {
    // Create the pyramid
    auto potLaplacianPyramid0 = LaplacianPyramid<float2>::create(
        std::string("flow") + "-" + getImIdString(0) + " - " + getImIdString(1), boundingInterRect0s[0].getWidth(),
        boundingInterRect0s[0].getHeight(), (int)(boundingInterRect0s.size() - 1),
        LaplacianPyramid<float2>::InternalFirstLevel, LaplacianPyramid<float2>::SingleShot, 2, 1, false);
    FAIL_RETURN(potLaplacianPyramid0.status());
    interToInputSpaceCoordMappingLaplacianPyramid0.reset(potLaplacianPyramid0.release());

    auto potWeightLaplacianPyramid0 = LaplacianPyramid<uint32_t>::create(
        std::string("weight") + "-" + getImIdString(0) + " - " + getImIdString(1), boundingInterRect0s[0].getWidth(),
        boundingInterRect0s[0].getHeight(), (int)(boundingInterRect0s.size() - 1),
        LaplacianPyramid<uint32_t>::InternalFirstLevel, LaplacianPyramid<uint32_t>::SingleShot, 2, 1, false);
    FAIL_RETURN(potWeightLaplacianPyramid0.status());
    interToInputSpaceWeightLaplacianPyramid0.reset(potWeightLaplacianPyramid0.release());
  } else {
    auto potLaplacianPyramid3 = LaplacianPyramid<float2>::create(
        std::string("flow") + "-" + getImIdString(0) + " - " + getImIdString(1), boundingInterRect0s[0].getWidth(),
        boundingInterRect0s[0].getHeight(), (int)(boundingInterRect0s.size() - 1),
        LaplacianPyramid<float2>::InternalFirstLevel, LaplacianPyramid<float2>::SingleShot, 2, 1, false);
    FAIL_RETURN(potLaplacianPyramid3.status());
    interToPanoSpaceCoordMappingLaplacianPyramid0.reset(potLaplacianPyramid3.release());

    auto potWeightLaplacianPyramid3 = LaplacianPyramid<uint32_t>::create(
        std::string("weight") + "-" + getImIdString(0) + " - " + getImIdString(1), boundingInterRect0s[0].getWidth(),
        boundingInterRect0s[0].getHeight(), (int)(boundingInterRect0s.size() - 1),
        LaplacianPyramid<uint32_t>::InternalFirstLevel, LaplacianPyramid<uint32_t>::SingleShot, 2, 1, false);
    FAIL_RETURN(potWeightLaplacianPyramid3.status());
    interToPanoSpaceWeightLaplacianPyramid0.reset(potWeightLaplacianPyramid3.release());
  }

  auto potLaplacianPyramid1 = LaplacianPyramid<float2>::create(
      std::string("flow") + "-" + getImIdString(0) + " - " + getImIdString(1), boundingInterRect1s[0].getWidth(),
      boundingInterRect1s[0].getHeight(), (int)(boundingInterRect1s.size() - 1),
      LaplacianPyramid<float2>::InternalFirstLevel, LaplacianPyramid<float2>::SingleShot, 2, 1, false);
  FAIL_RETURN(potLaplacianPyramid1.status());
  interToInputSpaceCoordMappingLaplacianPyramid1.reset(potLaplacianPyramid1.release());

  auto potWeightLaplacianPyramid1 = LaplacianPyramid<uint32_t>::create(
      std::string("weight") + "-" + getImIdString(0) + " - " + getImIdString(1), boundingInterRect1s[0].getWidth(),
      boundingInterRect1s[0].getHeight(), (int)(boundingInterRect1s.size() - 1),
      LaplacianPyramid<uint32_t>::InternalFirstLevel, LaplacianPyramid<uint32_t>::SingleShot, 2, 1, false);
  FAIL_RETURN(potWeightLaplacianPyramid1.status());
  interToInputSpaceWeightLaplacianPyramid1.reset(potWeightLaplacianPyramid1.release());

  GPU::UniqueBuffer<float2> interToPanoSpaceCoordMapping0;
  GPU::UniqueBuffer<uint32_t> mask0;
  FAIL_RETURN(mask0.alloc(boundingInterRect0s[0].getArea(), "Tmp Merger Pair"));
  FAIL_RETURN(Util::ImageProcessingGPU::binarizeMask(
      make_int2((int)boundingInterRect0s[0].getWidth(), (int)boundingInterRect0s[0].getHeight()),
      interToInputSpaceMask0.borrow_const(), mask0.borrow(), stream));

  GPU::UniqueBuffer<uint32_t> mask1;
  FAIL_RETURN(mask1.alloc(boundingInterRect1s[0].getArea(), "Tmp Merger Pair"));
  FAIL_RETURN(Util::ImageProcessingGPU::binarizeMask(
      make_int2((int)boundingInterRect1s[0].getWidth(), (int)boundingInterRect1s[0].getHeight()),
      interToInputSpaceMask1.borrow_const(), mask1.borrow(), stream));

  if (useInterToPano) {
    FAIL_RETURN(findMappingFromInterToPanoSpace(panoDef, id0s, interToInputSpaceCoordMapping0.borrow_const(),
                                                interToInputSpaceMask0.borrow_const(), boundingInterRect0s[0],
                                                interToPanoSpaceCoordMapping0, stream));
  }
  stream.synchronize();

#ifdef MERGER_PAIR_MAPPING_DEBUG
  GPU::UniqueBuffer<uint32_t> flowOutput;
  FAIL_RETURN(
      flowOutput.alloc(boundingInterRect0s[0].getWidth() * boundingInterRect0s[0].getHeight(), "Tmp Merger Pair"));

  if (useInterToPano) {
    FAIL_RETURN(Util::OpticalFlow::convertFlowToRGBA(
        make_int2(boundingInterRect0s[0].getWidth(), boundingInterRect0s[0].getHeight()),
        interToPanoSpaceCoordMapping0.borrow_const(), make_int2(panoDef.getWidth(), panoDef.getHeight()),
        flowOutput.borrow(), stream));
    ss.str("");
    ss << "interToPanoFlow-";
    ss << getImIdString(0) << " - " << getImIdString(1) << "image index" << 0 << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), flowOutput.borrow_const(), boundingInterRect0s[0].getWidth(),
                                boundingInterRect0s[0].getHeight());
  }

  FAIL_RETURN(Util::OpticalFlow::convertFlowToRGBA(
      make_int2(boundingInterRect0s[0].getWidth(), boundingInterRect0s[0].getHeight()),
      interToInputSpaceCoordMapping0.borrow_const(), make_int2(panoDef.getWidth(), panoDef.getHeight()),
      flowOutput.borrow(), stream));
  ss.str("");
  ss << "interToInputFlow-";
  ss << getImIdString(0) << " - " << getImIdString(1) << "image index" << 0 << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), flowOutput.borrow_const(), boundingInterRect0s[0].getWidth(),
                              boundingInterRect0s[0].getHeight());
#endif

  // Allocate and construct pyramid 's memory
  if (!useInterToPano) {
    FAIL_RETURN(GPU::memcpyBlocking(interToInputSpaceCoordMappingLaplacianPyramid0->getLevel(0).data(),
                                    interToInputSpaceCoordMapping0.borrow(),
                                    boundingRect0.getWidth() * boundingRect0.getHeight() * sizeof(float2)));
    FAIL_RETURN(GPU::memcpyBlocking(interToInputSpaceWeightLaplacianPyramid0->getLevel(0).data(), mask0.borrow_const(),
                                    boundingRect0.getWidth() * boundingRect0.getHeight() * sizeof(uint32_t)));
  } else {
    FAIL_RETURN(GPU::memcpyBlocking(interToPanoSpaceCoordMappingLaplacianPyramid0->getLevel(0).data(),
                                    interToPanoSpaceCoordMapping0.borrow_const(),
                                    boundingRect0.getWidth() * boundingRect0.getHeight() * sizeof(float2)));
    FAIL_RETURN(GPU::memcpyBlocking(interToPanoSpaceWeightLaplacianPyramid0->getLevel(0).data(), mask0.borrow_const(),
                                    boundingRect0.getWidth() * boundingRect0.getHeight() * sizeof(uint32_t)));
  }

  FAIL_RETURN(GPU::memcpyBlocking(interToInputSpaceCoordMappingLaplacianPyramid1->getLevel(0).data(),
                                  interToInputSpaceCoordMapping1.borrow(),
                                  boundingRect1.getWidth() * boundingRect1.getHeight() * sizeof(float2)));
  FAIL_RETURN(GPU::memcpyBlocking(interToInputSpaceWeightLaplacianPyramid1->getLevel(0).data(), mask1.borrow_const(),
                                  boundingRect1.getWidth() * boundingRect1.getHeight() * sizeof(uint32_t)));

  for (size_t i = 0; i < boundingInterRect0s.size() - 1; i++) {
    if (!useInterToPano) {
      FAIL_RETURN(VideoStitch::Image::subsample22Mask<float2>(
          interToInputSpaceCoordMappingLaplacianPyramid0->getLevel((unsigned int)i + 1).data(),
          interToInputSpaceWeightLaplacianPyramid0->getLevel((unsigned int)i + 1).data(),
          interToInputSpaceCoordMappingLaplacianPyramid0->getLevel((unsigned int)i).data(),
          interToInputSpaceWeightLaplacianPyramid0->getLevel((unsigned int)i).data(),
          (size_t)boundingInterRect0s[i].getWidth(), (size_t)boundingInterRect0s[i].getHeight(),
          (unsigned int)(ImageMerger::CudaBlockSize), stream));
    } else {
      FAIL_RETURN(VideoStitch::Image::subsample22Mask<float2>(
          interToPanoSpaceCoordMappingLaplacianPyramid0->getLevel((unsigned int)i + 1).data(),
          interToPanoSpaceWeightLaplacianPyramid0->getLevel((unsigned int)i + 1).data(),
          interToPanoSpaceCoordMappingLaplacianPyramid0->getLevel((unsigned int)i).data(),
          interToPanoSpaceWeightLaplacianPyramid0->getLevel((unsigned int)i).data(),
          (size_t)boundingInterRect0s[i].getWidth(), (size_t)boundingInterRect0s[i].getHeight(),
          (unsigned int)(ImageMerger::CudaBlockSize), stream));
    }

    FAIL_RETURN(VideoStitch::Image::subsample22Mask<float2>(
        interToInputSpaceCoordMappingLaplacianPyramid1->getLevel((unsigned int)i + 1).data(),
        interToInputSpaceWeightLaplacianPyramid1->getLevel((unsigned int)i + 1).data(),
        interToInputSpaceCoordMappingLaplacianPyramid1->getLevel((unsigned int)i).data(),
        interToInputSpaceWeightLaplacianPyramid1->getLevel((unsigned int)i).data(),
        (size_t)boundingInterRect1s[i].getWidth(), (size_t)boundingInterRect1s[i].getHeight(),
        (unsigned int)(ImageMerger::CudaBlockSize), stream));
  }
  return stream.synchronize();
}

Vector3<double> MergerPair::getAverageSphericalCoord(const PanoDefinition& panoDef, const videoreaderid_t id0,
                                                     const videoreaderid_t id1) {
  Vector3<double> avgCoord0 = Core::SpaceTransform::getAverageSphericalCoord(panoDef, panoDef.getInput(id0));
  Vector3<double> avgCoord1 = Core::SpaceTransform::getAverageSphericalCoord(panoDef, panoDef.getInput(id1));
  Vector3<double> avgCoord = (avgCoord0 + avgCoord1) / 2;
  return avgCoord;
}

bool MergerPair::UseInterToPano() const { return useInterToPano; }

Status MergerPair::findMappingToInputSpace(const PanoDefinition& panoDef, const StereoRigDefinition* rigDef,
                                           const std::vector<videoreaderid_t>& ids, const Vector3<double>& oldCoord,
                                           const Vector3<double>& newCoord, GPU::UniqueBuffer<float2>& toInputMapping,
                                           GPU::UniqueBuffer<uint32_t>& weight, Rect& boundingRect, GPU::Stream stream,
                                           const bool usePassedBoundingRect) {
  // Now find the mapping mask
  if (!usePassedBoundingRect) {
    const int erectWidth = (int)panoDef.getWidth();
    const int erectHeight = (int)panoDef.getHeight();
    // Allocate memory for coordinate mapping process
    GPU::UniqueBuffer<uint32_t> maskBuffer;
    FAIL_RETURN(maskBuffer.alloc(erectWidth * erectHeight, "Tmp Merger Pair"));
    FAIL_RETURN(GPU::memsetToZeroBlocking(maskBuffer.borrow(), erectWidth * erectHeight * sizeof(uint32_t)));

    GPU::UniqueBuffer<float2> tmpCoordBuffer;
    FAIL_RETURN(tmpCoordBuffer.alloc(erectWidth * erectHeight, "Tmp Merger Pair"));

    for (size_t i = 0; i < ids.size(); i++) {
      const videoreaderid_t id = ids[i];
      std::unique_ptr<VideoStitch::Core::SpaceTransform> toInputTransform(
          VideoStitch::Core::SpaceTransform::create(panoDef.getInput(id), oldCoord, newCoord));
      // Find the right ERect for mapping buffer coordinate
      FAIL_RETURN(toInputTransform->mapCoordOutputToInput(0, 0, 0, erectWidth, erectHeight, tmpCoordBuffer.borrow(),
                                                          maskBuffer.borrow(), panoDef, id, stream));
    }
#ifdef MERGER_PAIR_DEBUG
    static videoreaderid_t id = 0;
    std::stringstream ss;
    ss.str("");
    ss << "bitmask";
    for (size_t i = 0; i < ids.size(); i++) {
      ss << "-" << ids[i];
    }
    ss << ".png";
    Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), maskBuffer.borrow_const(), panoDef.getWidth(),
                                     panoDef.getHeight());
#endif
    // Now find the bounding rect
    FAIL_RETURN(Util::ImageProcessingGPU::findBBox(EQUIRECTANGULAR, true, rigDef, erectWidth, erectHeight,
                                                   maskBuffer.borrow_const(), boundingRect, stream));

    // Make sure that the bounding rect is bounded the pano size
    const int oldRight = (int)boundingRect.right();
    const int oldBottom = (int)boundingRect.bottom();
    boundingRect.growToMultipleSizeOf(ImageMerger::CudaBlockSize, ImageMerger::CudaBlockSize);
    if (oldRight < panoDef.getWidth() && boundingRect.right() >= panoDef.getWidth()) {
      boundingRect.setRight(oldRight);
    }
    if (boundingRect.bottom() >= panoDef.getHeight()) {
      boundingRect.setBottom(oldBottom);
    }
  }

  // Allocated memory for the cropped buffer
  FAIL_RETURN(toInputMapping.alloc(boundingRect.getWidth() * boundingRect.getHeight(), "Tmp Merger Pair"));
  FAIL_RETURN(weight.alloc(boundingRect.getWidth() * boundingRect.getHeight(), "Tmp Merger Pair"));
  FAIL_RETURN(Util::ImageProcessingGPU::setConstantBuffer<float2>(
      make_int2((int)boundingRect.getWidth(), (int)boundingRect.getHeight()), toInputMapping.borrow(),
      make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE), stream));
  FAIL_RETURN(Util::ImageProcessingGPU::setConstantBuffer<uint32_t>(
      make_int2((int)boundingRect.getWidth(), (int)boundingRect.getHeight()), weight.borrow(), 0, stream));

  for (size_t i = 0; i < ids.size(); i++) {
    const videoreaderid_t id = ids[i];
    std::unique_ptr<VideoStitch::Core::SpaceTransform> toInputTransform(
        VideoStitch::Core::SpaceTransform::create(panoDef.getInput(id), oldCoord, newCoord));

    // Map values to the cropped buffer
    FAIL_RETURN(toInputTransform->mapCoordOutputToInput(0, (int)boundingRect.left(), (int)boundingRect.top(),
                                                        (int)boundingRect.getWidth(), (int)boundingRect.getHeight(),
                                                        toInputMapping.borrow(), weight.borrow(), panoDef, id, stream));
  }

  return stream.synchronize();
}

Rect MergerPair::getBoundingPanosIRect() const {
  if (!useInterToPano) {
    // This is the case where full image overlapping were used in lib/src/tests
    return getBoundingPanoRect(0);
  }
  const Rect rect0 = getBoundingPanoRect(0);
  const Rect rect1 = getBoundingPanoRect(1);
  Rect iRect;
  Rect uRect = Rect::fromInclusiveTopLeftBottomRight(0, 0, 0, 0);

  bool isSpecial = false;
  if (std::max(rect0.right(), rect1.right()) > getWrapWidth()) {
    if (std::min(rect0.left(), rect1.left()) < std::max(rect0.right(), rect1.right()) % getWrapWidth()) {
      iRect.setTop(std::max(rect0.top(), rect1.top()));
      iRect.setBottom(std::min(rect0.bottom(), rect1.bottom()));
      iRect.setLeft(std::min(rect0.left(), rect1.left()));
      iRect.setRight(std::min(rect0.right(), rect1.right()));
      isSpecial = true;
    }
  }

  if (!isSpecial) {
    Rect::getInterAndUnion(rect0, rect1, iRect, uRect, (int)wrapWidth);
  }

  if (iRect.empty()) {
    return iRect;
  }

  // If the 2 are not intersecting then
  const int extendedSize = (int)(extendedRatio * iRect.getWidth());
  iRect.setBottom(std::min<int>((int)(iRect.bottom() + extendedSize), (int)wrapHeight - 1));
  iRect.setTop(std::max<int>((int)(iRect.top() - extendedSize), 0));
  iRect.setLeft(std::max<int>((int)(iRect.left() - extendedSize), 0));
  iRect.setRight(std::min<int>((int)(iRect.right() + extendedSize), (int)wrapWidth - 1));

  int paddedWidth = ((int)(iRect.getWidth() / VideoStitch::Core::ImageMerger::CudaBlockSize) + 1) *
                    VideoStitch::Core::ImageMerger::CudaBlockSize;
  int paddedHeight = ((int)(iRect.getHeight() / VideoStitch::Core::ImageMerger::CudaBlockSize) + 1) *
                     VideoStitch::Core::ImageMerger::CudaBlockSize;
  iRect.setRight(iRect.left() + paddedWidth - 1);
  iRect.setBottom(iRect.top() + paddedHeight - 1);
  return iRect;
}

#ifdef MERGER_PAIR_DEBUG
Status MergerPair::debugMergerPair(const int2 panoSize, const GPU::Buffer<const uint32_t> panoBuffer,
                                   const int2 bufferSize0, const GPU::Buffer<const uint32_t> buffer0,
                                   const int2 bufferSize1, const GPU::Buffer<const uint32_t> buffer1,
                                   GPU::Stream gpuStream) const {
  GPU::UniqueBuffer<uint32_t> packedBuffer;
  FAIL_RETURN(packedBuffer.alloc(panoSize.x * panoSize.y, "Merger Pair Tmp"));

  for (int i = 0; i <= std::min(0, getInterToInputSpaceCoordMappingLaplacianPyramid(1)->numLevels()); i++) {
    {
      const LaplacianPyramid<float2>::LevelSpec<float2>& level0 =
          getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(i);
      GPU::UniqueBuffer<uint32_t> image0;
      image0.alloc(level0.width() * level0.height(), "Merger Pair Tmp");

      FAIL_RETURN(Util::OpticalFlow::coordLookup(level0.width(), level0.height(), level0.data(), bufferSize0.x,
                                                 bufferSize0.y, buffer0, image0.borrow(), gpuStream));

      FAIL_RETURN(Util::ImageProcessingGPU::packBuffer<uint32_t>(
          panoSize.x, 0, getBoundingInterRect(0, i), image0.borrow_const(),
          Core::Rect(0, 0, panoSize.y - 1, panoSize.x - 1), packedBuffer.borrow(), gpuStream));
      FAIL_RETURN(gpuStream.synchronize());

      std::stringstream ss;
      ss.str("");
      ss << "mergerPair-";
      ss << getImIdString(0) << " - " << i << "_level"
         << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), packedBuffer.borrow(), panoSize.x, panoSize.y);
    }

    {
      const LaplacianPyramid<float2>::LevelSpec<float2>& level1 =
          getInterToInputSpaceCoordMappingLaplacianPyramid(1)->getLevel(i);
      GPU::UniqueBuffer<uint32_t> image1;
      image1.alloc(level1.width() * level1.height(), "Merger Pair Tmp");

      FAIL_RETURN(Util::OpticalFlow::coordLookup(level1.width(), level1.height(), level1.data(), bufferSize1.x,
                                                 bufferSize1.y, buffer1, image1.borrow(), gpuStream));

      FAIL_RETURN(Util::ImageProcessingGPU::packBuffer<uint32_t>(
          panoSize.x, 0, getBoundingInterRect(1, i), image1.borrow_const(),
          Core::Rect(0, 0, panoSize.y - 1, panoSize.x - 1), packedBuffer.borrow(), gpuStream));
      FAIL_RETURN(gpuStream.synchronize());

      std::stringstream ss;
      ss.str("");
      ss << "mergerPair-";
      ss << getImIdString(1) << " - " << i << "_level"
         << ".png";
      Debug::dumpRGBADeviceBuffer(ss.str().c_str(), packedBuffer.borrow(), panoSize.x, panoSize.y);
    }
  }

  // Write remapping from pano to input, these images are used to check whether the space transformations perform
  // properly
  // @NOTE: turn it off now

  const int input0Width = bufferSize0.x;
  const int input0Height = bufferSize0.y;

  GPU::UniqueBuffer<uint32_t> inputBackwardBuffer;
  GPU::UniqueBuffer<uint32_t> panoBackwardBuffer;
  FAIL_RETURN(inputBackwardBuffer.alloc(input0Width * input0Height, "Merger Pair"));
  FAIL_RETURN(panoBackwardBuffer.alloc(panoSize.x * panoSize.y, "Merger Pair"));

  FAIL_RETURN(Util::OpticalFlow::coordLookup(panoSize.x, panoSize.y, panoToInputCoordMapping0.borrow_const(),
                                             bufferSize0.x, bufferSize0.y, buffer0, panoBackwardBuffer.borrow(),
                                             gpuStream));

  FAIL_RETURN(Util::OpticalFlow::coordLookup(input0Width, input0Height, inputToPanoCoordMapping0.borrow_const(),
                                             panoSize.x, panoSize.y, panoBackwardBuffer.borrow_const(),
                                             inputBackwardBuffer.borrow(), gpuStream));

  std::stringstream ss;

  ss.str("");
  ss << "inputBackward-";
  ss << getImIdString(0) << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), inputBackwardBuffer.borrow(), input0Width, input0Height);

  ss.str("");
  ss << "panoBackward-";
  ss << getImIdString(0) << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), panoBackwardBuffer.borrow(), panoSize.x, panoSize.y);

  ss.str("");
  ss << "panoOriward-";
  ss << getImIdString(0) << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), panoBuffer, panoSize.x, panoSize.y);

  return Status::OK();
}
#else
Status MergerPair::debugMergerPair(const int2, const GPU::Buffer<const uint32_t>, const int2,
                                   const GPU::Buffer<const uint32_t>, const int2, const GPU::Buffer<const uint32_t>,
                                   GPU::Stream) const {
  return Status::OK();
}
#endif

#endif

}  // namespace Core
}  // namespace VideoStitch
