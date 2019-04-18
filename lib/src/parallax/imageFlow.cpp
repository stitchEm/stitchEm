// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "./imageFlow.hpp"

#include "./mergerPair.hpp"
#include "./noFlow.hpp"
#ifndef VS_OPENCL
#include "./simpleFlow.hpp"
#endif

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"
#include "gpu/memcpy.hpp"

#ifndef NDEBUG
#include "util/debugUtils.hpp"
#include <sstream>
#endif

namespace VideoStitch {
namespace Core {

Potential<ImageFlow> ImageFlow::factor(const ImageFlowAlgorithm e, std::shared_ptr<MergerPair>& mergerPair,
                                       const std::map<std::string, float>& parameters) {
  std::unique_ptr<ImageFlow> flowPtr;
  switch (e) {
#ifndef VS_OPENCL
    case ImageFlowAlgorithm::SimpleFlow:
      flowPtr.reset(new SimpleFlow(parameters));
      break;
#endif
    case ImageFlowAlgorithm::NoFlow:
      flowPtr.reset(new NoFlow(parameters));
      break;
    default:
      flowPtr.reset(new NoFlow(parameters));
      break;
  }
  FAIL_RETURN(flowPtr->init(mergerPair));
  return Potential<ImageFlow>(flowPtr.release());
}

ImageFlow::ImageFlow(const std::map<std::string, float>& parameters) : parameters(parameters) {}

ImageFlow::~ImageFlow() {}

Status ImageFlow::findExtrapolatedImageFlow(const int2&, const int2&, const GPU::Buffer<const uint32_t>&,
                                            const GPU::Buffer<const float2>&, const int2&, const int2&,
                                            const GPU::Buffer<const uint32_t>&, const int2&, const int2&,
                                            GPU::Buffer<float2>, GPU::Stream) {
  return Status::OK();
}

Rect ImageFlow::getFlowRect(const int level) const { return mergerPair->getBoundingInterRect(0, level); }

int2 ImageFlow::getLookupOffset(const int level) const {
  return make_int2((int)mergerPair->getBoundingInterRect(1, level).left(),
                   (int)mergerPair->getBoundingInterRect(1, level).top());
}

Status ImageFlow::init(std::shared_ptr<MergerPair>& inMergerPair) {
  this->mergerPair = inMergerPair;

  int firstLevelWidth = (int)mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(0).width();
  int firstLevelHeight = (int)mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(0).height();
  int numLevels = (int)mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->numLevels();

  auto potflowLaplacianPyramid = LaplacianPyramid<float2>::create(
      std::string("flow"), firstLevelWidth, firstLevelHeight, numLevels, LaplacianPyramid<float2>::InternalFirstLevel,
      LaplacianPyramid<float2>::SingleShot, 2, 1, false);
  FAIL_RETURN(potflowLaplacianPyramid.status());
  flowLaplacianPyramid.reset(potflowLaplacianPyramid.release());

  const Rect interRect1 = mergerPair->getBoundingInterRect(1, 0);
  auto potExtraFlowLaplacianPyramid = LaplacianPyramid<float2>::create(
      std::string("extraFlow"), interRect1.getWidth(), interRect1.getHeight(), numLevels,
      LaplacianPyramid<float2>::InternalFirstLevel, LaplacianPyramid<float2>::SingleShot, 2, 1, false);
  FAIL_RETURN(potExtraFlowLaplacianPyramid.status());
  extrapolatedFlowLaplacianPyramid.reset(potExtraFlowLaplacianPyramid.release());
  extrapolatedImage1.alloc(interRect1.getArea(), "Optical Flow Pair");
  // Set the extrapolated flow rect equals to the intermediate rect 1.
  // Smaller size can be considered for better performance but I strongly recommend "no"
  extrapolatedFlowRects = mergerPair->getBoundingInterRect1s();
  FAIL_RETURN(allocMemory());
  return Status::OK();
}

Status ImageFlow::allocMemory() {
  const int maxSize0 = (int)(mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(0).width() *
                             mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(0)->getLevel(0).height());
  const int maxSize1 = (int)(mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(1)->getLevel(0).width() *
                             mergerPair->getInterToInputSpaceCoordMappingLaplacianPyramid(1)->getLevel(0).height());

  FAIL_RETURN(image0.alloc(maxSize0, "Optical Flow Pair"));
  FAIL_RETURN(image1.alloc(maxSize1, "Optical Flow Pair"));
  FAIL_RETURN(finalFlow.alloc(maxSize0, "Optical Flow Pair"));
  return Status::OK();
}

Status ImageFlow::findMultiScaleImageFlow(const frameid_t, const int, const int2&, const GPU::Buffer<const uint32_t>&,
                                          const int2&, const GPU::Buffer<const uint32_t>&, GPU::Stream) {
  return Status::OK();
}

Status ImageFlow::cacheFlowSequence(const frameid_t, const int, const int2&, const GPU::Buffer<const uint32_t>&,
                                    const int2&, const GPU::Buffer<const uint32_t>&, GPU::Stream) const {
  return {Origin::ImageFlow, ErrType::ImplementationError, "Implementation Error"};
}

Status ImageFlow::findMultiScaleImageFlow(const frameid_t, const int, const int2&, const GPU::Buffer<const uint32_t>&,
                                          const int2&, const GPU::Buffer<const uint32_t>&, GPU::Buffer<float2>,
                                          GPU::Stream) {
  return Status::OK();
}

Status ImageFlow::findTemporalCoherentFlow(const frameid_t, const int2&, GPU::Buffer<float2>, GPU::Stream) {
  return Status::OK();
}

const MergerPair* ImageFlow::getMergerPair() const { return mergerPair.get(); }

const GPU::Buffer<const float2> ImageFlow::getFinalFlowBuffer() const { return finalFlow.borrow_const(); }

const GPU::Buffer<const float2> ImageFlow::getFinalExtrapolatedFlowBuffer() const {
  return extrapolatedFlowLaplacianPyramid->getLevel(0).data();
}

Rect ImageFlow::getExtrapolatedFlowRect(const int level) const { return extrapolatedFlowRects[level]; }

#ifndef NDEBUG
Status ImageFlow::dumpDebugImages(const int, const int, const GPU::Buffer<const uint32_t>&, const int, const int,
                                  const GPU::Buffer<const uint32_t>&, GPU::Stream) const {
  return Status::OK();
}
#endif

}  // namespace Core
}  // namespace VideoStitch
