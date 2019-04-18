// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "imageProcessingGPUUtils.hpp"
#include "core1/imageMerger.hpp"
#include "core1/imageMapping.hpp"
#include "core1/bounds.hpp"
#include "gpu/image/sampling.hpp"

//#define MERGER_PAIR_DEBUG

#if defined(MERGER_PAIR_DEBUG)
#include "util/debugUtils.hpp"
#include <sstream>
#endif

namespace VideoStitch {
namespace Util {

template <>
Status ImageProcessingGPU::subsampleImage<uint32_t>(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> src,
                                                    std::size_t srcWidth, std::size_t srcHeight, GPU::Stream gpuStream,
                                                    const bool isNearest) {
  if (isNearest) {
    assert(false);
    /*
    return VideoStitch::Image::subsample22Nearest<uint32_t>(
      dst, src, srcWidth, srcHeight,
      Core::ImageMerger::CudaBlockSize, gpuStream);
      */

    return {Origin::Stitcher, ErrType::ImplementationError, "Unsupported subsampling mode"};
  } else {
    return VideoStitch::Image::subsample22RGBA(dst, src, srcWidth, srcHeight, gpuStream);
  }
}

template <>
Status ImageProcessingGPU::subsampleImage<unsigned char>(GPU::Buffer<unsigned char> dst,
                                                         GPU::Buffer<const unsigned char> src, std::size_t srcWidth,
                                                         std::size_t srcHeight, GPU::Stream gpuStream,
                                                         const bool isNearest) {
  if (isNearest) {
    assert(false);
    /*
    return VideoStitch::Image::subsample22Nearest<unsigned char>(
      dst, src, srcWidth, srcHeight,
      Core::ImageMerger::CudaBlockSize, gpuStream);
      */
    return {Origin::Stitcher, ErrType::ImplementationError, "Unsupported subsampling mode"};
  } else {
    return VideoStitch::Image::subsample22<unsigned char>(dst, src, srcWidth, srcHeight, gpuStream);
  }
}

template <typename T>
Status ImageProcessingGPU::downSampleImages(const int levelCount, int& bufferWidth, int& bufferHeight,
                                            GPU::Buffer<T> buffer, GPU::Stream stream, const bool isNearest) {
  GPU::UniqueBuffer<T> tmpBuffer;
  FAIL_RETURN(tmpBuffer.alloc(bufferWidth * bufferHeight, "ImageProcessingGPU"));
  int width = bufferWidth;
  int height = bufferHeight;
  for (int t = 0; t < levelCount; t++) {
    // Down-sample the image as well
    if (t % 2 == 0) {
      FAIL_RETURN(subsampleImage<T>(tmpBuffer.borrow(), buffer.as_const(), width, height, stream, isNearest));
    } else {
      FAIL_RETURN(subsampleImage<T>(buffer, tmpBuffer.borrow_const(), width, height, stream, isNearest));
    }
    const int dstWidth = (width + 1) / 2;
    const int dstHeight = (height + 1) / 2;
    width = dstWidth;
    height = dstHeight;
  }
  if (levelCount % 2 == 1) {
    FAIL_RETURN(GPU::memcpyBlocking<T>(buffer, tmpBuffer.borrow_const(), width * height * sizeof(T)));
  }
  bufferWidth = width;
  bufferHeight = height;
  return stream.synchronize();
}

template Status ImageProcessingGPU::downSampleImages<uint32_t>(const int levelCount, int& bufferWidth,
                                                               int& bufferHeight, GPU::Buffer<uint32_t> buffer,
                                                               GPU::Stream stream, const bool isNearest);

template Status ImageProcessingGPU::downSampleImages<unsigned char>(const int levelCount, int& bufferWidth,
                                                                    int& bufferHeight,
                                                                    GPU::Buffer<unsigned char> buffer,
                                                                    GPU::Stream stream, const bool isNearest);

Status ImageProcessingGPU::findBBox(Core::TextureTarget t, const bool canWarp, const Core::StereoRigDefinition* rigDef,
                                    const int width, const int height, const GPU::Buffer<const uint32_t>& maskBuffer,
                                    Core::Rect& boundingRect, GPU::Stream gpuStream) {
  // Now find the bounding rect
  GPU::UniqueBuffer<uint32_t> tmpBinarizedMaskBuffer;
  GPU::UniqueBuffer<uint32_t> tmpDevBuffer;
  FAIL_RETURN(tmpDevBuffer.alloc(size_t(std::max(width, height)), "Image Processing GPU"));
  FAIL_RETURN(tmpBinarizedMaskBuffer.alloc(width * height, "Image Processing GPU"));

  FAIL_RETURN(binarizeMask(make_int2(width, height), maskBuffer, tmpBinarizedMaskBuffer.borrow(), gpuStream));

  auto tmpHostBuffer =
      GPU::HostBuffer<uint32_t>::allocate(size_t(std::max(width, height) * sizeof(uint32_t)), "Image Processing GPU");
  FAIL_RETURN(tmpHostBuffer.status());

  std::map<readerid_t, Core::ImageMapping*> imageMappings;
  imageMappings[0] = new Core::ImageMapping(0);
  // Compute H-Bound
  FAIL_RETURN(Core::computeHBounds(t, width, height, imageMappings, rigDef, Eye::LeftEye,
                                   tmpBinarizedMaskBuffer.borrow_const(), tmpHostBuffer.value(), tmpDevBuffer.borrow(),
                                   gpuStream, canWarp));
  // Compute V-Bound
  FAIL_RETURN(Core::computeVBounds(t, width, height, imageMappings, tmpBinarizedMaskBuffer.borrow_const(),
                                   tmpHostBuffer.value(), tmpDevBuffer.borrow(), gpuStream));

  FAIL_RETURN(tmpHostBuffer.value().release());

  boundingRect = imageMappings[0]->getOutputRect(t);
  delete imageMappings[0];
  imageMappings.clear();
  return Status::OK();
}

Status ImageProcessingGPU::downSampleCoordImage(const int inputWidth, const int inputHeight, const int levelCount,
                                                GPU::UniqueBuffer<float2>& coordBuffer,
                                                GPU::UniqueBuffer<uint32_t>& weightBuffer, GPU::Stream stream) {
  GPU::UniqueBuffer<float2> tmpCoordBuffer;
  FAIL_RETURN(tmpCoordBuffer.alloc(inputWidth * inputHeight, "Image Processing GPU"));

  GPU::UniqueBuffer<uint32_t> tmpWeightBuffer;
  FAIL_RETURN(tmpWeightBuffer.alloc(inputWidth * inputHeight, "Image Processing GPU"));

  int width = inputWidth;
  int height = inputHeight;
  for (int t = 0; t < levelCount; t++) {
    // Down-sample the image as well
    if (t % 2 == 0) {
      FAIL_RETURN(VideoStitch::Image::subsample22Mask<float2>(tmpCoordBuffer.borrow(), tmpWeightBuffer.borrow(),
                                                              coordBuffer.borrow_const(), weightBuffer.borrow_const(),
                                                              width, height, Core::ImageMerger::CudaBlockSize, stream));
    } else {
      FAIL_RETURN(VideoStitch::Image::subsample22Mask<float2>(
          coordBuffer.borrow(), weightBuffer.borrow(), tmpCoordBuffer.borrow_const(), tmpWeightBuffer.borrow_const(),
          width, height, Core::ImageMerger::CudaBlockSize, stream));
    }
    const int dstWidth = (width + 1) / 2;
    const int dstHeight = (height + 1) / 2;
    width = dstWidth;
    height = dstHeight;
  }
  if (levelCount % 2 == 1) {
    FAIL_RETURN(
        GPU::memcpyBlocking(coordBuffer.borrow(), tmpCoordBuffer.borrow_const(), width * height * sizeof(float2)));
    FAIL_RETURN(
        GPU::memcpyBlocking(weightBuffer.borrow(), tmpWeightBuffer.borrow_const(), width * height * sizeof(uint32_t)));
  }
  return stream.synchronize();
}

Status ImageProcessingGPU::computeTightOverlappingRect(Core::TextureTarget t, const int warpWidth,
                                                       const Core::Rect& boundingRect0,
                                                       const GPU::Buffer<const uint32_t>& buffer0,
                                                       const Core::Rect& boundingRect1,
                                                       const GPU::Buffer<const uint32_t>& buffer1,
                                                       Core::Rect& overlappingRect, GPU::Stream stream) {
  Core::Rect iRect, uRect;

  if (boundingRect0.right() < warpWidth) {
    uRect = boundingRect0;
  } else if (boundingRect1.right() < warpWidth) {
    uRect = boundingRect1;
  } else {
    Core::Rect::getInterAndUnion(boundingRect0, boundingRect1, iRect, uRect, warpWidth);
    uRect.setLeft(0);
    uRect.setRight(warpWidth - 1);
  }
  GPU::UniqueBuffer<uint32_t> maskBuffer;

  FAIL_RETURN(maskBuffer.alloc(uRect.getArea(), "Image Processing GPU"));
  FAIL_RETURN(onBothBufferOperator(warpWidth, boundingRect0, buffer0, boundingRect1, buffer1, uRect,
                                   maskBuffer.borrow(), stream));
  FAIL_RETURN(stream.synchronize());

#ifdef MERGER_PAIR_DEBUG
  static videoreaderid_t id = 0;
  std::stringstream ss;
  ss.str("");
  ss << "andMaskOut-" << id << "_1.png";
  Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), buffer1, boundingRect1.getWidth(), boundingRect1.getHeight());
  id++;
#endif

  FAIL_RETURN(findBBox(t, false, nullptr, (int)uRect.getWidth(), (int)uRect.getHeight(), maskBuffer.borrow_const(),
                       overlappingRect, stream));

  overlappingRect.setTop(uRect.top() + overlappingRect.top());
  overlappingRect.setBottom(uRect.top() + overlappingRect.bottom());
  overlappingRect.setLeft(uRect.left() + overlappingRect.left());
  overlappingRect.setRight(uRect.left() + overlappingRect.right());
  return Status::OK();
}

}  // namespace Util
}  // namespace VideoStitch
