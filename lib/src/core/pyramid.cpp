// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pyramid.hpp"

#include "gpu/memcpy.hpp"
#include "gpu/image/blur.hpp"
#include "gpu/image/imageOps.hpp"
#include "gpu/image/sampling.hpp"
#include "gpu/buffer.hpp"

#include "libvideostitch/status.hpp"
#include <memory>

//#define PYRAMID_DEBUG

//#define PYRAMID_BLUR
//#define PYRAMID_MULTIBAND
//#define PYRAMID_COLLAPSE
//#define PYRAMID_UP
//#define PYRAMID_ALPHA

#ifdef PYRAMID_DEBUG
#define PYRAMID_BLUR
#define PYRAMID_MULTIBAND
#define PYRAMID_COLLAPSE
#define PYRAMID_UP
#define PYRAMID_ALPHA
#endif

#if defined PYRAMID_BLUR || defined PYRAMID_COLLAPSE || defined PYRAMID_MULTIBAND || defined PYRAMID_ALPHA
#include "util/debugUtils.hpp"
#include <sstream>
#endif

namespace VideoStitch {
namespace Core {

template <typename T>
Potential<LaplacianPyramid<T>> LaplacianPyramid<T>::create(std::string name, int64_t width, int64_t height,
                                                           int numLevels, LevelLocation levelLocation,
                                                           Reconstruction reconstruction, int gaussianRadius,
                                                           int filterPasses, bool wrap) {
  std::unique_ptr<LaplacianPyramid<T>> pyr(new LaplacianPyramid<T>(name, computeBufferSize(width, height, numLevels),
                                                                   levelLocation, reconstruction, gaussianRadius,
                                                                   filterPasses, wrap));
  FAIL_RETURN(pyr->init(width, height, numLevels));
  return Potential<LaplacianPyramid<T>>(pyr.release());
}

template <typename T>
Status LaplacianPyramid<T>::init(int64_t width, int64_t height, int numLevels) {
  const int64_t alignment = 256;  // XXX TODO FIXME

  // Compute the total needed buffer size:
  switch (levelLocation) {
    case ExternalFirstLevel: {
      int64_t levelSize = width * height;
      if (levelSize * sizeof(T) % alignment != 0) {
        levelSize = ((levelSize * sizeof(T) / alignment + 1) * alignment) / sizeof(T);
      }
      devBufferSizeInPixels = bufferSizeInPixels - levelSize;
      break;
    }
    case InternalFirstLevel:
      devBufferSizeInPixels = bufferSizeInPixels;
      break;
    default:
      return {Origin::Stitcher, ErrType::ImplementationError, "Invalid pyramid level"};
  }

  // Allocate temp memory
  FAIL_RETURN(devTmp.alloc((size_t)(width * height), std::string("LaplacianPyramid-" + name).c_str()));
  FAIL_RETURN(devTmp2.alloc((size_t)(width * height), std::string("LaplacianPyramid-" + name).c_str()));

  // Allocate internal pyramid memory "only" when needed
  if (devBufferSizeInPixels > 0) {
    FAIL_RETURN(pyramid.alloc(devBufferSizeInPixels, std::string("LaplacianPyramid-" + name).c_str()));
  }

  // Allocate reconstruction pyramid memory only when not in place
  if (reconstruction == Multiple) {
    FAIL_RETURN(reconstructedPyramid.alloc(devBufferSizeInPixels, std::string("LaplacianPyramid-" + name).c_str()));
  }

  // Fetch memory into the appropriate LevelSpec
  int64_t levelSize;
  int64_t pyramidOffset = 0, reconstructedPyramidOffset = 0;

  levelSize = width * height;
  if (levelSize * sizeof(T) % alignment != 0) {
    levelSize = ((levelSize * sizeof(T) / alignment + 1) * alignment) / sizeof(T);
  }

  switch (levelLocation) {
    case ExternalFirstLevel: {
      levels.push_back(LevelSpec<T>(width, height, GPU::Buffer<T>()));
      reconstructedLevels.push_back(LevelSpec<T>(width, height, GPU::Buffer<T>()));
    } break;
    case InternalFirstLevel: {
      switch (reconstruction) {
        case SingleShot:
          levels.push_back(LevelSpec<T>(width, height, pyramid.borrow()));
          reconstructedLevels.push_back(LevelSpec<T>(width, height, pyramid.borrow()));
          pyramidOffset += levelSize;
          break;
        case Multiple:
          levels.push_back(LevelSpec<T>(width, height, pyramid.borrow()));
          pyramidOffset += levelSize;
          reconstructedLevels.push_back(LevelSpec<T>(width, height, reconstructedPyramid.borrow()));
          reconstructedPyramidOffset += levelSize;
          break;
        default:
          return {Origin::Stitcher, ErrType::ImplementationError, "Invalid pyramid reconstruction mode"};
      }
    } break;
    default:
      return {Origin::Stitcher, ErrType::ImplementationError, "Invalid pyramid level"};
  }

  // Compute buffer offsets.
  int64_t lWidth = (width + 1) / 2;
  int64_t lHeight = (height + 1) / 2;

  for (int level = 1; level < numLevels; ++level) {
    levels.push_back(LevelSpec<T>(lWidth, lHeight, pyramid.borrow().createSubBuffer(pyramidOffset)));
    switch (reconstruction) {
      case SingleShot:
        reconstructedLevels.push_back(LevelSpec<T>(lWidth, lHeight, pyramid.borrow().createSubBuffer(pyramidOffset)));
        break;
      case Multiple:
        reconstructedLevels.push_back(
            LevelSpec<T>(lWidth, lHeight, reconstructedPyramid.borrow().createSubBuffer(reconstructedPyramidOffset)));
        break;
      default:
        return {Origin::Stitcher, ErrType::ImplementationError, "Invalid pyramid reconstruction mode"};
    }

    levelSize = lWidth * lHeight;
    if (levelSize * sizeof(T) % alignment != 0) {
      levelSize = ((levelSize * sizeof(T) / alignment + 1) * alignment) / sizeof(T);
    }

    pyramidOffset += levelSize;
    reconstructedPyramidOffset += levelSize;

    lWidth = (lWidth + 1) / 2;
    lHeight = (lHeight + 1) / 2;
  }

  // Base level:
  if (numLevels > 0) {
    levels.push_back(LevelSpec<T>(lWidth, lHeight, pyramid.borrow().createSubBuffer(pyramidOffset)));
    switch (reconstruction) {
      case SingleShot:
        reconstructedLevels.push_back(LevelSpec<T>(lWidth, lHeight, pyramid.borrow().createSubBuffer(pyramidOffset)));
        break;
      case Multiple:
        reconstructedLevels.push_back(
            LevelSpec<T>(lWidth, lHeight, reconstructedPyramid.borrow().createSubBuffer(reconstructedPyramidOffset)));
        break;
      default:
        return {Origin::Stitcher, ErrType::ImplementationError, "Invalid pyramid reconstruction mode"};
    }
  }
  return Status::OK();
}

template <typename T>
LaplacianPyramid<T>::LaplacianPyramid(std::string name, int64_t bufferSizeInPixels, LevelLocation levelLocation,
                                      Reconstruction reconstruction, int gaussianRadius, int filterPasses, bool wrap)
    : name(name),
      wrap(wrap),
      bufferSizeInPixels(bufferSizeInPixels),
      levelLocation(levelLocation),
      reconstruction(reconstruction),
      devBufferSizeInPixels(0),
      gaussianRadius(gaussianRadius),
      filterPasses(filterPasses) {}

template <typename T>
int64_t LaplacianPyramid<T>::computeBufferSize(int64_t width, int64_t height, int numLevels) {
  int64_t result = 0;
  const int64_t alignment = 256;  // XXX TODO FIXME
  for (int level = 0; level <= numLevels; ++level) {
    int64_t levelSize = width * height;
    if (levelSize * sizeof(T) % alignment != 0) {
      levelSize = ((levelSize * sizeof(T) / alignment + 1) * alignment) / sizeof(T);
    }
    result += levelSize;

    width = (width + 1) / 2;
    height = (height + 1) / 2;
  }
  return result;
}

template <>
Status LaplacianPyramid<uint32_t>::computeGaussian(GPU::Stream stream) {
  assert(levels[0].data().wasAllocated());
  // Gaussian pyramid
  for (int level = 0; level < numLevels(); ++level) {
    const LevelSpec<uint32_t>& curLevel = levels[level];
    // Blur, store in devTmp
    FAIL_RETURN(Image::gaussianBlur2DRGBA(devTmp.borrow(), curLevel.data(), devTmp2.borrow(), curLevel.width(),
                                          curLevel.height(), gaussianRadius, filterPasses, wrap, stream));

    // Subsample, store in next level.
    LevelSpec<uint32_t>& nextLevel = levels[level + 1];
    FAIL_RETURN(Image::subsample22RGBA(nextLevel.data(), devTmp.borrow(), (unsigned)curLevel.width(),
                                       (unsigned)curLevel.height(), stream));
  }

#ifdef PYRAMID_BLUR
  stream.synchronize();
  for (int level = 1; level <= numLevels(); ++level) {
    std::stringstream ss;
    ss << "testBlurLevel-" << name << "-" << level << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), levels[level].data(), levels[level].width(), levels[level].height());
  }
#endif

  return Status::OK();
}

template <>
Status LaplacianPyramid<unsigned char>::computeGaussian(GPU::Stream stream) {
  // Gaussian pyramid
  for (int level = 0; level < numLevels(); ++level) {
    const LevelSpec<unsigned char>& curLevel = levels[level];
    // Blur, store in devTmp
    FAIL_RETURN(Image::gaussianBlur2D(devTmp.borrow(), curLevel.data(), devTmp2.borrow(), curLevel.width(),
                                      curLevel.height(), gaussianRadius, filterPasses, wrap, stream));
    // Subsample, store in next level.
    LevelSpec<unsigned char>& nextLevel = levels[level + 1];
    FAIL_RETURN(Image::subsample22(nextLevel.data(), devTmp.borrow_const(), (unsigned)curLevel.width(),
                                   (unsigned)curLevel.height(), stream));
  }

#ifdef PYRAMID_ALPHA
  stream.synchronize();
  for (int level = 1; level <= numLevels(); ++level) {
    std::stringstream ss;
    ss << "testBlurLevel-" << name << "-" << level << ".png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), levels[level].data().as<const unsigned char>(),
                                                     levels[level].width(), levels[level].height());
  }
#endif

  return Status::OK();
}

template <typename T>
Status LaplacianPyramid<T>::computeGaussian(GPU::Stream stream) {
  // Gaussian pyramid
  for (int level = 0; level < numLevels(); ++level) {
    const LevelSpec<T>& curLevel = levels[level];
    // Blur, store in devTmp
    FAIL_RETURN(Image::gaussianBlur2D(devTmp.borrow(), curLevel.data(), devTmp2.borrow(), curLevel.width(),
                                      curLevel.height(), gaussianRadius, filterPasses, wrap, stream));
    // Subsample, store in next level.
    LevelSpec<T>& nextLevel = levels[level + 1];
    FAIL_RETURN(Image::subsample22(nextLevel.data(), devTmp.borrow_const(), (unsigned)curLevel.width(),
                                   (unsigned)curLevel.height(), stream));
  }

  return Status::OK();
}

template <>
Status LaplacianPyramid<uint32_t>::compute(GPU::Stream stream) {
  assert(levels[0].data().wasAllocated());
  computeGaussian(stream);
  // Laplacian pyramid
  for (int level = 0; level < numLevels(); ++level) {
    // Upsample next level, store in devTmp
    // Subtract with current level

    LevelSpec<uint32_t>& curLevel = levels[level];
    const LevelSpec<uint32_t>& nextLevel = levels[level + 1];
    FAIL_RETURN(Image::upsample22RGBA(devTmp.borrow(), nextLevel.data(), (unsigned)curLevel.width(),
                                      (unsigned)curLevel.height(), wrap, stream));
#ifdef PYRAMID_UP
    stream.synchronize();
    std::stringstream ss;
    ss << "testUpLevel-" << name << "-" << level << ".png";
    Debug::dumpRGBADeviceBuffer(ss.str().c_str(), devTmp.borrow(), (unsigned)curLevel.width(),
                                (unsigned)curLevel.height());
#endif
    Image::subtract(curLevel.data(), devTmp.borrow(), curLevel.width() * curLevel.height(), stream);
  }

#ifdef PYRAMID_MULTIBAND
  stream.synchronize();
  for (int level = 0; level < numLevels(); ++level) {
    std::stringstream ss;
    ss << "testBandLevel-" << name << "-" << level << ".png";
    Debug::dumpRGB210DeviceBuffer(ss.str().c_str(), levels[level].data(), levels[level].width(),
                                  levels[level].height());
  }
  std::stringstream ss;
  ss << "testBandLevel-" << name << "-" << numLevels() << ".png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), levels[numLevels()].data(), levels[numLevels()].width(),
                              levels[numLevels()].height());
#endif
  return Status::OK();
}

template <typename T>
Status LaplacianPyramid<T>::compute(GPU::Stream stream) {
  assert(levels[0].data().wasAllocated());
  computeGaussian(stream);

  // Laplacian pyramid
  for (int level = 0; level < numLevels(); ++level) {
    LevelSpec<T>& curLevel = levels[level];
    // Upsample next level, store in devTmp:
    const LevelSpec<T>& nextLevel = levels[level + 1];
    FAIL_RETURN(Image::upsample22(devTmp.borrow(), nextLevel.data(), (unsigned)curLevel.width(),
                                  (unsigned)curLevel.height(), wrap, stream));

    // Subtract with current level:
    FAIL_RETURN(
        Image::subtractRaw(curLevel.data(), devTmp.borrow_const(), curLevel.width() * curLevel.height(), stream));
  }
  return Status::OK();
}

template <typename T>
Status LaplacianPyramid<T>::compute(GPU::Buffer<const T> src, GPU::Stream stream) {
  assert(levelLocation == InternalFirstLevel);
  // Copy src to first level:
  PROPAGATE_FAILURE_STATUS(
      GPU::memcpyAsync(levels[0].data(), src, (size_t)(levels[0].width() * levels[0].height() * sizeof(T)), stream));
  return compute(stream);
}

template <>
Status LaplacianPyramid<uint32_t>::collapse(bool final, GPU::Stream stream) {
  assert(levels[0].data().wasAllocated());

#ifdef PYRAMID_COLLAPSE
  stream.synchronize();
  for (int level = 0; level < numLevels(); ++level) {
    std::stringstream ss3;
    ss3 << "testCollapsePre-" << name << "-" << level << ".png";
    Debug::dumpRGB210DeviceBuffer(ss3.str().c_str(), levels[level].data(), levels[level].width(),
                                  levels[level].height());
  }
  std::stringstream ss4;
  ss4 << "testCollapsePre-" << name << "-" << numLevels() << ".png";
  Debug::dumpRGBADeviceBuffer(ss4.str().c_str(), levels[numLevels()].data(), levels[numLevels()].width(),
                              levels[numLevels()].height());
#endif

  std::vector<LevelSpec<uint32_t>>& lvls = final ? levels : reconstructedLevels;

  if (reconstruction == Multiple && !final) {
    for (int level = 0; level <= numLevels(); ++level) {
      GPU::memcpyAsync(
          reconstructedLevels[level].data(), levels[level].data().as_const(),
          (size_t)(reconstructedLevels[level].width() * reconstructedLevels[level].height() * sizeof(uint32_t)),
          stream);
    }
  }

  for (int level = numLevels() - 1; level >= 0; --level) {
    LevelSpec<uint32_t>& curLevel = lvls[level];
    const LevelSpec<uint32_t>& nextLevel = lvls[level + 1];

    // Upsample lower level, store in devTmp
    // Add with current level

    if (level == numLevels() - 1) {
      // coarsest level is RGBA
      Image::upsample22RGBA(devTmp.borrow(), nextLevel.data(), curLevel.width(), curLevel.height(), wrap, stream);
    } else {
      Image::upsample22RGBA210(devTmp.borrow(), nextLevel.data(), curLevel.width(), curLevel.height(), wrap, stream);
    }

    if (level == numLevels() - 1) {
      // coarsest level is RGBA
      if (level == 0) {
        // finest level to RGBA
        Image::add10n8Clamp(curLevel.data(), devTmp.borrow(), curLevel.width() * curLevel.height(), stream);
      } else {
        Image::add10n8(curLevel.data(), devTmp.borrow(), curLevel.width() * curLevel.height(), stream);
      }
    } else if (level == 0) {
      // finest level to RGBA
      Image::addClamp(curLevel.data(), devTmp.borrow(), curLevel.width() * curLevel.height(), stream);
    } else {
      Image::add10(curLevel.data(), devTmp.borrow(), curLevel.width() * curLevel.height(), stream);
    }
  }

#ifdef PYRAMID_COLLAPSE
  stream.synchronize();
  std::stringstream ss;
  ss << "testCollapse-" << name << "-0.png";
  Debug::dumpRGBADeviceBuffer(ss.str().c_str(), lvls[0].data(), lvls[0].width(), lvls[0].height());
  for (int level = 1; level < numLevels(); ++level) {
    std::stringstream ss;
    ss << "testCollapse-" << name << "-" << level << ".png";
    Debug::dumpRGB210DeviceBuffer(ss.str().c_str(), lvls[level].data(), lvls[level].width(), lvls[level].height());
  }
  std::stringstream ss1;
  ss1 << "testCollapse-" << name << "-" << numLevels() << ".png";
  Debug::dumpRGBADeviceBuffer(ss1.str().c_str(), lvls[numLevels()].data(), lvls[numLevels()].width(),
                              lvls[numLevels()].height());
#endif

  return Status::OK();
}

template <typename T>
Status LaplacianPyramid<T>::collapse(bool final, GPU::Stream stream) {
  assert(levels[0].data().wasAllocated());

  std::vector<LevelSpec<T>>& lvls = final ? levels : reconstructedLevels;

  for (int level = numLevels() - 1; level >= 0; --level) {
    LevelSpec<T>& curLevel = lvls[level];
    // Upsample next level, store in devTmp:
    const LevelSpec<T>& nextLevel = levels[level + 1];
    FAIL_RETURN(
        Image::upsample22(devTmp.borrow(), nextLevel.data(), curLevel.width(), curLevel.height(), wrap, stream));
    // Add with current level:
    FAIL_RETURN(Image::addRaw(curLevel.data(), devTmp.borrow_const(), curLevel.width() * curLevel.height(), stream));
  }
  return Status::OK();
}

template <typename T>
void LaplacianPyramid<T>::start(GPU::Buffer<T> result, GPU::Buffer<T> reconstruct, GPU::Stream stream) {
  if (levelLocation == ExternalFirstLevel) {
    if (reconstruction == Multiple) {
      reconstructedLevels[0].setDataBuffer(reconstruct);
    } else {
      reconstructedLevels[0].setDataBuffer(result);
    }
    levels[0].setDataBuffer(result);
  }
  GPU::memsetToZeroAsync(pyramid.borrow(), devBufferSizeInPixels * sizeof(T), stream);
}

template class LaplacianPyramid<uint32_t>;
template class LaplacianPyramid<unsigned char>;
template class LaplacianPyramid<float2>;
}  // namespace Core
}  // namespace VideoStitch
