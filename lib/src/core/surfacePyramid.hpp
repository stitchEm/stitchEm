// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

#include <cstddef>
#include <stdint.h>
#include <vector>

// TODO support downsampling from odd width/height (width+1)/2, (height+1)/2
// TODO replace bilinear interpolation with proper gaussian blur and subsampling

namespace VideoStitch {

namespace GPU {
class Stream;
class Surface;
}  // namespace GPU

namespace Core {

class SourceSurface;

enum class PyramidType { Source, Depth };

// TODO common base class with LaplacianPyramid?
template <PyramidType surfType>
class SurfacePyramid {
 public:
  class LevelSpec {
   public:
    /**
     * Returns the width of the level.
     */
    int64_t width() const { return _width; }
    /**
     * Returns the height of the level.
     */
    int64_t height() const { return _height; }
    /**
     * Returns the buffer for the level.
     */
    Core::SourceSurface* surf() const { return _surf; }

    GPU::Surface& gpuSurf() const;

    int64_t scale() const { return _scale; }

   private:
    LevelSpec(Core::SourceSurface* surf, int64_t width, int64_t height, int64_t scale)
        : _surf(surf), _width(width), _height(height), _scale(scale) {}
    Core::SourceSurface* _surf;
    const int64_t _width;
    const int64_t _height;
    const int64_t _scale;
    friend class SurfacePyramid;
  };

  SurfacePyramid(int numLevels, int fullWidth, int fullHeight);

  SurfacePyramid(const SurfacePyramid& other) = delete;
  SurfacePyramid& operator=(const SurfacePyramid& other) = delete;

#if (_MSC_VER && _MSC_VER < 1900)
  SurfacePyramid(SurfacePyramid&& other)
      : fullWidth(other.fullWidth),
        fullHeight(other.fullHeight),
        original(std::move(other.original)),
        downscaledData(std::move(other.downscaledData)),
        levelSpecs(std::move(other.levelSpecs)) {}
#else
  SurfacePyramid(SurfacePyramid&& other) = default;
  SurfacePyramid& operator=(SurfacePyramid&& other) = default;
#endif

  virtual ~SurfacePyramid();

  LevelSpec getLevel(size_t level) const;

  size_t numLevels() const;

 protected:
  Potential<Core::SourceSurface> createSurface(int width, int height);

  LevelSpec spec(Core::SourceSurface* surf, int64_t width, int64_t height, int64_t scale) const {
    return LevelSpec{surf, width, height, scale};
  }

  int fullWidth;
  int fullHeight;

  // owned by this pyramid
  std::vector<Core::SourceSurface*> downscaledData;

  // only references to the surfaces
  std::vector<LevelSpec> levelSpecs;

  // careful: not copied, not owned
  Core::SourceSurface* original;
};

class InputPyramid : public SurfacePyramid<PyramidType::Source> {
 public:
  InputPyramid(int numLevels, int fullWidth, int fullHeight) : SurfacePyramid(numLevels, fullWidth, fullHeight) {}

  Status compute(Core::SourceSurface* fullResolution, GPU::Stream& stream);

  InputPyramid(const InputPyramid& other) = delete;
  InputPyramid& operator=(const InputPyramid& other) = delete;

#if (_MSC_VER && _MSC_VER < 1900)
  InputPyramid(InputPyramid&& other) : SurfacePyramid(std::move(other)) {}
#else
  InputPyramid(InputPyramid&& other) = default;
  InputPyramid& operator=(InputPyramid&& other) = default;
#endif
};

class DepthPyramid : public SurfacePyramid<PyramidType::Depth> {
 public:
  DepthPyramid(int numLevels, int fullWidth, int fullHeight) : SurfacePyramid(numLevels, fullWidth, fullHeight) {}

#if (_MSC_VER && _MSC_VER < 1900)
  DepthPyramid(DepthPyramid&& other) : SurfacePyramid(std::move(other)) {}
#else
  DepthPyramid(DepthPyramid&& other) = default;
  DepthPyramid& operator=(DepthPyramid&& other) = default;
#endif
};

}  // namespace Core
}  // namespace VideoStitch
