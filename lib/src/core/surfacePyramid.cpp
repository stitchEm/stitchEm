// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "surfacePyramid.hpp"

#include "gpu/allocator.hpp"
#include "gpu/image/downsampler.hpp"
#include "gpu/stream.hpp"
#include "gpu/surface.hpp"

#include "common/container.hpp"

namespace VideoStitch {
namespace Core {

template <PyramidType surfType>
SurfacePyramid<surfType>::SurfacePyramid(int numLevels, int fullWidth, int fullHeight)
    : fullWidth(fullWidth), fullHeight(fullHeight), original(nullptr) {
  // allocate all levels
  int64_t levelWidth = fullWidth;
  int64_t levelHeight = fullHeight;
  int64_t scale = 1;
  for (int level = 1; level < numLevels; level++) {
    assert(levelWidth % 2 == 0 && levelHeight % 2 == 0);
    scale *= 2;
    levelWidth = fullWidth / scale;
    levelHeight = fullHeight / scale;
    auto potSurf = createSurface((int)levelWidth, (int)levelHeight);

    // TODO
    assert(potSurf.ok());

    levelSpecs.push_back(spec(potSurf.object(), levelWidth, levelHeight, scale));
    downscaledData.push_back(potSurf.release());
  }
}

template <PyramidType surfType>
GPU::Surface& SurfacePyramid<surfType>::LevelSpec::gpuSurf() const {
  return *_surf->pimpl->surface;
}

template <PyramidType surfType>
SurfacePyramid<surfType>::~SurfacePyramid() {
  deleteAll(downscaledData);
}

template <PyramidType surfType>
size_t SurfacePyramid<surfType>::numLevels() const {
  return downscaledData.size() + 1;
}

template <PyramidType surfType>
auto SurfacePyramid<surfType>::getLevel(size_t level) const -> LevelSpec {
  assert(level < numLevels());
  if (level == 0) {
    return spec(original, fullWidth, fullHeight, 1);
  } else {
    return levelSpecs[level - 1];
  }
}

template <>
Potential<Core::SourceSurface> SurfacePyramid<PyramidType::Source>::createSurface(int width, int height) {
  return Core::OffscreenAllocator::createSourceSurface(width, height, "SurfacePyramid");
}

template <>
Potential<Core::SourceSurface> SurfacePyramid<PyramidType::Depth>::createSurface(int width, int height) {
  return Core::OffscreenAllocator::createDepthSurface(width, height, "SurfacePyramid");
}

template class SurfacePyramid<PyramidType::Source>;
template class SurfacePyramid<PyramidType::Depth>;

Status InputPyramid::compute(Core::SourceSurface* fullResolution, GPU::Stream& stream) {
  original = fullResolution;
  for (size_t level = 1; level < numLevels(); level++) {
    LevelSpec currentLevel = getLevel(level - 1);
    LevelSpec nextLevel = getLevel(level);
    FAIL_RETURN(Image::downsampleRGBASurf2x(nextLevel.gpuSurf(), currentLevel.gpuSurf(), (unsigned)nextLevel.width(),
                                            (unsigned)nextLevel.height(), stream));
  }
  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
