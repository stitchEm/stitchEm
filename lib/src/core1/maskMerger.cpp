// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "maskMerger.hpp"

#include "imageMapping.hpp"
#include "voronoiMaskMerger.hpp"
#include "panoRemapper.hpp"

#include "gpu/image/imageOps.hpp"
#include "gpu/memcpy.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/panoDef.hpp"

#include <algorithm>

//#define DEBUGALPHA

#ifdef DEBUGALPHA
#ifndef _MSC_VER
static const std::string DEBUG_FOLDER = "/tmp/voronoi/";
#else
static const std::string DEBUG_FOLDER = "";
#endif
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "util/debugUtils.hpp"
#include <sstream>
#endif

namespace VideoStitch {
namespace Core {

MaskMerger::MaskMergerType MaskMerger::getDefaultMaskMerger() { return MaskMerger::MaskMergerType::VoronoiMask; }

MaskMerger* MaskMerger::factor(const MaskMergerType maskMergerType) {
  switch (maskMergerType) {
    case MaskMerger::MaskMergerType::VoronoiMask:
      return new VoronoiMaskMerger();
    default:
      return nullptr;
  }
}

MaskMerger::MaskMerger() {}

MaskMerger::~MaskMerger() {}

GPU::Buffer<unsigned char> MaskMerger::getAlpha(TextureTarget t) const {
  if (alpha[t]) {
    return alpha[t].borrow();
  } else {
    return GPU::Buffer<unsigned char>();
  }
}

namespace {
Status reproject(int panoWidth, int panoHeight, int faceLength, bool equiangular,
                 GPU::Buffer<const unsigned char> alpha, Rect rect, GPU::Buffer<unsigned char> posX, Rect posXrect,
                 GPU::Buffer<unsigned char> posY, Rect posYrect, GPU::Buffer<unsigned char> posZ, Rect posZrect,
                 GPU::Buffer<unsigned char> negX, Rect negXrect, GPU::Buffer<unsigned char> negY, Rect negYrect,
                 GPU::Buffer<unsigned char> negZ, Rect negZrect, GPU::Stream stream) {
  Potential<SourceSurface> potSurf =
      OffscreenAllocator::createAlphaSurface(rect.getWidth(), rect.getHeight(), "Equirectangular Alpha Layer");
  if (!potSurf.ok()) {
    return {Origin::GPU, ErrType::OutOfResources, "Can't allocate alpha channel"};
  }
  SourceSurface* alphaSurf = potSurf.release();

  GPU::memcpyAsync(*alphaSurf->pimpl->surface, alpha, stream);

  FAIL_RETURN(reprojectAlphaToCubemap(panoWidth, panoHeight, faceLength, *alphaSurf->pimpl->surface, rect, posX,
                                      posXrect, negX, negXrect, posY, posYrect, negY, negYrect, posZ, posZrect, negZ,
                                      negZrect, equiangular, stream));

  delete alphaSurf;

  return Status::OK();
}
}  // namespace

Status MaskMerger::setupMask(const PanoDefinition& pano, GPU::Buffer<const uint32_t> panoDevOut,
                             const ImageMapping& fromIm, const ImageMerger* const to, GPU::Stream stream) {
  return setup(pano, panoDevOut, fromIm, to, stream);
}

Status MaskMerger::setupMaskCubemap(const PanoDefinition& pano, GPU::Buffer<const uint32_t> panoDevOut,
                                    const ImageMapping& fromIm, const ImageMerger* const to, GPU::Stream stream) {
  FAIL_RETURN(setup(pano, panoDevOut, fromIm, to, stream));

  // reproject on every cubemap face
  if (!fromIm.getOutputRect(CUBE_MAP_POSITIVE_X).empty())
    FAIL_RETURN(alpha[CUBE_MAP_POSITIVE_X].recreate(fromIm.getOutputRect(CUBE_MAP_POSITIVE_X).getArea(),
                                                    "Voronoi Mask Merger"));
  if (!fromIm.getOutputRect(CUBE_MAP_NEGATIVE_X).empty())
    FAIL_RETURN(alpha[CUBE_MAP_NEGATIVE_X].recreate(fromIm.getOutputRect(CUBE_MAP_NEGATIVE_X).getArea(),
                                                    "Voronoi Mask Merger"));
  if (!fromIm.getOutputRect(CUBE_MAP_POSITIVE_Y).empty())
    FAIL_RETURN(alpha[CUBE_MAP_POSITIVE_Y].recreate(fromIm.getOutputRect(CUBE_MAP_POSITIVE_Y).getArea(),
                                                    "Voronoi Mask Merger"));
  if (!fromIm.getOutputRect(CUBE_MAP_NEGATIVE_Y).empty())
    FAIL_RETURN(alpha[CUBE_MAP_NEGATIVE_Y].recreate(fromIm.getOutputRect(CUBE_MAP_NEGATIVE_Y).getArea(),
                                                    "Voronoi Mask Merger"));
  if (!fromIm.getOutputRect(CUBE_MAP_POSITIVE_Z).empty())
    FAIL_RETURN(alpha[CUBE_MAP_POSITIVE_Z].recreate(fromIm.getOutputRect(CUBE_MAP_POSITIVE_Z).getArea(),
                                                    "Voronoi Mask Merger"));
  if (!fromIm.getOutputRect(CUBE_MAP_NEGATIVE_Z).empty())
    FAIL_RETURN(alpha[CUBE_MAP_NEGATIVE_Z].recreate(fromIm.getOutputRect(CUBE_MAP_NEGATIVE_Z).getArea(),
                                                    "Voronoi Mask Merger"));

  FAIL_RETURN(reproject((int)pano.getWidth(), (int)pano.getHeight(), (int)pano.getLength(),
                        pano.getProjection() == PanoProjection::EquiangularCubemap,
                        alpha[EQUIRECTANGULAR].borrow().as_const(), fromIm.getOutputRect(EQUIRECTANGULAR),
                        alpha[CUBE_MAP_POSITIVE_X].borrow(), fromIm.getOutputRect(CUBE_MAP_POSITIVE_X),
                        alpha[CUBE_MAP_POSITIVE_Y].borrow(), fromIm.getOutputRect(CUBE_MAP_POSITIVE_Y),
                        alpha[CUBE_MAP_POSITIVE_Z].borrow(), fromIm.getOutputRect(CUBE_MAP_POSITIVE_Z),
                        alpha[CUBE_MAP_NEGATIVE_X].borrow(), fromIm.getOutputRect(CUBE_MAP_NEGATIVE_X),
                        alpha[CUBE_MAP_NEGATIVE_Y].borrow(), fromIm.getOutputRect(CUBE_MAP_NEGATIVE_Y),
                        alpha[CUBE_MAP_NEGATIVE_Z].borrow(), fromIm.getOutputRect(CUBE_MAP_NEGATIVE_Z), stream));

#ifdef DEBUGALPHA
  {
    stream.synchronize();
    for (int t = CUBE_MAP_POSITIVE_X; t <= CUBE_MAP_NEGATIVE_Z; ++t) {
      TextureTarget target = (TextureTarget)t;
      if (!fromIm.getOutputRect(target).empty()) {
        std::stringstream ss;
        ss << DEBUG_FOLDER << "Voronoi Mask-" << fromIm.getImId() << "-" << toString(target) << ".png";
        Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str(), alpha[target].borrow(),
                                                         fromIm.getOutputRect(target).getWidth(),
                                                         fromIm.getOutputRect(target).getHeight());
      }
    }
  }
#endif

  return Status::OK();
}

Status MaskMerger::buildPyramidMask(const ImageMapping& fromIm, std::string name, const int numLevels,
                                    const int gaussianRadius, const int filterPasses, const bool warp,
                                    GPU::Stream stream) {
  if (!alpha[EQUIRECTANGULAR]) {
    return Status{Origin::Stitcher, ErrType::ImplementationError, "Mask merger not set up"};
  }

  // Construct mask and sharp mask
  Potential<LaplacianPyramid<unsigned char>> fMaskStatus = LaplacianPyramid<unsigned char>::create(
      std::string("alpha-equirectangular-") + name, fromIm.getOutputRect(EQUIRECTANGULAR).getWidth(),
      fromIm.getOutputRect(EQUIRECTANGULAR).getHeight(), numLevels, LaplacianPyramid<unsigned char>::ExternalFirstLevel,
      LaplacianPyramid<unsigned char>::SingleShot, gaussianRadius, filterPasses, warp);
  FAIL_RETURN(fMaskStatus.status());

  alphaPyramids[EQUIRECTANGULAR].reset(fMaskStatus.release());
  alphaPyramids[EQUIRECTANGULAR]->start(alpha[EQUIRECTANGULAR].borrow(), GPU::Buffer<unsigned char>(), stream);

  FAIL_RETURN(alphaPyramids[EQUIRECTANGULAR]->computeGaussian(stream));

  return Status::OK();
}

Status MaskMerger::buildPyramidMaskCubemap(const PanoDefinition& pano, const ImageMapping& fromIm, std::string name,
                                           const int numLevels, const int gaussianRadius, const int filterPasses,
                                           const bool warp, GPU::Stream stream) {
  // make a gaussian pyramid for the equirectangular alpha layer
  FAIL_RETURN(buildPyramidMask(fromIm, name, numLevels, gaussianRadius, filterPasses, warp, stream));

  // initiate a pyramid for each face with the reprojected equirectangular alpha layer
  for (int i = CUBE_MAP_POSITIVE_X; i <= CUBE_MAP_NEGATIVE_Z; ++i) {
    TextureTarget target = (TextureTarget)i;

    if (fromIm.getOutputRect(target).empty()) {
      continue;
    }

    Potential<LaplacianPyramid<unsigned char>> fMaskStatus = LaplacianPyramid<unsigned char>::create(
        std::string("alpha") + "-" + toString(target) + "-" + name, fromIm.getOutputRect(target).getWidth(),
        fromIm.getOutputRect(target).getHeight(), numLevels, LaplacianPyramid<unsigned char>::ExternalFirstLevel,
        LaplacianPyramid<unsigned char>::SingleShot, gaussianRadius, filterPasses, warp);
    FAIL_RETURN(fMaskStatus.status());

    alphaPyramids[target].reset(fMaskStatus.release());
    alphaPyramids[target]->start(alpha[target].borrow(), GPU::Buffer<unsigned char>(), stream);
  }

  // reproject every sublevel of the equirect pyramid on every cubemap face (ie. make the gaussian pyramid for each face
  // manually)
  Rect px = fromIm.getOutputRect(CUBE_MAP_POSITIVE_X);
  Rect py = fromIm.getOutputRect(CUBE_MAP_POSITIVE_Y);
  Rect pz = fromIm.getOutputRect(CUBE_MAP_POSITIVE_Z);
  Rect nx = fromIm.getOutputRect(CUBE_MAP_NEGATIVE_X);
  Rect ny = fromIm.getOutputRect(CUBE_MAP_NEGATIVE_Y);
  Rect nz = fromIm.getOutputRect(CUBE_MAP_NEGATIVE_Z);
  Rect eq = fromIm.getOutputRect(EQUIRECTANGULAR);

  int panoWidth = (int)pano.getWidth();
  int panoHeight = (int)pano.getHeight();
  int faceLength = (int)pano.getLength();

  for (int level = 1; level <= alphaPyramids[EQUIRECTANGULAR]->numLevels(); ++level) {
    if (!px.empty())
      px = Rect::fromInclusiveTopLeftBottomRight(px.top() / 2, px.left() / 2, px.bottom() / 2, px.right() / 2);
    if (!py.empty())
      py = Rect::fromInclusiveTopLeftBottomRight(py.top() / 2, py.left() / 2, py.bottom() / 2, py.right() / 2);
    if (!pz.empty())
      pz = Rect::fromInclusiveTopLeftBottomRight(pz.top() / 2, pz.left() / 2, pz.bottom() / 2, pz.right() / 2);
    if (!nx.empty())
      nx = Rect::fromInclusiveTopLeftBottomRight(nx.top() / 2, nx.left() / 2, nx.bottom() / 2, nx.right() / 2);
    if (!ny.empty())
      ny = Rect::fromInclusiveTopLeftBottomRight(ny.top() / 2, ny.left() / 2, ny.bottom() / 2, ny.right() / 2);
    if (!nz.empty())
      nz = Rect::fromInclusiveTopLeftBottomRight(nz.top() / 2, nz.left() / 2, nz.bottom() / 2, nz.right() / 2);
    if (!eq.empty())
      eq = Rect::fromInclusiveTopLeftBottomRight(eq.top() / 2, eq.left() / 2, eq.bottom() / 2, eq.right() / 2);

    panoWidth /= 2;
    panoHeight /= 2;
    faceLength /= 2;

    FAIL_RETURN(
        reproject(panoWidth, panoHeight, faceLength, pano.getProjection() == PanoProjection::EquiangularCubemap,
                  alphaPyramids[EQUIRECTANGULAR]->getLevel(level).data().as_const(), eq,
                  alphaPyramids[CUBE_MAP_POSITIVE_X] ? alphaPyramids[CUBE_MAP_POSITIVE_X]->getLevel(level).data()
                                                     : GPU::Buffer<unsigned char>(),
                  px,
                  alphaPyramids[CUBE_MAP_POSITIVE_Y] ? alphaPyramids[CUBE_MAP_POSITIVE_Y]->getLevel(level).data()
                                                     : GPU::Buffer<unsigned char>(),
                  py,
                  alphaPyramids[CUBE_MAP_POSITIVE_Z] ? alphaPyramids[CUBE_MAP_POSITIVE_Z]->getLevel(level).data()
                                                     : GPU::Buffer<unsigned char>(),
                  pz,
                  alphaPyramids[CUBE_MAP_NEGATIVE_X] ? alphaPyramids[CUBE_MAP_NEGATIVE_X]->getLevel(level).data()
                                                     : GPU::Buffer<unsigned char>(),
                  nx,
                  alphaPyramids[CUBE_MAP_NEGATIVE_Y] ? alphaPyramids[CUBE_MAP_NEGATIVE_Y]->getLevel(level).data()
                                                     : GPU::Buffer<unsigned char>(),
                  ny,
                  alphaPyramids[CUBE_MAP_NEGATIVE_Z] ? alphaPyramids[CUBE_MAP_NEGATIVE_Z]->getLevel(level).data()
                                                     : GPU::Buffer<unsigned char>(),
                  nz, stream));
  }

#ifdef DEBUGALPHA
  stream.synchronize();
  for (int level = 0; level <= alphaPyramids[EQUIRECTANGULAR]->numLevels(); ++level) {
    for (int t = CUBE_MAP_POSITIVE_X; t <= CUBE_MAP_NEGATIVE_Z; ++t) {
      TextureTarget target = (TextureTarget)t;
      if (!fromIm.getOutputRect(target).empty()) {
        std::stringstream ss;
        ss << DEBUG_FOLDER << "testAlpha-alpha-" + toString(target) + "-" + name << "-" << level << ".png";
        Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str(), alphaPyramids[target]->getLevel(level).data(),
                                                         alphaPyramids[target]->getLevel(level).width(),
                                                         alphaPyramids[target]->getLevel(level).height());
      }
    }
  }
#endif

  return Status::OK();
}

LaplacianPyramid<unsigned char>* MaskMerger::getAlphaPyramid(TextureTarget t) const {
  if (alphaPyramids[t]) {
    return alphaPyramids[t].get();
  } else {
    return nullptr;
  }
}

}  // namespace Core
}  // namespace VideoStitch
