// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "allocator.hpp"

#include "core/transformGeoParams.hpp"

#include "core1/panoRemapper.hpp"
#include "core1/imageMapping.hpp"
#include "core1/imageMerger.hpp"

#include "gpu/image/imageOps.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/overlay.hpp"

//#define DUMP_FINAL
//#define DUMP_ORIGIN

#if defined(DUMP_FINAL) || defined DUMP_ORIGIN
#include "util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Core {

SourceSurface::Pimpl::Pimpl(GPU::Surface* s, GPU::Stream stream) : surface(s), stream(stream) {}

SourceSurface::Pimpl::~Pimpl() {
  stream.destroy();
  delete surface;
}

PanoSurface::Pimpl::Pimpl(GPU::Stream stream, GPU::Buffer<uint32_t> buffer, size_t w, size_t h)
    : buffer(buffer), stream(stream), width(w), height(h) {}

PanoPimpl::PanoPimpl(GPU::Stream stream, GPU::Buffer<uint32_t> buffer, GPU::Surface* remap, size_t w, size_t h)
    : PanoSurface::Pimpl(stream, buffer, w, h), remapBuffer(remap) {}

CubemapPimpl::CubemapPimpl(bool equiangular, GPU::Stream stream, GPU::Buffer<uint32_t>* bufs,
                           GPU::Buffer<uint32_t> buffer, GPU::CubemapSurface* cubemap, GPU::Buffer<uint32_t> t,
                           size_t w)
    : CubemapSurface::Pimpl(stream, buffer, w), equiangular(equiangular), remapBuffer(cubemap), tmp(t) {
  for (int i = 0; i < 6; i++) {
    buffers[i] = bufs[i];
  }
}

PanoSurface::Pimpl::~Pimpl() {
  stream.destroy();
  if (!externalAlloc) {
    buffer.release();
  }
}

PanoPimpl::~PanoPimpl() { delete remapBuffer; }

CubemapPimpl::~CubemapPimpl() {
  if (!externalAlloc) {
    for (int i = 0; i < 6; i++) {
      buffers[i].release();
    }
  }

  delete remapBuffer;
  tmp.release();
}

Potential<SourceSurface::Pimpl> SourceSurface::Pimpl::create(GPU::Surface* surface) {
  PotentialValue<GPU::Stream> stream = GPU::Stream::create();
  if (stream.ok()) {
    return Potential<Pimpl>(new Pimpl(surface, stream.value()));
  } else {
    return Potential<Pimpl>(stream.status());
  }
}

void SourceSurface::acquire() { pimpl->acquireReader(); }

void SourceSurface::release() { pimpl->releaseReader(); }

void SourceSurface::Pimpl::acquireWriter() {
  std::unique_lock<std::mutex> lk(mutex);
  cv.wait(lk, [this] { return renderers == 0; });
  stitcher = true;
}

void SourceSurface::Pimpl::releaseWriter() {
  {
    std::lock_guard<std::mutex> lk(mutex);
    stitcher = false;
  }
  cv.notify_all();
}

void SourceSurface::Pimpl::acquireReader() {
  std::unique_lock<std::mutex> lk(mutex);
  cv.wait(lk, [this] { return !stitcher; });
  renderers++;
}

void SourceSurface::Pimpl::releaseReader() {
  {
    std::lock_guard<std::mutex> lk(mutex);
    renderers--;
  }
  cv.notify_one();
}

size_t SourceSurface::getWidth() const { return pimpl->getWidth(); }

size_t SourceSurface::getHeight() const { return pimpl->getHeight(); }

size_t SourceSurface::Pimpl::getWidth() const { return surface->width(); }

size_t SourceSurface::Pimpl::getHeight() const { return surface->height(); }

Potential<PanoPimpl> PanoPimpl::create(GPU::Buffer<uint32_t> buffer, GPU::Surface* surface, size_t w, size_t h) {
  PotentialValue<GPU::Stream> stream = GPU::Stream::create();
  if (stream.ok()) {
    return Potential<PanoPimpl>(new PanoPimpl(stream.value(), buffer, surface, w, h));
  } else {
    return Potential<PanoPimpl>(stream.status());
  }
}

void PanoSurface::acquire() { pimpl->acquireReader(); }

void PanoSurface::release() { pimpl->releaseReader(); }

void PanoSurface::Pimpl::acquireWriter() {
  std::unique_lock<std::mutex> lk(mutex);
  cv.wait(lk, [this] { return renderers == 0; });
  stitcher = true;
}

void PanoSurface::Pimpl::releaseWriter() {
  {
    std::lock_guard<std::mutex> lk(mutex);
    stitcher = false;
  }
  cv.notify_all();
}

void PanoSurface::Pimpl::acquireReader() {
  std::unique_lock<std::mutex> lk(mutex);
  cv.wait(lk, [this] { return !stitcher; });
  renderers++;
}

void PanoSurface::Pimpl::releaseReader() {
  {
    std::lock_guard<std::mutex> lk(mutex);
    renderers--;
  }
  cv.notify_one();
}

size_t PanoSurface::getWidth() const { return pimpl->getWidth(); }

size_t PanoSurface::getHeight() const { return pimpl->getHeight(); }

SourceSurface::SourceSurface(Pimpl* pimpl) : pimpl(pimpl) {}

SourceSurface::~SourceSurface() { delete pimpl; }

PanoSurface::PanoSurface(Pimpl* pimpl) : pimpl(pimpl) {}

PanoSurface::~PanoSurface() { delete pimpl; }

CubemapSurface::CubemapSurface(Pimpl* pimpl) : PanoSurface(pimpl) {}

CubemapSurface::~CubemapSurface() {}

size_t CubemapSurface::getLength() const { return dynamic_cast<CubemapPimpl*>(pimpl)->getLength(); }

Status PanoPimpl::reset(const Core::ImageMerger* merger) {
  if (merger && (merger->warpMergeType() == Core::ImageMerger::Format::Gradient)) {
    return memsetToZeroAsync(*remapBuffer, stream);
  } else {
    return memsetToZeroAsync(buffer, stream);
  }
}

Status PanoPimpl::reproject(const Core::PanoDefinition& pano, const Matrix33<double>& perspective,
                            const Core::ImageMerger* merger) {
#if defined(DUMP_ORIGIN)
  if (merger && (merger->warpMergeType() == Core::ImageMerger::Format::Gradient)) {
    memcpyAsync(buffer, *remapBuffer, stream);
  }
  stream.synchronize();
  Debug::dumpRGBADeviceBuffer("origin.png", buffer, (unsigned)pano.getWidth(), (unsigned)pano.getHeight());
#endif

  if (width != (unsigned)pano.getWidth() || height != (unsigned)pano.getHeight()) {
    return {Origin::Surface, ErrType::InvalidConfiguration, "Surface unadapted to the current panorama configuration"};
  }

  if (!(merger && (merger->warpMergeType() == Core::ImageMerger::Format::Gradient))) {
    memcpyAsync(*remapBuffer, buffer.as_const(), stream);
  }

  float2 srcScale = {
      Core::TransformGeoParams::computePanoScale(Core::PanoProjection::Equirectangular, pano.getWidth(), 360.f),
      2 * Core::TransformGeoParams::computePanoScale(Core::PanoProjection::Equirectangular, pano.getHeight(), 360.f)};
  float2 dstScale = {
      Core::TransformGeoParams::computePanoScale(pano.getProjection(), pano.getWidth(), (float)pano.getHFOV()),
      Core::TransformGeoParams::computePanoScale(pano.getProjection(), pano.getWidth(), (float)pano.getHFOV())};

  switch (pano.getProjection()) {
    case Core::PanoProjection::Rectilinear:
      return Core::reprojectRectilinear(buffer, dstScale, *remapBuffer, srcScale, (unsigned)pano.getWidth(),
                                        (unsigned)pano.getHeight(), perspective, stream);
    case Core::PanoProjection::Cylindrical:
      assert(false);
      return Status::OK();
    case Core::PanoProjection::Equirectangular:
      return Core::reprojectEquirectangular(buffer, dstScale, *remapBuffer, srcScale, (unsigned)pano.getWidth(),
                                            (unsigned)pano.getHeight(), perspective, stream);
    case Core::PanoProjection::FullFrameFisheye:
      return Core::reprojectFullFrameFisheye(buffer, dstScale, *remapBuffer, srcScale, (unsigned)pano.getWidth(),
                                             (unsigned)pano.getHeight(), perspective, stream);
    case Core::PanoProjection::CircularFisheye:
      return Core::reprojectCircularFisheye(buffer, dstScale, *remapBuffer, srcScale, (unsigned)pano.getWidth(),
                                            (unsigned)pano.getHeight(), perspective, stream);
    case Core::PanoProjection::Stereographic:
      return Core::reprojectStereographic(buffer, dstScale, *remapBuffer, srcScale, (unsigned)pano.getWidth(),
                                          (unsigned)pano.getHeight(), perspective, stream);
    case Core::PanoProjection::Cubemap:
    case Core::PanoProjection::EquiangularCubemap:
      assert(false);
      return Status::OK();
  }
  return Status::OK();
}

Status CubemapPimpl::reproject(const Core::PanoDefinition& pano, const Matrix33<double>& perspective,
                               const Core::ImageMerger*) {
#if defined(DUMP_ORIGIN)
  stream.synchronize();
  Debug::dumpRGBADeviceBuffer("origin_face_+x.png", buffers[0], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("origin_face_-x.png", buffers[1], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("origin_face_+y.png", buffers[2], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("origin_face_-y.png", buffers[3], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("origin_face_+z.png", buffers[4], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("origin_face_-z.png", buffers[5], (unsigned)pano.getLength(), (unsigned)pano.getLength());
#endif

  memcpyCubemapAsync(*remapBuffer, buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5],
                     pano.getLength(), stream);

  stream.synchronize();

  rotateCubemap(pano, *remapBuffer, buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], perspective,
                equiangular, stream);

#if defined(DUMP_FINAL)
  stream.synchronize();
  Debug::dumpRGBADeviceBuffer("face_+x.png", buffers[0], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("face_-x.png", buffers[1], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("face_+y.png", buffers[2], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("face_-y.png", buffers[3], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("face_+z.png", buffers[4], (unsigned)pano.getLength(), (unsigned)pano.getLength());
  Debug::dumpRGBADeviceBuffer("face_-z.png", buffers[5], (unsigned)pano.getLength(), (unsigned)pano.getLength());
#endif

  return Status::OK();
}

Status PanoPimpl::warp(Core::ImageMapping* mapping, frameid_t frame, const Core::PanoDefinition& pano,
                       GPU::Stream& stream) {
  // warp the image to the destination space
  if (mapping->getMerger().isMultiScale()) {
    FAIL_RETURN(mapping->warp(frame, pano, progressivePbo.borrow(), *remapBuffer, stream));
  } else {
    FAIL_RETURN(mapping->warp(frame, pano, buffer, *remapBuffer, stream));
  }

  // analyze the image content if needed (eg. compute a multi-band pyramid)
  return mapping->getMerger().prepareMergeAsync(Core::EQUIRECTANGULAR, *mapping, stream);
}

Status CubemapPimpl::warp(Core::ImageMapping* mapping, frameid_t frame, const Core::PanoDefinition& pano,
                          GPU::Stream& stream) {
  FAIL_RETURN(mapping->warpCubemap(frame, pano, equiangular, stream));

  // analyze the image content if needed (eg. compute a multi-band pyramid)
  FAIL_RETURN(mapping->getMerger().prepareMergeAsync(Core::CUBE_MAP_POSITIVE_X, *mapping, stream));
  FAIL_RETURN(mapping->getMerger().prepareMergeAsync(Core::CUBE_MAP_NEGATIVE_X, *mapping, stream));
  FAIL_RETURN(mapping->getMerger().prepareMergeAsync(Core::CUBE_MAP_POSITIVE_Y, *mapping, stream));
  FAIL_RETURN(mapping->getMerger().prepareMergeAsync(Core::CUBE_MAP_NEGATIVE_Y, *mapping, stream));
  FAIL_RETURN(mapping->getMerger().prepareMergeAsync(Core::CUBE_MAP_POSITIVE_Z, *mapping, stream));
  return mapping->getMerger().prepareMergeAsync(Core::CUBE_MAP_NEGATIVE_Z, *mapping, stream);
}

Status PanoPimpl::blend(const Core::PanoDefinition& pano, const Core::ImageMapping& mapping, bool firstMerger,
                        GPU::Stream& stream) {
  return mapping.getMerger().mergeAsync(Core::EQUIRECTANGULAR, pano, buffer, progressivePbo, mapping, firstMerger,
                                        stream);
}

Status CubemapPimpl::blend(const Core::PanoDefinition& pano, const Core::ImageMapping& mapping, bool firstMerger,
                           GPU::Stream& stream) {
  GPU::UniqueBuffer<uint32_t> dummy;
  FAIL_RETURN(
      mapping.getMerger().mergeAsync(Core::CUBE_MAP_POSITIVE_X, pano, buffers[0], dummy, mapping, firstMerger, stream));
  FAIL_RETURN(
      mapping.getMerger().mergeAsync(Core::CUBE_MAP_NEGATIVE_X, pano, buffers[1], dummy, mapping, firstMerger, stream));
  FAIL_RETURN(
      mapping.getMerger().mergeAsync(Core::CUBE_MAP_POSITIVE_Y, pano, buffers[2], dummy, mapping, firstMerger, stream));
  FAIL_RETURN(
      mapping.getMerger().mergeAsync(Core::CUBE_MAP_NEGATIVE_Y, pano, buffers[3], dummy, mapping, firstMerger, stream));
  FAIL_RETURN(
      mapping.getMerger().mergeAsync(Core::CUBE_MAP_POSITIVE_Z, pano, buffers[4], dummy, mapping, firstMerger, stream));
  return mapping.getMerger().mergeAsync(Core::CUBE_MAP_NEGATIVE_Z, pano, buffers[5], dummy, mapping, firstMerger,
                                        stream);
}

Status PanoPimpl::flatten() { return Status::OK(); }

Status CubemapPimpl::flatten() {
  if (layout == YOUTUBE) {
    FAIL_RETURN(memcpy2DAsync(buffer, buffers[0], 0, 0, 0, 0, length, length, length, width, stream));
    FAIL_RETURN(memcpy2DAsync(buffer, buffers[1], 0, 0, length, 0, length, length, length, width, stream));
    FAIL_RETURN(memcpy2DAsync(buffer, buffers[2], 0, 0, 2 * length, 0, length, length, length, width, stream));
    FAIL_RETURN(memcpy2DAsync(buffer, buffers[3], 0, 0, 0, length, length, length, length, width, stream));
    FAIL_RETURN(memcpy2DAsync(buffer, buffers[4], 0, 0, length, length, length, length, length, width, stream));
    return memcpy2DAsync(buffer, buffers[5], 0, 0, 2 * length, length, length, length, length, width, stream);
  } else if (layout == ROT) {
    FAIL_RETURN(Image::rotate(tmp, buffers[1], length, stream));
    FAIL_RETURN(memcpy2DAsync(buffer, tmp, 0, 0, 0, 0, length, length, length, width, stream));
    FAIL_RETURN(Image::rotate(tmp, buffers[5], length, stream));
    FAIL_RETURN(memcpy2DAsync(buffer, tmp, 0, 0, length, 0, length, length, length, width, stream));
    FAIL_RETURN(Image::rotate(tmp, buffers[0], length, stream));
    FAIL_RETURN(memcpy2DAsync(buffer, tmp, 0, 0, 2 * length, 0, length, length, length, width, stream));
    FAIL_RETURN(Image::rotateLeft(tmp, buffers[2], length, stream));
    FAIL_RETURN(memcpy2DAsync(buffer, tmp, 0, 0, 0, length, length, length, length, width, stream));
    FAIL_RETURN(Image::rotateLeft(tmp, buffers[4], length, stream));
    FAIL_RETURN(memcpy2DAsync(buffer, tmp, 0, 0, length, length, length, length, length, width, stream));
    FAIL_RETURN(Image::rotateLeft(tmp, buffers[3], length, stream));
    return memcpy2DAsync(buffer, tmp, 0, 0, 2 * length, length, length, length, length, width, stream);
  }
  assert(false);
  return Status::OK();
}

Status PanoPimpl::reconstruct(const Core::PanoDefinition& pano, const Core::ImageMapping& mapping, GPU::Stream& stream,
                              bool final) {
  if (mapping.getMerger().isMultiScale()) {
    return mapping.reconstruct(Core::EQUIRECTANGULAR, pano, progressivePbo.borrow(), final, stream);
  } else {
    return mapping.reconstruct(Core::EQUIRECTANGULAR, pano, buffer, final, stream);
  }
}

Status CubemapPimpl::reconstruct(const Core::PanoDefinition& pano, const Core::ImageMapping& mapping, GPU::Stream&,
                                 bool final) {
  GPU::Buffer<uint32_t> dummy;

  FAIL_RETURN(mapping.reconstruct(Core::CUBE_MAP_POSITIVE_X, pano, dummy, final, stream));
  FAIL_RETURN(mapping.reconstruct(Core::CUBE_MAP_NEGATIVE_X, pano, dummy, final, stream));
  FAIL_RETURN(mapping.reconstruct(Core::CUBE_MAP_POSITIVE_Y, pano, dummy, final, stream));
  FAIL_RETURN(mapping.reconstruct(Core::CUBE_MAP_NEGATIVE_Y, pano, dummy, final, stream));
  FAIL_RETURN(mapping.reconstruct(Core::CUBE_MAP_POSITIVE_Z, pano, dummy, final, stream));
  return mapping.reconstruct(Core::CUBE_MAP_NEGATIVE_Z, pano, dummy, final, stream);
}

void SourceSurface::accept(std::shared_ptr<SourceRenderer>, mtime_t) {}

void SourceOpenGLSurface::accept(std::shared_ptr<SourceRenderer> renderer, mtime_t date) {
  renderer->render(std::dynamic_pointer_cast<SourceOpenGLSurface>(shared_from_this()), date);
}

void PanoSurface::accept(std::shared_ptr<PanoRenderer>, mtime_t) {}

void PanoSurface::accept(const std::shared_ptr<GPU::Overlayer>&, std::shared_ptr<PanoOpenGLSurface>, mtime_t) {}

void PanoOpenGLSurface::accept(std::shared_ptr<PanoRenderer> renderer, mtime_t date) {
  renderer->render(std::dynamic_pointer_cast<PanoOpenGLSurface>(shared_from_this()), date);
}

void PanoOpenGLSurface::accept(const std::shared_ptr<GPU::Overlayer>& compositor,
                               std::shared_ptr<PanoOpenGLSurface> oglSurf, mtime_t date) {
  compositor->computeOverlay(std::dynamic_pointer_cast<PanoOpenGLSurface>(shared_from_this()), oglSurf, date);
}

void CubemapSurface::accept(std::shared_ptr<PanoRenderer>, mtime_t) {}

void CubemapSurface::accept(const std::shared_ptr<GPU::Overlayer>&, std::shared_ptr<PanoOpenGLSurface>, mtime_t) {}

void CubemapOpenGLSurface::accept(std::shared_ptr<PanoRenderer> renderer, mtime_t date) {
  if ((static_cast<CubemapPimpl*>(pimpl))->equiangular) {
    renderer->renderEquiangularCubemap(std::dynamic_pointer_cast<CubemapOpenGLSurface>(shared_from_this()), date);
  } else {
    renderer->renderCubemap(std::dynamic_pointer_cast<CubemapOpenGLSurface>(shared_from_this()), date);
  }
}

void CubemapOpenGLSurface::accept(const std::shared_ptr<GPU::Overlayer>&, std::shared_ptr<PanoOpenGLSurface>, mtime_t) {
}

}  // namespace Core
}  // namespace VideoStitch
