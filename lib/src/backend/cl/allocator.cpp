// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../common/glAllocator.hpp"

#include "surface.hpp"
#include "deviceBuffer.hpp"
#include "context.hpp"
#include "deviceStream.hpp"

#include "../common/allocStats.hpp"

#include "gpu/allocator.hpp"
#include "gpu/memcpy.hpp"

namespace VideoStitch {
namespace Core {
cl_mem_flags getOpenCLGLMemAllocType(OpenGLAllocator::BufferAllocType flag) {
  switch (flag) {
    case OpenGLAllocator::BufferAllocType::ReadWrite:
      return CL_MEM_READ_WRITE;
    case OpenGLAllocator::BufferAllocType::ReadOnly:
      return CL_MEM_READ_ONLY;
    case OpenGLAllocator::BufferAllocType::WriteOnly:
      return CL_MEM_WRITE_ONLY;
  }

  assert(false);
  return CL_MEM_READ_WRITE;
}

Potential<SourceSurface> OffscreenAllocator::createAlphaSurface(size_t width, size_t height, const char* name) {
  const auto& potContext = GPU::getContext();
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  int err;
  cl_image_format image_format = {0};
  image_format.image_channel_order = CL_A;
  image_format.image_channel_data_type = CL_UNORM_INT8;
  cl_image_desc image_desc = {0};
  image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;  // TODO?
  image_desc.image_width = width;
  image_desc.image_height = height;

  cl_mem img = clCreateImage(ctx, CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
  PROPAGATE_CL_ERR(err);
  deviceStats.addPtr(name, img, width * height);

  auto surface = new GPU::Surface(new GPU::DeviceSurface(img, true), width, height);
  Potential<SourceSurface::Pimpl> impl = SourceSurface::Pimpl::create(surface);
  FAIL_RETURN(impl.status());
  return new SourceSurface(impl.release());
}

namespace {
Potential<GPU::Surface> makeSurface(std::string name, size_t width, size_t height) {
  const auto& potContext = GPU::getContext();
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  int err;
  cl_image_format image_format = {0};
  image_format.image_channel_order = CL_RGBA;
  image_format.image_channel_data_type = CL_UNORM_INT8;
  cl_image_desc image_desc = {0};
  image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;  // TODO?
  image_desc.image_width = width;
  image_desc.image_height = height;

  cl_mem img = clCreateImage(ctx, CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
  deviceStats.addPtr(name.c_str(), img, width * height * 4);
  PROPAGATE_CL_ERR(err);

  return new GPU::Surface(new GPU::DeviceSurface(img, true), width, height);
}

Potential<GPU::Surface> makeSurface_F32_C1(std::string name, size_t width, size_t height) {
  const auto& potContext = GPU::getContext();
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  int err;
  cl_image_format image_format = {0};
  image_format.image_channel_order = CL_DEPTH;
  image_format.image_channel_data_type = CL_FLOAT;
  cl_image_desc image_desc = {0};
  image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;  // TODO?
  image_desc.image_width = width;
  image_desc.image_height = height;

  cl_mem img = clCreateImage(ctx, CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
  deviceStats.addPtr(name.c_str(), img, width * height * sizeof(float));
  PROPAGATE_CL_ERR(err);

  return new GPU::Surface(new GPU::DeviceSurface(img, true), width, height);
}
}  // namespace

Potential<SourceSurface> OffscreenAllocator::createSourceSurface(size_t width, size_t height, const char* name) {
  auto surface = makeSurface(name, width, height);
  FAIL_RETURN(surface.status());

  Potential<SourceSurface::Pimpl> impl = SourceSurface::Pimpl::create(surface.release());
  FAIL_RETURN(impl.status());
  return new SourceSurface(impl.release());
}

Potential<SourceSurface> OffscreenAllocator::createDepthSurface(size_t width, size_t height, const char* name) {
  auto surface = makeSurface_F32_C1(name, width, height);
  FAIL_RETURN(surface.status());

  Potential<SourceSurface::Pimpl> impl = SourceSurface::Pimpl::create(surface.release());
  FAIL_RETURN(impl.status());
  return new SourceSurface(impl.release());
}

Potential<SourceSurface> OffscreenAllocator::createCoordSurface(size_t width, size_t height, const char* name) {
  const auto& potContext = GPU::getContext();
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  int err;
  cl_image_format image_format = {0};
  image_format.image_channel_order = CL_RG;
  image_format.image_channel_data_type = CL_FLOAT;
  cl_image_desc image_desc = {0};
  image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;  // TODO?
  image_desc.image_width = width;
  image_desc.image_height = height;

  cl_mem img = clCreateImage(ctx, CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
  deviceStats.addPtr(name, img, width * height * 8);
  PROPAGATE_CL_ERR(err);

  auto surface = new GPU::Surface(new GPU::DeviceSurface(img, true), width, height);
  Potential<SourceSurface::Pimpl> impl = SourceSurface::Pimpl::create(surface);
  FAIL_RETURN(impl.status());
  return new SourceSurface(impl.release());
}

Potential<SourceOpenGLSurface> OpenGLAllocator::createSourceSurface(size_t width, size_t height) {
  PotentialValue<GLuint> texture = createSourceSurfaceTexture(width, height);
  FAIL_RETURN(texture.status());

  const auto& potContext = GPU::getContext();
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  cl_int err;
  cl_mem img = clCreateFromGLTexture(ctx, CL_MEM_READ_WRITE, GL_TEXTURE_2D,
                                     0,  // miplevel
                                     texture.value(), &err);
  PROPAGATE_CL_ERR(err);

  auto surface = new GPU::Surface(new GPU::DeviceSurface(img, false), width, height);
  Potential<SourceOpenGLSurface::Pimpl> impl = SourceOpenGLSurface::Pimpl::create(surface);
  FAIL_RETURN(impl.status());

  SourceOpenGLSurface* surf = new SourceOpenGLSurface(impl.release());
  surf->texture = texture.value();
  return surf;
}

Potential<PanoOpenGLSurface> OpenGLAllocator::createPanoSurface(size_t width, size_t height, BufferAllocType flag) {
  auto pbtex = createPanoSurfacePB(width, height);
  FAIL_RETURN(pbtex.status());

  GLuint pixelbuffer = pbtex.value();

  const auto& potContext = GPU::getContext();
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  cl_int err;
  cl_mem_flags memFlag = getOpenCLGLMemAllocType(flag);
  cl_mem pbo = clCreateFromGLBuffer(ctx, memFlag, pixelbuffer, &err);
  PROPAGATE_CL_ERR(err);

  GPU::Buffer<uint32_t> buffer = GPU::DeviceBuffer<uint32_t>::createBuffer(pbo, width * height);
  FAIL_RETURN(GPU::memsetToZeroBlocking(buffer, width * height * sizeof(uint32_t)));

  auto remapSurf = makeSurface("Pano Surface", width, height);
  FAIL_RETURN(remapSurf.status());

  Potential<PanoOpenGLSurface::Pimpl> impl =
      PanoOpenGLSurface::Pimpl::create(buffer, remapSurf.release(), width, height);
  FAIL_RETURN(impl.status());

  PanoOpenGLSurface* surf = new PanoOpenGLSurface(impl.release());
  surf->pixelbuffer = pbtex.value();
  surf->pimpl->externalAlloc = true;

  return Potential<PanoOpenGLSurface>(surf);
}

Potential<CubemapSurface> OffscreenAllocator::createCubemapSurface(size_t width, const char* name, bool equiangular) {
  GPU::Buffer<uint32_t> faces[6];
  for (int i = 0; i < 6; ++i) {
    PotentialValue<GPU::Buffer<uint32_t>> buf = GPU::Buffer<uint32_t>::allocate(width * width, name);
    FAIL_RETURN(buf.status());
    GPU::memsetToZeroBlocking(buf.value(), width * width * sizeof(uint32_t));
    faces[i] = buf.value();
  }

  PotentialValue<GPU::Stream> stream = GPU::Stream::create();
  FAIL_RETURN(stream.status());

  PotentialValue<GPU::Buffer<uint32_t>> buf = GPU::Buffer<uint32_t>::allocate(6 * width * width, "Offscreen Surface");
  FAIL_RETURN(buf.status());

  PotentialValue<GPU::Buffer<uint32_t>> potBuf = GPU::Buffer<uint32_t>::allocate(width * width, "Cubemap");
  if (!potBuf.ok()) {
    return potBuf.status();
  }

  CubemapPimpl* impl =
      new CubemapPimpl(equiangular, stream.value(), &faces[0], buf.value(), nullptr, potBuf.value(), width);
  CubemapSurface* surf = new CubemapSurface(impl);

  surf->pimpl->externalAlloc = false;

  return Potential<CubemapSurface>(surf);
}

Potential<CubemapOpenGLSurface> OpenGLAllocator::createCubemapSurface(size_t width, bool equiangular,
                                                                      BufferAllocType flag) {
  GLuint pbo[6];

#ifndef GLEWLIB_UNSUPPORTED
  glewInit();
#endif
  glEnable(GL_TEXTURE_CUBE_MAP);
#ifdef GL_VERSION_3_2
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
#endif
  // clear error flag before mapping to OpenCL
  GLenum glerr = glGetError();
  while (glerr != GL_NO_ERROR) glerr = glGetError();

  size_t bufSize = width * width * 4;
  if (bufSize * 4 > std::numeric_limits<int32_t>::max()) {
    std::stringstream msg;
    msg << "Could not allocate OpenGL Surface of size " << bufSize;
    msg << ". Maximum supported texture size: " << std::numeric_limits<int32_t>::max();
    return Status{Origin::GPU, ErrType::InvalidConfiguration, msg.str()};
  }

  for (int i = 0; i < 6; ++i) {
    glGenBuffers(1, (GLuint*)&pbo[i]);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[i]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bufSize, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  }

  glerr = glGetError();
  if (glerr != GL_NO_ERROR) {
    return Potential<CubemapOpenGLSurface>(
        Status{Origin::GPU, ErrType::RuntimeError, "Could not allocate OpenGL Surface."});
  }

  const auto& potContext = GPU::getContext();
  FAIL_RETURN(potContext.status());
  const auto& ctx = potContext.value();
  cl_int err;
  cl_mem_flags memFlag = getOpenCLGLMemAllocType(flag);

  GPU::Buffer<uint32_t> buffers[6];
  for (int i = 0; i < 6; ++i) {
    cl_mem pb = clCreateFromGLBuffer(ctx, memFlag, pbo[i], &err);
    buffers[i] = GPU::DeviceBuffer<uint32_t>::createBuffer(pb, width * width);
  }

  PotentialValue<GPU::Buffer<uint32_t>> buf = GPU::Buffer<uint32_t>::allocate(6 * width * width, "Offscreen Surface");
  if (!buf.ok()) {
    return Potential<CubemapOpenGLSurface>(buf.status());
  }

  PotentialValue<GPU::Buffer<uint32_t>> potBuf = GPU::Buffer<uint32_t>::allocate(width * width, "Cubemap");
  if (!potBuf.ok()) {
    return potBuf.status();
  }

  Potential<CubemapOpenGLSurface::Pimpl> impl =
      CubemapOpenGLSurface::Pimpl::create(buffers, buf.value(), nullptr, potBuf.value(), width, equiangular);
  if (!impl.ok()) {
    return Potential<CubemapOpenGLSurface>(impl.status());
  }
  impl->externalAlloc = true;

  CubemapOpenGLSurface* surf = new CubemapOpenGLSurface(impl.release(), (int*)&pbo);

  return Potential<CubemapOpenGLSurface>(surf);
}

Potential<PanoSurface> OffscreenAllocator::createPanoSurface(size_t width, size_t height, const char* name) {
  PotentialValue<GPU::Buffer<uint32_t>> buf = GPU::Buffer<uint32_t>::allocate(width * height, name);
  FAIL_RETURN(buf.status());
  GPU::memsetToZeroBlocking(buf.value(), width * height * sizeof(uint32_t));

  auto remapSurf = makeSurface("Pano Surface", width, height);
  FAIL_RETURN(remapSurf.status());

  Potential<PanoPimpl> impl = PanoPimpl::create(buf.value(), remapSurf.release(), width, height);
  FAIL_RETURN(impl.status());

  PanoSurface* surf = new PanoSurface(impl.release());
  surf->pimpl->buffer = buf.value();
  surf->pimpl->externalAlloc = false;

  return Potential<PanoSurface>(surf);
}

SourceOpenGLSurface::SourceOpenGLSurface(Pimpl* pimpl) : SourceSurface(pimpl), texture(0) {}

SourceOpenGLSurface::~SourceOpenGLSurface() { glDeleteTextures(1, (GLuint*)&texture); }

PanoOpenGLSurface::PanoOpenGLSurface(Pimpl* pimpl) : PanoSurface(pimpl), pixelbuffer(0) {}

PanoOpenGLSurface::~PanoOpenGLSurface() { glDeleteBuffers(1, (GLuint*)&pixelbuffer); }

CubemapOpenGLSurface::CubemapOpenGLSurface(Pimpl* pimpl, int* f) : CubemapSurface(pimpl) {
  for (int i = 0; i < 6; ++i) {
    faces[i] = f[i];
  }
}

CubemapOpenGLSurface::~CubemapOpenGLSurface() {
  for (int i = 0; i < 6; ++i) {
    glDeleteBuffers(1, (GLuint*)&faces[i]);
  }
}

}  // namespace Core
}  // namespace VideoStitch
