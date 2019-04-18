// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __APPLE__
#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/gl.h>
#else
#include <GL/glew.h>
#include <OpenGL/gl.h>
#endif
#include "cuda/error.hpp"
#include "../common/glAllocator.hpp"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "surface.hpp"
#include "deviceBuffer.hpp"

#include "../common/allocStats.hpp"

#include "gpu/allocator.hpp"
#include "gpu/memcpy.hpp"

namespace VideoStitch {
namespace Core {

unsigned int getCudaGLMemAllocType(OpenGLAllocator::BufferAllocType flag) {
  switch (flag) {
    case OpenGLAllocator::BufferAllocType::ReadWrite:
      return cudaGraphicsMapFlagsNone;
    case OpenGLAllocator::BufferAllocType::ReadOnly:
      return cudaGraphicsMapFlagsReadOnly;
    case OpenGLAllocator::BufferAllocType::WriteOnly:
      return cudaGraphicsMapFlagsWriteDiscard;
  }

  assert(false);
  return cudaGraphicsMapFlagsNone;
}

Potential<SourceSurface> OffscreenAllocator::createAlphaSurface(size_t width, size_t height, const char* name) {
  cudaArray_t array;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
  FAIL_RETURN(CUDA_ERROR(cudaMallocArray(&array, &channelDesc, width, height, cudaArraySurfaceLoadStore)));
  deviceStats.addPtr(name, array, width * height);

  // Specify surface
  struct cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  // Create the surface objects
  resDesc.res.array.array = array;
  cudaSurfaceObject_t surface;
  FAIL_RETURN(CUDA_ERROR(cudaCreateSurfaceObject(&surface, &resDesc)));
  // create a texture
  cudaTextureObject_t tex;
  cudaTextureDesc texDesc = {};
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.normalizedCoords = false;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  cudaResourceViewDesc resViewDesc = {};
  resViewDesc.format = cudaResViewFormatUnsignedChar1;
  resViewDesc.width = width;
  resViewDesc.height = height;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, &resViewDesc);

  auto gsurface = new GPU::Surface(new GPU::DeviceSurface(array, tex, surface, true), width, height);

  Potential<SourceSurface::Pimpl> impl = SourceSurface::Pimpl::create(gsurface);
  FAIL_RETURN(impl.status());

  return new SourceSurface(impl.release());
}

namespace {
Potential<GPU::Surface> makeSurface(std::string name, size_t width, size_t height) {
  cudaArray_t array;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
  FAIL_RETURN(CUDA_ERROR(cudaMallocArray(&array, &channelDesc, width, height, cudaArraySurfaceLoadStore)));
  deviceStats.addPtr(name.c_str(), array, width * height * 4);

  // Specify surface
  struct cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  // Create the surface objects
  resDesc.res.array.array = array;
  cudaSurfaceObject_t surface;
  FAIL_RETURN(CUDA_ERROR(cudaCreateSurfaceObject(&surface, &resDesc)));
  // create a texture
  cudaTextureObject_t tex;
  cudaTextureDesc texDesc = {};
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.normalizedCoords = false;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  cudaResourceViewDesc resViewDesc = {};
  resViewDesc.format = cudaResViewFormatUnsignedChar4;
  resViewDesc.width = width;
  resViewDesc.height = height;
  FAIL_RETURN(CUDA_ERROR(cudaCreateTextureObject(&tex, &resDesc, &texDesc, &resViewDesc)));
  return new GPU::Surface(new GPU::DeviceSurface(array, tex, surface, true), width, height);
}

Potential<GPU::Surface> makeSurface_F32_C1(std::string name, size_t width, size_t height) {
  cudaArray_t array;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  FAIL_RETURN(CUDA_ERROR(cudaMallocArray(&array, &channelDesc, width, height, cudaArraySurfaceLoadStore)));
  deviceStats.addPtr(name.c_str(), array, width * height * sizeof(float));

  // Specify surface
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  // Create the surface objects
  resDesc.res.array.array = array;
  cudaSurfaceObject_t surface;
  FAIL_RETURN(CUDA_ERROR(cudaCreateSurfaceObject(&surface, &resDesc)));
  // create a texture
  cudaTextureObject_t tex;
  cudaTextureDesc texDesc = {};
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.normalizedCoords = false;
  texDesc.readMode = cudaReadModeElementType;
  cudaResourceViewDesc resViewDesc;
  memset(&resViewDesc, 0, sizeof(cudaResourceViewDesc));
  resViewDesc.format = cudaResViewFormatFloat1;
  resViewDesc.width = width;
  resViewDesc.height = height;
  FAIL_RETURN(CUDA_ERROR(cudaCreateTextureObject(&tex, &resDesc, &texDesc, &resViewDesc)));
  return new GPU::Surface(new GPU::DeviceSurface(array, tex, surface, true), width, height);
}
}  // namespace

Potential<SourceSurface> OffscreenAllocator::createSourceSurface(size_t width, size_t height, const char* name) {
  Potential<GPU::Surface> potSurf = makeSurface(name, width, height);
  if (!potSurf.ok()) {
    return potSurf.status();
  }

  Potential<SourceSurface::Pimpl> impl = SourceSurface::Pimpl::create(potSurf.release());
  FAIL_RETURN(impl.status());

  return new SourceSurface(impl.release());
}

Potential<SourceSurface> OffscreenAllocator::createDepthSurface(size_t width, size_t height, const char* name) {
  Potential<GPU::Surface> potSurf = makeSurface_F32_C1(name, width, height);
  if (!potSurf.ok()) {
    return potSurf.status();
  }

  Potential<SourceSurface::Pimpl> impl = SourceSurface::Pimpl::create(potSurf.release());
  FAIL_RETURN(impl.status());

  return new SourceSurface(impl.release());
}

Potential<SourceSurface> OffscreenAllocator::createCoordSurface(size_t width, size_t height, const char* name) {
  cudaArray_t array;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
  FAIL_RETURN(CUDA_ERROR(cudaMallocArray(&array, &channelDesc, width, height, cudaArraySurfaceLoadStore)));
  deviceStats.addPtr(name, array, width * height * 8);

  // Specify surface
  struct cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  // Create the surface objects
  resDesc.res.array.array = array;
  cudaSurfaceObject_t surface;
  FAIL_RETURN(CUDA_ERROR(cudaCreateSurfaceObject(&surface, &resDesc)));
  // create a texture
  cudaTextureObject_t tex;
  cudaTextureDesc texDesc = {};
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.normalizedCoords = false;
  texDesc.readMode = cudaReadModeElementType;
  cudaResourceViewDesc resViewDesc = {};
  resViewDesc.format = cudaResViewFormatFloat2;
  resViewDesc.width = width;
  resViewDesc.height = height;
  FAIL_RETURN(CUDA_ERROR(cudaCreateTextureObject(&tex, &resDesc, &texDesc, &resViewDesc)));

  auto gsurface = new GPU::Surface(new GPU::DeviceSurface(array, tex, surface, true), width, height);

  Potential<SourceSurface::Pimpl> impl = SourceSurface::Pimpl::create(gsurface);
  FAIL_RETURN(impl.status());

  return new SourceSurface(impl.release());
}

Potential<SourceOpenGLSurface> OpenGLAllocator::createSourceSurface(size_t width, size_t height) {
  auto allocPotSurf = [](GLuint texture, size_t width, size_t height) -> Potential<SourceOpenGLSurface> {
    cudaGraphicsResource* image;
    FAIL_RETURN(CUDA_ERROR(
        cudaGraphicsGLRegisterImage(&image, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore)))
    FAIL_RETURN(CUDA_ERROR(cudaGraphicsMapResources(1, &image, cudaStreamPerThread)))

    cudaArray_t array;
    FAIL_RETURN(CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, image, 0, 0)))
    FAIL_RETURN(CUDA_ERROR(cudaGraphicsUnmapResources(1, &image, cudaStreamPerThread)))

    // Specify surface
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    // Create the surface objects
    resDesc.res.array.array = array;
    cudaSurfaceObject_t surface;
    FAIL_RETURN(CUDA_ERROR(cudaCreateSurfaceObject(&surface, &resDesc)))

    // create a texture
    cudaTextureObject_t tex;
    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.normalizedCoords = false;
    texDesc.readMode = cudaReadModeNormalizedFloat;

    cudaResourceViewDesc resViewDesc = {};
    resViewDesc.format = cudaResViewFormatUnsignedChar4;
    resViewDesc.width = width;
    resViewDesc.height = height;
    FAIL_RETURN(CUDA_ERROR(cudaCreateTextureObject(&tex, &resDesc, &texDesc, &resViewDesc)))

    {
      auto gsurface = std::make_unique<GPU::Surface>(new GPU::DeviceSurface(array, tex, surface, false), width, height);
      Potential<SourceOpenGLSurface::Pimpl> impl = SourceOpenGLSurface::Pimpl::create(image, std::move(gsurface));
      FAIL_RETURN(impl.status())

      SourceOpenGLSurface* surf = new SourceOpenGLSurface(impl.release());
      surf->texture = texture;

      return surf;
    }
  };

  PotentialValue<GLuint> potTexture = createSourceSurfaceTexture(width, height);
  FAIL_RETURN(potTexture.status());

  GLuint texture = potTexture.value();

  auto potSurf = allocPotSurf(texture, width, height);

  if (!potSurf.ok()) {
    glDeleteTextures(1, &texture);
    return Status{Origin::GPU, ErrType::RuntimeError, "Could not map OpenGL surface to CUDA.", potSurf.status()};
  }

  return potSurf;
}

Potential<PanoOpenGLSurface> OpenGLAllocator::createPanoSurface(size_t width, size_t height, BufferAllocType flag) {
  auto allocPotSurf = [](GLuint pixelbuffer, size_t width, size_t height,
                         BufferAllocType flag) -> Potential<PanoOpenGLSurface> {
    cudaGraphicsResource* pbo;
    unsigned int memFlag = getCudaGLMemAllocType(flag);

    FAIL_RETURN(CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&pbo, pixelbuffer, memFlag)))
    FAIL_RETURN(CUDA_ERROR(cudaGraphicsMapResources(1, &pbo, cudaStreamPerThread)))

    void* devPtr;
    size_t size;
    FAIL_RETURN(CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, pbo)))

    FAIL_RETURN(CUDA_ERROR(cudaGraphicsUnmapResources(1, &pbo, cudaStreamPerThread)))

    GPU::Buffer<uint32_t> buffer = GPU::DeviceBuffer<uint32_t>::createBuffer((uint32_t*)devPtr, width * height);
    FAIL_RETURN(GPU::memsetToZeroBlocking(buffer, width * height * sizeof(uint32_t)))

    Potential<GPU::Surface> potremapSurf = makeSurface("Remap Buffer", width, height);
    FAIL_RETURN(potremapSurf.status())

    Potential<PanoOpenGLSurface::Pimpl> impl =
        PanoOpenGLSurface::Pimpl::create(pbo, buffer, potremapSurf.object(), width, height);
    FAIL_RETURN(impl.status())

    // ownership transferred to impl
    potremapSurf.release();

    PanoOpenGLSurface* surf = new PanoOpenGLSurface(impl.release());
    surf->pixelbuffer = pixelbuffer;
    surf->pimpl->externalAlloc = true;

    return surf;
  };

  auto pbtex = createPanoSurfacePB(width, height);
  FAIL_RETURN(pbtex.status());

  GLuint pixelbuffer = pbtex.value();

  auto potSurf = allocPotSurf(pixelbuffer, width, height, flag);

  if (!potSurf.ok()) {
    glDeleteBuffers(1, &pixelbuffer);
    return Status{Origin::GPU, ErrType::RuntimeError, "Could not map OpenGL surface to CUDA.", potSurf.status()};
  }

  return potSurf;
}

namespace {
Potential<GPU::CubemapSurface> makeCubemapSurface(std::string name, size_t width) {
  cudaArray_t array;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
  FAIL_RETURN(CUDA_ERROR(cudaMalloc3DArray(&array, &channelDesc, make_cudaExtent(width, width, 6),
                                           cudaArrayCubemap | cudaArraySurfaceLoadStore)));
  deviceStats.addPtr("Remap Buffer", array, width * width * 6);

  // Specify surface
  struct cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  // Create the surface objects
  resDesc.res.array.array = array;
  cudaSurfaceObject_t surface;
  FAIL_RETURN(CUDA_ERROR(cudaCreateSurfaceObject(&surface, &resDesc)));
  // create a texture
  cudaTextureObject_t tex;

  cudaTextureDesc texDesc = {};
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.normalizedCoords = false;
  texDesc.readMode = cudaReadModeNormalizedFloat;

  cudaResourceViewDesc resViewDesc = {};
  resViewDesc.format = cudaResViewFormatUnsignedChar4;
  resViewDesc.width = width;
  resViewDesc.height = width;
  resViewDesc.depth = 6;
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, &resViewDesc);

  return new GPU::CubemapSurface(new GPU::DeviceCubemapSurface(array, tex, surface, true), width);
}
}  // namespace

Potential<CubemapOpenGLSurface> OpenGLAllocator::createCubemapSurface(size_t width, bool equiangular,
                                                                      BufferAllocType flag) {
  std::array<GLuint, 6> pbo;

  glewInit();
  glEnable(GL_TEXTURE_CUBE_MAP);
#ifdef GL_VERSION_3_2
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
#endif
  {
    // clear error flag before mapping to CUDA
    GLenum glerr = glGetError();
    while (glerr != GL_NO_ERROR) glerr = glGetError();
  }

  auto allocPotSurf = [](const std::array<GLuint, 6>& pbo, size_t width, bool equiangular,
                         BufferAllocType flag) -> Potential<CubemapOpenGLSurface> {
    GLenum glerr = glGetError();
    if (glerr != GL_NO_ERROR) {
      return {Origin::GPU, ErrType::RuntimeError, "Could not allocate OpenGL buffer."};
    }

    cudaGraphicsResource* resources[6];
    GPU::Buffer<uint32_t> buffers[6];
    unsigned int memFlag = getCudaGLMemAllocType(flag);
    for (int i = 0; i < 6; ++i) {
      FAIL_RETURN(CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&resources[i], pbo[i], memFlag)))
      FAIL_RETURN(CUDA_ERROR(cudaGraphicsMapResources(1, &resources[i], cudaStreamPerThread)))

      void* devPtr;
      size_t size;
      FAIL_RETURN(CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, resources[i])))

      buffers[i] = GPU::DeviceBuffer<uint32_t>::createBuffer((uint32_t*)devPtr, width * width);
      FAIL_RETURN(CUDA_ERROR(cudaGraphicsUnmapResources(1, &resources[i], cudaStreamPerThread)))
    }

    PotentialValue<GPU::Buffer<uint32_t>> buf = GPU::Buffer<uint32_t>::allocate(6 * width * width, "Offscreen Surface");
    FAIL_RETURN(buf.status())

    PotentialValue<GPU::Buffer<uint32_t>> potBuf = GPU::Buffer<uint32_t>::allocate(width * width, "Cubemap");
    FAIL_RETURN(potBuf.status())

    Potential<GPU::CubemapSurface> remapSurf = makeCubemapSurface("Remap Buffer", width);
    FAIL_RETURN(remapSurf.status())

    Potential<CubemapOpenGLSurface::Pimpl> impl = CubemapOpenGLSurface::Pimpl::create(
        resources, buffers, buf.value(), remapSurf.release(), potBuf.value(), width, equiangular);
    FAIL_RETURN(impl.status())
    impl->externalAlloc = true;

    return new CubemapOpenGLSurface(impl.release(), (int*)pbo.data());
  };

  for (int i = 0; i < 6; ++i) {
    glGenBuffers(1, &pbo[i]);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[i]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * width * 4, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  }

  auto potSurf = allocPotSurf(pbo, width, equiangular, flag);

  if (!potSurf.ok()) {
    glDeleteBuffers(6, pbo.data());
    return Status{Origin::GPU, ErrType::RuntimeError, "Could not allocate OpenGL Surface.", potSurf.status()};
  }

  return potSurf;
}

Potential<PanoSurface> OffscreenAllocator::createPanoSurface(size_t width, size_t height, const char* name) {
  Status status;
  GPU::Buffer<uint32_t> buf;
  GPU::Surface* remapSurf = nullptr;

  PotentialValue<GPU::Buffer<uint32_t>> potbuf = GPU::Buffer<uint32_t>::allocate(width * height, name);
  status = potbuf.status();
  if (!status.ok()) {
    return status;
  }
  buf = potbuf.value();
  GPU::memsetToZeroBlocking(buf, width * height * sizeof(uint32_t));

  {
    Potential<GPU::Surface> potremapSurf = makeSurface("Remap Buffer", width, height);
    status = potremapSurf.status();
    if (!status.ok()) {
      goto error;
    }
    remapSurf = potremapSurf.release();
    Potential<PanoPimpl> impl = PanoPimpl::create(buf, remapSurf, width, height);
    status = impl.status();
    if (!status.ok()) {
      goto error;
    }
    PanoSurface* surf = new PanoSurface(impl.release());

    surf->pimpl->externalAlloc = false;

    return Potential<PanoSurface>(surf);
  }

error:
  buf.release();
  delete remapSurf;
  return status;
}

Potential<CubemapSurface> OffscreenAllocator::createCubemapSurface(size_t width, const char* name, bool equiangular) {
  GPU::Stream stream;
  GPU::Buffer<uint32_t> faces[6];
  GPU::Buffer<uint32_t> buf, tmp;
  Status status;

  for (int i = 0; i < 6; ++i) {
    PotentialValue<GPU::Buffer<uint32_t>> buf = GPU::Buffer<uint32_t>::allocate(width * width, name);
    status = buf.status();
    if (!status.ok()) {
      goto error_1;
    }
    GPU::memsetToZeroBlocking(buf.value(), width * width * sizeof(uint32_t));
    faces[i] = buf.value();
  }

  {
    PotentialValue<GPU::Stream> potStream = GPU::Stream::create();
    status = potStream.status();
    if (!status.ok()) {
      goto error_1;
    }
    stream = potStream.value();

    PotentialValue<GPU::Buffer<uint32_t>> potBuf =
        GPU::Buffer<uint32_t>::allocate(6 * width * width, "Offscreen Surface");
    status = potBuf.status();
    if (!status.ok()) {
      goto error_2;
    }
    buf = potBuf.value();

    potBuf = GPU::Buffer<uint32_t>::allocate(width * width, "Cubemap");
    status = potBuf.status();
    if (!status.ok()) {
      goto error_3;
    }
    tmp = potBuf.value();

    Potential<GPU::CubemapSurface> remapSurf = makeCubemapSurface("Remap Buffer", width);
    status = remapSurf.status();
    if (!status.ok()) {
      goto error_4;
    }

    CubemapPimpl* impl = new CubemapPimpl(equiangular, stream, &faces[0], buf, remapSurf.release(), tmp, width);
    CubemapSurface* surf = new CubemapSurface(impl);

    surf->pimpl->externalAlloc = false;

    return Potential<CubemapSurface>(surf);
  }

error_4:
  tmp.release();
error_3:
  buf.release();
error_2:
  stream.destroy();
error_1:
  for (int i = 0; i < 6; ++i) {
    faces[i].release();
  }

  return status;
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
