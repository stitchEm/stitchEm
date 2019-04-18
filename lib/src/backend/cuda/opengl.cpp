// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/opengl.hpp"

#include "gpu/buffer.hpp"
#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "cuda/memory.hpp"
#include "cuda/error.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/logging.hpp"

#ifndef __APPLE__
#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/gl.h>
#else
#include <GL/glew.h>
#include <OpenGL/gl.h>
#endif

#include <cuda.h>
#include <cudaGL.h>

namespace VideoStitch {

enum OpenGLInitState {
  NOT_INITIALIZED,
  CONTEXT_CREATED,
  GL_BUFFER_CREATED,
  CUDA_INTEROP,
  GL_TEXTURE_CREATED,
  PREPARED,
};

std::vector<int> getGLDevices() {
  std::vector<int> glDevices;
  unsigned int cudaDeviceCount;
  int cudaDevices[1];

  CUresult result = cuGLGetDevices(&cudaDeviceCount, cudaDevices, 1, CU_GL_DEVICE_LIST_ALL);
  if (result != CUDA_SUCCESS) {
    // iMac / OS X will return "operation not supported" on the cuGLGetDevices call
    // fall back to device 0
    glDevices.push_back(0);
    return glDevices;
  }

  for (unsigned i = 0; i < cudaDeviceCount; ++i) {
    glDevices.push_back(cudaDevices[i]);
  }

  return glDevices;
}

class OpenGLUpload::Pimpl {
 public:
  Pimpl();
  ~Pimpl();

  Status upload(PixelFormat fmt, int width, int height, const char *video);

  Status initializeStorageConfiguration(PixelFormat fmt, int frameWidth, int frameHeight);
  void cleanState();

  int textureWidth, textureHeight;
  GLuint textureId;

  OpenGLInitState state;
  Cuda::DeviceUniquePtr<uint32_t> transfer;
  Cuda::DeviceUniquePtr<uint32_t> rgbaPanorama;
  CUdevice openGLDevice;
  unsigned int bufferId;
  CUcontext ctx;
  GPU::Stream gpuStream;
  CUstream stream;
};

OpenGLUpload::Pimpl::Pimpl() : textureWidth(0), textureHeight(0), textureId(0), state(NOT_INITIALIZED), bufferId(0) {
  gpuStream = GPU::Stream::create().value();
  stream = gpuStream.get();
}

OpenGLUpload::Pimpl::~Pimpl() {
  cleanState();
  gpuStream.destroy();
}

void OpenGLUpload::Pimpl::cleanState() {
  if (state == NOT_INITIALIZED) return;

  GLenum err;
  if (state == GL_TEXTURE_CREATED || state == PREPARED) {
    err = glGetError();
    if (err != GL_NO_ERROR) {
      Logger::get(Logger::Error) << "OpenGL error " << err << " at cleanState" << std::endl;
    }
    // no need to unbind the texture, it is unbind
    // after glDeleteTextures
    glDeleteTextures(1, &textureId);
    err = glGetError();
    if (err != GL_NO_ERROR) {
      Logger::get(Logger::Error) << "OpenGL error " << err << " on glDeleteTextures" << std::endl;
    }
    state = CUDA_INTEROP;
  }

  if (state == CUDA_INTEROP) {
    cuGLUnregisterBufferObject(bufferId);
    state = GL_BUFFER_CREATED;
  }

  if (state == GL_BUFFER_CREATED) {
    // no need to unbind buffers, it is unbind
    // after glDeleteTextures
    glDeleteBuffers(1, &bufferId);
    err = glGetError();
    if (err != GL_NO_ERROR) {
      Logger::get(Logger::Error) << "OpenGL error " << err << " on glDeleteBuffers" << std::endl;
    }

    state = CONTEXT_CREATED;
  }

  if (state == CONTEXT_CREATED) {
    cuCtxDestroy(ctx);
    textureWidth = 0;
    textureHeight = 0;
  }

  state = NOT_INITIALIZED;
}

Status OpenGLUpload::Pimpl::initializeStorageConfiguration(PixelFormat fmt, int frameWidth, int frameHeight) {
  Status ret;

  cleanState();

  /*Enable OpenGL extensions*/
  GLenum glErr = glewInit();
  if (glErr != GLEW_OK) {
    return {Origin::GPU, ErrType::SetupFailure, "Unable to initialize glew"};
  }

  /*Enable texture 2d for opengl*/
  glEnable(GL_TEXTURE_2D);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::SetupFailure, "GL_TEXTURE_2D is not available."};
  }

  /*Get the maximum available texture size*/
  GLint maxTextureSize = 0;
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
  if (frameWidth > maxTextureSize || frameHeight > maxTextureSize) {
    return {Origin::GPU, ErrType::SetupFailure,
            "The frame size is not supported by OpenGL (maxTextureSize = " + std::to_string(maxTextureSize) + ")"};
  }

  /*Retrieve the device which do the openGL work*/
  std::vector<int> devices = getGLDevices();
  openGLDevice = devices[0];

  ret = CUDA_ERROR(cuGLCtxCreate(&ctx, CU_CTX_SCHED_AUTO, openGLDevice));
  if (!ret.ok()) {
    cleanState();
    return ret;
  }

  state = CONTEXT_CREATED;

  ret = CUDA_ERROR(cuCtxPushCurrent(ctx));
  if (!ret.ok()) {
    cleanState();
    return ret;
  }

  textureWidth = frameWidth;
  textureHeight = frameHeight;

  /*Intermediate buffer for YV12 conversion*/
  if (fmt == VideoStitch::YV12) {
    ret = rgbaPanorama.alloc(frameWidth * frameHeight, "OpenGL");
    if (!ret.ok()) {
      cleanState();
      return ret;
    }
  }

  /* 1- Allocate a GL buffer the size of the image. Cannot throw an error as the first parameter is a constant */
  glGenBuffers(1, &bufferId);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::UnsupportedAction, "Error on glGenBuffers"};
  }
  state = GL_BUFFER_CREATED;

  /* Make this the current UNPACK buffer (OpenGL is state-based).*/
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferId);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::UnsupportedAction, "GL_PIXEL_UNPACK_BUFFER is not available"};
  }

  /* Allocate data for the buffer. 4-channel 8-bit image */
  glBufferData(GL_PIXEL_UNPACK_BUFFER, frameWidth * frameHeight * 4, NULL, GL_DYNAMIC_DRAW);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::UnsupportedAction, "GL_PIXEL_UNPACK_BUFFER is not available"};
  }

  /*Associate GL buffer with cuda memory space */
  ret = CUDA_ERROR(cuGLRegisterBufferObject(bufferId));
  if (!ret.ok()) {
    cleanState();
    return ret;
  }

  state = CUDA_INTEROP;

  // 2- Allocate a GL texture the size of the image
  glGenTextures(1, &textureId);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::UnsupportedAction, "glGenTextures returned error"};
  }

  /*Activate texture*/
  glBindTexture(GL_TEXTURE_2D, textureId);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::UnsupportedAction, "GL_TEXTURE_2D is not available"};
  }

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, textureWidth, textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::SetupFailure, "Unable to initialize OpenGL texture"};
  }

  /*Texture parameters*/
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::SetupFailure, "Unable to set OpenGL texture min filter to GL_LINEAR"};
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::SetupFailure, "Unable to set OpenGL texture mag filter to GL_LINEAR"};
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::SetupFailure, "Unable to set OpenGL texture wrap S to GL_CLAMP_TO_EDGE"};
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::SetupFailure, "Unable to set OpenGL texture wrap T to GL_CLAMP_TO_EDGE"};
  }

  state = GL_TEXTURE_CREATED;

  PROPAGATE_FAILURE_STATUS(CUDA_ERROR(cuCtxPopCurrent(&ctx)));

  state = PREPARED;

  return Status::OK();
}

std::mutex opengl_mutex;

Status OpenGLUpload::Pimpl::upload(VideoStitch::PixelFormat fmt, int width, int height, const char *video) {
  // late storage configuration
  if (state == NOT_INITIALIZED) {
    /*
      Need to create a mutex locker here to prevent computer with old GPUs (for example: GT 640) from freezing.
      The problem likely comes from a bug of the old graphic card driver that hangs computer when multiple threads are
      calling the "cuGLCtxCreate" function.
      */
    std::lock_guard<std::mutex> guard(opengl_mutex);
    Status initStatus = initializeStorageConfiguration(fmt, width, height);
    if (!initStatus.ok()) {
      cleanState();
    }
    FAIL_RETURN(initStatus);
  }

  if (state != PREPARED) {
    return {Origin::GPU, ErrType::ImplementationError, "OpenGL output has not been initialized completely"};
  }

  Status ret;

  PROPAGATE_FAILURE_STATUS(CUDA_ERROR(cuCtxPushCurrent(ctx)));

  // 3- Map the GL buffer to CUDA memory
  CUdeviceptr glBuffer;
  size_t size;
  PROPAGATE_FAILURE_STATUS(CUDA_ERROR(cuGLMapBufferObject(&glBuffer, &size, bufferId)));

  // 4- Write the image from CUDA to the mapped memory

  // Which GPU owns the video buffer ?
  CUcontext origin;
  PROPAGATE_FAILURE_STATUS(
      CUDA_ERROR(cuPointerGetAttribute(&origin, CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr)video)));
  // If its not on the same GPU, first copy the video to the current gpu
  CUdeviceptr buffer = (CUdeviceptr)video;

  PROPAGATE_FAILURE_STATUS(CUDA_ERROR(cuCtxPushCurrent(origin)));
  CUdevice originDevice;
  PROPAGATE_FAILURE_STATUS(CUDA_ERROR(cuCtxGetDevice(&originDevice)));
  PROPAGATE_FAILURE_STATUS(CUDA_ERROR(cuCtxPopCurrent(&origin)));
  if (originDevice != openGLDevice) {
    buffer = (CUdeviceptr)transfer.get();

    // lazy alloc
    if (!buffer) {
      ret = transfer.alloc(width * height, "OpenGL");
      if (!ret.ok()) {
        cuGLUnmapBufferObject(bufferId);
        cuCtxPopCurrent(&ctx);
        return ret;
      }
      buffer = (CUdeviceptr)transfer.get();
    }
    size_t transfer_size;
    switch (fmt) {
      case VideoStitch::YV12: {
        transfer_size = (width * height * 3) / 2;
        break;
      }
      case VideoStitch::RGBA:
      default: {
        transfer_size = width * height * 4;
        break;
      }
    }
    ret = CUDA_ERROR(cuMemcpy(buffer, (CUdeviceptr)video, transfer_size));
    if (!ret.ok()) {
      cuGLUnmapBufferObject(bufferId);
      cuCtxPopCurrent(&ctx);
      return ret;
    }
  }

  CUdeviceptr rgbaPtr;
  switch (fmt) {
      /*
          case VideoStitch::YV12: {
            auto gpuRGBAGPano = GPU::DeviceBuffer<uint32_t>::createBuffer(rgbaPanorama.get(), width *height);
            auto gpuBuffer = GPU::DeviceBuffer<unsigned char>::createBuffer((unsigned char *)buffer, (width * height) *
         3 / 2);
            // Unpack and copy to texture
            VideoStitch::Input::VideoReader::unpackDevBuffer(VideoStitch::YV12,
                                                             gpuRGBAGPano,
                                                             gpuBuffer,
                                                             width, height, gpuStream);
            rgbaPtr = (CUdeviceptr) rgbaPanorama.get();
            break;
          }
          */
    case VideoStitch::RGBA: {
      rgbaPtr = buffer;
      break;
    }
    default:
      rgbaPtr = buffer;
      break;
  }
  PROPAGATE_FAILURE_STATUS(CUDA_ERROR(cuMemcpy(glBuffer, rgbaPtr, width * height * 4)));

  // 5- Unmap the GL buffer
  PROPAGATE_FAILURE_STATUS(CUDA_ERROR(cuGLUnmapBufferObject(bufferId)));

  // 6- Create a Texture From the Buffer
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferId);
  GLenum glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::RuntimeError, "Unable to bind buffer"};
  }
  glBindTexture(GL_TEXTURE_2D, textureId);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::RuntimeError, "Unable to bind texture"};
  }
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::RuntimeError, "Unable to tex sub image"};
  }
  // It's necessary to flush our texture upload commands here,
  // else the rendering might end up using an outdated texture
  // (eg. you sync, and everything but the openGL display report
  // the correct frame)
  glFlush();
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::RuntimeError, "Unable to flush"};
  }
  // unbind openGL texture
  glBindTexture(GL_TEXTURE_2D, 0);

  PROPAGATE_FAILURE_STATUS(CUDA_ERROR(cuCtxPopCurrent(&ctx)));
  return Status::OK();
}

OpenGLUpload::OpenGLUpload() { pimpl = new Pimpl; }
OpenGLUpload::~OpenGLUpload() { delete pimpl; }

Status OpenGLUpload::upload(VideoStitch::PixelFormat fmt, int width, int height, const char *video) {
  return pimpl->upload(fmt, width, height, video);
}

void OpenGLUpload::cleanState() { pimpl->cleanState(); }

int OpenGLUpload::getTexWidth() const { return pimpl->textureWidth; }
int OpenGLUpload::getTexHeight() const { return pimpl->textureHeight; }
int OpenGLUpload::getTexId() const { return (int)pimpl->textureId; }

}  // namespace VideoStitch
