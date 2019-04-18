// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/opengl.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/logging.hpp"

#include "gpu/buffer.hpp"
#include "gpu/stream.hpp"

#ifndef __APPLE__
#ifndef __ANDROID__
#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/gl.h>
#else
#ifndef GLEWLIB_UNSUPPORTED
#define GLEW_STATIC
#include <GL/glew.h>
#else
#include <GLES3/gl3.h>
#endif
#endif
#else
#include <GL/glew.h>
#include <OpenGL/gl.h>
#endif

#include "cl_error.hpp"

#include "opencl.h"

#include <mutex>

namespace VideoStitch {

enum OpenGLInitState {
  NOT_INITIALIZED,
  HOST_BUFFER_CREATED,
  GL_TEXTURE_CREATED,
  PREPARED,
};

std::vector<int> getGLDevices() {
  std::vector<int> glDevices = {0};
  // unsigned int cudaDeviceCount;
  // int cudaDevices[1];

  // CUresult result = cuGLGetDevices(&cudaDeviceCount, cudaDevices, 1, CU_GL_DEVICE_LIST_ALL);
  // if (result != CUDA_SUCCESS) {
  //   // iMac / OS X will return "operation not supported" on the cuGLGetDevices call
  //   // fall back to device 0
  //   glDevices.push_back(0);
  //   return glDevices;
  // }

  // for (unsigned i = 0; i < cudaDeviceCount; ++i) {
  //   glDevices.push_back(cudaDevices[i]);
  // }

  return glDevices;
}

class OpenGLUpload::Pimpl {
 public:
  Pimpl();
  ~Pimpl();

  Status upload(PixelFormat fmt, int width, int height, const char *video);

  Status initializeStorageConfiguration(int frameWidth, int frameHeight, const char *video);
  void cleanState();

  int textureWidth, textureHeight;
  GLuint textureId;

  OpenGLInitState state;
  unsigned int bufferId;
  cl_command_queue command_queue = nullptr;
  std::unique_ptr<unsigned char[]> hostMem;
};

OpenGLUpload::Pimpl::Pimpl() : textureWidth(0), textureHeight(0), textureId(0), state(NOT_INITIALIZED), bufferId(0) {}

OpenGLUpload::Pimpl::~Pimpl() { cleanState(); }

void OpenGLUpload::Pimpl::cleanState() {
  if (state == NOT_INITIALIZED) return;

  Status st;
  GLenum glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    Logger::get(Logger::Error) << "OpenGL error " << glErr << " at cleanState" << std::endl;
  }

  if (state == PREPARED) {
    // release command queue
    st = CL_ERROR(clReleaseCommandQueue(command_queue));
    if (!st.ok()) {
      Logger::get(Logger::Error) << "OpenCL error " << st.getErrorMessage() << " on clReleaseCommandQueue" << std::endl;
    }
    command_queue = nullptr;
    state = GL_TEXTURE_CREATED;
  }

  if (state == GL_TEXTURE_CREATED) {
    // no need to unbind the texture, it is unbind
    // after glDeleteTextures
    glDeleteTextures(1, &textureId);
    glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
      Logger::get(Logger::Error) << "OpenGL error " << glErr << " on glDeleteTextures" << std::endl;
    }
    state = HOST_BUFFER_CREATED;
  }

  if (state == HOST_BUFFER_CREATED) {
    hostMem.reset(nullptr);
    textureWidth = 0;
    textureHeight = 0;
  }

  state = NOT_INITIALIZED;
}

Status OpenGLUpload::Pimpl::initializeStorageConfiguration(int frameWidth, int frameHeight, const char *video) {
  cleanState();

  /*Enable OpenGL extensions*/
#ifndef GLEWLIB_UNSUPPORTED
  GLenum glErr = glewInit();
  if (glErr != GLEW_OK) {
    return {Origin::GPU, ErrType::SetupFailure, "Unable to initialize glew"};
  }
#else
  GLenum glErr;
#endif

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

  // 1- Allocate a host buffer the size of the image
  hostMem.reset(new unsigned char[frameWidth * frameHeight * 4]);

  textureWidth = frameWidth;
  textureHeight = frameHeight;

  state = HOST_BUFFER_CREATED;

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

  cl_mem buf = (cl_mem)video;

  // Command queue needs to be created in OpenCL context that is used for stitching
  // Uploader does not know the stitching device, let's query the context
  cl_context buf_context;
  PROPAGATE_CL_ERR(clGetMemObjectInfo(buf, CL_MEM_CONTEXT, sizeof(cl_context), &buf_context, nullptr));

  // pick the first device in the context
  // TODO if the 2nd device is used for stitching, does this work?
  cl_device_id context_first_device;
  PROPAGATE_CL_ERR(
      clGetContextInfo(buf_context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &context_first_device, nullptr));

  cl_int err;
  command_queue = clCreateCommandQueue(buf_context, context_first_device, 0, &err);
  PROPAGATE_CL_ERR(err);

  state = PREPARED;

  return Status::OK();
}

std::mutex opengl_mutex;

Status OpenGLUpload::Pimpl::upload(VideoStitch::PixelFormat /*fmt*/, int width, int height, const char *video) {
  GLenum glErr;
  // late storage configuration
  if (textureWidth != width || textureHeight != height) {
    PROPAGATE_FAILURE_STATUS(initializeStorageConfiguration(width, height, video));
  }

  if (state != PREPARED) {
    return {Origin::GPU, ErrType::SetupFailure, "OpenGL output has not been initialized completely"};
  }

  // 1-Enqueue commands to read from a buffer object to host memory
  cl_mem buf = (cl_mem)video;
  PROPAGATE_CL_ERR(clEnqueueReadBuffer(command_queue, buf, CL_TRUE, 0, width * height * 4, (void *)hostMem.get(), 0,
                                       nullptr, nullptr));
  PROPAGATE_CL_ERR(clFinish(command_queue));

  // 2-Enqueue commands to read from a buffer object to host memory
  glBindTexture(GL_TEXTURE_2D, textureId);
  glErr = glGetError();
  if (glErr != GL_NO_ERROR) {
    return {Origin::GPU, ErrType::RuntimeError, "Unable to bind texture"};
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, textureWidth, textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, hostMem.get());
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
