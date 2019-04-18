// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "glAllocator.hpp"

#include <sstream>

namespace VideoStitch {
namespace Core {

Status initGlew() {
#ifndef GLEWLIB_UNSUPPORTED
  GLenum glerr = glewInit();
  if (glerr != GLEW_OK) {
    return Status{Origin::GPU, ErrType::RuntimeError, "Unable to initialize glew"};
  }
#endif
  return Status::OK();
}

PotentialValue<GLuint> createSourceSurfaceTexture(size_t width, size_t height) {
  // clear error flag before mapping to CUDA/OpenCL
  GLenum glerr = glGetError();
  while (glerr != GL_NO_ERROR) {
    glerr = glGetError();
  }

  FAIL_RETURN(initGlew());

  glEnable(GL_TEXTURE_2D);

  GLuint texture;
  glGenTextures(1, (GLuint*)&texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)width, (GLsizei)height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

  glerr = glGetError();
  if (glerr != GL_NO_ERROR) {
    return Status{Origin::GPU, ErrType::RuntimeError,
                  "Could not allocate OpenGL Surface. OpenGL error " + std::to_string(glerr)};
  }

  return texture;
}

PotentialValue<GLuint> createPanoSurfacePB(size_t width, size_t height) {
  FAIL_RETURN(initGlew());

  glEnable(GL_TEXTURE_2D);
  // clear error flag before mapping to CUDA/OpenCL
  GLenum glerr = glGetError();
  while (glerr != GL_NO_ERROR) {
    glerr = glGetError();
  }

  if (width * height * 4 > std::numeric_limits<int32_t>::max()) {
    std::stringstream msg;
    msg << "Could not allocate OpenGL Surface of size " << width * height * 4;
    msg << ". Maximum supported texture size: " << std::numeric_limits<int32_t>::max();
    return Status{Origin::GPU, ErrType::OutOfResources, msg.str()};
  }

  GLuint pixelbuffer;

  glGenBuffers(1, (GLuint*)&pixelbuffer);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelbuffer);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_STREAM_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  glerr = glGetError();
  if (glerr != GL_NO_ERROR) {
    return Status{Origin::GPU, ErrType::RuntimeError,
                  "Could not allocate OpenGL Surface. OpenGL error " + std::to_string(glerr)};
  }

  return pixelbuffer;
}

}  // namespace Core
}  // namespace VideoStitch
