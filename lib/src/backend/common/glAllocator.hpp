// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/status.hpp"

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

namespace VideoStitch {
namespace Core {

PotentialValue<GLuint> createSourceSurfaceTexture(size_t width, size_t height);
PotentialValue<GLuint> createPanoSurfacePB(size_t width, size_t height);

}  // namespace Core
}  // namespace VideoStitch
