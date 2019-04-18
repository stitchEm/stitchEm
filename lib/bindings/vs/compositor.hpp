// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "../../include/libvideostitch/overlay.hpp"
#include "../../include/libvideostitch/allocator.hpp"
#include "../../include/libvideostitch/stitchOutput.hpp"

#include <GL/glew.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#ifdef __linux__
#include <GLFW/glfw3.h>
#else
#include <glfw/glfw3.h>
#endif

#include <mutex>

static std::string COMPMtag = "OpenGLRenderer";

class Compositor : public VideoStitch::GPU::Overlayer {
 public:
  Compositor(GLFWwindow* window, const VideoStitch::Core::PanoDefinition* pano, const VideoStitch::FrameRate& frameRate)
      : Overlayer() {
    this->ctx.window = window;
    this->initialize(pano, frameRate);
  }

  void attachContext() {
    ctx.lock.lock();
    glfwMakeContextCurrent(ctx.window);
  }
  void detachContext() {
    glfwMakeContextCurrent(nullptr);
    ctx.lock.unlock();
  }

 private:
  struct LockableContext {
    GLFWwindow* window;
    std::mutex lock;
  };

  LockableContext ctx;
};
