// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "frame.hpp"
#include "status.hpp"
#include <vector>

namespace VideoStitch {

/**
 * Convenience function uploading a managed pixel buffer
 * to an OpenGL texture.
 * The transfers occuring never use the Host memory, although in multiple
 * GPUs setups some might be controlled by a CPU and not by the PCI controller.
 *
 * OpenGL context management is left to the user.
 * upload(...) needs an OpenGL context to be active on the current thread.
 *
 * This function is meant to be used in an VideoWriter implementation.
 * Beware, the thread calling pushFrame(...) is not necessarily the same
 * one as the thread calling stitch(...).
 *
 * Example below for GLX:
 *
 * class GLXCallback : public Output::VideoWriter, OpenGLUpload {
 * public:
 *   GLXCallback(GLXContext& ctx, unsigned width, unsigned height)
 *   : Output::VideoWriter("display 0", width, height, 0, 0, VideoStitch::RGBA, Device) {
 *     dpy  = XOpenDisplay(nullptr);
 *     root = DefaultRootWindow(dpy);
 *     vi   = glXChooseVisual(dpy, 0, attr);
 *     glc  = glXCreateContext(dpy, vi, ctx, GL_TRUE);
 *   }
 *
 *   ~GLXCallback() {
 *     glXDestroyContext(dpy, glc);
 *   }
 *
 *   void pushFrame(int frame, const char* videoFrame, size_t nbAudioSamples, uint8_t* const* audioSamples) {
 *     glXMakeCurrent(dpy, root, glc);
 *     upload(fmt, width, height, videoFrame);
 *   }
 *
 * private:
 *   Display *dpy;
 *   Window root;
 *   GLint attr[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
 *   XVisualInfo *vi;
 *   GLXContext glc;
 * };
 *
 */
class VS_EXPORT OpenGLUpload {
 public:
  class Pimpl;

  OpenGLUpload();
  virtual ~OpenGLUpload();

  /**
   * Allocates a texture in the current context if necessary and uploads the video frame
   * to it. getTexId() to retrieve and bind the associated texture.
   */
  Status upload(PixelFormat fmt, int width, int height, const char* video);

  // valid after the first call to pushFrame
  /** Texture dimension */
  int getTexWidth() const;
  /** Texture dimension */
  int getTexHeight() const;
  /** Texture identifier */
  int getTexId() const;

 protected:
  void cleanState();

 private:
  Pimpl* pimpl;
};

/**
 * Return the GPGPU-compatible devices corresponding to the current OpenGL context.
 * There can be several of them if the setup is using SLI fir example (not recommended
 * for VideoStitch).
 *
 * When using multiple GPUs, display performances will peak if using
 * the OpenGL device as a primary target for the Output::VideoWriter callbacks,
 * by diminishing the number of frame transfers between GPUs.
 *
 * Use the returned value is recommended as:
 * - the primary device for composite outputs such as the Anaglyph and
 *   TopDown / LeftRight stereo callbacks.
 * - the device for your main Stitcher, through the Controller's DeviceDefinition
 *
 * Needs an active OpenGL context for introspection, like the upload(...) function.
 */
VS_EXPORT std::vector<int> getGLDevices();
}  // namespace VideoStitch
