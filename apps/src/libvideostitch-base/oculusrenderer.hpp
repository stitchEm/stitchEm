// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "renderer.hpp"

#include "common-config.hpp"

#include <QElapsedTimer>
#include <QOpenGLFunctions>
#include <array>

#include "OVR_CAPI_GL.h"

struct DepthBuffer : public QOpenGLFunctions {
  GLuint texId;

  explicit DepthBuffer(ovrSizei size) {
    initializeOpenGLFunctions();
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, size.w, size.h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);
  }
  ~DepthBuffer() {
    if (texId) {
      glDeleteTextures(1, &texId);
      texId = 0;
    }
  }
};

struct TextureBuffer : public QOpenGLFunctions {
  ovrSession session;
  ovrTextureSwapChain textureSwapChain;
  GLuint fboId;
  ovrSizei texSize;

  TextureBuffer(ovrSession session, ovrSizei size) : session(session), texSize(size) {
    initializeOpenGLFunctions();

    ovrTextureSwapChainDesc desc = {};
    desc.Type = ovrTexture_2D;
    desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
    desc.ArraySize = 1;
    desc.Width = size.w;
    desc.Height = size.h;
    desc.MipLevels = 1;
    desc.SampleCount = 1;
    desc.StaticImage = ovrFalse;
    ovr_CreateTextureSwapChainGL(session, &desc, &textureSwapChain);

    int length = 0;
    ovr_GetTextureSwapChainLength(session, textureSwapChain, &length);

    for (int i = 0; i < length; ++i) {
      GLuint texId = 0;
      ovr_GetTextureSwapChainBufferGL(session, textureSwapChain, i, &texId);
      glBindTexture(GL_TEXTURE_2D, texId);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glGenFramebuffers(1, &fboId);
  }

  ~TextureBuffer() {
    if (textureSwapChain) {
      ovr_DestroyTextureSwapChain(session, textureSwapChain);
      textureSwapChain = nullptr;
    }
    if (fboId) {
      glDeleteFramebuffers(1, &fboId);
      fboId = 0;
    }
  }

  ovrSizei GetSize(void) const { return texSize; }

  void SetAndClearRenderSurface(DepthBuffer* dbuffer) {
    GLuint texId = 0;
    ovr_GetTextureSwapChainBufferGL(session, textureSwapChain, -1, &texId);

    glBindFramebuffer(GL_FRAMEBUFFER, fboId);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dbuffer->texId, 0);

    glViewport(0, 0, texSize.w, texSize.h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }

  void UnsetRenderSurface() {
    glBindFramebuffer(GL_FRAMEBUFFER, fboId);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
  }

  void commitTexture() { ovr_CommitTextureSwapChain(session, textureSwapChain); }
};

class VS_COMMON_EXPORT OculusRenderer : public Renderer {
  Q_OBJECT
 public:
  OculusRenderer();
  ~OculusRenderer();

  bool initializeOculus();
  void configureRendering(int w, int h, bool mirror);
  virtual void render();

  void startTimer() { renderTimer.start(); }

  GLuint getMirrorTextureId() const;

 signals:
  void fpsChanged(const QString);
  void orientationChanged(double yaw, double pitch, double roll);
  void renderingConfigured();

 private:
  void notifyOrientation(ovrQuatf newOrientation);

 private:
  std::array<TextureBuffer*, 2> eyeRenderTexture;
  std::array<DepthBuffer*, 2> eyeDepthBuffer;

  ovrMirrorTexture mirrorTexture;

  ovrSession session;
  ovrGraphicsLuid luid;
  ovrHmdDesc hmdDesc;

  // eye pose and fov
  ovrVector3f viewOffset[2];
  ovrEyeRenderDesc eyeRenderDesc[2];

  int nbRenderedFrames;
  QElapsedTimer renderTimer;

  bool openglInitialized;
};
