// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <memory>

#include "lambdathread.hpp"
#include "oculusrenderer.hpp"
#include "mirrorwidget.hpp"
#include "texture.hpp"

#include <QWindow>
#include <QOpenGLContext>

class VS_COMMON_EXPORT OculusWindow : public QWindow {
 public:
  explicit OculusWindow(bool stereoscopic, bool mirror = true);
  virtual ~OculusWindow();

  bool start();
  // Should only be called from the primary thread
  virtual void stop();

  OculusRenderer& getRenderer();

 private:
  void renderLoop();
  void displayMirrorWindow();

 protected:
  bool shuttingDown;
  LambdaThread renderThread;
  std::unique_ptr<QOpenGLContext> context;
  std::unique_ptr<MirrorWidget> mirrorWidget;
  OculusRenderer* oculus;
  bool started;
  bool mirror;
};
