// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "lambdathread.hpp"
#include "steamvrrenderer.hpp"

#include <QWindow>
class QOpenGLContext;

class VS_COMMON_EXPORT SteamVRWindow : public QWindow {
 public:
  explicit SteamVRWindow(bool stereoscopic);
  virtual ~SteamVRWindow();

  bool start();
  // Should only be called from the primary thread
  virtual void stop();

  SteamVRRenderer& getRenderer();

 private:
  void renderLoop();

 protected:
  bool shuttingDown;
  LambdaThread renderThread;
  QOpenGLContext* context;
  SteamVRRenderer renderer;
  bool started;
};
