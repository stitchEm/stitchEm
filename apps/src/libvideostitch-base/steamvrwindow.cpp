// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "steamvrwindow.hpp"
#include <QApplication>
#include <QOpenGLContext>

SteamVRWindow::SteamVRWindow(bool stereoscopic)
    : QWindow(),
      shuttingDown(false),
      renderThread([&] { renderLoop(); }),
      context(nullptr),
      renderer(),
      started(false) {
  setSurfaceType(QSurface::OpenGLSurface);

  QSurfaceFormat format;
  // Qt Quick may need a depth and stencil buffer. Always make sure these are available.
  format.setDepthBufferSize(16);
  format.setStencilBufferSize(8);
  format.setVersion(4, 3);
  format.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
  setFormat(format);

  context = new QOpenGLContext;
  context->setFormat(format);
  context->setShareContext(QOpenGLContext::globalShareContext());
  context->create();

  setFlags(Qt::FramelessWindowHint);
  show();

  context->doneCurrent();
  context->moveToThread(&renderThread);
  renderThread.setObjectName("render-to-vive thread");
}

SteamVRWindow::~SteamVRWindow() {
  if (started) {
    stop();
  }
}

bool SteamVRWindow::start() {
  if (!renderer.initializeSteamVR()) {
    return false;
  }

  // start the render thread after the context has been shared
  renderThread.start(QThread::HighestPriority);

  started = true;
  return true;
}

// Should only be called from the primary thread
void SteamVRWindow::stop() {
  if (!shuttingDown) {
    shuttingDown = true;
    renderThread.quit();
    renderThread.wait();
  }
  started = false;
  renderer.uninitializeSteamVR();
}

SteamVRRenderer& SteamVRWindow::getRenderer() { return renderer; }

void SteamVRWindow::renderLoop() {
  // the render context can be shared with texture uploaders, they must already have acquired the context here
  // now the render context can be made current
  context->makeCurrent(this);
  renderer.configureRendering(0, 0);
  while (!shuttingDown) {
    renderer.render();
  }
  context->doneCurrent();
  context->moveToThread(QApplication::instance()->thread());
}
