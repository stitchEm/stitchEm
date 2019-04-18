// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include <QApplication>
#include <QDesktopWidget>
#include <QCoreApplication>

#include "oculuswindow.hpp"

inline QRect getSecondaryScreenGeometry(const ovrSizei& size) {
  QDesktopWidget desktop;
  const int primary = desktop.primaryScreen();
  int monitorCount = desktop.screenCount();
  int best = -1;
  for (int i = 0; i < monitorCount; ++i) {
    if (primary == i) {
      continue;
    }
    QRect geometry = desktop.screenGeometry(i);
    QSize screenSize = geometry.size();
    if (best < 0 && (screenSize.width() >= (int)size.w && screenSize.height() >= (int)size.h)) {
      best = i;
    }
  }

  if (best < 0) {
    best = primary;
  }
  return desktop.screenGeometry(best);
}

inline QRect getPrimaryScreenGeometry() {
  auto desktop = QApplication::desktop();
  return desktop->screenGeometry(desktop->primaryScreen());
}

OculusWindow::OculusWindow(bool stereoscopic, bool mirror)
    : QWindow(),
      shuttingDown(false),
      renderThread([&] { renderLoop(); }),
      mirrorWidget(new MirrorWidget),
      context(new QOpenGLContext),
      oculus(new OculusRenderer),
      started(false),
      mirror(mirror) {
  setSurfaceType(QSurface::OpenGLSurface);

  QSurfaceFormat format;
  // Qt Quick may need a depth and stencil buffer. Always make sure these are available.
  format.setDepthBufferSize(16);
  format.setStencilBufferSize(8);
  format.setVersion(4, 3);
  format.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
  setFormat(format);

  const auto screen = getPrimaryScreenGeometry();
  mirrorWidget->setOculusSize(screen.width(), screen.height());

  context->setFormat(format);
  context->setShareContext(QOpenGLContext::globalShareContext());
  context->create();

  setFlags(Qt::FramelessWindowHint);
  show();

  context->doneCurrent();
  context->moveToThread(&renderThread);
  renderThread.setObjectName("render-to-oculus thread");

  oculus->moveToThread(&renderThread);
  connect(oculus, &OculusRenderer::renderingConfigured, this, &OculusWindow::displayMirrorWindow);
}

OculusWindow::~OculusWindow() {
  if (started) {
    stop();
    context->makeCurrent(this);
  }

  context->doneCurrent();
}

bool OculusWindow::start() {
  if (!oculus->initializeOculus()) {
    return false;
  }

  renderThread.start(QThread::HighestPriority);

  mirrorWidget->startTimer();

  started = true;
  return true;
}

void OculusWindow::displayMirrorWindow() {
  if (mirror) {
    auto mirrorTextureId = oculus->getMirrorTextureId();
    if (mirrorTextureId) {
      mirrorWidget->showMaximized();
      mirrorWidget->Init(mirrorTextureId);
    }
  }
}

// Should only be called from the primary thread
void OculusWindow::stop() {
  if (!shuttingDown) {
    shuttingDown = true;
    renderThread.quit();
    renderThread.wait();
  }
  started = false;
}

OculusRenderer& OculusWindow::getRenderer() { return *oculus; }

void OculusWindow::renderLoop() {
  // the render context is shared with the texture uploaders, they must already have acquired the context here
  // now the render context can be made current
  context->makeCurrent(this);

  const auto screen = getPrimaryScreenGeometry();
  oculus->configureRendering(screen.width(), screen.height(), mirror);
  oculus->startTimer();

  while (!shuttingDown) {
    oculus->render();
  }
  delete oculus;  // Delete the oculus renderer in its thread when the context is current
  context->doneCurrent();
  context->moveToThread(QApplication::instance()->thread());
}
