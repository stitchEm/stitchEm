// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mirrorwidget.hpp"

#include <QOpenGLContext>
#include <QTimer>

#include <iostream>

MirrorWidget::MirrorWidget(QWidget* const parent)
    : QOpenGLWidget(parent),
      oculusWidth(0),
      oculusHeight(0),
      outputWidth(0),
      outputHeight(0),
      updateTimer(new QTimer(this)),
      mirrorFBO(0) {
  connect(updateTimer, SIGNAL(timeout), this, SLOT(update));
}

MirrorWidget::~MirrorWidget() {
  if (mirrorFBO) {
    glDeleteFramebuffers(1, &mirrorFBO);
  }
}

void MirrorWidget::setOculusSize(int newOculusWidth, int newOculusHeight) {
  oculusWidth = newOculusWidth;
  oculusHeight = newOculusHeight;
}

void MirrorWidget::paintGL() {
  // Blit mirror texture to back buffer
  if (mirrorFBO) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, oculusHeight, oculusWidth, 0, 0, 0, outputWidth, outputHeight, GL_COLOR_BUFFER_BIT,
                      GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  }
}

void MirrorWidget::initializeGL() { initializeOpenGLFunctions(); }

void MirrorWidget::resizeGL(int width, int height) {
  outputWidth = width;
  outputHeight = height;
}

void MirrorWidget::Init(GLuint mirrorTextureId) {
  if (mirrorTextureId) {
    makeCurrent();

    glGenFramebuffers(1, &mirrorFBO);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
    glBindTexture(GL_TEXTURE_2D, mirrorTextureId);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mirrorTextureId, 0);
    glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    doneCurrent();
  }
}

void MirrorWidget::startTimer() { updateTimer->start(updateRate); }
