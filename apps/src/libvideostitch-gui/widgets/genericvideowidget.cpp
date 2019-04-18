// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "genericvideowidget.hpp"

#include "libvideostitch-gui/utils/sourcewidgetlayoututil.hpp"

#include <QThread>

GenericVideoWidget::GenericVideoWidget(QWidget *parent)
    : QOpenGLWidget(parent), ref_vs(0), placeHolderTex(QOpenGLTexture::TargetRectangle), name("source view") {
  clk.start();
  connect(this, SIGNAL(gotFrame(mtime_t)), this, SLOT(update()));
}

GenericVideoWidget::~GenericVideoWidget() {}

void GenericVideoWidget::clearTextures() {
  std::lock_guard<std::mutex> lock(textureMutex);
  textures.clear();
}

void GenericVideoWidget::setName(const std::string &n) { name = n; }

void GenericVideoWidget::initializeGL() {
  initializeOpenGLFunctions();
  const QColor clearColor(Qt::black);
  glClearColor(clearColor.redF(), clearColor.greenF(), clearColor.blueF(), clearColor.alphaF());
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glShadeModel(GL_FLAT);
  glEnable(GL_TEXTURE_2D);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  QImage logo(":/assets/images/home_grey");
  placeHolderTex.setData(logo);
}

std::string GenericVideoWidget::getName() const { return name; }

int GenericVideoWidget::getNumbTextures() {
  std::lock_guard<std::mutex> lock(textureMutex);
  return int(textures.size());
}

void GenericVideoWidget::resizeGL(int w, int h) {
  const qreal retinaScale = devicePixelRatio();
  glViewport(0, 0, w * retinaScale, h * retinaScale);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, w * retinaScale, h * retinaScale, 0, 0, 1);
  glMatrixMode(GL_MODELVIEW);
  update();
}

void GenericVideoWidget::paintFrame(GLuint texture, float quadWidth, float quadHeight, float verticalMargin,
                                    float horizontalMargin) {
  glBindTexture(GL_TEXTURE_2D, texture);

  glBegin(GL_QUADS);
  glTexCoord2f(0, 0);
  glVertex2f(verticalMargin, horizontalMargin);
  glTexCoord2f(0, 1);
  glVertex2f(verticalMargin, quadHeight - horizontalMargin);
  glTexCoord2f(1, 1);
  glVertex2f(quadWidth - verticalMargin, quadHeight - horizontalMargin);
  glTexCoord2f(1, 0);
  glVertex2f(quadWidth - verticalMargin, horizontalMargin);
  glEnd();

  glBindTexture(GL_TEXTURE_2D, 0);
}
