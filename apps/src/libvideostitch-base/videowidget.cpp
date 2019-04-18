// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videowidget.hpp"
#include "yprsignalcaps.hpp"
#include "geometry.hpp"

VideoWidget::VideoWidget(QWidget* parent)
    : QOpenGLWidget(parent),
      yprsignal(nullptr),
      yaw(0.0),
      pitch(0.0),
      roll(0.0),
      button(Qt::MouseButton::NoButton),
      proj(VideoStitch::equirectangular),
      HFOV(360.0),
      zoom(MIN_ZOOM),
      editOrientation(false),
      enableZoom(false),
      width_(0),
      height_(0),
      gridSizeX(1),
      gridSizeY(1) {}

VideoWidget::~VideoWidget() {}

void VideoWidget::initializeGL() {
  initializeOpenGLFunctions();
  const QColor clearColor(Qt::black);
  glClearColor(clearColor.redF(), clearColor.greenF(), clearColor.blueF(), clearColor.alphaF());
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glShadeModel(GL_FLAT);
  glMatrixMode(GL_MODELVIEW);

  glGenTextures(1, &Texture::get().id);
}

void VideoWidget::paintGL() {
  std::lock_guard<std::mutex> textureLock(*(Texture::get().lock));
  Texture::Type textureType = Texture::get().getType();
  switch (textureType) {
    case Texture::Type::PANORAMIC:
      paintPano();
      break;
    case Texture::Type::CUBEMAP:
    case Texture::Type::EQUIANGULAR_CUBEMAP:
      paintCubemap();
      break;
  }

  paintGrid();
}

void VideoWidget::paintGrid() {
  if (editOrientation) {
    glLineWidth(2);
    glColor3f(gridColor.redF(), gridColor.greenF(), gridColor.blueF());
    glBegin(GL_LINES);

    // Horizontals
    float step_y = height() / gridSizeY;
    for (int i = 1; i != gridSizeY; ++i) {
      glVertex2f(-width() / 2, -height() / 2 + step_y * i);
      glVertex2f(width() / 2, -height() / 2 + step_y * i);
    }

    // Verticals
    float step_x = width() / gridSizeX;
    for (int i = 1; i != gridSizeX; ++i) {
      glVertex2f(-width() / 2 + step_x * i, -height() / 2);
      glVertex2f(-width() / 2 + step_x * i, height() / 2);
    }

    // Middle
    glColor3f(gridColorHighlight.redF(), gridColorHighlight.greenF(), gridColorHighlight.blueF());
    glVertex2f(-width() / 2, -height() / 2 + step_y * gridSizeY / 2);
    glVertex2f(width() / 2, -height() / 2 + step_y * gridSizeY / 2);
    glVertex2f(-width() / 2 + step_x * gridSizeX / 2, -height() / 2);
    glVertex2f(-width() / 2 + step_x * gridSizeX / 2, height() / 2);

    glEnd();
    glColor3f(1, 1, 1);
    glLineWidth(1);
  }
}

void VideoWidget::paintPano() {
  Texture::get().latePanoramaDef();

  glEnable(GL_TEXTURE_2D);

  glClear(GL_COLOR_BUFFER_BIT);
  glBindTexture(GL_TEXTURE_2D, Texture::get().id);

  width_ = Texture::get().getWidth();
  height_ = Texture::get().getHeight();

  if (Texture::get().getWidth() != 0) {
    if (Texture::get().pixelBuffer != 0) {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Texture::get().pixelBuffer);
      glBindTexture(GL_TEXTURE_2D, Texture::get().id);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Texture::get().getWidth(), Texture::get().getHeight(), GL_RGBA,
                      GL_UNSIGNED_BYTE, nullptr);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    glLoadIdentity();
    glTranslatef(width() / 2., height() / 2., 0.);
    // normalize pan
    QPointF maxPan((width() * (zoom - 100.) / 100.) / 2., (height() * (zoom - 100.) / 100.) / 2.);
    float x = 0., y = 0.;
    if (pan.x() >= 0) {
      x = std::min(pan.x(), maxPan.x());
    } else {
      x = std::max(pan.x(), -maxPan.x());
    }
    if (pan.y() >= 0) {
      y = std::min(pan.y(), maxPan.y());
    } else {
      y = std::max(pan.y(), -maxPan.y());
    }
    pan = QPointF(x, y);
    glTranslatef(pan.x(), pan.y(), 0.);
    // zoom in
    glScalef(zoom / 100., zoom / 100., 1.);

    glBegin(GL_QUADS);
    if (getWidgetAspectRatio() > getTextureAspectRatio()) {
      const float margin = (width() - Texture::get().getWidth() * height() / Texture::get().getHeight()) / 2;
      glTexCoord2f(0, 0);
      glVertex2f(margin - width() / 2., -height() / 2.);
      glTexCoord2f(0, 1);
      glVertex2f(margin - width() / 2., height() / 2.);
      glTexCoord2f(1, 1);
      glVertex2f(width() / 2. - margin, height() / 2);
      glTexCoord2f(1, 0);
      glVertex2f(width() / 2. - margin, -height() / 2.);
    } else {
      const float margin = (height() - Texture::get().getHeight() * width() / Texture::get().getWidth()) / 2;
      glTexCoord2f(0, 0);
      glVertex2f(-width() / 2, margin - height() / 2.);
      glTexCoord2d(0, 1);
      glVertex2f(-width() / 2, height() / 2. - margin);
      glTexCoord2d(1, 1);
      glVertex2f(width() / 2., height() / 2. - margin);
      glTexCoord2d(1, 0);
      glVertex2f(width() / 2., margin - height() / 2.);
    }
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
  }

  glDisable(GL_TEXTURE_2D);
}

void VideoWidget::paintCubemap() {
  Texture::get().lateCubemapDef();

  glEnable(GL_TEXTURE_CUBE_MAP);

  glClear(GL_COLOR_BUFFER_BIT);
  glBindTexture(GL_TEXTURE_CUBE_MAP, Texture::get().id);

  width_ = Texture::get().getWidth();
  height_ = Texture::get().getHeight();

  if (Texture::get().getWidth() != 0) {
    for (int i = 0; i < 6; ++i) {
      if (Texture::get().pbo[i] != 0) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Texture::get().pbo[i]);
        glTexSubImage2D(cube[i], 0, 0, 0, Texture::get().getLength(), Texture::get().getLength(), GL_RGBA,
                        GL_UNSIGNED_BYTE, nullptr);
      }
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // paintDice();
    paintCompact();

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
  }

  glDisable(GL_TEXTURE_CUBE_MAP);
}

void VideoWidget::paintDice() {
  glBegin(GL_QUADS);
  float hMargin = 0, vMargin = 0;
  float width;
  if (getWidgetAspectRatio() > 4. / 3.) {
    hMargin = (this->width() - 4 * Texture::get().getLength() * height() / (3 * Texture::get().getLength())) / 2;
    width = height() / 3;
  } else {
    vMargin = (height() - 3 * Texture::get().getLength() * this->width() / (4 * Texture::get().getLength())) / 2;
    width = this->width() / 4.;
  }

  // +x
  glTexCoord3f(1, -1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin + width);
  glTexCoord3f(1., 1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin + 2 * width);
  glTexCoord3f(1., 1., 1.);
  glVertex2f(hMargin + 3 * width, vMargin + 2 * width);
  glTexCoord3f(1., -1., 1.);
  glVertex2f(hMargin + 3 * width, vMargin + width);
  // -x
  glTexCoord3f(-1., -1., 1.);
  glVertex2f(hMargin, vMargin + width);
  glTexCoord3f(-1., 1., 1.);
  glVertex2f(hMargin, vMargin + 2 * width);
  glTexCoord3f(-1., 1., -1.);
  glVertex2f(hMargin + width, vMargin + 2 * width);
  glTexCoord3f(-1., -1., -1.);
  glVertex2f(hMargin + width, vMargin + width);
  // +y
  glTexCoord3f(-1., 1., -1.);
  glVertex2f(hMargin + width, vMargin + 2 * width);
  glTexCoord3f(-1., 1., 1.);
  glVertex2f(hMargin + width, vMargin + 3 * width);
  glTexCoord3f(1., 1., 1.);
  glVertex2f(hMargin + 2 * width, vMargin + 3 * width);
  glTexCoord3f(1., 1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin + 2 * width);
  // -y
  glTexCoord3f(-1., -1., 1.);
  glVertex2f(hMargin + width, vMargin);
  glTexCoord3f(-1., -1., -1.);
  glVertex2f(hMargin + width, vMargin + width);
  glTexCoord3f(1., -1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin + width);
  glTexCoord3f(1., -1., 1.);
  glVertex2f(hMargin + 2 * width, vMargin);
  // +z
  glTexCoord3f(-1., -1., -1.);
  glVertex2f(hMargin + width, vMargin + width);
  glTexCoord3f(-1., 1., -1.);
  glVertex2f(hMargin + width, vMargin + 2 * width);
  glTexCoord3f(1., 1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin + 2 * width);
  glTexCoord3f(1., -1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin + width);
  // -z
  glTexCoord3f(1., -1., 1.);
  glVertex2f(hMargin + 3 * width, vMargin + width);
  glTexCoord3f(1., 1., 1.);
  glVertex2f(hMargin + 3 * width, vMargin + 2 * width);
  glTexCoord3f(-1., 1., 1.);
  glVertex2f(hMargin + 4 * width, vMargin + 2 * width);
  glTexCoord3f(-1., -1., 1.);
  glVertex2f(hMargin + 4 * width, vMargin + width);

  glEnd();
}

void VideoWidget::paintCompact() {
  glBegin(GL_QUADS);
  float hMargin = 0, vMargin = 0;
  float width;
  if (getWidgetAspectRatio() > 3. / 2.) {
    hMargin = (this->width() - 3 * Texture::get().getLength() * height() / (2 * Texture::get().getLength())) / 2;
    width = height() / 2;
  } else {
    vMargin = (height() - 2 * Texture::get().getLength() * this->width() / (3 * Texture::get().getLength())) / 2;
    width = this->width() / 3.;
  }

  // +x
  glTexCoord3f(-1., -1., 1.);
  glVertex2f(hMargin, vMargin);
  glTexCoord3f(-1., 1., 1.);
  glVertex2f(hMargin, vMargin + width);
  glTexCoord3f(-1., 1., -1.);
  glVertex2f(hMargin + width, vMargin + width);
  glTexCoord3f(-1, -1., -1.);
  glVertex2f(hMargin + width, vMargin);
  // -x
  glTexCoord3f(1., -1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin);
  glTexCoord3f(1., 1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin + width);
  glTexCoord3f(1., 1., 1.);
  glVertex2f(hMargin + 3 * width, vMargin + width);
  glTexCoord3f(1., -1., 1.);
  glVertex2f(hMargin + 3 * width, vMargin);
  // -y
  glTexCoord3f(1., 1., -1.);
  glVertex2f(hMargin, vMargin + width);
  glTexCoord3f(-1., 1., -1.);
  glVertex2f(hMargin, vMargin + 2 * width);
  glTexCoord3f(-1., 1., 1.);
  glVertex2f(hMargin + width, vMargin + 2 * width);
  glTexCoord3f(1., 1., 1.);
  glVertex2f(hMargin + width, vMargin + width);
  // +y
  glTexCoord3f(1., -1., 1.);
  glVertex2f(hMargin + 2 * width, vMargin + width);
  glTexCoord3f(-1., -1., 1.);
  glVertex2f(hMargin + 2 * width, vMargin + 2 * width);
  glTexCoord3f(-1., -1., -1.);
  glVertex2f(hMargin + 3 * width, vMargin + 2 * width);
  glTexCoord3f(1., -1., -1.);
  glVertex2f(hMargin + 3 * width, vMargin + width);
  // +z
  glTexCoord3f(1., 1., 1.);
  glVertex2f(hMargin + width, vMargin + width);
  glTexCoord3f(1., -1., 1.);
  glVertex2f(hMargin + 2 * width, vMargin + width);
  glTexCoord3f(-1., -1., 1.);
  glVertex2f(hMargin + 2 * width, vMargin + 2 * width);
  glTexCoord3f(-1., 1., 1.);
  glVertex2f(hMargin + width, vMargin + 2 * width);
  // -z
  glTexCoord3f(-1., -1., -1.);
  glVertex2f(hMargin + width, vMargin);
  glTexCoord3f(-1., 1., -1.);
  glVertex2f(hMargin + width, vMargin + width);
  glTexCoord3f(1., 1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin + width);
  glTexCoord3f(1., -1., -1.);
  glVertex2f(hMargin + 2 * width, vMargin);

  glEnd();
}

void VideoWidget::resizeGL(int w, int h) {
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, w, h, 0, 0, 1);
  glMatrixMode(GL_MODELVIEW);
}

void VideoWidget::mousePressEvent(QMouseEvent* e) {
  if (editOrientation) {
    prevPos = e->pos();
    std::lock_guard<std::mutex> textureLock(*(Texture::get().lock));
    point2pano(prevPos);
    button = e->button();
  } else {
    prevPos = e->pos();
  }
  if (!((e->buttons() & Qt::RightButton) && (e->buttons() & Qt::LeftButton))) {
    yprsignal = YPRSignalCaps::create();
  }
}

void VideoWidget::mouseMoveEvent(QMouseEvent* event) {
  if (editOrientation) {
    // accessing texture width and height
    std::lock_guard<std::mutex> textureLock(*(Texture::get().lock));
    curPos = event->pos();
    point2pano(curPos);
    QVector3D v1, v2;
    switch (proj) {
      case VideoStitch::equirectangular:
        equirectangular2sphere(curPos, v2, sphereRadius());
        equirectangular2sphere(prevPos, v1, sphereRadius());
        break;
      case VideoStitch::fullframe_fisheye:
      case VideoStitch::circular_fisheye:
        spherical2sphere(curPos, v2, sphereRadius());
        spherical2sphere(prevPos, v1, sphereRadius());
        break;
      case VideoStitch::rectilinear:
        rectilinear2sphere(curPos, v2, sphereRadius());
        rectilinear2sphere(prevPos, v1, sphereRadius());
        break;
      case VideoStitch::stereographic:
        stereographic2sphere(curPos, v2, sphereRadius());
        stereographic2sphere(prevPos, v1, sphereRadius());
        break;
      case VideoStitch::equiangular_cubemap:
      case VideoStitch::cubemap:
      case VideoStitch::interactive:
      case VideoStitch::unknownProjection:
        break;
    }
    qreal y, p, r;
    sphere2orientation(v2, v1, y, p, r);
    roll = radToDeg(r);
    if (button == Qt::RightButton) {
      yaw = 0.0;
      pitch = 0.0;
    } else {
      yaw = radToDeg(y);
      pitch = radToDeg(p);
    }
    prevPos = curPos;
  } else {
    pan += event->pos() - prevPos;
    prevPos = event->pos();
    update();
  }
  if (editOrientation && yprsignal) {
    emit rotatePanorama(yprsignal->add(yaw, pitch, roll));
  }
}

void VideoWidget::mouseReleaseEvent(QMouseEvent*) {
  if (yprsignal) {
    yprsignal->terminate();
  }
  yprsignal = nullptr;
  if (editOrientation) {
    emit applyOrientation();
  }
}

void VideoWidget::wheelEvent(QWheelEvent* event) {
  if (enableZoom) {
    if (event->delta() > 0 && zoom < MAX_ZOOM) {
      zoom += WHEEL_STEP;
    } else if (event->delta() < 0 && zoom > MIN_ZOOM) {
      zoom -= WHEEL_STEP;
    }
    update();
  } else {
    event->ignore();
  }
}

void VideoWidget::keyPressEvent(QKeyEvent* event) { event->ignore(); }

void VideoWidget::setEditOrientationActivated(bool active) {
  editOrientation = active;
  update();
}

void VideoWidget::setZoomActivated(bool active) { enableZoom = active; }

void VideoWidget::setProjection(VideoStitch::Projection p, double hfov) {
  proj = p;
  HFOV = hfov;
}

void VideoWidget::point2pano(QPointF& p) {
  // coords inside the quad
  QPoint topLeftCorner(0, 0);
  int w = width();
  int h = height();
  if (getWidgetAspectRatio() > getTextureAspectRatio()) {
    int margin = (width() - Texture::get().getWidth() * h / Texture::get().getHeight()) / 2;
    margin *= 100.0f / zoom;
    topLeftCorner.rx() = margin;
    w -= 2 * margin;
  } else {
    int margin = (height() - Texture::get().getHeight() * w / Texture::get().getWidth()) / 2;
    margin *= 100.0f / zoom;
    topLeftCorner.ry() = margin;
    h -= 2 * margin;
  }
  p -= topLeftCorner;
  // coords inside the panorama
  p.rx() *= Texture::get().getWidth() / (float)w;
  p.ry() *= Texture::get().getHeight() / (float)h;
  p.rx() -= Texture::get().getWidth() / 2;
  p.ry() -= Texture::get().getHeight() / 2;
}

float VideoWidget::getTextureAspectRatio() const {
  return (Texture::get().getWidth() / (float)Texture::get().getHeight());
}

float VideoWidget::getWidgetAspectRatio() const { return width() / (float)height(); }

float VideoWidget::sphereRadius() {
  switch (proj) {
    case VideoStitch::equirectangular:
    case VideoStitch::fullframe_fisheye:
    case VideoStitch::circular_fisheye:
      return Texture::get().getWidth() / degToRad((qreal)HFOV);
    case VideoStitch::rectilinear:
      return Texture::get().getWidth() / (2.0f * qTan(degToRad((qreal)HFOV) / 2.0f));
    case VideoStitch::stereographic:
      return Texture::get().getWidth() / (4.0f * qTan(degToRad((qreal)HFOV) / 4.0f));
    default:
      Q_ASSERT(false);
      break;
  }

  return -1;
}
