// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef MIRRORWIDGET_HPP
#define MIRRORWIDGET_HPP

#include "common-config.hpp"
#include "videostitchqopenglfunctions.hpp"

#include <QOpenGLWidget>

class VS_COMMON_EXPORT MirrorWidget : public QOpenGLWidget, public VideoStitchQOpenGLFunctions {
  Q_OBJECT
 public:
  explicit MirrorWidget(QWidget* const parent = nullptr);
  ~MirrorWidget();

  void setOculusSize(int newOculusWidth, int newOculusHeight);
  void Init(GLuint mirrorTextureId);
  void startTimer();

 protected:
  void paintGL();
  void initializeGL();
  void resizeGL(int width, int height);

 private:
  static const int updateRate = 16;

  int oculusWidth;
  int oculusHeight;
  int outputWidth;
  int outputHeight;

  QTimer* updateTimer;
  GLuint mirrorFBO;
};

#endif  // MIRRORWIDGET_HPP
