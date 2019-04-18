// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "renderer.hpp"

#include "common-config.hpp"

#include <QMatrix4x4>

class VS_COMMON_EXPORT InteractiveRenderer : public Renderer {
  Q_OBJECT
 public:
  explicit InteractiveRenderer();
  ~InteractiveRenderer();
  virtual void render();

  void drawSphere();
  void drawSkybox(QGLShaderProgram&);

  void resize(int width, int height);
  void updateZoom();

  float latRot;
  float lngRot;
  float roll;
  float fov;
  int rWidth;
  int rHeight;

 private:
  QMatrix4x4 projection;
};
