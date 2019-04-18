// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "appslogging.hpp"
#include "common-config.hpp"
#include "videostitchqopenglfunctions.hpp"

#include <QElapsedTimer>
#include <QGLShaderProgram>

#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>
#include <QMutex>

#include <memory>

class Texture;

class VS_COMMON_EXPORT Renderer : public QObject, public VideoStitchQOpenGLFunctions {
  Q_OBJECT
 public:
  Renderer();
  virtual ~Renderer();  // The context should be current to be able to delete Renderer::placeHolderTex

  virtual void initialize();
  virtual void renderSphere();
  virtual void renderSkybox(QGLShaderProgram &);

 signals:
  void logMessage(const QString &mess, VideoStitch::E_logLevel level);

 protected:
  void checkOpenglError();

  QOpenGLTexture placeHolderTex;

  QOpenGLBuffer sphereVbo, skyboxVbo;
  QOpenGLVertexArrayObject sphereVao, skyboxVao;
  QGLShaderProgram sphereProgram, skyboxProgram, equiangularSkyboxProgram;

 private:
  QVector<GLfloat> buildSphereLines() const;
  QVector<GLfloat> buildCube(float side) const;
};
