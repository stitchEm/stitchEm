// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "interactiverenderer.hpp"

#include "texture.hpp"

#include <QMouseEvent>

#include <math.h>

#define RADIANS_TO_DEGREES (180.0 / M_PI)
#define DEFAULT_FOV 120
#define FOV_MAX 140
#define FOV_MIN 5

InteractiveRenderer::InteractiveRenderer()
    : Renderer(), latRot(0.0f), lngRot(0.0f), roll(0.0f), fov(90), rWidth(1), rHeight(1) {}

InteractiveRenderer::~InteractiveRenderer() {}

void InteractiveRenderer::drawSphere() {
  Texture::getLeft().latePanoramaDef();

  glEnable(GL_TEXTURE_2D);

  sphereProgram.bind();
  if (Texture::getLeft().getWidth() != 0) {
    glBindTexture(GL_TEXTURE_2D, Texture::getLeft().id);
    if (Texture::getLeft().pixelBuffer != 0) {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Texture::getLeft().pixelBuffer);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, Texture::getLeft().getWidth(), Texture::getLeft().getHeight(), GL_RGBA,
                      GL_UNSIGNED_BYTE, nullptr);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
  } else {
    placeHolderTex.bind();
  }
  sphereProgram.setUniformValue("texture", 0);

  static const QVector3D defaultCenter(0.0f, 0.0f, 0.0f);
  static const QVector3D defaultLookedPoint(-1.0f, 0.0f, 0.0f);
  static const QVector3D defaultUp(0.0f, 0.0f, -1.0f);
  static const QVector3D yawAxis(0, 0, 1);
  static const QVector3D pitchAxis(0, -1, 0);
  static const QVector3D rollAxis(1, 0, 0);

  QMatrix4x4 viewMatrix;
  viewMatrix.lookAt(defaultCenter, defaultLookedPoint, defaultUp);
  viewMatrix.rotate(latRot * RADIANS_TO_DEGREES, pitchAxis);
  viewMatrix.rotate(roll * RADIANS_TO_DEGREES, rollAxis);
  viewMatrix.rotate(lngRot * RADIANS_TO_DEGREES, yawAxis);

  sphereProgram.setUniformValue("mvp_matrix", projection * viewMatrix);
  sphereProgram.release();

  Renderer::renderSphere();
  glBindTexture(GL_TEXTURE_2D, 0);

  glDisable(GL_TEXTURE_2D);
}

void InteractiveRenderer::drawSkybox(QGLShaderProgram& program) {
  Texture::getLeft().lateCubemapDef();

  glEnable(GL_TEXTURE_CUBE_MAP);

  program.bind();
  if (Texture::getLeft().getLength() != 0) {
    glBindTexture(GL_TEXTURE_CUBE_MAP, Texture::getLeft().id);
    for (int i = 0; i < 6; ++i) {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Texture::getLeft().pbo[i]);
      glTexSubImage2D(cube[i], 0, 0, 0, Texture::getLeft().getLength(), Texture::getLeft().getLength(), GL_RGBA,
                      GL_UNSIGNED_BYTE, nullptr);
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    program.setUniformValue("texture", 0);
  } else {
    // TODO
  }
  static const QVector3D defaultCenter(0.0f, 0.0f, 0.0f);
  static const QVector3D defaultLookedPoint(0.0f, 0.0f, 1.0f);
  static const QVector3D defaultUp(0.0f, -1.0f, 0.0f);
  static const QVector3D yawAxis(0, 1, 0);
  static const QVector3D pitchAxis(1, 0, 0);
  static const QVector3D rollAxis(0, 0, 1);

  QMatrix4x4 viewMatrix;
  viewMatrix.lookAt(defaultCenter, defaultLookedPoint, defaultUp);
  viewMatrix.rotate(latRot * RADIANS_TO_DEGREES, pitchAxis);
  viewMatrix.rotate(roll * RADIANS_TO_DEGREES, rollAxis);
  viewMatrix.rotate(lngRot * RADIANS_TO_DEGREES, yawAxis);

  program.setUniformValue("mvp_matrix", projection * viewMatrix);
  program.release();

  Renderer::renderSkybox(program);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

  glDisable(GL_TEXTURE_CUBE_MAP);
}

void InteractiveRenderer::render() {
  glClearColor(0., 1., 0., 0.);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  std::lock_guard<std::mutex> textureLock(*(Texture::get().lock));
  Texture::Type textureType = Texture::getLeft().getType();
  switch (textureType) {
    case Texture::Type::PANORAMIC:
      return drawSphere();
    case Texture::Type::CUBEMAP:
      return drawSkybox(skyboxProgram);
    case Texture::Type::EQUIANGULAR_CUBEMAP:
      return drawSkybox(equiangularSkyboxProgram);
  }
}

void InteractiveRenderer::resize(int width, int height) {
  rWidth = width;
  rHeight = height;
  glViewport(0, 0, rWidth, rHeight);
  projection.setToIdentity();
  projection.perspective(fov, rWidth / (float)rHeight, 0.1f, 100.0f);  // origin = bottom left
}

void InteractiveRenderer::updateZoom() { resize(rWidth, rHeight); }
