// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "renderer.hpp"

#include "frame.hpp"

#include <locale.h>
#include <math.h>

#define NUM_PARALLELS 80
#define NUM_MERIDIANS 60

Renderer::Renderer()
    : placeHolderTex(QOpenGLTexture::Target2D),
      sphereVbo(QOpenGLBuffer::VertexBuffer),
      skyboxVbo(QOpenGLBuffer::VertexBuffer),
      sphereProgram(this),
      skyboxProgram(this) {}

Renderer::~Renderer() {
  sphereVbo.destroy();
  skyboxVbo.destroy();
  sphereVao.destroy();
  skyboxVao.destroy();
  placeHolderTex.destroy();
}

void Renderer::initialize() {
  initializeOpenGLFunctions();
  // Enable depth buffer
  glDisable(GL_DEPTH_TEST);
  // Disable back face culling
  glDisable(GL_CULL_FACE);

  sphereVbo.create();
  skyboxVbo.create();
  sphereVao.create();
  skyboxVao.create();

  // idle texture
  placeHolderTex.setData(QImage(":/assets/home_grey").mirrored());
  placeHolderTex.setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
  placeHolderTex.setMagnificationFilter(QOpenGLTexture::Linear);
  glBindTexture(GL_TEXTURE_2D, 0);

  checkOpenglError();

  // Override system locale until shaders are compiled
  setlocale(LC_NUMERIC, "C");

  // Spherical video

  QVector<GLfloat> data = buildSphereLines();
  sphereVbo.bind();
  sphereVbo.allocate(&data.front(), data.size() * 4);
  sphereVbo.release();

  // Compile vertex shader
  if (!sphereProgram.addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/sphereVertexShader")) {
    checkOpenglError();
    return;
  }
  // Compile fragment shader
  if (!sphereProgram.addShaderFromSourceFile(QGLShader::Fragment, ":/shaders/sphereFragmentShader")) {
    checkOpenglError();
    return;
  }
  // Link shader pipeline
  if (!sphereProgram.link()) {
    checkOpenglError();
    return;
  }

  // Skybox video

  data = buildCube(1.);  // less computation for the equiangular shader with this cube
  skyboxVbo.bind();
  skyboxVbo.allocate(&data.front(), data.size() * 4);
  skyboxVbo.release();

  if (!skyboxProgram.addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/skyboxVertexShader")) {
    checkOpenglError();
    return;
  }
  // Compile fragment shader
  if (!skyboxProgram.addShaderFromSourceFile(QGLShader::Fragment, ":/shaders/skyboxFragmentShader")) {
    checkOpenglError();
    return;
  }
  // Link shader pipeline
  if (!skyboxProgram.link()) {
    checkOpenglError();
    return;
  }

  // Equiangular skybox video

  if (!equiangularSkyboxProgram.addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/skyboxVertexShader")) {
    checkOpenglError();
    return;
  }
  // Compile fragment shader
  if (!equiangularSkyboxProgram.addShaderFromSourceFile(QGLShader::Fragment,
                                                        ":/shaders/equiangularSkyboxFragmentShader")) {
    checkOpenglError();
    return;
  }
  // Link shader pipeline
  if (!equiangularSkyboxProgram.link()) {
    checkOpenglError();
    return;
  }

  // Restore system locale
  setlocale(LC_ALL, "");
}

void Renderer::renderSphere() {
  sphereProgram.bind();
  sphereVbo.bind();
  sphereVao.bind();

  int vertexLocation = sphereProgram.attributeLocation("a_position");
  sphereProgram.enableAttributeArray(vertexLocation);
  glVertexAttribPointer(vertexLocation, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);

  int texcoordLocation = sphereProgram.attributeLocation("a_texcoord");
  sphereProgram.enableAttributeArray(texcoordLocation);
  glVertexAttribPointer(texcoordLocation, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (const void*)(3 * sizeof(float)));

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 2 * (NUM_PARALLELS + 1) * (NUM_MERIDIANS + 1));

  sphereVao.release();
  sphereVbo.release();
  sphereProgram.release();
}

void Renderer::renderSkybox(QGLShaderProgram& program) {
  program.bind();
  skyboxVbo.bind();
  skyboxVao.bind();

  int vertexLocation = program.attributeLocation("vertex");
  program.enableAttributeArray(vertexLocation);
  glVertexAttribPointer(vertexLocation, 3, GL_FLOAT, GL_FALSE, 0, 0);

  glDrawArrays(GL_TRIANGLES, 0, 36);

  skyboxVao.release();
  skyboxVbo.release();
  program.release();
}

void Renderer::checkOpenglError() {
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    QString error;

    switch (err) {
      case GL_INVALID_OPERATION:
        error = " INVALID_OPERATION";
        break;
      case GL_INVALID_ENUM:
        error = " INVALID_ENUM";
        break;
      case GL_INVALID_VALUE:
        error = " INVALID_VALUE";
        break;
      case GL_OUT_OF_MEMORY:
        error = " OUT_OF_MEMORY";
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        error = " INVALID_FRAMEBUFFER_OPERATION";
        break;
      default:
        error = " UNKNOWN";
        break;
    }

    error.prepend(__FUNCTION__);
    emit logMessage(error, VideoStitch::LOG_ERROR);
  }
}

QVector<GLfloat> Renderer::buildCube(float side) const {
  QVector<GLfloat> data;

  // Top of cube
  data.push_back(side);
  data.push_back(side);
  data.push_back(side);

  data.push_back(side);
  data.push_back(side);
  data.push_back(-side);

  data.push_back(-side);
  data.push_back(side);
  data.push_back(-side);

  data.push_back(side);
  data.push_back(side);
  data.push_back(side);

  data.push_back(-side);
  data.push_back(side);
  data.push_back(-side);

  data.push_back(-side);
  data.push_back(side);
  data.push_back(side);

  // Bottom of cube
  data.push_back(-side);
  data.push_back(-side);
  data.push_back(-side);

  data.push_back(side);
  data.push_back(-side);
  data.push_back(-side);

  data.push_back(side);
  data.push_back(-side);
  data.push_back(side);

  data.push_back(-side);
  data.push_back(-side);
  data.push_back(side);

  data.push_back(-side);
  data.push_back(-side);
  data.push_back(-side);

  data.push_back(side);
  data.push_back(-side);
  data.push_back(side);

  // Left side of cube
  data.push_back(-side);
  data.push_back(side);
  data.push_back(side);

  data.push_back(-side);
  data.push_back(side);
  data.push_back(-side);

  data.push_back(-side);
  data.push_back(-side);
  data.push_back(-side);

  data.push_back(-side);
  data.push_back(side);
  data.push_back(side);

  data.push_back(-side);
  data.push_back(-side);
  data.push_back(-side);

  data.push_back(-side);
  data.push_back(-side);
  data.push_back(side);

  // Right side of cube
  data.push_back(side);
  data.push_back(-side);
  data.push_back(-side);

  data.push_back(side);
  data.push_back(side);
  data.push_back(-side);

  data.push_back(side);
  data.push_back(side);
  data.push_back(side);

  data.push_back(side);
  data.push_back(side);
  data.push_back(side);

  data.push_back(side);
  data.push_back(-side);
  data.push_back(side);

  data.push_back(side);
  data.push_back(-side);
  data.push_back(-side);

  // Front
  data.push_back(side);
  data.push_back(-side);
  data.push_back(side);

  data.push_back(side);
  data.push_back(side);
  data.push_back(side);

  data.push_back(-side);
  data.push_back(side);
  data.push_back(side);

  data.push_back(-side);
  data.push_back(side);
  data.push_back(side);

  data.push_back(-side);
  data.push_back(-side);
  data.push_back(side);

  data.push_back(side);
  data.push_back(-side);
  data.push_back(side);

  // Back
  data.push_back(side);
  data.push_back(-side);
  data.push_back(-side);

  data.push_back(-side);
  data.push_back(-side);
  data.push_back(-side);

  data.push_back(-side);
  data.push_back(side);
  data.push_back(-side);

  data.push_back(-side);
  data.push_back(side);
  data.push_back(-side);

  data.push_back(side);
  data.push_back(side);
  data.push_back(-side);

  data.push_back(side);
  data.push_back(-side);
  data.push_back(-side);

  return data;
}

QVector<GLfloat> Renderer::buildSphereLines() const {
  float hfov = 360.f, vfov = 180.f;
  float rHFov = hfov * M_PI / 180;
  float rVFov = vfov * M_PI / 180;
  QVector<GLfloat> data;

  for (int i = 0; i <= NUM_PARALLELS; i++) {
    float v0 = (i - 1) / (float)(NUM_PARALLELS);
    float lat0 = rVFov * (-0.5f + v0);
    float z0 = sinf(lat0);
    float zr0 = cosf(lat0);

    float v1 = i / (float)(NUM_PARALLELS);
    float lat1 = rVFov * (-0.5f + v1);
    float z1 = sinf(lat1);
    float zr1 = cosf(lat1);

    for (int j = 0; j <= NUM_MERIDIANS; j++) {
      float u = j / (float)NUM_MERIDIANS;
      float lng = rHFov * u;
      float x = cosf(lng);
      float y = sinf(lng);

      data.push_back(x * zr0 * 20);  // X
      data.push_back(y * zr0 * 20);  // Y
      data.push_back(z0 * 20);       // Z

      data.push_back(u);   // U
      data.push_back(v0);  // V

      data.push_back(x * zr1 * 20);  // X
      data.push_back(y * zr1 * 20);  // Y
      data.push_back(z1 * 20);       // Z

      data.push_back(u);   // U
      data.push_back(v1);  // V
    }
  }

  return data;
}
