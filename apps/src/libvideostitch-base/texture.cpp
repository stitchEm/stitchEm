// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "texture.hpp"

Texture& Texture::getLeft() {
  static Texture leftTexture;
  return leftTexture;
}
Texture& Texture::getRight() {
  static Texture rightTexture;
  return rightTexture;
}

void activateTexture(GLenum targetTexture, GLuint textureId) {
  glEnable(targetTexture);
  glClear(GL_COLOR_BUFFER_BIT);
  glBindTexture(targetTexture, textureId);
}

void deactivateTexture(GLenum targetTexture) {
  glBindTexture(targetTexture, 0);
  glDisable(targetTexture);
}

void configureTextureParams(GLenum targetTexture) {
  glTexParameteri(targetTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(targetTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(targetTexture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(targetTexture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void Texture::setPanoramicSize(int width, int height) {
  _type = Texture::Type::PANORAMIC;
  _needToRedefinePano |= (_width != width || _height != height);
  _width = width;
  _height = height;
}

void Texture::setCubemapSize(int width, int height, int length, Type type) {
  _type = type;
  _width = width;
  _height = height;
  _needToRedefineCubemap |= _length != length;
  _length = length;
}

void Texture::latePanoramaDef() {
  if (_needToRedefinePano) {
    activateTexture(GL_TEXTURE_2D, id);
    configureTextureParams(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    deactivateTexture(GL_TEXTURE_2D);
    _needToRedefinePano = false;
  }
}

void Texture::lateCubemapDef() {
  if (_needToRedefineCubemap) {
    activateTexture(GL_TEXTURE_CUBE_MAP, id);
    configureTextureParams(GL_TEXTURE_CUBE_MAP);
    for (int i = 0; i < 6; ++i) {
      glTexImage2D(cube[i], 0, GL_RGBA8, (GLsizei)_length, (GLsizei)_length, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
    deactivateTexture(GL_TEXTURE_CUBE_MAP);
    _needToRedefineCubemap = false;
  }
}
