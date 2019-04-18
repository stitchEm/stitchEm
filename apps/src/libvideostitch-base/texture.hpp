// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common-config.hpp"

#include "libvideostitch/config.hpp"

#include <qopengl.h>

#include <condition_variable>
#include <mutex>
#include <memory>

static GLenum cube[6] = {GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                         GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                         GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z};

class VS_COMMON_EXPORT Texture {
 public:
  enum Type { PANORAMIC, CUBEMAP, EQUIANGULAR_CUBEMAP };

  const static GLuint ID_NONE = -1;

  Texture()
      : id(ID_NONE),
        pixelBuffer(0),
        date(0),
        lock(std::make_shared<std::mutex>()),
        _type(PANORAMIC),
        _width(0),
        _height(0),
        _length(0) {
    for (int i = 0; i < 6; ++i) {
      pbo[i] = 0;
    }
  }

  static Texture& get() { return getLeft(); }

  static Texture& getLeft();
  static Texture& getRight();

  inline Texture::Type getType() const { return _type; }
  inline int getWidth() const { return _width; }
  inline int getHeight() const { return _height; }
  inline int getLength() const { return _length; }

  void setPanoramicSize(int width, int height);
  void setCubemapSize(int width, int height, int length, Type type);

  void latePanoramaDef();
  void lateCubemapDef();

  GLuint id;

  GLuint pixelBuffer;
  GLuint pbo[6];

  mtime_t date;
  std::shared_ptr<std::mutex> lock;

  Texture(Texture const&) = delete;
  void operator=(Texture const&) = delete;

 private:
  Type _type;
  int _width;
  int _height;
  int _length;
  bool _needToRedefinePano = false;
  bool _needToRedefineCubemap = false;
};
