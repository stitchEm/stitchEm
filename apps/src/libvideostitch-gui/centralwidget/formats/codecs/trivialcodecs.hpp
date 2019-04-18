// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "codec.hpp"

/**
 * @brief The TrivialCodec class represents a codec with no properties
 */
class TrivialCodec : public Codec {
  Q_OBJECT

 public:
  explicit TrivialCodec(QWidget* const parent = nullptr) : Codec(parent) {}

  virtual void setup() override {}
  virtual bool hasConfiguration() const override { return false; }

  virtual VideoStitch::Ptv::Value* getOutputConfig() const override {
    VideoStitch::Ptv::Value* outputConfig = VideoStitch::Ptv::Value::emptyObject();
    return outputConfig;
  }

  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override {
    Q_UNUSED(config)
    return true;
  }
};

class RawCodec : public TrivialCodec {
  Q_OBJECT

 public:
  explicit RawCodec(QWidget* const parent = nullptr) : TrivialCodec(parent) {}

  virtual QString getKey() const override { return QStringLiteral("raw"); }
};

class NullCodec : public TrivialCodec {
  Q_OBJECT

 public:
  explicit NullCodec(QWidget* const parent = nullptr) : TrivialCodec(parent) {}

  virtual QString getKey() const override { return QStringLiteral("null"); }
};

class PamCodec : public TrivialCodec {
  Q_OBJECT

 public:
  explicit PamCodec(QWidget* const parent = nullptr) : TrivialCodec(parent) {}

  virtual QString getKey() const { return QStringLiteral("pam"); }
};

class PpmCodec : public TrivialCodec {
  Q_OBJECT

 public:
  explicit PpmCodec(QWidget* const parent = nullptr) : TrivialCodec(parent) {}

  virtual QString getKey() const override { return QStringLiteral("ppm"); }
};

class PngCodec : public TrivialCodec {
  Q_OBJECT

 public:
  explicit PngCodec(QWidget* const parent = nullptr) : TrivialCodec(parent) {}

  virtual QString getKey() const override { return QStringLiteral("png"); }
};

class Yuv420Codec : public TrivialCodec {
  Q_OBJECT

 public:
  explicit Yuv420Codec(QWidget* const parent = nullptr) : TrivialCodec(parent) {}

  virtual QString getKey() const override { return QStringLiteral("yuv420p"); }
};
