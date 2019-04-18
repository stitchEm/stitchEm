// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "codec.hpp"
#include "libvideostitch/parse.hpp"
#include <QGridLayout>
#include <QComboBox>
#include <QLabel>
#include <QApplication>

/**
 * @brief The TiffCodec class represents a widget holding the properties of the tiff codec
 */
class TiffCodec : public Codec {
  Q_OBJECT

  enum class CompressionEnum { NONE, LZW, PACKBITS, JPEG, DEFLATE };

 public:
  explicit TiffCodec(QWidget* const parent = nullptr)
      : Codec(parent),
        mainLayout(new QGridLayout(this)),
        compressionComboBox(new QComboBox(this)),
        labelCompression(new QLabel(tr("Compression:"), this)) {
    addCompression(CompressionEnum::NONE);
    addCompression(CompressionEnum::LZW);
    // FIXME: compression modes not working properly. Disabled for the moment
    // addCompression(CompressionEnum::PACKBITS);
    // addCompression(CompressionEnum::JPEG);
    // addCompression(CompressionEnum::DEFLATE);
    connect(compressionComboBox, SIGNAL(currentIndexChanged(int)), this, SIGNAL(valueChanged()));
  }

  virtual void setup() override {
    setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(CONTROLS_SPACING);
    mainLayout->setContentsMargins(0, 0, 0, 1);
    labelCompression->setFixedSize(LABEL_WIDTH, CONTROL_HEIGHT);
    compressionComboBox->setFixedSize(CONTROL_WIDTH, CONTROL_HEIGHT);
    mainLayout->addWidget(labelCompression, 0, 0);
    mainLayout->addWidget(compressionComboBox, 0, 1);
    setLayout(mainLayout);
  }

  virtual bool hasConfiguration() const override { return true; }

  virtual VideoStitch::Ptv::Value* getOutputConfig() const override {
    VideoStitch::Ptv::Value* outputConfig = VideoStitch::Ptv::Value::emptyObject();
    outputConfig->get("compression")->asString() = compressionComboBox->currentData().toString().toStdString();
    return outputConfig;
  }

  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override {
    std::string compression;
    if (VideoStitch::Parse::populateString("Ptv", *config, "compression", compression, false) !=
        VideoStitch::Parse::PopulateResult_Ok) {
      return false;
    }
    const int index = compressionComboBox->findData(QString::fromStdString(compression));
    if (index < 0) {
      compressionComboBox->setCurrentIndex(0);
    } else {
      compressionComboBox->setCurrentIndex(index);
    }
    return true;
  }

  virtual QString getKey() const override { return QStringLiteral("tif"); }

 private:
  QGridLayout* mainLayout;
  QComboBox* compressionComboBox;
  QLabel* labelCompression;

  QString getDisplayNameFromEnum(const CompressionEnum& value) const {
    switch (value) {
      case CompressionEnum::NONE:
        return tr("No compression");
      case CompressionEnum::LZW:
        return tr("LZW (lossless)");
      case CompressionEnum::PACKBITS:
        return tr("PackBits (lossless)");
      case CompressionEnum::JPEG:
        return tr("JPEG (lossy)");
      case CompressionEnum::DEFLATE:
        return tr("Deflate");
      default:
        return QString();
    }
  }

  QString getStringFromEnum(const CompressionEnum& value) const {
    switch (value) {
      case CompressionEnum::NONE:
        return QStringLiteral("none");
      case CompressionEnum::LZW:
        return QStringLiteral("lzw");
      case CompressionEnum::PACKBITS:
        return QStringLiteral("packbits");
      case CompressionEnum::JPEG:
        return QStringLiteral("jpeg");
      case CompressionEnum::DEFLATE:
        return QStringLiteral("deflate");
      default:
        return QString();
    }
  }

  void addCompression(const CompressionEnum& profile) {
    const QString name = getDisplayNameFromEnum(profile);
    const QString data = getStringFromEnum(profile);
    compressionComboBox->addItem(name, data);
  }
};
