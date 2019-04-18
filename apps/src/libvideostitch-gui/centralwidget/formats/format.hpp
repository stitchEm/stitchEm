// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "codecs/codecfactory.hpp"
#include "extensionhandlers/extensionhandler.hpp"
#include "libvideostitch/parse.hpp"
#include "codecs/codec.hpp"
#include <QObject>

#include <memory>

/**
 * @brief The Format class is a class used to represent a container (mp4, mov, jpg, tiff...)
 */
class Format : public QObject {
  Q_OBJECT
 public:
  explicit Format(QWidget* const parent = nullptr) : handler(nullptr), container(parent) {}

  virtual ~Format() { delete handler; }

  /**
   * @brief Gets the list of supported Codecs by this format.
   * @return A list with zero or more Codecs.
   */
  QStringList getSupportedCodecs() const { return supportedCodecs; }

  /**
   * @brief Gets the current configured Codec.
   * @return The current Codec.
   */
  Codec* getCodec() const { return codec.get(); }

  /**
   * @brief Sets the current Codec associated to this format.
   * @param codec Codec name.
   */
  void setCodec(const QString& codecString) {
    codec.reset(CodecFactory::create(codecString, container));
    if (codec) {
      connect(codec.get(), SIGNAL(valueChanged()), this, SIGNAL(valueChanged()));
    }
  }

  /**
   * @brief Gets the PTV configuration from the widget format.
   * @note The caller has ownership.
   * @return The PTV configuration.
   */
  virtual VideoStitch::Ptv::Value* getOutputConfig() const = 0;

  /**
   * @brief Sets the widget configuration from the PTV.
   * @param config A PTV value.
   * @return True if the configuration was successfull.
   */
  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) = 0;

  /**
   * @brief Sometimes a Format can be a Codec too.
   * @return True if is a Codec and a format.
   */
  virtual bool isACodecToo() const = 0;

 signals:
  /**
   * @brief Trigger this signal when a value from the widget changes.
   */
  void valueChanged();

 protected:
  QStringList supportedCodecs;
  ExtensionHandler* handler;
  QWidget* container;
  std::unique_ptr<Codec> codec;
};
