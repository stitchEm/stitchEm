// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/widgets/stylablewidget.hpp"
#include "libvideostitch/ptv.hpp"

/**
 * @brief The Codec class represent a widget that holds the properties of a codec.
 */
class Codec : public QWidget {
  Q_OBJECT
  Q_MAKE_STYLABLE

 public:
  explicit Codec(QWidget* const parent = nullptr) : QWidget(parent) {}

  virtual ~Codec() {}

  /**
   * @brief this function will set up the widget representing the codec
   */
  virtual void setup() = 0;
  /**
   * @brief Checks if the current Codec has parameters to configure.
   * @return True if it has parameters.
   */
  virtual bool hasConfiguration() const = 0;

  /**
   * @brief Gets the PTV configuration from the widget.
   * @return Returns a PTV value with the configuration.
   */
  virtual VideoStitch::Ptv::Value* getOutputConfig() const = 0;

  /**
   * @brief Sets the widget configuration from the PTV.
   * @param config A PTV object.
   * @return True if the configuration was loaded successfully.
   */
  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) = 0;

  /**
   * @brief Gets the Codec key (name).
   * @return A string representing the codec name.
   */
  virtual QString getKey() const = 0;

  /**
   * @brief Checks if the output size is supported by the Codec.
   * @param width The current output width.
   * @param height The current output height.
   * @return True if the size is supported.
   */
  virtual bool meetsSizeRequirements(int /*width*/, int /*height*/) const { return true; }

  /**
   * @brief Gets the correct size for the current Codec.
   * @param width The supporter output width.
   * @param height The supporter output height.
   */
  virtual void correctSizeToMeetRequirements(int& /*width*/, int& /*height*/) {}

 signals:
  /**
   * @brief Trigger this signal when a value from the widget changes.
   */
  void valueChanged();
};
