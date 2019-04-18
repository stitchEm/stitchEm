// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CONFIGPANORAMA_HPP
#define CONFIGPANORAMA_HPP

#include <QWidget>
#include "ui_configurationpanorama.h"
#include "iconfigurationvalue.hpp"
#include <QOpenGLFunctions>

class ConfigPanoramaWidget : public IConfigurationCategory, public Ui::ConfigurationPanoramaClass, QOpenGLFunctions {
  Q_OBJECT
 public:
  explicit ConfigPanoramaWidget(QWidget* const parent = nullptr);
  virtual ~ConfigPanoramaWidget();

  void recoverPanoFromError();
  void updateEditability(bool outputIsActivated, bool algorithmIsActivated);

 protected:
  virtual void reactToChangedProject();
  virtual void saveData();
  virtual void restoreData();

 private slots:
  void checkOutputParameters();

 private:
  unsigned int validWidth;
  unsigned int validHeight;
};

#endif  // CONFIGPANORAMA_HPP
