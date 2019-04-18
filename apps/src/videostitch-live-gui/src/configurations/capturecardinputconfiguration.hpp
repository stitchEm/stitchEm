// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CAPTURECARDINPUTCONFIGURATION_HPP
#define CAPTURECARDINPUTCONFIGURATION_HPP

#include "inputconfigurationwidget.hpp"

#include <memory>

class CaptureCardLiveInput;

namespace Ui {
class CaptureCardInputConfiguration;
}

class CaptureCardInputConfiguration : public InputConfigurationWidget {
  Q_OBJECT
 public:
  explicit CaptureCardInputConfiguration(std::shared_ptr<const CaptureCardLiveInput> liveInput, QWidget *parent = 0);
  ~CaptureCardInputConfiguration();

 protected:
  virtual void saveData();
  virtual void reactToChangedProject();
  virtual bool hasValidConfiguration() const;

 private slots:
  void updateAvailableDevices();

 private:
  QScopedPointer<Ui::CaptureCardInputConfiguration> ui;
  std::shared_ptr<const CaptureCardLiveInput> templateInput;  // This input is usefull only to display its parameters
};

#endif  // CAPTURECARDINPUTCONFIGURATION_HPP
