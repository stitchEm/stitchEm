// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "inputconfigurationwidget.hpp"

#include <memory>

class LiveInputAJA;

namespace Ui {
class AjaInputConfiguration;
}

class AjaInputConfiguration : public InputConfigurationWidget {
  Q_OBJECT
 public:
  explicit AjaInputConfiguration(std::shared_ptr<const LiveInputAJA> liveInput, QWidget *parent = 0);
  ~AjaInputConfiguration();

 protected:
  virtual void saveData();
  virtual void reactToChangedProject();
  virtual bool hasValidConfiguration() const;

 private slots:
  void updateAvailableDevices();

 private:
  QScopedPointer<Ui::AjaInputConfiguration> ui;
  std::shared_ptr<const LiveInputAJA> templateInput;  // This input is usefull only to display its parameters
};
