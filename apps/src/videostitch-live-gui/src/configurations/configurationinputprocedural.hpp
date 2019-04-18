// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CONFIGURATIONINPUTPROCEDURAL_H
#define CONFIGURATIONINPUTPROCEDURAL_H

#include "inputconfigurationwidget.hpp"

#include <memory>

class LiveInputProcedural;

namespace Ui {
class ConfigurationInputProcedural;
}

class ConfigurationInputProcedural : public InputConfigurationWidget {
  Q_OBJECT

 public:
  explicit ConfigurationInputProcedural(std::shared_ptr<const LiveInputProcedural> liveInput,
                                        QWidget* parent = nullptr);
  ~ConfigurationInputProcedural();

 protected:
  virtual void saveData();
  virtual void reactToChangedProject();

 private:
  QScopedPointer<Ui::ConfigurationInputProcedural> ui;
  std::shared_ptr<const LiveInputProcedural> templateInput;  // This input is usefull only to display its parameters
};

#endif  // CONFIGURATIONINPUTPROCEDURAL_H
