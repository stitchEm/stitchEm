// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "iconfigurationvalue.hpp"

#include "videostitcher/liveinputfactory.hpp"

struct AudioConfiguration;
class PluginsController;

class InputConfigurationWidget : public IConfigurationCategory {
  Q_OBJECT

 public:
  explicit InputConfigurationWidget(QWidget* parent = nullptr);
  virtual ~InputConfigurationWidget() {}

  void setPluginsController(const PluginsController* newPluginsController);

  LiveInputList getEditedInputs() const;

 protected:
  const PluginsController* pluginsController;
  LiveInputList editedInputs;  // We save the parameters in these inputs
};
