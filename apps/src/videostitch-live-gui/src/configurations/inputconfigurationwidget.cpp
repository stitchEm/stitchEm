// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputconfigurationwidget.hpp"

#include "videostitcher/audioconfiguration.hpp"

InputConfigurationWidget::InputConfigurationWidget(QWidget* parent)
    : IConfigurationCategory(parent), pluginsController(nullptr) {}

void InputConfigurationWidget::setPluginsController(const PluginsController* newPluginsController) {
  pluginsController = newPluginsController;
}

LiveInputList InputConfigurationWidget::getEditedInputs() const { return editedInputs; }
