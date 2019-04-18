// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configurationoutputsdi.hpp"
#include "videostitcher/liveoutputsdi.hpp"

#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

ConfigurationOutputSDI::ConfigurationOutputSDI(LiveOutputSDI* output, VideoStitch::OutputFormat::OutputFormatEnum type,
                                               QWidget* const parent)
    : OutputConfigurationWidget(parent), outputRef(output) {
  setupUi(this);
  verticalLayout->addLayout(buttonsLayout);
  connect(comboDisplayMode, &QComboBox::currentTextChanged, this, &ConfigurationOutputSDI::onConfigurationChanged);
  connect(audioConfig, &ConfigOutputAudio::notifyConfigChanged, this, &ConfigurationOutputSDI::onConfigurationChanged);
  audioConfig->setLiveAudio(outputRef);
  audioConfig->setType(type);
  outputRef->getIdentifier();
}

ConfigurationOutputSDI::~ConfigurationOutputSDI() {}

void ConfigurationOutputSDI::toggleWidgetState() { configurationWidget->setEnabled(!configurationWidget->isEnabled()); }

LiveOutputFactory* ConfigurationOutputSDI::getOutput() const { return outputRef; }

void ConfigurationOutputSDI::reactToChangedProject() {
  if (projectDefinition) {
    fillWidgetWithValue();
  }
}

void ConfigurationOutputSDI::fillWidgetWithValue() {
  OutputConfigurationWidget::fillWidgetWithValue();

  labelDeviceName->setText(outputRef->getOutputDisplayName());
  comboDisplayMode->setCurrentText(displayModeToString(outputRef->getDisplayMode()));

  // TODO: not supported audio output for SDI.
  audioConfig->loadParameters();
  //  audioConfig->displayMessage(tr("Audio not supported"));
}

void ConfigurationOutputSDI::saveData() {
  if (outputRef != nullptr) {
    outputRef->setDisplayMode(comboDisplayMode->currentData().value<VideoStitch::Plugin::DisplayMode>());
    audioConfig->saveConfiguration();
  }
}

void ConfigurationOutputSDI::setSupportedDisplayModes(std::vector<VideoStitch::Plugin::DisplayMode> displayModes) {
  for (VideoStitch::Plugin::DisplayMode displayMode : displayModes) {
    comboDisplayMode->addItem(displayModeToString(displayMode), QVariant::fromValue(displayMode));
  }
}
