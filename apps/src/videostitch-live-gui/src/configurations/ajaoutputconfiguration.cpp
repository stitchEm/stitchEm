// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ajaoutputconfiguration.hpp"
#include "videostitcher/liveoutputaja.hpp"
#include "videostitcher/liveprojectdefinition.hpp"

#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

AjaOutputConfiguration::AjaOutputConfiguration(LiveOutputAJA* output, QWidget* const parent)
    : OutputConfigurationWidget(parent), outputRef(output) {
  setupUi(this);
  verticalLayout->addLayout(buttonsLayout);
  connect(comboDisplayMode, &QComboBox::currentTextChanged, this, &AjaOutputConfiguration::updateOffset);
  connect(enableAudioBox, &QCheckBox::toggled, this, &AjaOutputConfiguration::onConfigurationChanged);
  connect(offsetXBox, &QSpinBox::editingFinished, this, &AjaOutputConfiguration::onConfigurationChanged);
  connect(offsetYBox, &QSpinBox::editingFinished, this, &AjaOutputConfiguration::onConfigurationChanged);
}

AjaOutputConfiguration::~AjaOutputConfiguration() {}

void AjaOutputConfiguration::toggleWidgetState() { configurationWidget->setEnabled(!configurationWidget->isEnabled()); }

LiveOutputFactory* AjaOutputConfiguration::getOutput() const { return outputRef; }

void AjaOutputConfiguration::reactToChangedProject() {
  if (projectDefinition) {
    fillWidgetWithValue();
  }
}

void AjaOutputConfiguration::fillWidgetWithValue() {
  OutputConfigurationWidget::fillWidgetWithValue();

  labelDeviceName->setText(outputRef->getOutputDisplayName());
  comboDisplayMode->setCurrentText(displayModeToString(outputRef->getDisplayMode()));
  offsetXBox->setValue(outputRef->getHorizontalOffset());
  offsetYBox->setValue(outputRef->getVerticalOffset());

  // Audio settings
  const StitcherController* controller = GlobalController::getInstance().getController();
  boxAudioSettings->setVisible(controller->hasInputAudio());
  enableAudioBox->setChecked(boxAudioSettings->isVisible() && outputRef->getAudioIsEnabled());
}

void AjaOutputConfiguration::updateOffset() {
  if (projectDefinition) {
    int maxOffsetX = comboDisplayMode->currentData().value<VideoStitch::Plugin::DisplayMode>().width -
                     projectDefinition->getPanoConst()->getWidth();
    int maxOffsetY = comboDisplayMode->currentData().value<VideoStitch::Plugin::DisplayMode>().height -
                     projectDefinition->getPanoConst()->getHeight();
    offsetXBox->setMaximum(maxOffsetX);
    offsetYBox->setMaximum(maxOffsetY);
    offsetXBox->setValue(maxOffsetX / 2);
    offsetYBox->setValue(maxOffsetY / 2);
  }
  emit onConfigurationChanged();
}

void AjaOutputConfiguration::saveData() {
  if (outputRef != nullptr) {
    outputRef->setDisplayMode(comboDisplayMode->currentData().value<VideoStitch::Plugin::DisplayMode>());
    outputRef->setAudioIsEnabled(boxAudioSettings->isVisible() && enableAudioBox->isChecked());
    outputRef->setHorizontalOffset(offsetXBox->value());
    outputRef->setVerticalOffset(offsetYBox->value());
  }
}

void AjaOutputConfiguration::setSupportedDisplayModes(std::vector<VideoStitch::Plugin::DisplayMode> displayModes) {
  for (VideoStitch::Plugin::DisplayMode displayMode : displayModes) {
    comboDisplayMode->addItem(displayModeToString(displayMode), QVariant::fromValue(displayMode));
  }
}
