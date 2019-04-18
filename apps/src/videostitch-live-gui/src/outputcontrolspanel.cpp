// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputcontrolspanel.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "videostitcher/liveoutputfactory.hpp"
#include "videostitcher/liveoutputlist.hpp"
#include "videostitcher/globallivecontroller.hpp"
#include "widgetsmanager.hpp"
#include "guiconstants.hpp"
#include <QScrollBar>

OutputControlsPanel::OutputControlsPanel(QWidget* const parent)
    : QFrame(parent),
      buttonMapper(new QSignalMapper(this)),
      labelMapper(new QSignalMapper(this)),
      projectDefinition(nullptr) {
  setupUi(this);
  setProperty("vs-button-bar", true);
  buttonNewOutput->setProperty("vs-button-medium", true);
  buttonCalibration->setProperty("vs-button-medium", true);
  buttonExposure->setProperty("vs-button-medium", true);
  buttonOrientation->setProperty("vs-button-medium", true);
  buttonPanorama->setProperty("vs-button-medium", true);
  buttonSnapshot->setProperty("vs-button-medium", true);
  buttonCropInputs->setProperty("vs-button-medium", true);
  buttonCalibrationImprove->setProperty("vs-button-medium", true);
  buttonCalibrationToggleControlPoints->setProperty("vs-button-medium", true);
  buttonCalibrationClear->setProperty("vs-button-medium", true);
  buttonCalibrationAdapt->setProperty("vs-button-medium", true);
  buttonCalibrationBack->setProperty("vs-button-medium", true);
  buttonExposureApply->setProperty("vs-button-medium", true);
  buttonExposureClear->setProperty("vs-button-medium", true);
  buttonExposureSettings->setProperty("vs-button-medium", true);
  buttonExposureBack->setProperty("vs-button-medium", true);
  buttonAudioProcessors->setProperty("vs-button-medium", true);
  showMainTab();

  connect(buttonNewOutput, &QPushButton::clicked, this, &OutputControlsPanel::onButtonAddOutputClicked);
  connect(buttonCalibration, &QPushButton::clicked, this, &OutputControlsPanel::onCalibrationButtonClicked);
  connect(buttonExposure, &QPushButton::clicked, this, &OutputControlsPanel::onButtonExposureClicked);
  connect(buttonPanorama, &QPushButton::clicked, this, &OutputControlsPanel::notifyPanoramaEdition);
  connect(buttonSnapshot, &QPushButton::clicked, this, &OutputControlsPanel::notifyTakePanoSnapshot);
  connect(buttonScrollDown, &QPushButton::clicked, this, &OutputControlsPanel::onButtonDownScrollClicked);
  connect(buttonScrollUp, &QPushButton::clicked, this, &OutputControlsPanel::onButtonUpScrollClicked);
  connect(buttonAudioProcessors, &QPushButton::clicked, this, &OutputControlsPanel::notifyAudioProcessorsEdition);
  connect(buttonCalibrationBack, &QPushButton::clicked, this, &OutputControlsPanel::showMainTab);
  connect(buttonExposureBack, &QPushButton::clicked, this, &OutputControlsPanel::showMainTab);

  connect(labelMapper, static_cast<void (QSignalMapper::*)(const QString&)>(&QSignalMapper::mapped), this,
          &OutputControlsPanel::onOutputEditClicked);
  connect(buttonMapper, static_cast<void (QSignalMapper::*)(const QString&)>(&QSignalMapper::mapped), this,
          &OutputControlsPanel::onOutputButtonClicked);
}

OutputControlsPanel::~OutputControlsPanel() {
  delete buttonMapper;
  delete labelMapper;
}

void OutputControlsPanel::configure(Options options) {
  bool displayOrientation = !options.testFlag(HideOrientation);
  buttonOrientation->setVisible(displayOrientation);
  labelOrientation->setVisible(displayOrientation);

  bool displayPanorama = !options.testFlag(HidePanorama);
  buttonPanorama->setVisible(displayPanorama);
  labelPanorama->setVisible(displayPanorama);

  bool displaySnapshot = !options.testFlag(HideSnapshot);
  buttonSnapshot->setVisible(displaySnapshot);
  labelSnapshot->setVisible(displaySnapshot);

  bool displayAudioProcessors = !options.testFlag(HideAudioProcessors);
  const StitcherController* controller = GlobalController::getInstance().getController();
  const bool hasAudio = controller->hasInputAudio();
  buttonAudioProcessors->setVisible(displayAudioProcessors && hasAudio);
  labelAudioProcessors->setVisible(displayAudioProcessors && hasAudio);
}

void OutputControlsPanel::toggleOutput(const QString& id) {
  auto button = outputsMap.value(id);
  if (!button.isEmpty()) {
    button.at(0)->setChecked(!button.at(0)->isChecked());
  }
}

void OutputControlsPanel::setOutputActionable(const QString& id, bool actionable) {
  auto button = outputsMap.value(id);
  if (!button.isEmpty()) {
    button.at(0)->setEnabled(actionable);
  }
}

void OutputControlsPanel::addOutputButton(LiveOutputFactory* output) {
  QPushButton* button(new QPushButton(this));
  button->setProperty("vs-button-medium", true);
  button->setFixedSize(BUTTON_SIDE, BUTTON_SIDE);
  button->setCheckable(true);
  button->setChecked(false);
  button->setFocusPolicy(Qt::NoFocus);
  button->setIcon(QIcon(output->getIcon()));
  button->setIconSize(QSize(BUTTON_SIDE, BUTTON_SIDE));
  scrollAreaLayout->insertWidget(2, button);

  QPushButton* label(new QPushButton(output->getOutputTypeDisplayName()));
  label->setProperty("vs-textbutton-small", true);
  label->setFocusPolicy(Qt::NoFocus);
  if (output->isConfigurable()) {
    label->setLayoutDirection(Qt::RightToLeft);
    label->setIcon(QIcon(":/live/icons/assets/icon/live/data-edit.png"));
    label->setIconSize(QSize(ICON_SIZE, ICON_SIZE));
    labelMapper->setMapping(label, output->getIdentifier());
    connect(label, &QPushButton::clicked, labelMapper, static_cast<void (QSignalMapper::*)(void)>(&QSignalMapper::map));
  } else {
    label->setStyleSheet("background-color: none;");
  }
  scrollAreaLayout->insertWidget(3, label);

  buttonMapper->setMapping(button, output->getIdentifier());
  connect(button, &QPushButton::clicked, buttonMapper, static_cast<void (QSignalMapper::*)(void)>(&QSignalMapper::map));
  outputsMap[output->getIdentifier()] = QList<QPushButton*>({button, label});
  widgetsToDelete.push_back(button);
  widgetsToDelete.push_back(label);
}

void OutputControlsPanel::setProject(LiveProjectDefinition* project) {
  projectDefinition = project;
  addOutputButtons();
}

void OutputControlsPanel::clearProject() {
  projectDefinition = nullptr;
  removeOutputButtons();
  showMainTab();
}

void OutputControlsPanel::resizeEvent(QResizeEvent*) { updateScrollButtons(); }

void OutputControlsPanel::showEvent(QShowEvent*) { updateScrollButtons(); }

void OutputControlsPanel::onOutputButtonClicked(const QString& id) {
  outputsMap.value(id).at(0)->setEnabled(false);
  emit notifyOutputActivated(id);
}

void OutputControlsPanel::onButtonAddOutputClicked() {
  WidgetsManager::getInstance()->showConfiguration(ConfigIdentifier::CONFIG_OUTPUT);
  emit notifyStartAddingOutput();
}

void OutputControlsPanel::onOutputEditClicked(const QString& id) {
  WidgetsManager::getInstance()->showConfiguration(ConfigIdentifier::CONFIG_OUTPUT);
  emit notifyConfigureOutput(id);
}

void OutputControlsPanel::onOutputIdChanged(const QString& oldId, const QString& newId) {
  QList<QPushButton*> button = outputsMap.value(oldId);
  if (!button.isEmpty()) {
    outputsMap.remove(oldId);
    outputsMap[newId] = button;
    buttonMapper->removeMappings(button.at(0));
    buttonMapper->setMapping(button.at(0), newId);
    labelMapper->removeMappings(button.at(1));
    labelMapper->setMapping(button.at(1), newId);
  }
}

void OutputControlsPanel::updateEditability(bool outputIsActivated, bool algorithmIsActivated) {
  buttonCalibration->setEnabled(!algorithmIsActivated && !outputIsActivated);
  labelCalibration->setEnabled(!algorithmIsActivated);
  buttonNewOutput->setEnabled(!outputIsActivated);
}

void OutputControlsPanel::onButtonExposureClicked() { stackedWidget->setCurrentWidget(pageExposure); }

void OutputControlsPanel::onCalibrationButtonClicked() {
  if (projectDefinition->getPanoConst()->hasBeenCalibrated()) {
    stackedWidget->setCurrentWidget(pageCalibration);
  } else {
    emit notifyNewCalibration();
  }
}

void OutputControlsPanel::showMainTab() { stackedWidget->setCurrentWidget(pageMain); }

void OutputControlsPanel::onProjectClosed() {
  removeOutputButtons();
  showMainTab();
}

void OutputControlsPanel::addOutputButtons() {
  removeOutputButtons();
  if (projectDefinition && projectDefinition->isInit() && !projectDefinition->getOutputConfigs()->isEmpty()) {
    for (LiveOutputFactory* outputDef : projectDefinition->getOutputConfigs()->getValues()) {
      addOutputButton(outputDef);
    }
  }
}

void OutputControlsPanel::removeOutputButtons() {
  foreach (QWidget* widget, widgetsToDelete) {
    verticalLayout->removeWidget(widget);
    delete widget;
  }
  for (auto button : outputsMap) {
    buttonMapper->removeMappings(button.at(0));
    labelMapper->removeMappings(button.at(1));
  }
  outputsMap.clear();
  widgetsToDelete.clear();
}

void OutputControlsPanel::updateScrollButtons() {
  bool visible = scrollAreaWidgetContents->size().height() > height();
  buttonScrollDown->setVisible(visible);
  buttonScrollUp->setVisible(visible);
}

void OutputControlsPanel::onButtonDownScrollClicked() {
  scrollArea->verticalScrollBar()->setValue(scrollArea->verticalScrollBar()->value() + SCROLL_DISP);
}

void OutputControlsPanel::onButtonUpScrollClicked() {
  scrollArea->verticalScrollBar()->setValue(scrollArea->verticalScrollBar()->value() - SCROLL_DISP);
}
