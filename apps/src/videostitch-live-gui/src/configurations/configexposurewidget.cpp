// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configexposurewidget.hpp"
#include "widgetsmanager.hpp"
#include "projectworkwidget.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "exposure/liveexposure.hpp"

ConfigExposureWidget::ConfigExposureWidget(QWidget* const parent, LiveProjectDefinition* projectDefinition)
    : QWidget(parent),
      background(new BackgroundContainer(this, tr("Exposure compensation settings"), parent)),
      projectDefinition(projectDefinition) {
  setupUi(this);
  buttonSave->setProperty("vs-button-medium", true);
  buttonSave->setEnabled(false);
  connect(background, &BackgroundContainer::notifyWidgetClosed, this, &ConfigExposureWidget::onWidgetClosed);
  connect(buttonSave, &QPushButton::clicked, this, &ConfigExposureWidget::save);
  connect(comboAnchor, static_cast<void (QComboBox::*)(int)>(&QComboBox::activated), this,
          &ConfigExposureWidget::onConfigurationChanged);
  load();
  background->show();
  background->raise();
  setAttribute(Qt::WA_DeleteOnClose);
}

ConfigExposureWidget::~ConfigExposureWidget() {}

void ConfigExposureWidget::updateEditability(bool outputIsActivated, bool algorithmIsActivated) {
  Q_UNUSED(outputIsActivated);
  configWidget->setEnabled(!algorithmIsActivated);
  labelWarning->setVisible(algorithmIsActivated);
}

void ConfigExposureWidget::save() {
  if (projectDefinition != nullptr) {
    LiveExposure& configuration = projectDefinition->getDelegate()->getExposureConfiguration();
    configuration.setAnchor(comboAnchor->currentIndex() - 1);
  }
  onWidgetClosed();
}

void ConfigExposureWidget::load() {
  reactToChangedProject();
  ProjectWorkWidget* project = WidgetsManager::getInstance()->getProjectWorkWidget();
  if (project != nullptr) {
    bool outputIsActivated = project->outputIsActivated();
    bool algorithmIsActivated = project->algorithmIsActivated();
    updateEditability(outputIsActivated, algorithmIsActivated);
  }
}

void ConfigExposureWidget::onConfigurationChanged() { buttonSave->setEnabled(true); }

void ConfigExposureWidget::onWidgetClosed() {
  background->hide();
  background->deleteLater();
  close();
}

void ConfigExposureWidget::resetAnchorValues() {
  comboAnchor->clear();
  comboAnchor->addItem(tr("None"));
}

void ConfigExposureWidget::reactToChangedProject() {
  resetAnchorValues();
  if (projectDefinition && projectDefinition->isInit()) {
    comboAnchor->addItems(projectDefinition->getVideoInputNames());
    LiveExposure& configuration = projectDefinition->getDelegate()->getExposureConfiguration();
    comboAnchor->setCurrentIndex(configuration.getAnchor() + 1);
  } else {
    comboAnchor->setCurrentIndex(0);
  }
}

void ConfigExposureWidget::reactToClearedProject() {
  resetAnchorValues();
  comboAnchor->setCurrentIndex(0);
}
