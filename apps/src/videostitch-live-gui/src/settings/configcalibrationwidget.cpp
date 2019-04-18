// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videostitch-live-gui/src/guiconstants.hpp"

#include "configcalibrationwidget.hpp"
#include "widgetsmanager.hpp"
#include "livesettings.hpp"
#include "projectworkwidget.hpp"
#include "videostitcher/liveprojectdefinition.hpp"

#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"

#include "libvideostitch-base/file.hpp"

#include <QFileDialog>

ConfigCalibrationWidget::ConfigCalibrationWidget(QWidget* const parent) : IAppSettings(parent) {
  setupUi(this);
  buttonBrowse->setProperty("vs-button-medium", true);
  connect(buttonBrowse, &QPushButton::clicked, this, &ConfigCalibrationWidget::onButtonBrowseClicked);
}

ConfigCalibrationWidget::~ConfigCalibrationWidget() {}

void ConfigCalibrationWidget::updateEditability(bool outputIsActivated, bool algorithmIsActivated) {
  configWidget->setEnabled(!outputIsActivated && !algorithmIsActivated);
}

void ConfigCalibrationWidget::load() {
  lineSnapSource->setText(LiveSettings::getLiveSettings()->getSnapshotPath());
  checkCountdown->setChecked(LiveSettings::getLiveSettings()->getShowCalibrationCounter());
  ProjectWorkWidget* project = WidgetsManager::getInstance()->getProjectWorkWidget();
  if (project != nullptr) {
    bool outputIsActivated = project->outputIsActivated();
    bool algorithmIsActivated = project->algorithmIsActivated();
    updateEditability(outputIsActivated, algorithmIsActivated);
  }
}

void ConfigCalibrationWidget::save() {
  LiveSettings::getLiveSettings()->setSnapshotPath(lineSnapSource->text());
  LiveSettings::getLiveSettings()->setShowCalibrationCounter(checkCountdown->isChecked());
}

void ConfigCalibrationWidget::onButtonBrowseClicked() {
  const QString& path = QFileDialog::getExistingDirectory(this, tr("Select a snapshot directory"),
                                                          ProjectFileHandler::getInstance()->getWorkingDirectory(),
                                                          QFileDialog::ShowDirsOnly);
  lineSnapSource->setText(path);
}
