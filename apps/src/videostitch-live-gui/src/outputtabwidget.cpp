// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputtabwidget.hpp"

#include "configurations/configoutputswidget.hpp"
#include "guiconstants.hpp"
#include "widgetsmanager.hpp"
#include "livesettings.hpp"
#include "videostitcher/livestitchercontroller.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"

#include <QDesktopWidget>
#include <QShortcut>
#include <QWindow>

OutPutTabWidget::OutPutTabWidget(QWidget* const parent)
    : QFrame(parent), outputControls(new OutputControlsPanel(this)), projectDefinition(nullptr) {
  setupUi(this);
  videoWidget->syncOff();
  showVideoWidgetPage();
  videoWidget->setZoomActivated(true);
  fullScreenShortCut = new QShortcut(QKeySequence(Qt::Key_F), videoWidget);
  connect(fullScreenShortCut, &QShortcut::activated, this, &OutPutTabWidget::onFullScreenActivated);
  connect(outputControls->buttonOrientation, &QPushButton::toggled, videoWidget,
          &DeviceVideoWidget::setEditOrientationActivated);
  connect(outputControls, &OutputControlsPanel::notifyOutputActivated, this, &OutPutTabWidget::notifyOutputActivated);
  connect(outputControls, &OutputControlsPanel::notifyTakePanoSnapshot, this,
          &OutPutTabWidget::onButtonSnapshotClicked);
  connect(outputControls, &OutputControlsPanel::notifyPanoramaEdition, this, &OutPutTabWidget::showPanoramaEditionPage);
  connect(outputControls, &OutputControlsPanel::notifyAudioProcessorsEdition, this,
          &OutPutTabWidget::showAudioProcessorsPage);
  connect(QApplication::desktop(), &QDesktopWidget::screenCountChanged, this, &OutPutTabWidget::reSetupVideoWidget);
  connect(panoramaConfigurationWidget, &ConfigPanoramaWidget::reqBack, this, &OutPutTabWidget::showVideoWidgetPage);
  connect(panoramaConfigurationWidget, &ConfigPanoramaWidget::saved, this, &OutPutTabWidget::onFileHasTobeSaved);
  connect(audioProcessorsConfigurationWidget, &AudioProcessorsWidget::saved, this,
          &OutPutTabWidget::onFileHasTobeSaved);
  connect(audioProcessorsConfigurationWidget, &AudioProcessorsWidget::reqBack, this,
          &OutPutTabWidget::showVideoWidgetPage);
}

OutPutTabWidget::~OutPutTabWidget() {}

void OutPutTabWidget::setOutputWidgetReference(OutputControlsPanel* controlBar) {
  outputControls = controlBar;
  outputControls->configure(OutputControlsPanel::NoOption);
  horizontalLayout->insertWidget(1, controlBar);
}

void OutPutTabWidget::onFileHasTobeSaved() {
  emit reqSaveProject(ProjectFileHandler::getInstance()->getFilename());
  emit reqStitcherReload();
}

void OutPutTabWidget::toggleOutput(const QString& id) { outputControls->toggleOutput(id); }

void OutPutTabWidget::setOutputActionable(const QString& id, bool actionable) {
  outputControls->setOutputActionable(id, actionable);
}

void OutPutTabWidget::updateOutputId(const QString oldName, const QString newName) {
  outputControls->onOutputIdChanged(oldName, newName);
}

void OutPutTabWidget::restore() {
  outputControls->buttonOrientation->setChecked(false);
  panoramaConfigurationWidget->restore();
}

DeviceVideoWidget* OutPutTabWidget::getVideoWidget() const { return videoWidget; }

OutputControlsPanel* OutPutTabWidget::getControlsBar() const { return outputControls; }

ConfigPanoramaWidget* OutPutTabWidget::getConfigPanoramaWidget() const { return panoramaConfigurationWidget; }

AudioProcessorsWidget* OutPutTabWidget::getAudioProcessorWidget() const { return audioProcessorsConfigurationWidget; }

void OutPutTabWidget::onFullScreenActivated() {
  if (videoWidget->isFullScreen()) {
    videoWidget->setParent(this);
    layoutOutputControl->insertWidget(0, videoWidget);
    videoWidget->show();
    videoWidget->update();
    // Recover focus after fullscreen off
    QApplication::setActiveWindow(WidgetsManager::getInstance()->getMainWindowRef());
  } else {
    layoutOutputControl->removeWidget(videoWidget);
    videoWidget->setParent(nullptr);
    videoWidget->show();
    // Show the fullscreen widget in the same screen as the containing widget
    QWidget* screen = QApplication::desktop()->screen(QApplication::desktop()->screenNumber(this));
    videoWidget->windowHandle()->setScreen(screen->windowHandle()->screen());
    videoWidget->showFullScreen();
  }
}

void OutPutTabWidget::onInvalidPano() { panoramaConfigurationWidget->recoverPanoFromError(); }

void OutPutTabWidget::onButtonSnapshotClicked() {
  WidgetsManager::getInstance()->showSnapshotDialog(videoWidget);

  const QString timestamp = QDateTime::currentDateTime().toString(Qt::ISODate).replace(':', "_");
  QString snapFileName = LiveSettings::getLiveSettings()->getSnapshotPath() + QString("/snap_%1.jpg").arg(timestamp);
  emit notifyTakePanoSnapshot(snapFileName);
}

void OutPutTabWidget::setProject(ProjectDefinition* project) {
  projectDefinition = qobject_cast<LiveProjectDefinition*>(project);
  outputControls->setProject(projectDefinition);
  panoramaConfigurationWidget->setProject(projectDefinition);
  audioProcessorsConfigurationWidget->setProject(projectDefinition);
}

void OutPutTabWidget::clearProject() {
  projectDefinition = nullptr;
  outputControls->clearProject();
  panoramaConfigurationWidget->clearProject();
  audioProcessorsConfigurationWidget->clearProject();
}

void OutPutTabWidget::reSetupVideoWidget() { videoWidget->windowHandle()->setScreen(this->windowHandle()->screen()); }

void OutPutTabWidget::showVideoWidgetPage() { stackedWidget->setCurrentWidget(pageVideoWidget); }

void OutPutTabWidget::showPanoramaEditionPage() { stackedWidget->setCurrentWidget(panoramaConfigurationWidget); }

void OutPutTabWidget::showAudioProcessorsPage() { stackedWidget->setCurrentWidget(audioProcessorsConfigurationWidget); }
