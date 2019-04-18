// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "topinformationbarwidget.hpp"
#include "widgetsmanager.hpp"
#include "generic/genericdialog.hpp"
#include "videostitcher/liveoutputfactory.hpp"
#include "videostitcher/liveoutputlist.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "videostitcher/globallivecontroller.hpp"

#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"

#include <QMovie>

static const QString& DEFAULT_TIME_VALUE("00:00:00");  // HH:MM:SS
static const QString& DEFAULT_FPS_VALUE("0");
static const QString& CONNECTING_ICON(":/live/icons/assets/icon/live/loadconnecting.gif");

TopInformationBarWidget::TopInformationBarWidget(QWidget* const parent)
    : QFrame(parent), loadingAnimation(new QMovie(CONNECTING_ICON)), quitDialog(nullptr), projectDefinition(nullptr) {
  setupUi(this);
  labelEXposure->setVisible(false);
  labelLoadingAnimation->setMovie(loadingAnimation.data());
  loadingAnimation->start();
  showLoading(false);
  connect(buttonQuitProject, &QPushButton::clicked, this, &TopInformationBarWidget::onQuitButtonClicked);
  labelCurrentTime->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  labelTimeText->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());

  StateManager::getInstance()->registerObject(this);
}

TopInformationBarWidget::~TopInformationBarWidget() {}

void TopInformationBarWidget::setDefaultTime() {
  labelCurrentTime->setText(DEFAULT_TIME_VALUE);
  labelTargetFPS->setText(DEFAULT_FPS_VALUE);
  labelCurrentFPS->setText(DEFAULT_FPS_VALUE);
}

void TopInformationBarWidget::setCurrentProjectName(const QString& fileName) { labelProjectTitle->setText(fileName); }

void TopInformationBarWidget::showLoading(const bool show) {
  labelConnecting->setVisible(show);
  labelLoadingAnimation->setVisible(show);
}

void TopInformationBarWidget::setProject(const ProjectDefinition* project) {
  projectDefinition = qobject_cast<const LiveProjectDefinition*>(project);
  // TOOD: the label will show the Display Mode when the output preset is implemented
  if (projectDefinition->isInit()) {
    const uint64_t width = projectDefinition->getPanoConst()->getWidth();
    const uint64_t height = projectDefinition->getPanoConst()->getHeight();
    labelPanoSize->setText(tr("Panorama: %0x%1 @").arg(QString::number(width), QString::number(height)));

    // Initialize vumeter
    if (projectDefinition->hasAudio()) {
      audioLevelMeter->clear();
      int nbAudioChannels =
          static_cast<int>(projectDefinition->getAudioPipeConst()->getSelectedInput().value()->numSources());
      audioLevelMeter->addMeter(nbAudioChannels);
    } else {
      audioLevelMeter->clear();
    }
  }
}

void TopInformationBarWidget::clearProject() {
  projectDefinition = nullptr;
  cleanOutputs();
  setDefaultTime();
  audioLevelMeter->clear();
}

void TopInformationBarWidget::onActivateOutputs(const QString id) {
  if (projectDefinition != nullptr && !projectDefinition->getOutputConfigs()->isEmpty()) {
    LiveOutputFactory* liveOutput = projectDefinition->getOutputById(id);
    if (liveOutput != nullptr) {
      QWidget* statusWidget = liveOutput->createStatusWidget(this);
      outputsMap[id] = statusWidget;
      layoutOutputs->addWidget(statusWidget);
    }
    labelNoOutputsActive->setVisible(false);
  }
}

void TopInformationBarWidget::onDeactivateOutputs(const QString id) {
  if (projectDefinition != nullptr && !projectDefinition->getOutputConfigs()->isEmpty()) {
    LiveOutputFactory* liveOutput = projectDefinition->getOutputById(id);
    if (liveOutput != nullptr && outputsMap.contains(id)) {
      QWidget* statusWidget = outputsMap[id];
      if (statusWidget != nullptr) {
        layoutOutputs->removeWidget(statusWidget);
        delete statusWidget;
        outputsMap.remove(id);
      }
    }
    labelNoOutputsActive->setVisible(layoutOutputs->count() == 0);
  }
}

void TopInformationBarWidget::onActivateExposure(const bool activate) { labelEXposure->setVisible(activate); }

void TopInformationBarWidget::onQuitButtonClicked() {
  if (quitDialog.isNull()) {
    quitDialog.reset(
        new GenericDialog(tr("Close project"), tr("Would you like to close the current camera configuration?"),
                          GenericDialog::DialogMode::ACCEPT_CANCEL, WidgetsManager::getInstance()->getMainWindowRef()));
    connect(quitDialog.data(), &GenericDialog::notifyAcceptClicked, this, &TopInformationBarWidget::onQuitConfirmed);
    connect(quitDialog.data(), &GenericDialog::notifyCancelClicked, this, &TopInformationBarWidget::onQuitCancelled);
    quitDialog->show();
  }
}

void TopInformationBarWidget::onQuitConfirmed() {
  quitDialog->close();
  quitDialog.reset();
  emit notifyQuitProject();
}

void TopInformationBarWidget::onQuitCancelled() {
  quitDialog->close();
  quitDialog.reset();
}

void TopInformationBarWidget::updateCurrentTime(const QString& time, const double estimatedFramerate,
                                                VideoStitch::FrameRate targetFramerate) {
  labelCurrentTime->setText(time);
  const double targetFramerateDbl = double(targetFramerate.num) / double(targetFramerate.den);
  labelCurrentFPS->setText(tr("%0 FPS").arg(QString::number(estimatedFramerate, 'g', 3)));
  labelTargetFPS->setText(tr("(Target %0 FPS)").arg(QString::number(targetFramerateDbl, 'g', 3)));
}

void TopInformationBarWidget::changeState(GUIStateCaps::State state) {
  buttonQuitProject->setEnabled(state != GUIStateCaps::State::disabled);
  if (state != GUIStateCaps::State::stitch) {
    setDefaultTime();
  }
}

void TopInformationBarWidget::cleanOutputs() {
  qDeleteAll(outputsMap);
  outputsMap.clear();
}

void TopInformationBarWidget::updateVuMeterValues() {
  if (projectDefinition == nullptr || projectDefinition->getAudioPipeConst().get() == nullptr) {
    return;
  }
  QString inputName = QString::fromStdString(projectDefinition->getAudioPipeConst()->getSelectedAudio());
  audioLevelMeter->levelsChanged(GlobalController::getInstance().getController()->getRMSValues(inputName),
                                 GlobalController::getInstance().getController()->getPeakValues(inputName));
}
