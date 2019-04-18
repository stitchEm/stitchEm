// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <functional>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

#include "googleapis/client/auth/oauth2_authorization.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"

#include "generic/backgroundcontainer.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "videostitcher/liveoutputrtmp.hpp"
#include "videostitcher/liveoutputyoutube.hpp"

#include "widgetsmanager.hpp"

#include "youtubeconfiguration.hpp"
#include "youtubebaseconfiguration.hpp"
#include "googleauthenticationmanager.hpp"
#include "googlecredentialmodel.hpp"
#include "youtubebroadcastmodel.hpp"

#include "ui_youtubebaseconfiguration.h"

const std::array<int, 6> YoutubeBaseConfiguration::widthArray = {426, 640, 854, 1280, 1920, 2560};
const std::array<int, 6> YoutubeBaseConfiguration::heightArray = {240, 360, 480, 720, 1080, 1440};
const int YoutubeBaseConfiguration::HFR_THRESHOLD = 720;
const int YoutubeBaseConfiguration::FPS_THRESHOLD = 48;

YoutubeBaseConfiguration::YoutubeBaseConfiguration(QWidget* parent, LiveOutputRTMP* outputRef,
                                                   LiveProjectDefinition* projectDefinition)
    : IStreamingServiceConfiguration(parent, outputRef, projectDefinition),
      ui(new Ui::YoutubeBaseConfiguration),
      youtubeBroadcastModel(new YoutubeBroadcastModel()),
      googleCredentialModel(GoogleAuthenticationManager::getInstance().getCredentialModel()) {
  ui->setupUi(this);

  ui->broadcastsView->setModel(youtubeBroadcastModel.get());

  connect(youtubeBroadcastModel.get(), &YoutubeBroadcastModel::modelReset, this,
          &YoutubeBaseConfiguration::handleStateChanged);

  connect(ui->configurationButton, &QPushButton::clicked, this, &YoutubeBaseConfiguration::openMoreConfiguration);
}

YoutubeBaseConfiguration::~YoutubeBaseConfiguration() {}

bool YoutubeBaseConfiguration::loadConfiguration() {
  if (!outputRef->getPubUser().isEmpty()) {
    if (!googleCredentialModel->authorizeAndSetCurrentCredential(outputRef->getPubUser())) {
      ui->labelMessage->setText(tr("Please authenticate or reconfigure the event."));
      return false;
    }

    ui->accountNameLabel->setText(outputRef->getPubUser());
    auto broadcastId = static_cast<LiveOutputYoutube*>(outputRef)->getBroadcastId();
    if (!broadcastId.isEmpty()) {
      updateYoutubeData(broadcastId);
    }
  }

  return true;
}

void YoutubeBaseConfiguration::saveConfiguration() {
  auto profile = currentYoutubeProfile();
  auto saveIndex = youtubeBroadcastModel->index(0, 0);
  outputRef->setUrl(youtubeBroadcastModel->getUrlForIndex(saveIndex, profile.first, profile.second).toString());

  if (GoogleAuthenticationManager::getInstance().authorized()) {
    outputRef->setPubUser(QString::fromStdString(GoogleAuthenticationManager::getInstance().getCredential()->email()));

    static_cast<LiveOutputYoutube*>(outputRef)->setBroadcastId(
        youtubeBroadcastModel->getBroadcastIdForIndex(saveIndex));
  }
}

bool YoutubeBaseConfiguration::hasValidConfiguration() const { return youtubeBroadcastModel->rowCount(); }

void YoutubeBaseConfiguration::startBaseConfiguration() { openMoreConfiguration(); }

void YoutubeBaseConfiguration::updateYoutubeData(const QString& broadcastId) {
  if (GoogleAuthenticationManager::getInstance().authorized()) {
    ui->accountNameLabel->setText(
        QString::fromStdString(GoogleAuthenticationManager::getInstance().getCredential()->email()));

    QString errorMessage;
    bool success = youtubeBroadcastModel->updateUserData(GoogleAuthenticationManager::getInstance().getCredential(),
                                                         broadcastId, &errorMessage);
    if (!success) {
      ui->labelMessage->setText(errorMessage);
    }
  }
}

void YoutubeBaseConfiguration::openMoreConfiguration() {
  auto youtubeConfig = new YoutubeConfiguration(this);

  auto backgroundContainer = new BackgroundContainer(youtubeConfig, tr("YouTube configuration"),
                                                     WidgetsManager::getInstance()->getMainWindowRef());
  backgroundContainer->show();
  backgroundContainer->raise();

  connect(backgroundContainer, &BackgroundContainer::notifyWidgetClosed, this, [this, backgroundContainer]() {
    emit this->basicConfigurationCanceled();
    backgroundContainer->close();
  });

  connect(youtubeConfig, &YoutubeConfiguration::notifySettingsSaved, this,
          [this, backgroundContainer](const QString& broadcastId) {
            updateYoutubeData(broadcastId);
            emit stateChanged();
            emit this->basicConfigurationComplete();
            backgroundContainer->close();
          });
}

void YoutubeBaseConfiguration::handleStateChanged() {
  ui->stackedWidget->setCurrentIndex(youtubeBroadcastModel->rowCount());
}

std::pair<std::string, std::string> YoutubeBaseConfiguration::currentYoutubeProfile() const {
  const auto frameRate = GlobalController::getInstance().getController()->getFrameRate();
  const auto ask60 = frameRate.num / frameRate.den > FPS_THRESHOLD;

  const auto downFactor = outputRef->getDownsamplingFactor();
  Q_ASSERT(downFactor != 0);

  const auto outWidth = liveProjectDefinition->getPanoConst()->getWidth() / downFactor;
  const auto outHeight = liveProjectDefinition->getPanoConst()->getHeight() / downFactor;

  const auto lowerBoundWidth = std::lower_bound(std::begin(widthArray), std::end(widthArray), outWidth);
  const auto lowerBoundHeight = std::lower_bound(std::begin(heightArray), std::end(heightArray), outHeight);
  const auto maxDiff = std::max(lowerBoundWidth - std::begin(widthArray), lowerBoundHeight - std::begin(heightArray));

  auto resolution = std::to_string(heightArray[maxDiff]) + "p";
  std::string fps = "30fps";
  if (ask60 && heightArray[maxDiff] >= HFR_THRESHOLD) {
    fps = "60fps";
  }

  return std::make_pair(resolution, fps);
}
