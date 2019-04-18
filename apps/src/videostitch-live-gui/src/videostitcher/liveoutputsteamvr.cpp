// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputsteamvr.hpp"
#include "liveprojectdefinition.hpp"
#include "libvideostitch-gui/mainwindow/stitchersteamvrwindow.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch-base/logmanager.hpp"
#include "libvideostitch/parse.hpp"
#include <QLabel>

LiveOutputSteamVR::LiveOutputSteamVR(const VideoStitch::OutputFormat::OutputFormatEnum type)
    : LiveRendererFactory(type) {
  // the SteamVR display must be instantiated in the GUI thread
  windowMaker.moveToThread(QApplication::instance()->thread());
}

const QString LiveOutputSteamVR::getIdentifier() const { return QString(StitcherSteamVRWindow::name.c_str()); }

VideoStitch::Ptv::Value *LiveOutputSteamVR::serialize() const {
  VideoStitch::Ptv::Value *value = VideoStitch::Ptv::Value::emptyObject();
  value->get("type")->asString() = "vive";
  value->get("filename")->asString() = getIdentifier().toStdString();
  return value;
}

VideoStitch::PotentialValue<std::shared_ptr<VideoStitch::Core::PanoRenderer>> LiveOutputSteamVR::createRenderer() {
  std::shared_ptr<StitcherSteamVRWindow>
      result;  // For result to be returned correctly - connection type should not be asynchronous
  QMetaObject::invokeMethod(&windowMaker, "createSteamVRWindow", Qt::BlockingQueuedConnection,
                            Q_RETURN_ARG(std::shared_ptr<StitcherSteamVRWindow>, result));
  std::shared_ptr<VideoStitch::Core::PanoRenderer> renderer =
      std::dynamic_pointer_cast<VideoStitch::Core::PanoRenderer>(result);
  if (renderer != nullptr) {
    return VideoStitch::PotentialValue<std::shared_ptr<VideoStitch::Core::PanoRenderer>>(renderer);
  } else {
    return VideoStitch::Status(VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
                               "Could not create HMD display");
  }
}

void LiveOutputSteamVR::destroyRenderer(bool wait) {
  QMetaObject::invokeMethod(&windowMaker, "closeSteamVRWindow", wait ? Qt::DirectConnection : Qt::QueuedConnection);
}

QWidget *LiveOutputSteamVR::createStatusWidget(QWidget *const parent) { return createStatusIcon(parent); }

QPixmap LiveOutputSteamVR::getIcon() const { return QPixmap(":/live/icons/assets/icon/live/oculus.png"); }

std::shared_ptr<StitcherSteamVRWindow> SteamVRWindowFactory::createSteamVRWindow() {
#if defined(Q_OS_WIN)
  if (!stitcherSteamVRWindow) {
    stitcherSteamVRWindow = std::shared_ptr<StitcherSteamVRWindow>(new StitcherSteamVRWindow(true));
    if (!stitcherSteamVRWindow->start()) {
      VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
          "Failed to initialize HMD. Please check that your HMD is plugged in and switched on.");
      stitcherSteamVRWindow.reset();
      return nullptr;
    }
  }
  return stitcherSteamVRWindow;
#else
  Q_UNUSED(writer)
#endif
  return nullptr;
}

void SteamVRWindowFactory::closeSteamVRWindow() {
#if defined(Q_OS_WIN)
  if (stitcherSteamVRWindow) {
    stitcherSteamVRWindow->stop();
    stitcherSteamVRWindow.reset();
  }
#endif
}
