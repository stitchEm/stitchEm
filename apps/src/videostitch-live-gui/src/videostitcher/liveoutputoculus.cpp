// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveoutputoculus.hpp"

#include "liveprojectdefinition.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/parse.hpp"

#include "livesettings.hpp"

#include "libvideostitch-gui/utils/outputformat.hpp"

#include <QLabel>
#include <QApplication>

LiveRendererOculus::LiveRendererOculus(const VideoStitch::OutputFormat::OutputFormatEnum type)
    : LiveRendererFactory(type) {
  // the Oculus display must be instantiated in the GUI thread
  windowMaker.moveToThread(QApplication::instance()->thread());
}

const QString LiveRendererOculus::getIdentifier() const {
#if defined(Q_OS_WIN)
  return QString(StitcherOculusWindow::name.c_str());
#else
  return QString();
#endif
}

VideoStitch::Ptv::Value* LiveRendererOculus::serialize() const {
  VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
  value->get("type")->asString() = "oculus";
  value->get("filename")->asString() = getIdentifier().toStdString();
  return value;
}

VideoStitch::PotentialValue<std::shared_ptr<VideoStitch::Core::PanoRenderer>> LiveRendererOculus::createRenderer() {
#if defined(Q_OS_WIN)
  std::shared_ptr<StitcherOculusWindow>
      result;  // For result to be returned correctly - connection type should not be asynchronous
  QMetaObject::invokeMethod(&windowMaker, "createOculusWindow", Qt::BlockingQueuedConnection,
                            Q_RETURN_ARG(std::shared_ptr<StitcherOculusWindow>, result));
  std::shared_ptr<VideoStitch::Core::PanoRenderer> renderer =
      std::dynamic_pointer_cast<VideoStitch::Core::PanoRenderer>(result);
  if (renderer != nullptr) {
    return VideoStitch::PotentialValue<std::shared_ptr<VideoStitch::Core::PanoRenderer>>(renderer);
  }
#endif
  return VideoStitch::Status(VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
                             "Could not create Oculus window");
}

void LiveRendererOculus::destroyRenderer(bool wait) {
  QMetaObject::invokeMethod(&windowMaker, "closeOculusWindow", wait ? Qt::DirectConnection : Qt::QueuedConnection);
}

QWidget* LiveRendererOculus::createStatusWidget(QWidget* const parent) { return createStatusIcon(parent); }

QPixmap LiveRendererOculus::getIcon() const { return QPixmap(":/live/icons/assets/icon/live/oculus.png"); }

#if defined(Q_OS_WIN)
std::shared_ptr<StitcherOculusWindow> OculusWindowFactory::createOculusWindow() {
  if (!stitcherOculusWindow) {
    // Check oculus can be initialized
    if (!StitcherOculusWindow::checkOculusCanBeInitialized()) {
      return false;
    }

    stitcherOculusWindow = std::shared_ptr<StitcherOculusWindow>(
        new StitcherOculusWindow(false, LiveSettings::getLiveSettings()->getMirrorModeEnabled()),
        [](StitcherOculusWindow* dataPtr) { dataPtr->deleteLater(); });
    if (!stitcherOculusWindow->start()) {
      VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
          "Failed to initialize the Oculus. Please check that your Oculus is plugged in and switched on.");
      stitcherOculusWindow.reset();
      return nullptr;
    }
  }
  return stitcherOculusWindow;
}
#endif

// This method should be called from main thread.
void OculusWindowFactory::closeOculusWindow() {
#if defined(Q_OS_WIN)
  if (stitcherOculusWindow) {
    stitcherOculusWindow->stop();
    stitcherOculusWindow.reset();
  }
#endif
}
