// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpuhelper.hpp"

#include "libvideostitch-base/msgboxhandler.hpp"

#include <QCoreApplication>

namespace VideoStitch {
namespace GPU {
bool checkGPUFrameworkAvailable(VideoStitch::Discovery::Framework framework) {
  QString message;
  switch (VideoStitch::Discovery::getFrameworkStatus(framework)) {
    case VideoStitch::Discovery::FrameworkStatus::Ok:
      return true;
    case VideoStitch::Discovery::FrameworkStatus::GenericError:
      switch (framework) {
        case VideoStitch::Discovery::Framework::CUDA:
          message = QCoreApplication::translate("GPU device check", "Unable to load CUDA runtime.");
          break;
        case VideoStitch::Discovery::Framework::OpenCL:
          message = QCoreApplication::translate("GPU device check", "Something went wrong with your graphics card.");
          break;
        case VideoStitch::Discovery::Framework::Unknown:
          Q_ASSERT(false);
          return false;
      }
      MsgBoxHandler::getInstance()->genericSync(
          message, QCoreApplication::translate("GPU device check", "Missing GPU driver"), CRITICAL_ERROR_ICON);
      return false;

    case VideoStitch::Discovery::FrameworkStatus::MissingDriver:
      switch (framework) {
        case VideoStitch::Discovery::Framework::CUDA:
          message = QCoreApplication::translate("GPU device check", "Unable to load CUDA driver.");
          break;
        case VideoStitch::Discovery::Framework::OpenCL:
          message = QCoreApplication::translate("GPU device check", "Unable to load GPU driver.");
          break;
        case VideoStitch::Discovery::Framework::Unknown:
          Q_ASSERT(false);
          return false;
      }
      MsgBoxHandler::getInstance()->genericSync(
          message, QCoreApplication::translate("GPU device check", "Missing GPU driver"), CRITICAL_ERROR_ICON);
      return false;

    case VideoStitch::Discovery::FrameworkStatus::OutdatedDriver:
      switch (framework) {
        case VideoStitch::Discovery::Framework::CUDA:
          message = QCoreApplication::translate("GPU device check", "Your CUDA driver appears to be outdated.");
          break;
        case VideoStitch::Discovery::Framework::OpenCL:
          message = QCoreApplication::translate("GPU device check", "Your GPU driver appears to be outdated.");

          break;
        case VideoStitch::Discovery::Framework::Unknown:
          Q_ASSERT(false);
          return false;
      }
      MsgBoxHandler::getInstance()->genericSync(
          message, QCoreApplication::translate("GPU device check", "Outdated GPU driver"), CRITICAL_ERROR_ICON);
      return false;

    case VideoStitch::Discovery::FrameworkStatus::NoCompatibleDevice:
      switch (framework) {
        case VideoStitch::Discovery::Framework::CUDA:
          message = QCoreApplication::translate("GPU device check", "No CUDA capable GPU detected on your system.");
          break;
        case VideoStitch::Discovery::Framework::OpenCL:
          message = QCoreApplication::translate("GPU device check", "No OpenCL capable GPU detected on your system.");
          break;
        case VideoStitch::Discovery::Framework::Unknown:
          Q_ASSERT(false);
          return false;
      }
      MsgBoxHandler::getInstance()->genericSync(
          message, QCoreApplication::translate("GPU device check", "No compatible device found"), CRITICAL_ERROR_ICON);
      return false;
  }
  Q_ASSERT(false);
  return false;
}

void showGPUInitializationError(int device, std::string const& error) {
  VideoStitch::Discovery::DeviceProperties prop;
  if (!VideoStitch::Discovery::getDeviceProperties(device, prop)) {
    Q_ASSERT(false);
  }
  QString deviceName = prop.name;
  QString message;
  if (VideoStitch::Discovery::getNumberOfDevices() == 1) {
    //: GPU error. %0 is the app name, %1 is the GPU name, %2 is the FAQ link
    message =
        QCoreApplication::translate("GPU Initialization", "%0 is unable to open the project with the graphics card.")
            .arg(QCoreApplication::applicationName());
  } else {
    //: GPU error. %0 is the app name, %1 is the GPU name, %2 is the FAQ link
    message = QCoreApplication::translate("GPU Initialization",
                                          "%0 is unable to open a project with the selected graphics card.")
                  .arg(QCoreApplication::applicationName());
  }
  MsgBoxHandler::getInstance()->generic(message, QCoreApplication::translate("GPU Initialization", error.c_str()),
                                        CRITICAL_ERROR_ICON);
}

}  // namespace GPU
}  // namespace VideoStitch
