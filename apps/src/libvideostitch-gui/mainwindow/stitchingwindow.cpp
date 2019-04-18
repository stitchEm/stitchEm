// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stitchingwindow.hpp"

#include "libvideostitch-gui/dialogs/modalprogressdialog.hpp"
#include "vssettings.hpp"

#include "ui_header/progressreporterwrapper.hpp"

#include "libvideostitch/gpu_device.hpp"

#include "libgpudiscovery/delayLoad.hpp"
#include "libgpudiscovery/genericDeviceInfo.hpp"

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

static const size_t BytesInMB = 1024 * 1024;

StitchingWindow::StitchingWindow(QWidget* parent, Qt::WindowFlags flags) : QMainWindow(parent, flags) {
  buildDeviceList();
}

StitchingWindow::~StitchingWindow() {}

QVector<int> StitchingWindow::getCurrentDevices() const { return currentDevices; }

QString StitchingWindow::getGpuNames() const {
  QStringList gpuNames;
  for (int d : currentDevices) {
    VideoStitch::Discovery::DeviceProperties props;
    if (!VideoStitch::Discovery::getDeviceProperties(d, props)) {
      return "";
    }
    QString gpuLabel;
    // if more than one GPU, add an index at the beginning of the name
    if (currentDevices.size() > 1) {
      gpuLabel = tr("%0: %1 (%2 MB)").arg(d).arg(props.name).arg(props.globalMemSize / BytesInMB);
    } else {
      gpuLabel = tr("%0 (%1 MB)").arg(props.name).arg(props.globalMemSize / BytesInMB);
    }
    gpuNames.push_back(gpuLabel);
  }
  return gpuNames.join(" | ");
}

QString StitchingWindow::getGpuInfo(QList<size_t> usedBytesByDevices) {
  // format info strings
  QStringList infoStrings;
  for (int index = 0; index < currentDevices.size(); ++index) {
    // at Studio start, when no process is running on gpu, nvml retrieve no info, so don't crash and wait
    if (usedBytesByDevices.size() == 0) {
      infoStrings.append(tr("%0 MB").arg(0));
    }
    // if more than one GPU, add an index at the beginning of the name
    else if (currentDevices.size() > 1) {
      infoStrings.append(tr("GPU %0: %1 MB").arg(currentDevices[index]).arg(usedBytesByDevices[index] / BytesInMB));
    } else {
      infoStrings.append(tr("%0 MB").arg(usedBytesByDevices[index] / BytesInMB));
    }
  }

  return tr("GPU memory usage: %0").arg(infoStrings.join(" | "));
}

void StitchingWindow::onKernelProgressChanged(const QString& message, double progress) {
  if (kernelCompilationProgressWindow.isNull()) {
    kernelCompilationProgressWindow = new ModalProgressDialog(tr("Preparing your GPU"), this);
    kernelCompilationProgressWindow->getReporter()->setRange(0, 0);
  }

  if (kernelCompilationProgressWindow->getReporter()->notify(message.toStdString(), progress * 100)) {
    emit reqCancelKernelCompile();
    kernelCompilationProgressWindow->getReporter()->reset();
  }

  if (kernelCompilationProgressWindow->isHidden()) {
    kernelCompilationProgressWindow->show();
  }
}

void StitchingWindow::onKernelCompileDone() {
  if (!kernelCompilationProgressWindow.isNull()) {
    kernelCompilationProgressWindow->hide();
    kernelCompilationProgressWindow->deleteLater();
  }
}

void StitchingWindow::buildDeviceList() {
  currentDevices.clear();
  int nbDevices = VideoStitch::Discovery::getNumberOfDevices();
  // Initialize with previously selected devices
  QVector<int> selectedDevices = VSSettings::getSettings()->getDevices();

  for (int device : selectedDevices) {
    VideoStitch::Discovery::DeviceProperties prop;
    if (device < nbDevices && VideoStitch::Discovery::getDeviceProperties(device, prop) &&
        prop.supportedFramework == VideoStitch::GPU::getFramework()) {
      currentDevices.append(device);
    }
  }

  // If the selected GPU was removed, initialize with the first valid GPU
  if (currentDevices.isEmpty()) {
    for (int device = 0; device < nbDevices; ++device) {
      VideoStitch::Discovery::DeviceProperties prop;
      if (VideoStitch::Discovery::getDeviceProperties(device, prop) &&
          prop.supportedFramework == VideoStitch::GPU::getFramework()) {
        currentDevices.append(device);
        break;
      }
    }
  }

  if (currentDevices != selectedDevices) {
    VSSettings::getSettings()->setDevices(currentDevices);
  }
}
