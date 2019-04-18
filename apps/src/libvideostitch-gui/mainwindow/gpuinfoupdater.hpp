// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/gpu_device.hpp"

#include "libgpudiscovery/genericDeviceInfo.hpp"

#include "vssettings.hpp"

#include <QThread>
#include <QTimer>
#include <QList>

#include <vector>

#define REFRESH_INTERVAL 5000

using namespace VideoStitch;

/**
 * @brief This class is the class which polls the GPU to update get its used memory
 */
class VS_GUI_EXPORT GPUInfoUpdater : public QThread {
  Q_OBJECT
 signals:
  /**
   * @brief Signal called when the gpu informations have been updated. The main window will catch that signal and update
   * the UI/
   * @param usedBytes Used memory on the GPU (in bytes) by Studio.
   * @param totalBytes Total memory in the GPU (in bytes).
   */
  void reqUpdateGPUInfo(size_t usedBytes, size_t totalBytes, QList<size_t> usedBytesByDevices);

  /**
   * @brief Starts the polling timer.
   */
  void startTimer();

 public slots:
  /**
   * @brief Slot called when the refresh timer times out. This slot is called every REFRESH_INTERVAL milliseconds.
   */
  void refreshTick() {
    PotentialValue<size_t> used = GPU::getMemoryUsage();
    std::size_t totalBytes = 0;
    // sum the total GPU memory across devices
    for (int deviceId : VSSettings::getSettings()->getDevices()) {
      VideoStitch::Discovery::DeviceProperties prop;
      if (VideoStitch::Discovery::getDeviceProperties(deviceId, prop)) {
        totalBytes += prop.globalMemSize;
      }
    }
    Q_ASSERT(used.ok());
    // retrive the used memory by devices
    PotentialValue<std::vector<size_t> > vectorUsedByDevices = GPU::getMemoryUsageByDevices();
    Q_ASSERT(vectorUsedByDevices.ok());
    // turn it into a QList
    QList<size_t> listUsedByDevices;
    for (auto &it : vectorUsedByDevices.value()) {
      listUsedByDevices.push_back(it);
    }
    emit reqUpdateGPUInfo(used.value(), totalBytes, listUsedByDevices);
  }

 private:
  /**
   * @brief Overloaded method run of QThread. It is called by the start() slot and initializes the updater thread.
   */
  void run() {
    refreshTimer.setInterval(REFRESH_INTERVAL);
    connect(&refreshTimer, SIGNAL(timeout()), this, SLOT(refreshTick()));
    connect(this, SIGNAL(startTimer()), &refreshTimer, SLOT(start()));
    emit startTimer();
    exec();
  }

  QTimer refreshTimer;
};
