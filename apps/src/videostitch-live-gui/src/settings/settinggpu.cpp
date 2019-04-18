// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "settinggpu.hpp"
#include "libgpudiscovery/cudaDeviceInfo.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"
#include "libvideostitch/gpu_device.hpp"

SettingGPU::SettingGPU(QWidget* const parent) : IAppSettings(parent) {
  setupUi(this);
  // fill gpuBox
  VideoStitch::PotentialValue<int> gpuCount = VideoStitch::Discovery::getNumberOfCudaDevices();
  Q_ASSERT(gpuCount.ok());
  for (int device = 0; device < gpuCount.value(); ++device) {
    VideoStitch::Discovery::DeviceProperties prop;
    if (!VideoStitch::Discovery::getCudaDeviceProperties(device, prop)) {
      Q_ASSERT(false);
    }
    gpuBox->addItem(prop.name);
  }
}

SettingGPU::~SettingGPU() {}

void SettingGPU::load() {
  // VSA-4810: At the moment, multi-GPU is not officially supported in VahanaVR

  auto selectedGpus = VSSettings::getSettings()->getDevices();
  if (!selectedGpus.isEmpty()) {
    gpuBox->setCurrentIndex(selectedGpus.first());
  }
  connect(gpuBox, &QComboBox::currentTextChanged, this, &SettingGPU::checkForChanges);
}

void SettingGPU::save() {
  // VSA-4810: At the moment, multi-GPU is not officially supported in VahanaVR
  // Save only if the configuration changed
  QVector<int> selectedGpus = QVector<int>() << gpuBox->currentIndex();
  if (VSSettings::getSettings()->getDevices() != selectedGpus) {
    VSSettings::getSettings()->setDevices(selectedGpus);
  }
}

void SettingGPU::checkForChanges() {
  // VSA-4810: At the moment, multi-GPU is not officially supported in VahanaVR
  // Require restart only if the original configuration changed
  QVector<int> selectedGpus = QVector<int>() << gpuBox->currentIndex();
  emit notifyNeedToSave(VSSettings::getSettings()->getDevices() != selectedGpus);
}
