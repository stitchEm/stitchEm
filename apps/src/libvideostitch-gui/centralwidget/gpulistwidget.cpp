// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpulistwidget.hpp"

#include "libvideostitch/gpu_device.hpp"

#include <libgpudiscovery/genericDeviceInfo.hpp>

GpuListWidget::GpuListWidget(QWidget *parent) : VSListWidget(parent) {
  int gpuCount = VideoStitch::Discovery::getNumberOfDevices();

  for (int device = 0; device < gpuCount; ++device) {
    VideoStitch::Discovery::DeviceProperties prop;
    if (VideoStitch::Discovery::getDeviceProperties(device, prop)) {
      QListWidgetItem *listItem = new QListWidgetItem(prop.name);
      listItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
      listItem->setCheckState(Qt::Unchecked);
      addItem(listItem);
    }
  }

  // Let's disable the device selection if there is only one device
  if (count() == 1) {
    setDisabled(true);
  }

  checkGpus();
  connect(this, &QListWidget::itemClicked, this, &GpuListWidget::toggleItemCheckState);
}

GpuListWidget::~GpuListWidget() {}

QVector<int> GpuListWidget::getSelectedGpus() const {
  QVector<int> selectedGpus;
  for (int index = 0; index < count(); ++index) {
    if (item(index)->checkState() == Qt::Checked) {
      selectedGpus.append(index);
    }
  }
  return selectedGpus;
}

void GpuListWidget::toggleItemCheckState(QListWidgetItem *item) {
  item->setCheckState(item->checkState() == Qt::Checked ? Qt::Unchecked : Qt::Checked);
}

void GpuListWidget::checkGpus() { checkGpusFor(nullptr); }

void GpuListWidget::setSelectedGpus(QVector<int> selectedGpus) {
  disconnect(this, &QListWidget::itemChanged, this, &GpuListWidget::checkGpusFor);
  for (auto index = 0; index < count(); ++index) {
    item(index)->setCheckState(selectedGpus.contains(index) ? Qt::Checked : Qt::Unchecked);
  }
  connect(this, &QListWidget::itemChanged, this, &GpuListWidget::checkGpusFor, Qt::UniqueConnection);
}

void GpuListWidget::checkGpusFor(QListWidgetItem *changedItem) {
  disconnect(this, &QListWidget::itemChanged, this, &GpuListWidget::checkGpusFor);

  QList<QListWidgetItem *> selectedGpus;
  for (int index = 0; index < count(); ++index) {
    if (item(index)->checkState() == Qt::Checked) {
      selectedGpus.append(item(index));
    }
  }

  if (changedItem) {
    // find changed item's backend
    VideoStitch::Discovery::DeviceProperties changedItemProp;
    const int changedItemIndex = row(changedItem);
    bool changedItemFound = VideoStitch::Discovery::getDeviceProperties(changedItemIndex, changedItemProp);
    if (changedItemFound) {
      // filter selected GPUs by changed item's backend
      for (int index = 0; index < count(); ++index) {
        if (item(index)->checkState() == Qt::Checked) {
          VideoStitch::Discovery::DeviceProperties prop;
          if (VideoStitch::Discovery::getDeviceProperties(index, prop) &&
              prop.supportedFramework != changedItemProp.supportedFramework) {
            item(index)->setCheckState(Qt::Unchecked);
            selectedGpus.removeOne(item(index));
          }
        }
      }
    }
  }

  // Control min gpus (= 1)
  if (selectedGpus.isEmpty()) {
    if (changedItem) {
      changedItem->setCheckState(Qt::Checked);
      selectedGpus.append(changedItem);
    } else {
      item(0)->setCheckState(Qt::Checked);
      selectedGpus.append(item(0));
    }
  }

  connect(this, &QListWidget::itemChanged, this, &GpuListWidget::checkGpusFor, Qt::UniqueConnection);
  emit selectedGpusChanged();
}
