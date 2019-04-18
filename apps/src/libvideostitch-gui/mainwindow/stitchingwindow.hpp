// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/common.hpp"
#include "libvideostitch-gui/mainwindow/versionHelper.hpp"

#include <QMainWindow>
#include <QPointer>
#include <QThread>

#include <memory>

class ModalProgressDialog;
class Updater;

class VS_GUI_EXPORT StitchingWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit StitchingWindow(QWidget* parent = nullptr, Qt::WindowFlags flags = 0);
  ~StitchingWindow();

  QVector<int> getCurrentDevices() const;
  QString getGpuNames() const;
  QString getGpuInfo(QList<size_t> usedBytesByDevices);

 signals:
  void reqCancelKernelCompile();
  void reqManualUpdate();

 protected slots:
  void onKernelProgressChanged(const QString& message, double progress);
  void onKernelCompileDone();

 private:
  void buildDeviceList();

 protected:
 private:
  /**
   * List of GPUs
   */
  QVector<int> currentDevices;
  QPointer<ModalProgressDialog> kernelCompilationProgressWindow;
  QThread updaterThread;
};
