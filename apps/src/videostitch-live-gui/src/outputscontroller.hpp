// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <functional>
#include <memory>

#include <QHash>
#include <QString>
#include <QTimer>

#include "libvideostitch/status.hpp"

class ConfigurationTabWidget;
class GenericDialog;
class OutPutTabWidget;
class LiveProjectDefinition;
class ProjectDefinition;
class QtDeviceWriter;
class QtHostWriter;
class TopInformationBarWidget;

class OutputsController : public QObject {
  Q_OBJECT
 public:
  OutputsController(TopInformationBarWidget* topBar, OutPutTabWidget* outputTab, ConfigurationTabWidget* configTab);

 public slots:
  void onOutputActivatedFromSideBar(const QString& id);
  void onOutputError();
  void onOutputTryingToActivate();
  void onOutputCreated(const QString& id);
  void onWriterCreated(const QString& id);
  void onWriterRemoved(const QString& id);
  void setProject(ProjectDefinition* project);
  void onOutputChanged(const QString oldName, const QString newName);
  void clearProject();
  void stopTimers();
  void stopTimer(const QString& id);
  void disableOutputsUi();
  void onOutputDisconnected(const QString& outputId);
  void onOutputConnected(const QString& outputId);
  void toggleOutputTimer(const QString& outputId);

 signals:
  void reqActivateOutput(const QString& id);
  void reqToggleOutput(const QString& id);
  void orientationChanged(double yaw, double pitch, double roll);
  void outputActivationChanged();

 private slots:
  void outputConnectionTimerExpired(const QString& outputId, bool reconnectionMode = false);

 private:
  TopInformationBarWidget* informationBarRef;
  OutPutTabWidget* outputTabRef;
  ConfigurationTabWidget* configTabRef;
  QString lastActivated;
  LiveProjectDefinition* projectDefinition;
  QHash<QString, std::shared_ptr<QTimer>> startingOutputs;
  std::unique_ptr<GenericDialog, std::function<void(GenericDialog*)>> connectionDialog;
};
