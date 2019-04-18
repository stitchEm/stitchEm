// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "iconfigurationvalue.hpp"
#include "ui_audioprocessorswidget.h"
#include <QPointer>
#include <QPushButton>

class GenericTableWidget;
class NewAudioProcessorWidget;
class LiveAudioProcessFactory;
class AudioProcessorConfigurationWidget;
class LiveProjectDefinition;

class AudioProcessorsWidget : public IConfigurationCategory, Ui::ConfigAudioProcessorsWidgetClass {
  Q_OBJECT
 public:
  explicit AudioProcessorsWidget(QWidget* const parent = nullptr);

 signals:
  void injectProject(LiveProjectDefinition* project);
  void projectCleared();
  void notifyRemoveProcessor(const QString name);
  void notifyEditProcessor(LiveAudioProcessFactory* liveAudioProcess);

 protected:
  virtual void reactToChangedProject();
  virtual void reactToClearedProject();

 private slots:
  void onProcessorClicked(int row, int col);
  void onButtonAddProcessorClicked();
  void onNewProcessorBackClicked();
  void onNewProcessorSelected(const QString displayName, const QString type, const bool isUsed);
  void onProcessorWidgetFinished();

 private:
  void removeProcessors();
  void addProcessors(QList<LiveAudioProcessFactory*> liveAudioProcessors);
  void addSingleProcessor(LiveAudioProcessFactory* processor);
  void createAddButton();
  void showConfigurationWidget(AudioProcessorConfigurationWidget* widget);
  GenericTableWidget* tableProcessors;
  NewAudioProcessorWidget* newAudioProcessorWidget;
  QPointer<QPushButton> buttonAddProcessor;
};
