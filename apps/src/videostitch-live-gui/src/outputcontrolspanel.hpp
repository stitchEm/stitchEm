// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_outputcontrolspanel.h"
#include <QSignalMapper>

class LiveOutputFactory;
class LiveProjectDefinition;

class OutputControlsPanel : public QFrame, public Ui::OutputControlsPanelClass {
  Q_OBJECT

 public:
  enum Option {
    NoOption = 0x0,
    HideOrientation = 0x1,
    HidePanorama = 0x2,
    HideSnapshot = 0x4,
    HideAudioProcessors = 0x8
  };
  Q_DECLARE_FLAGS(Options, Option);

 public:
  explicit OutputControlsPanel(QWidget* const parent = nullptr);
  ~OutputControlsPanel();

  void configure(Options options);
  void toggleOutput(const QString& id);
  void setOutputActionable(const QString& id, bool actionable);
  void setProject(LiveProjectDefinition* project);
  void clearProject();
  void onOutputIdChanged(const QString& oldId, const QString& newId);
  void updateEditability(bool outputIsActivated, bool algorithmIsActivated);
  void showMainTab();

 public slots:
  void onProjectClosed();
  void onCalibrationButtonClicked();

 signals:
  void notifyOutputActivated(const QString& id);
  void notifyConfigureOutput(const QString& id);
  void notifyTakePanoSnapshot();
  void notifyNewCalibration();
  void notifyPanoramaEdition();
  void notifyAudioProcessorsEdition();
  void notifyStartAddingOutput();

 protected:
  virtual void resizeEvent(QResizeEvent*);
  virtual void showEvent(QShowEvent*);

 private slots:
  void onButtonDownScrollClicked();
  void onButtonUpScrollClicked();
  void onButtonExposureClicked();
  void onOutputButtonClicked(const QString& id);
  void onButtonAddOutputClicked();
  void onOutputEditClicked(const QString& id);

 private:
  void addOutputButtons();
  void addOutputButton(LiveOutputFactory* output);
  void removeOutputButtons();
  void updateScrollButtons();
  void showAudioProcessButton();
  QMap<QString, QList<QPushButton*>> outputsMap;
  QList<QWidget*> widgetsToDelete;
  QSignalMapper* buttonMapper;
  QSignalMapper* labelMapper;
  LiveProjectDefinition* projectDefinition;
};

Q_DECLARE_OPERATORS_FOR_FLAGS(OutputControlsPanel::Options);
