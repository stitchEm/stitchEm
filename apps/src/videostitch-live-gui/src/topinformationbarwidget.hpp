// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef TOPINFORMATIONBARWIDGET_H
#define TOPINFORMATIONBARWIDGET_H

#include "ui_topinformationbarwidget.h"

#include "libvideostitch-gui/caps/guistatecaps.hpp"
#include "libvideostitch-gui/widgets/levelmeter.hpp"

#include "libvideostitch/frame.hpp"

#include <QTimer>

class GenericDialog;
class LiveProjectDefinition;
class ProjectDefinition;

class TopInformationBarWidget : public QFrame, public GUIStateCaps, public Ui::TopInformationBarWidgetClass {
  Q_OBJECT

 public:
  explicit TopInformationBarWidget(QWidget* const parent = nullptr);
  virtual ~TopInformationBarWidget();

  void setDefaultTime();
  void setCurrentProjectName(const QString& fileName);
  void showLoading(const bool show);
  void onActivateOutputs(const QString id);
  void onDeactivateOutputs(const QString id);
 public slots:
  void updateCurrentTime(const QString& time, const double estimatedFramerate, VideoStitch::FrameRate targetFramerate);
  void setProject(const ProjectDefinition* project);
  void clearProject();
  void onQuitButtonClicked();
  void onQuitConfirmed();
  void onQuitCancelled();
  void onActivateExposure(const bool activate);
  void updateVuMeterValues();
 protected slots:
  virtual void changeState(GUIStateCaps::State state);

 signals:
  void notifyQuitProject();
  void reqChangeState(GUIStateCaps::State s);

 private:
  void cleanOutputs();
  QScopedPointer<QMovie> loadingAnimation;
  QScopedPointer<GenericDialog> quitDialog;
  const LiveProjectDefinition* projectDefinition;
  QMap<QString, QWidget*> outputsMap;
};

#endif  // TOPINFORMATIONBARWIDGET_H
