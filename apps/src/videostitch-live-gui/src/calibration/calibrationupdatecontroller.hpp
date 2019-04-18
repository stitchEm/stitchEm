// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "utils/maintabs.hpp"
#include "../videostitcher/livestitchercontroller.hpp"

#include "libvideostitch-gui/utils/inputlensenum.hpp"
#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/panoramaDefinitionUpdater.hpp"

#include <QTimer>

class OutputControlsPanel;
class ConfigCalibrationWidget;

class CalibrationInterpolator : public QTimer {
  Q_OBJECT
 public:
  explicit CalibrationInterpolator(LiveProjectDefinition* projectDefinition);

 public:
  void setupInterpolation(VideoStitch::Core::PanoramaDefinitionUpdater& panoUpdater);

 signals:
  void reqResetPanoramaWithoutSave(VideoStitch::Core::PanoramaDefinitionUpdater*, bool save = false);

 public slots:
  void start();

 private slots:
  void onInterpolationStep();

 private:
  LiveProjectDefinition* projectDefinition;
  std::unique_ptr<VideoStitch::Core::PanoramaDefinitionUpdater> intermediatePanoramaUpdater;
  unsigned int step;
};

class CalibrationUpdateController : public QObject, public VideoStitch::Core::AlgorithmOutput::Listener {
  Q_OBJECT
 public:
  explicit CalibrationUpdateController(OutputControlsPanel* widget);
  ~CalibrationUpdateController();

  virtual void onPanorama(VideoStitch::Core::PanoramaDefinitionUpdater& pano);
  virtual void onError(const VideoStitch::Status& error);

 private:
  OutputControlsPanel* outputControlsPanel;
  LiveProjectDefinition* projectDefinition;
  CalibrationInterpolator* interpolator;

 public slots:
  void onCalibrationAdaptationAsked();
  void setProject(ProjectDefinition*);
  void clearProject();

 signals:
  void reqCalibrationAdaptationProcess(LiveStitcherController::Callback);
  void reqResetPanoramaWithoutSave(VideoStitch::Core::PanoramaDefinitionUpdater*, bool saveProject = false);
  void reqStartInterpolation();

  friend class ProjectWorkWidget;
};
