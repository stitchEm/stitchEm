// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videostitcher/liveprojectdefinition.hpp"
#include "videostitcher/livestitchercontroller.hpp"
#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/panoramaDefinitionUpdater.hpp"

#include <QTimer>

class OutputControlsPanel;
class OutPutTabWidget;
class ConfigExposureWidget;
/**
 * @brief The ExposureActionController class for creating, saving and loading an exposure configuration
 */
class ExposureActionController : public QObject, public VideoStitch::Core::AlgorithmOutput::Listener {
  Q_OBJECT
 public:
  ExposureActionController(OutPutTabWidget* widget);
  ~ExposureActionController();

  virtual void onPanorama(VideoStitch::Core::PanoramaDefinitionUpdater& pano);
  virtual void onError(const VideoStitch::Status&);

  bool exposureIsActivated() const;

 private:
  /**
   * @brief Starts the continious exposure algorithm
   */
  void startExposure();
  /**
   * @brief Stops the continious exposure algorithm
   */
  void stopExposure();
  /**
   * @brief Shows an error message for the exposure algorithm
   */
  void showExposureErrorMessage();
  /**
   * @brief Runs the exposure algorithm if this was on autorun mode
   */
  void checkAutoExposure();

  OutputControlsPanel* controlsPanel;
  QTimer expoContTimer;
  LiveProjectDefinition* projectDefinition;
  OutPutTabWidget* outputTabRef;

 public slots:
  void onExposureApplied();
  void onExposureFailed();
  void setProject(ProjectDefinition*);
  void clearProject();
  void onExposureSettings();
  void onActivationFromControlBar(const bool active);
  void toggleExposure(const bool active);

 signals:
  void reqReplacePanorama(VideoStitch::Core::PanoramaDefinitionUpdater* panorama, bool saveProject = false);
  void reqCompensateExposure(LiveStitcherController::Callback);
  void reqCancelExposure();
  void reqClearExposure();
  void exposureActivationChanged(bool active);
};
