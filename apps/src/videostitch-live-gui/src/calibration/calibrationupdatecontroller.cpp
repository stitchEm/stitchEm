// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationupdatecontroller.hpp"
#include "widgetsmanager.hpp"
#include "outputcontrolspanel.hpp"
#include "guiconstants.hpp"

#include "libvideostitch-gui/videostitcher/presetsmanager.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/cameraDef.hpp"
#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"

CalibrationInterpolator::CalibrationInterpolator(LiveProjectDefinition* projectDefinition)
    : projectDefinition(projectDefinition) {
  /*Define behavior for each timer tick*/
  connect(this, &CalibrationInterpolator::timeout, this, &CalibrationInterpolator::onInterpolationStep);
}

void CalibrationInterpolator::setupInterpolation(VideoStitch::Core::PanoramaDefinitionUpdater& panoUpdater) {
  /*
   * Create a curve for geometries defined on time t E [0; CALIBRATION_NUM_STEPS_INTERPOLATION]
   * The initial calibration in the input panodef is the t=0 calibration
   * The destination calibration in the input panodef is the constant value calibration
   */
  intermediatePanoramaUpdater = std::make_unique<VideoStitch::Core::PanoramaDefinitionUpdater>(std::move(panoUpdater));

  for (int idinput = 0; idinput < (int)intermediatePanoramaUpdater->numInputs(); ++idinput) {
    VideoStitch::Core::GeometryDefinition destinationCalib =
        intermediatePanoramaUpdater->getInput(idinput).getGeometries().getConstantValue();
    VideoStitch::Core::GeometryDefinition initialCalib =
        intermediatePanoramaUpdater->getInput(idinput).getGeometries().at(0);

    /*Let's create a linear spline*/
    VideoStitch::Core::SplineTemplate<VideoStitch::Core::GeometryDefinition>* spline;
    spline = VideoStitch::Core::SplineTemplate<VideoStitch::Core::GeometryDefinition>::point(0, initialCalib);
    spline->lineTo(CALIBRATION_NUM_STEPS_INTERPOLATION, destinationCalib);

    /*Replace the curve with the newly created one*/
    intermediatePanoramaUpdater->getInput(idinput).replaceGeometries(
        new VideoStitch::Core::GeometryDefinitionCurve(spline));
  }
}

void CalibrationInterpolator::start() {
  /*Start timer*/
  step = 0;
  QTimer::start(CALIBRATION_INTERPOLATION_TIMEOUT);
}

void CalibrationInterpolator::onInterpolationStep() {
  if (step >= CALIBRATION_NUM_STEPS_INTERPOLATION) {
    /*Once we ticked more than CALIBRATION_NUM_STEPS_INTERPOLATION times, the interpolator is at the destination
     * calibration. Therefore, we do not need to interpolate anymore and quit the timer "process"
     */
    QTimer::stop();
    return;
  }

  /*Vahana is not aware of time, so let's interpolate geometries*/

  /*For each input, the current calibration is a function of time (step)*/
  for (int idinput = 0; idinput < (int)intermediatePanoramaUpdater->numInputs(); ++idinput) {
    VideoStitch::Core::GeometryDefinition currentCalib =
        intermediatePanoramaUpdater->getInput(idinput).getGeometries().at(step);
    VideoStitch::Core::GeometryDefinitionCurve* curve = new VideoStitch::Core::GeometryDefinitionCurve(
        VideoStitch::Core::SplineTemplate<VideoStitch::Core::GeometryDefinition>::point(0, currentCalib));
    intermediatePanoramaUpdater->getInput(idinput).replaceGeometries(curve);
  }

  /*Ask the lib to update the panoDef for the next stitching iteration*/
  emit reqResetPanoramaWithoutSave(intermediatePanoramaUpdater.get());

  // Todo: ugly, consider introduction of ability to apply updates multiple times.
  intermediatePanoramaUpdater = std::make_unique<VideoStitch::Core::PanoramaDefinitionUpdater>(
      (VideoStitch::Core::PanoDefinition&)intermediatePanoramaUpdater);

  step++;
}

CalibrationUpdateController::CalibrationUpdateController(OutputControlsPanel* widget)
    : outputControlsPanel(widget), projectDefinition(nullptr), interpolator(nullptr) {
  /*
   * Connect the button click event
   */
  connect(outputControlsPanel->buttonCalibrationAdapt, &QPushButton::clicked, this,
          &CalibrationUpdateController::onCalibrationAdaptationAsked);
}

CalibrationUpdateController::~CalibrationUpdateController() {}

void CalibrationUpdateController::onPanorama(VideoStitch::Core::PanoramaDefinitionUpdater& pano) {
  /*
   * This function is called whenever the calibration adaptation process has succeeded.
   * In this case, what we want is smoothly update the panodefinition over time to use this new calibration
   */

  if (interpolator) {
    interpolator->setupInterpolation(pano);
    emit reqStartInterpolation();
  }
}

void CalibrationUpdateController::onError(const VideoStitch::Status& error) {
  Q_UNUSED(error);
  /*
   * This function is called whenever the calibration adaptation process has failed.
   * Because it's may be an automated process, we don't want the user to have a feedback.
   */
}

void CalibrationUpdateController::onCalibrationAdaptationAsked() {
  /*
   * A calibration adaptation has been required.
   * Let's initialize the calibration adaptation algorithm.
   */

  /* Load calibration configuration from default presets */
  std::unique_ptr<VideoStitch::Ptv::Value> calibrationConfig =
      PresetsManager::getInstance()->clonePresetContent("calibration", "default_vahana");
  if (!calibrationConfig) {
    return;
  }

  /* Configuration for the adaptation process */
  calibrationConfig->get("adaptation_mode")->asBool() = true;

  /*
   * Create online algorithm for calibration
   */
  VideoStitch::Potential<VideoStitch::Util::OnlineAlgorithm> fStatusCalib =
      VideoStitch::Util::OnlineAlgorithm::create("calibration", calibrationConfig.get());
  if (!fStatusCalib.ok()) {
    return;
  }

  /* Any completion information will be sent back to the current (this) object */
  emit reqCalibrationAdaptationProcess(LiveStitcherController::Callback(fStatusCalib.release(), this, nullptr));
}

void CalibrationUpdateController::setProject(ProjectDefinition* p) {
  /*
   * The project changed.
   * This means many object may have changed.
   */
  clearProject();

  projectDefinition = qobject_cast<LiveProjectDefinition*>(p);
  interpolator = new CalibrationInterpolator(projectDefinition);

  /*Recreate signals connections*/
  connect(this, &CalibrationUpdateController::reqStartInterpolation, interpolator, &CalibrationInterpolator::start,
          static_cast<Qt::ConnectionType>(Qt::QueuedConnection | Qt::UniqueConnection));
  connect(interpolator, &CalibrationInterpolator::reqResetPanoramaWithoutSave, this,
          &CalibrationUpdateController::reqResetPanoramaWithoutSave, Qt::UniqueConnection);
}

void CalibrationUpdateController::clearProject() {
  /*
   * The project changed.
   * This means many object may have changed.
   */
  delete interpolator;
  interpolator = nullptr;
  projectDefinition = nullptr;
}
