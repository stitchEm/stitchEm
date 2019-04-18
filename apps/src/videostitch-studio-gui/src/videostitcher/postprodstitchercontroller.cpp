// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "postprodstitchercontroller.hpp"
#include "commands/externcalibrationappliedcommand.hpp"
#include "commands/orientationchangedcommand.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"
#include "libvideostitch-gui/mainwindow/timeconverter.hpp"
#include "libvideostitch-gui/videostitcher/stitchercontrollerprogressreporter.hpp"
#include "libvideostitch-gui/videostitcher/videostitcher.hpp"

#include "libvideostitch-base/file.hpp"
#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <QApplication>
#include <QFileDialog>

#define VIDEO_WRITER_DEFAULT_FRAMERATE 25.0

PostProdStitcherController::PostProdStitcherController(DeviceDefinition& device)
    : StitcherController(device), project(nullptr), currentProjection(VideoStitch::unknownProjection) {}

PostProdStitcherController::~PostProdStitcherController() {
  delete project;
  project = nullptr;
}

const ProjectDefinition* PostProdStitcherController::getProjectPtr() const { return project; }

ProjectDefinition* PostProdStitcherController::getProjectPtr() { return project; }

frameid_t PostProdStitcherController::getLastStitchableFrame() const {
  if (controller != nullptr) {
    return controller->getLastStitchableFrame();
  } else {
    return 0;
  }
}

std::vector<frameid_t> PostProdStitcherController::getLastFrames() const {
  if (controller) {
    return controller->getLastFrames();
  } else {
    return std::vector<frameid_t>();
  }
}

bool PostProdStitcherController::hasAudioInput() const { return controller->hasAudio(); }

void PostProdStitcherController::informOnPano() {
  if (controller) {
    emit panoChanged(controller->getLastStitchableFrame(), controller->getCurrentFrame());
  } else {
    emit panoChanged(0, 0);
  }
}

void PostProdStitcherController::switchOutputAndRestitch(const QString& outputType) {
  this->switchOutput(outputType, true);
}

// -------------------------- Stitching -------------------------------------

bool PostProdStitcherController::seekOnSignal(frameid_t frame, SignalCompressionCaps* comp) {
  if (comp && comp->pop() > 0) {
    return false;
  }
  if (controller) {
    frameid_t currentFrame = controller->getCurrentFrame();

    VideoStitch::Status seekStatus = controller->seekFrame(frame);
    if (seekStatus.ok()) {
      return true;
    }

    // if seeking failed, reader state is undefined; readers may be out of sync
    // (some may have seeked to the correct position, some may have not moved at all, some may have moved to a random
    // position) let's try to get back to a defined state and sync them again, even if it's not where we wanted to go
    seekStatus = controller->seekFrame(currentFrame);
    if (seekStatus.ok()) {
      return true;
    }

    // hail mary
    seekStatus = controller->seekFrame(controller->getFirstReadableFrame());
    if (seekStatus.ok()) {
      return true;
    }

    forwardStitcherError(
        VideoStitch::Core::ControllerStatus::fromError(
            {VideoStitch::Origin::PostProcessor, VideoStitch::ErrType::RuntimeError,
             PostProdStitcherController::tr("Seeking Error: Unable to seek in the video stream. Please check the input "
                                            "file encoding and reload the project.")
                 .toStdString(),
             seekStatus}),
        false);
  }
  return false;
}

void PostProdStitcherController::stitch(frameid_t frame, SignalCompressionCaps* comp) {
  // one-time stitch commands should not be given while stitch event loop (requireNextFrame) runs
  assert(!isPlaying());
  if (seekOnSignal(frame, comp)) {
    StitcherController::stitchOnce();
  }
}

void PostProdStitcherController::restitch(frameid_t frame, SignalCompressionCaps* comp) {
  if (seekOnSignal(frame, comp)) {
    StitcherController::restitchOnce();
  }
}

void PostProdStitcherController::extract(frameid_t frame, SignalCompressionCaps* comp) {
  if (seekOnSignal(frame, comp)) {
    StitcherController::extractOnce();
  }
}

void PostProdStitcherController::reextract(frameid_t frame, SignalCompressionCaps* comp) {
  if (seekOnSignal(frame, comp)) {
    StitcherController::reextractOnce();
  }
}

void PostProdStitcherController::stitchAndExtract(frameid_t frame, SignalCompressionCaps* comp) {
  // one-time stitch commands should not be given while stitch event loop (requireNextFrame) runs
  assert(!isPlaying());
  if (seekOnSignal(frame, comp)) {
    StitcherController::stitchAndExtractOnce();
  }
}

// -------------------------- Open ------------------------------------------

void PostProdStitcherController::finishProjectOpening() {
  // set the projection
  setProjection(VideoStitch::mapPTVStringToIndex[project->getProjection()], project->getHFOV());
  // adjust the GUI according to the first/last frames
  project->checkRange(controller->getFirstReadableFrame(), controller->getLastStitchableFrame());
  informOnPano();
}

void PostProdStitcherController::openInputs(QList<QUrl> urls, int customWidth, int customHeight) {
  StitcherControllerProgressReporter progressReporter(this);
  std::unique_ptr<VideoStitch::Input::DefaultReaderFactory> factory(
      new VideoStitch::Input::DefaultReaderFactory(0, NO_LAST_FRAME));
  QList<VideoStitch::Ptv::Value*> userInputs;
  foreach (QUrl url, urls) {
    VideoStitch::Ptv::Value* val = VideoStitch::Ptv::Value::emptyObject();
    val->asString() = url.toString().toStdString();
    VideoStitch::Input::ProbeResult result = factory->probe(*val);
    if (result.valid) {
      VideoStitch::Ptv::Value* input = VideoStitch::Ptv::Value::emptyObject();
      input->get("width")->asInt() = result.width;
      input->get("height")->asInt() = result.height;
      // all inputs are in the same group in order not to assume
      // the audio and video track begin at the same instant
      // (in practice, the audio might start a few frame before the
      // video in the container)
      input->get("group")->asInt() = 0;
      input->push("reader_config", val);
      input->get("video_enabled")->asBool() = true;
      input->get("audio_enabled")->asBool() = result.hasAudio;
      userInputs.push_back(input);
    } else {
      MsgBoxHandler::getInstance()->generic(
          PostProdStitcherController::tr("The file '%1' cannot be handled by %2. Please check the media file validity.")
              .arg(url.toString())
              .arg(QCoreApplication::applicationName()),
          PostProdStitcherController::tr("Importing media file failed"), WARNING_ICON);
    }
  }
  if (userInputs.isEmpty()) {
    emit openFromInputFailed();
    return;
  }

  if (!project) {
    createProject();
  }

  if (project->isInit()) {
    if ((customHeight != 0) || (customWidth != 0)) {
      project->updateSize(customWidth, customHeight);
    }
    project->addInputs(userInputs);
  } else if (project->setDefaultValues(userInputs)) {
    if ((customHeight != 0) || (customWidth != 0)) {
      project->updateSize(customWidth, customHeight);
    }
    // start a new project from scratch
    bool success = open(&progressReporter);
    project->setModified(true);
    if (!success) {
      return;
    }

    StitcherController::extractOnce();
  }
  progressReporter.finishProgress();
  emit notifyInputsOpened();
}

bool PostProdStitcherController::checkProject() {
  // set the group attribute in the InputDefinition:
  // all inputs are assumed to share a common time origin
  for (readerid_t inputId = 0; inputId < project->getNumInputs(); ++inputId) {
    project->getPano()->getInput(inputId).setGroup(0);
  }

  // normalize paths
  project->fixInputPaths();

  QString newFolder = ProjectFileHandler::getInstance()->getWorkingDirectory();

  project->fixMissingInputs(newFolder);

  // Check blender (laplacian is disabled for OpenCL)
  const std::vector<std::string>& availableMergers = VideoStitch::Core::ImageMergerFactory::availableMergers();
  std::unique_ptr<VideoStitch::Ptv::Value> factoryValue(project->getImageMergerFactory()->serialize());
  const std::string& merger = factoryValue->has("type")->asString();
  if (std::find(availableMergers.cbegin(), availableMergers.cend(), merger) == availableMergers.cend()) {
    bool fallBackToLinearBlending = false;
    emit mergerNotSupportedByGpu(merger, fallBackToLinearBlending);
    if (fallBackToLinearBlending) {
      std::unique_ptr<VideoStitch::Ptv::Value> mergerValue(VideoStitch::Ptv::Value::emptyObject());
      mergerValue->get("type")->asString() = "gradient";
      VideoStitch::Potential<VideoStitch::Core::ImageMergerFactory> mergerFactory =
          VideoStitch::Core::ImageMergerFactory::createMergerFactory(*mergerValue);
      project->setMergerFactory(mergerFactory.release());
      project->setModified(true);
    } else {
      return false;
    }
  }

  return true;
}

// --------------------------------- Reconfigure ------------------------------------------

VideoStitch::Status PostProdStitcherController::resetPano() {
  if (!controller) {
    return VideoStitch::Status::OK();  // we call stupidly reset when switching tab...
  }

  std::stringstream msg;
  if (!project->validatePanorama(msg)) {
    // using translation for static error message
    std::string msgString = PostProdStitcherController::tr("Invalid panorama parameters").toStdString();
    if (!msg.str().empty()) {
      msgString += ": " + msg.str();
    }
    const VideoStitch::Status st = {VideoStitch::Origin::PostProcessor, VideoStitch::ErrType::UnsupportedAction,
                                    msgString};

    forwardStitcherError(st, false);
    return st;
  }

  setupLock.lockForWrite();
  frameid_t frame = controller->getCurrentFrame();
  const auto& panoDef = *project->getPanoConst().get();

  if (!controller->isPanoChangeCompatible(panoDef)) {
    StitcherControllerProgressReporter progressReporter(this);
    progressReporter.setProgress(10);
    setupLock.unlock();
    this->closeProject();
    progressReporter.setProgress(40);
    bool success = this->open(&progressReporter, frame);
    if (success) {
      progressReporter.finishProgress();
    }
  } else {
    VideoStitch::Status st = controller->updatePanorama(*project->getPanoConst().get());
    setupLock.unlock();

    if (!st.ok()) {
      forwardStitcherError(st, false);
      return st;
    }

    // TODO don't return OK on error from restitch
    stitcher->restitch();
  }
  informOnPano();

  return VideoStitch::Status::OK();
}

void PostProdStitcherController::reset() { resetPano(); }

void PostProdStitcherController::onResetRig() {
  setupLock.lockForWrite();
  controller->resetRig(*project->getStereoRigConst().get());
  setupLock.unlock();

  stitcher->restitch();
  informOnPano();
}

void PostProdStitcherController::resetProject() {
  // inform everyone that the project is no longer valid
  emit projectReset();
  delete project;
  project = nullptr;
}

void PostProdStitcherController::resetMerger(SignalCompressionCaps* signalCompressor) {
  if (signalCompressor && signalCompressor->pop() > 0) {
    return;
  }
  stitcher->resetMerger();
  emit resetMergerApplied();
  delayedUpdate();
}

void PostProdStitcherController::resetAdvancedBlending(SignalCompressionCaps* signalCompressor) {
  if (signalCompressor && signalCompressor->pop() > 0) {
    return;
  }
  stitcher->resetAdvancedBlending();
  emit resetAdvancedBlendingApplied();
  delayedUpdate();
}

void PostProdStitcherController::applySynchronization(SignalCompressionCaps* comp,
                                                      VideoStitch::Core::PanoDefinition* panoDef) {
  if (comp && comp->pop() > 0) {
    return;
  }
  project->setPano(panoDef);
  if (resetPano().ok()) {
    // from the controller documentation:
    //   * Status resetPano(const PanoDefinition& newPano):
    //   * For performance reasons, this does not change the current state of the readers.
    //   * Therefore, if you change frame offsets, you'll have to seekFrame() to the current frame to resynchronize the
    //   readers.
    stitchAndExtract(controller->getCurrentFrame());

    // When synchronizing, the project last frame can become invalid
    if (project->getLastFrame() > getLastStitchableFrame()) {
      project->setLastFrame(getLastStitchableFrame());
    }
  }
}

void PostProdStitcherController::applyExposure(VideoStitch::Core::PanoDefinition* panoDef) {
  project->setPano(panoDef);
  if (resetPano().ok()) {
    for (int i = 0; i < (int)project->getNumInputs(); ++i) {
      emit reqUpdateCurve(panoDef->getInput(i).getExposureValue().clone(), CurveGraphicsItem::InputExposure, i);
      emit reqUpdateCurve(panoDef->getInput(i).getBlueCB().clone(), CurveGraphicsItem::BlueCorrection, i);
      emit reqUpdateCurve(panoDef->getInput(i).getRedCB().clone(), CurveGraphicsItem::RedCorrection, i);
    }
    emit exposureApplied();
  }
}

void PostProdStitcherController::applyPhotometricCalibration(VideoStitch::Core::PanoDefinition* panoDef) {
  project->setPano(panoDef);
  if (resetPano().ok()) {
    emit reqUpdatePhotometry();
  }
}

void PostProdStitcherController::applyStabilization(VideoStitch::Core::PanoDefinition* panoDef) {
  project->setPano(panoDef);
  if (resetPano().ok()) {
    emit reqUpdateQuaternionCurve(panoDef->getStabilization().clone(), CurveGraphicsItem::Stabilization);
  }
}

void PostProdStitcherController::applyCalibration(VideoStitch::Core::PanoDefinition* panoDef) {
  project->setPano(panoDef);
  if (resetPano().ok()) {
    emit calibrationApplied();
  }
}

void PostProdStitcherController::applyBlendingMask(VideoStitch::Core::PanoDefinition* panoDef) {
  project->setPano(panoDef);
  if (resetPano().ok()) {
    emit blendingMaskApplied();
  }
}

void PostProdStitcherController::applyAdvancedBlending(VideoStitch::Core::PanoDefinition* panoDef) {
  project->setPano(panoDef);
  if (resetPano().ok()) {
    emit advancedBlendingApplied();
  }
}

void PostProdStitcherController::applyCrops(const QVector<Crop>& crops, const InputLensClass::LensType lensType) {
  StitcherController::applyCrops(crops, lensType);
  reset();
}
// -------------------------- Apply an external calibration ---------------------

void PostProdStitcherController::importCalibration(const QString& templateFile) {
  VideoStitch::Core::PanoDefinition* oldPanoDef = project->getPanoConst()->clone();

  QString errorString;
  VideoStitch::Potential<VideoStitch::Core::PanoDefinition> templatePanoDef =
      StitcherController::createAPanoTemplateFromCalibration(templateFile, errorString);
  if (!templatePanoDef.ok()) {
    forwardStitcherError(templatePanoDef.status(), false);
    return;
  }

  ExternCalibrationAppliedCommand* command =
      new ExternCalibrationAppliedCommand(oldPanoDef, templatePanoDef.release(), project);
  PostProdStitcherController::connect(command, SIGNAL(reqApplyCalibration(VideoStitch::Core::PanoDefinition*)),
                                      static_cast<PostProdStitcherController*>(this),
                                      SLOT(applyExternalCalibration(VideoStitch::Core::PanoDefinition*)));
  qApp->findChild<QUndoStack*>()->push(command);
}

void PostProdStitcherController::applyExternalCalibration(VideoStitch::Core::PanoDefinition* panoDef) {
  project->setPano(panoDef);
  if (resetPano().ok()) {
    MsgBoxHandler::getInstance()->generic(PostProdStitcherController::tr("Calibration template applied successfully"),
                                          PostProdStitcherController::tr("Import calibration"), INFORMATION_ICON);
    emit calibrationApplied();
  }
}

void PostProdStitcherController::importTemplate(const QString& templateFile) {
  std::unique_ptr<VideoStitch::Core::PanoDefinition> oldPanoDef(project->getPanoConst()->clone());

  VideoStitch::Potential<VideoStitch::Core::PanoDefinition> panoDefinitionNew =
      StitcherController::createAPanoTemplateFromProject(templateFile);
  if (!panoDefinitionNew.ok()) {
    forwardStitcherError(panoDefinitionNew.status(), false);
    return;
  }

  ExternCalibrationAppliedCommand* command =
      new ExternCalibrationAppliedCommand(oldPanoDef.release(), panoDefinitionNew.release(), project);
  PostProdStitcherController::connect(command, SIGNAL(reqApplyCalibration(VideoStitch::Core::PanoDefinition*)),
                                      static_cast<PostProdStitcherController*>(this),
                                      SLOT(applyTemplate(VideoStitch::Core::PanoDefinition*)));
  qApp->findChild<QUndoStack*>()->push(command);
}

void PostProdStitcherController::applyTemplate(VideoStitch::Core::PanoDefinition* panoDef) {
  project->setPano(panoDef);
  if (!resetPano().ok()) {
    return;
  }

  MsgBoxHandler::getInstance()->generic(PostProdStitcherController::tr("Calibration template applied successfully"),
                                        PostProdStitcherController::tr("Import calibration"), INFORMATION_ICON);
  emit calibrationApplied();
}

// ------------------------------------ Projection handling --------------------------------------

bool PostProdStitcherController::allowsYPRModifications() const {
  return currentProjection == VideoStitch::Projection::equirectangular && currentHFovs[currentProjection] == 360.0 &&
         project->getPanoConst()->getWidth() == 2 * project->getPanoConst()->getHeight();
}

void PostProdStitcherController::ensureProjectionIsValid() {
  if (currentProjection == VideoStitch::interactive) {
    setProjection(previousProjection, currentHFovs[previousProjection]);
  }
}

void PostProdStitcherController::setProjection(VideoStitch::Projection proj, double fov) {
  previousProjection = currentProjection;
  currentProjection = proj;
  currentHFovs[currentProjection] = fov;
  QString projectionName;
  switch (proj) {
    case VideoStitch::Projection::rectilinear:
      projectionName = "rectilinear";
      break;
    case VideoStitch::Projection::fullframe_fisheye:
      projectionName = "ff_fisheye";
      break;
    case VideoStitch::Projection::circular_fisheye:
      projectionName = "circular_fisheye";
      break;
    case VideoStitch::Projection::stereographic:
      projectionName = "stereographic";
      break;
    case VideoStitch::Projection::equirectangular:
    case VideoStitch::Projection::interactive:
      projectionName = "equirectangular";
      break;
    case VideoStitch::Projection::cubemap:
      projectionName = "cubemap";
      break;
    case VideoStitch::Projection::equiangular_cubemap:
      projectionName = "equiangular_cubemap";
      break;
    default:
      Q_ASSERT(0);
  }
  project->changeProjection(projectionName, fov);
  emit projectOrientable(allowsYPRModifications());
  emit reqChangeProjection(proj, fov);
}

// -------------------------- Snapshots -------------------------------

void PostProdStitcherController::snapshotSources(const QString& directory, bool forCalibration) {
  QStringList snapshots = onSnapshotSources(directory);
  if (forCalibration) {
    emit snapshotsDone(snapshots);
  }
}
// -------------------------- Orientation ------------------------------------

void PostProdStitcherController::selectOrientation() {
  int frame = controller->getCurrentFrame();
  std::unique_ptr<VideoStitch::Core::QuaternionCurve> orientationCurve(
      project->getPano()->getGlobalOrientation().clone());
  VideoStitch::Quaternion<double> currO = orientationCurve->at(frame);

  // get the interactive rotation from a stitcher
  VideoStitch::Quaternion<double> interOrient = stitcher->getRotation();

  // get the operator's rotation from the project
  // now chain the two rotations to get to the final orientation
  VideoStitch::Quaternion<double> newOrientation = currO * interOrient;

  OrientationChangedCommand* command =
      new OrientationChangedCommand(frame, currO, newOrientation, orientationCurve.release());
  PostProdStitcherController::connect(command, &OrientationChangedCommand::reqFinishOrientation,
                                      static_cast<PostProdStitcherController*>(this),
                                      &PostProdStitcherController::finishOrientation);
  qApp->findChild<QUndoStack*>()->push(command);
}

void PostProdStitcherController::finishOrientation(int frame, VideoStitch::Quaternion<double> orientation,
                                                   VideoStitch::Core::QuaternionCurve* curve) {
  // reset orientation on the stitcher
  stitcher->resetOrientation();
  // then update the curves and notify the rest of the application
  std::unique_ptr<VideoStitch::Core::QuaternionCurve> oc(
      new VideoStitch::Core::QuaternionCurve(VideoStitch::Core::SphericalSpline::point(frame, orientation)));
  oc->extend(curve);
  std::vector<std::pair<VideoStitch::Core::Curve*, CurveGraphicsItem::Type> > dummy;
  std::vector<std::pair<VideoStitch::Core::QuaternionCurve*, CurveGraphicsItem::Type> > curves;
  curves.push_back(std::make_pair(oc->clone(), CurveGraphicsItem::GlobalOrientation));

  emit reqUpdateQuaternionCurve(oc->clone(), CurveGraphicsItem::GlobalOrientation);
  project->curvesChanged(nullptr, dummy, curves);
}

void PostProdStitcherController::configureRig(const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                                              const VideoStitch::Core::StereoRigDefinition::Geometry geometry,
                                              const double diameter, const double ipd, const QVector<int> leftInputs,
                                              const QVector<int> rightInputs) {
  project->setRigConfiguration(orientation, geometry, diameter, ipd, leftInputs, rightInputs);
  reset();
}

VideoStitch::Input::ReaderFactory* PostProdStitcherController::createReaderFactory() const {
  return new VideoStitch::Input::DefaultReaderFactory(0, NO_LAST_FRAME);
}

void PostProdStitcherController::clearCalibration() {
  StitcherController::clearCalibration();
  reset();
}

void PostProdStitcherController::createProject() {
  if (project) {
    delete project;
    project = nullptr;
  }
  project = new PostProdProjectDefinition();
  connect(project, SIGNAL(reqToggleInputNumbers(bool)), this, SLOT(toggleInputNumbers(bool)));
  connect(project, SIGNAL(reqDisplayOrientationGrid(bool)), this, SLOT(setDisplayGrid(bool)));
  connect(project, SIGNAL(reqReset(SignalCompressionCaps*)), this, SLOT(onReset(SignalCompressionCaps*)));
  connect(project, SIGNAL(reqReextract(SignalCompressionCaps*)), this, SLOT(reextract(SignalCompressionCaps*)));
  connect(project, SIGNAL(reqResetRig(SignalCompressionCaps*)), this, SLOT(onResetRig(SignalCompressionCaps*)));
  connect(project, SIGNAL(reqMissingInputs(QString&)), this, SIGNAL(reqCheckInputsDialog(QString&)));
  connect(project, SIGNAL(reqWarnWrongInputSize(uint, uint, uint, uint)), this,
          SIGNAL(reqWarnWrongInputSize(uint, uint, uint, uint)));
}
