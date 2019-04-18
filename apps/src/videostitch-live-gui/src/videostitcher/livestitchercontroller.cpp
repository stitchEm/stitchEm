// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "livestitchercontroller.hpp"

#include "liveoutputfactory.hpp"
#include "liveoutputlist.hpp"

#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"
#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"
#include "libvideostitch-gui/videostitcher/videostitcher.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/controller.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/preprocessor.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <QApplication>

LiveStitcherController::LiveStitcherController(DeviceDefinition& device)
    : StitcherController(device), project(), globalFramerate({1, 1}) {}

LiveStitcherController::~LiveStitcherController() {
  closeProject();
  project.reset();
}

void LiveStitcherController::createNewProject() {
  createProject();
  emit this->projectInitialized(project.data());
}

void LiveStitcherController::resetProject() {
  project.reset();
  emit StitcherController::projectReset();
}

void LiveStitcherController::preCloseProject() {
  // This code is intended to be run only by the main Thread
  Q_ASSERT(QThread::currentThread() == QApplication::instance()->thread());

  if (!project || !project->getOutputConfigs()) {
    return;
  }
  QList<QString> outputsToRemove;
  for (LiveOutputFactory* output : project->getOutputConfigs()->getValues()) {
    if (output->earlyClosingRequired()) {
      outputsToRemove.append(output->getIdentifier());
    }
  }
  for (QString outputId : outputsToRemove) {
    disableOutput(outputId, false, true);
  }
}

void LiveStitcherController::closeProject() {
  if (project != nullptr) {
    disableActiveOutputs();
    project->clearOutputs();
    project->close();
  }
  resetControlPoints();
  StitcherController::closeProject();
}

// --------------------------------- Reconfiguration -------------------------------------

void LiveStitcherController::finishProjectOpening() {
  // Check if the audio input is loaded. If not, show a message.
  /*
VideoStitch::Core::InputDefinition* audioInput = getProjectPtr()->getPanoConst()->getAudioInput();
if (!controller->hasAudio() ) {
  VideoStitch::Ptv::Value* config = audioInput->getReaderConfig().clone();
  std::string name;
  QString translatedString = QApplication::translate("LiveStitcherController", "Unknown audio device");
  if (VideoStitch::Parse::populateString("Ptv", *config, "device", name, false) ==
VideoStitch::Parse::PopulateResult_Ok) { translatedString = QString::fromStdString(name);
  }
  const QString errorString = QApplication::translate("LiveStitcherController",
                                                      "Could not find the audio input:\n"
                                                      "%0.\n"
                                                      "If you click OK, the audio input will be removed from this
project.\n" "Click on Cancel to close the project").arg(translatedString); const QString errorTitle  =
QApplication::translate("LiveStitcherController", "Input Error"); emit notifyAudioInputNotFound(errorTitle,
errorString);
}
*/
}

void LiveStitcherController::reset() {
  std::stringstream msg;
  if (!project->validatePanorama(msg)) {
    // TODOLATERSTATUS get Status cause, not message
    VideoStitch::Status cause{VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                              msg.str()};
    VideoStitch::Status invalid{VideoStitch::Origin::PanoramaConfiguration, VideoStitch::ErrType::InvalidConfiguration,
                                StitcherController::tr("Invalid panorama parameters").toStdString(), cause};
    emit this->notifyInvalidPano();
    return;
  }
  setupLock.lockForWrite();
  const VideoStitch::Core::PanoDefinition& pano = *project->getPanoConst().get();
  if (controller->isPanoChangeCompatible(pano)) {
    VideoStitch::Status st = controller->updatePanorama(pano);
    setupLock.unlock();
    if (!st.ok()) {
      forwardStitcherError(st, false);
      return;
    }
  } else {
    setupLock.unlock();
    closeProject();
    this->openProject(ProjectFileHandler::getInstance()->getFilename());
  }
}

void LiveStitcherController::onResetRig() {
  setupLock.lockForWrite();
  VideoStitch::Status st = controller->updatePanorama(*project->getPanoConst().get());
  setupLock.unlock();
  if (!st.ok()) {
    forwardStitcherError(st, false);
  }
}

void LiveStitcherController::applyCrops(const QVector<Crop>& crops, const InputLensClass::LensType lensType) {
  StitcherController::applyCrops(crops, lensType);
  this->saveProject(ProjectFileHandler::getInstance()->getFilename());
  reset();
  emit notifyInputsCropped();
}

void LiveStitcherController::configureRig(const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                                          const VideoStitch::Core::StereoRigDefinition::Geometry geometry,
                                          const double diameter, const double ipd, const QVector<int> leftInputs,
                                          const QVector<int> rightInputs) {
  project->setRigConfiguration(orientation, geometry, diameter, ipd, leftInputs, rightInputs);
  this->saveProject(ProjectFileHandler::getInstance()->getFilename());
  reset();
  emit notifyRigConfigureSuccess();
}

void LiveStitcherController::updateAudioProcessor(LiveAudioProcessFactory* liveProcessor) {
  Q_ASSERT(liveProcessor != nullptr);
  if (!project->updateAudioProcessor(liveProcessor)) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        LiveStitcherController::tr("Could not update audio processor."));
  }
  project->setModified();
  this->saveProject(ProjectFileHandler::getInstance()->getFilename());
  reset();
  if (!resetAudioPipe()) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        LiveStitcherController::tr("Could not reset audio pipe."));
  }
}

void LiveStitcherController::removeAudioProcessor(const QString name) {
  if (!project->removeAudioProcessor(name)) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        LiveStitcherController::tr("Could not remove audio processor."));
  }
  project->setModified();
  this->saveProject(ProjectFileHandler::getInstance()->getFilename());
  emit this->projectInitialized(project.data());
}

void LiveStitcherController::setAudioProcessorConfiguration(LiveAudioProcessFactory* liveProcessor) {
  Q_ASSERT(liveProcessor != nullptr);
  if (!project->setAudioProcessorConfiguration(liveProcessor)) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        LiveStitcherController::tr("Could not modify audio processor."));
  }
}

bool LiveStitcherController::resetAudioPipe() {
  return controller->applyAudioProcessorParam(*project->getAudioPipeConst().get()).ok();
}

// ------------------------------ IO Management ------------------------------------------

void LiveStitcherController::configureInputsWith(const LiveInputList inputs) {
  if (project->updateInputs(inputs, project->getAudioConfiguration())) {
    if (project->isInit()) {
      if (updateProjectAfterConfigurationChanged()) {
        VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
            LiveStitcherController::tr("Inputs added to project"));
        emit notifyInputsConfigurationSuccess(LiveStitcherController::tr("Inputs edited successfully"));
      } else {
        VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
            LiveStitcherController::tr("Failed to apply the configuration to the inputs"));
      }
    } else {
      this->open();
      this->saveProject(ProjectFileHandler::getInstance()->getFilename());
    }
  } else {
    QString message = LiveStitcherController::tr("Error while updating the inputs. Preset file not found");
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(message);
    emit notifyInputsConfigurationError(message);
  }
}

void LiveStitcherController::configureAudioInput(AudioConfiguration audioConfiguration) {
  Q_ASSERT(project && project->isInit());
  project->updateAudioInput(audioConfiguration);
  if (updateProjectAfterConfigurationChanged()) {
    const QString message = LiveStitcherController::tr("Audio input configured successfully");
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(message);
    emit notifyInputsConfigurationSuccess(message);
  } else {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        LiveStitcherController::tr("Failed to configure the audio input"));
  }
}

void LiveStitcherController::insertOutput(LiveOutputFactory* output) {
  Q_ASSERT(output != nullptr);
  if (project->addOutput(output)) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        LiveStitcherController::tr("Output successfully added: new output ") + output->getIdentifier());
    this->saveProject(ProjectFileHandler::getInstance()->getFilename());
    emit this->projectInitialized(project.data());
  } else {
    delete output;
  }
}

void LiveStitcherController::removeOutput(const QString& id) {
  project->deleteOutput(id);
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
      LiveStitcherController::tr("The output was successfully removed: ") + id);
  this->saveProject(ProjectFileHandler::getInstance()->getFilename());
  emit this->projectInitialized(project.data());
}

bool LiveStitcherController::disableOutput(const QString& id, bool notifyUI, bool wait) {
  auto output = project->getOutputById(id);
  if (!output) {
    return false;
  }
  if (output->getOutputState() == LiveOutputFactory::OutputState::DISABLED) {
    return true;
  }

  output->setOutputState(LiveOutputFactory::OutputState::DISABLED);

  LiveWriterFactory* factory = dynamic_cast<LiveWriterFactory*>(output);
  if (factory != nullptr) {
    stitchOutput->removeWriter(factory->getIdentifier().toStdString());
    setupLock.lockForWrite();
    controller->removeAudioOutput(factory->getIdentifier().toStdString());
    setupLock.unlock();
    factory->destroyWriter();
  }

  LiveRendererFactory* rendererFactory = dynamic_cast<LiveRendererFactory*>(output);
  if (rendererFactory != nullptr) {
    stitchOutput->removeRenderer(rendererFactory->getIdentifier().toStdString());
    rendererFactory->destroyRenderer(wait);
  }

  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QWidget::tr("Output deactivated: ") + id);
  if (notifyUI) {
    emit notifyOutputRemoved(id);
  }

  return true;
}

bool LiveStitcherController::updateOutput(const QString& id) {
  auto output = project->getOutputById(id);
  if (!output) {
    return false;
  }
  if (output->getOutputState() != LiveOutputFactory::OutputState::ENABLED) {
    return false;
  }

  LiveWriterFactory* factory = dynamic_cast<LiveWriterFactory*>(output);
  if (factory != nullptr) {
    stitchOutput->updateWriter(factory->getIdentifier().toStdString(), *output->serialize());
  }

  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QWidget::tr("Output updated: ") + id);

  return true;
}

void LiveStitcherController::disableActiveOutputs(bool notifyUI) {
  if (!project || !project->getOutputConfigs()) {
    return;
  }

  for (const auto& output : project->getOutputConfigs()->getValues()) {
    if (output->getOutputState() != LiveOutputFactory::OutputState::DISABLED) {
      disableOutput(output->getIdentifier(), notifyUI);
    }
  }
}

namespace {
template <typename Writer>
VideoStitch::Potential<Writer> make_writer(LiveWriterFactory* output, LiveProjectDefinition* project,
                                           VideoStitch::FrameRate framerate);
template <>
VideoStitch::Potential<VideoStitch::Output::VideoWriter> make_writer(LiveWriterFactory* output,
                                                                     LiveProjectDefinition* project,
                                                                     VideoStitch::FrameRate framerate) {
  auto pot = output->createWriter(project, framerate);
  if (pot.ok()) {
    return VideoStitch::Potential<VideoStitch::Output::VideoWriter>(pot.release()->getVideoWriter());
  } else {
    return VideoStitch::Potential<VideoStitch::Output::VideoWriter>(pot.status());
  }
}
}  // namespace

void LiveStitcherController::activateOutputAsync(const QString outputSelected) {
  emit notifyOutputActivated(outputSelected);
  LiveOutputFactory* output = project->getOutputById(outputSelected);
  if (output != nullptr) {
    LiveWriterFactory* factory = dynamic_cast<LiveWriterFactory*>(output);
    if (factory != nullptr) {
      VideoStitch::Potential<Controller::Output::Writer> writer =
          make_writer<Controller::Output::Writer>(factory, project.data(), controller->getFrameRate());
      if (asyncActivation.isCanceled()) {
        stitchOutput->removeWriter(writer->getName());
        return;
      }
      if (writer.ok()) {
        std::shared_ptr<Controller::Output::Writer> sharedWriter(writer.release());
        factory->setOutputState(LiveOutputFactory::OutputState::ENABLED);
        connectWriterEvents(*sharedWriter, outputSelected);
        stitchOutput->addWriter(sharedWriter);
        emit notifyOutputWriterCreated(outputSelected);
        if (sharedWriter->getAudioWriter()) {
          setupLock.lockForWrite();
          controller->addAudioOutput(std::dynamic_pointer_cast<VideoStitch::Output::AudioWriter>(sharedWriter));
          setupLock.unlock();
        }
      } else {
        disableOutput(output->getIdentifier(), true);
        VideoStitch::Status error{VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
                                  "Unable to activate output", writer.status()};
        emit StitcherController::notifyOutputError();
        emit StitcherController::notifyErrorMessage(error, false);
      }
    }

    LiveRendererFactory* rendererFactory = dynamic_cast<LiveRendererFactory*>(output);
    if (rendererFactory != nullptr) {
      VideoStitch::PotentialValue<std::shared_ptr<VideoStitch::Core::PanoRenderer>> renderer =
          rendererFactory->createRenderer();
      if (renderer.ok()) {
        output->setOutputState(LiveOutputFactory::OutputState::ENABLED);
        stitchOutput->addRenderer(renderer.value());
        emit notifyOutputWriterCreated(outputSelected);
      } else {
        disableOutput(output->getIdentifier(), true);
        VideoStitch::Status error{VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
                                  "Unable to activate output", renderer.status()};
        emit StitcherController::notifyOutputError();
        emit StitcherController::notifyErrorMessage(error, false);
      }
    }
  }
}

void LiveStitcherController::toggleOutputActivation(const QString& id) {
  LiveOutputFactory* output = project->getOutputById(id);
  // DISABLED ->Try to connect
  if (output->getOutputState() == LiveOutputFactory::OutputState::DISABLED) {
    output->setOutputState(LiveOutputFactory::OutputState::INITIALIZATION);
    asyncActivation = QtConcurrent::run(std::bind(&LiveStitcherController::activateOutputAsync, this, id));
    emit notifyOutputTrying();
    // INITIALIZATION -> Disable it and cancel activation
  } else if (output->getOutputState() == LiveOutputFactory::OutputState::INITIALIZATION) {
    asyncActivation.cancel();
    QtConcurrent::run(std::bind(&LiveStitcherController::disableOutput, this, id, true, false));
    emit notifyOutputActivationCancelled(id);
    // ENABLED -> Disable it
  } else {
    QtConcurrent::run(std::bind(&LiveStitcherController::disableOutput, this, id, true, false));
  }
}

// ----------------------- Online algorithms ----------------------------------

void LiveStitcherController::compensateExposure(Callback callback) {
  // just to be sure
  algoOutput.toggle.test_and_set();
  delete algoOutput.algoOutput;
  exposureCallback = callback;
  algoOutput.algoOutput = new VideoStitch::Core::AlgorithmOutput(callback.callback, *(project->getPanoConst().get()),
                                                                 *callback.listener, callback.context);
  algoOutput.toggle.clear();
}

void LiveStitcherController::clearExposure() {
  project->getPano()->resetExposure();
  this->saveProject(ProjectFileHandler::getInstance()->getFilename());
  reset();
}

void LiveStitcherController::calibrate(Callback callback) {
  // just to be sure
  algoOutput.toggle.test_and_set();
  delete algoOutput.algoOutput;
  calibrationCallback = callback;
  algoOutput.algoOutput = new VideoStitch::Core::AlgorithmOutput(callback.callback, *(project->getPanoConst().get()),
                                                                 *callback.listener, callback.context);
  algoOutput.toggle.clear();
}

void LiveStitcherController::clearCalibration() {
  StitcherController::clearCalibration();
  resetControlPoints();
  this->saveProject(ProjectFileHandler::getInstance()->getFilename());
  reset();
}

void LiveStitcherController::onCalibrationAdaptationProcess(Callback callback) {
  // just to be sure
  algoOutput.toggle.test_and_set();
  delete algoOutput.algoOutput;
  calibrationCallback = callback;
  algoOutput.algoOutput = new VideoStitch::Core::AlgorithmOutput(callback.callback, *(project->getPanoConst().get()),
                                                                 *callback.listener, callback.context);
  algoOutput.toggle.clear();
}

bool LiveStitcherController::activeOutputs() const {
  if (project->isInit()) {
    return project->areActiveOutputs();
  } else {
    return false;
  }
}

void LiveStitcherController::testInputs(const int id, const QString filename) {
  std::unique_ptr<VideoStitch::Input::DefaultReaderFactory> factory(
      new VideoStitch::Input::DefaultReaderFactory(0, NO_LAST_FRAME));
  std::unique_ptr<VideoStitch::Ptv::Value> val(VideoStitch::Ptv::Value::emptyObject());
  val->asString() = filename.toStdString();
  const VideoStitch::Input::ProbeResult result = factory->probe(*val);
  emit notifyInputTested(id, result.valid, result.width, result.height);
}

void LiveStitcherController::onResetPanorama(VideoStitch::Core::PanoramaDefinitionUpdater* panoramaUpdater,
                                             bool saveProject) {
  // resetting the panorama is not thread-safe relative to stitching
  // lock the stitchers while we're reconfiguring
  setupLock.lockForWrite();
  VideoStitch::Status st = controller->updatePanorama(panoramaUpdater->getCloneUpdater());

  if (!st.ok()) {
    forwardStitcherError(st, false);
    setupLock.unlock();
    return;
  }

  project->setPano(controller->getPano().clone());  // Todo: consider making ownership transfers more obvious
  setupLock.unlock();
  if (saveProject) {
    this->saveProject(ProjectFileHandler::getInstance()->getFilename());
    emit this->projectInitialized(project.data());
  }
}

// -------------------------- Processors ----------------------------------------

void LiveStitcherController::toggleControlPoints(bool draw) {
  if (!project->isInit()) {
    Q_ASSERT(0);
    return;
  }
  setupLock.lockForWrite();
  for (auto inputIndex = 0; inputIndex < (int)project->getNumInputs(); ++inputIndex) {
    if (draw) {
      // skip non-video inputs
      if (!project->getPanoConst()->getInput(inputIndex).getIsVideoEnabled()) {
        continue;
      }
      VideoStitch::Ptv::Value* value = VideoStitch::Ptv::Value::emptyObject();
      value->get("type")->asString() = "controlpoints";
      VideoStitch::Potential<VideoStitch::Core::PreProcessor> p =
          VideoStitch::Core::PreProcessor::create(*value, calibrationCallback.context);
      if (!p.ok()) {
        VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
            LiveStitcherController::tr("Could not initialize the preprocessor."));
      }
      controller->setPreProcessor(inputIndex, p.release());
      delete value;
    } else {
      controller->setPreProcessor(inputIndex, nullptr);
    }
  }
  // force reading the input frames again
  controller->seekFrame(controller->getCurrentFrame());
  setupLock.unlock();
}

void LiveStitcherController::resetControlPoints() {
  if (calibrationCallback.context) {
    delete *(calibrationCallback.context);  // Delete control points
    *(calibrationCallback.context) = nullptr;
  }
  calibrationCallback.context = nullptr;
}

VideoStitch::Input::ReaderFactory* LiveStitcherController::createReaderFactory() const {
  return new VideoStitch::Input::DefaultReaderFactory(0, NO_LAST_FRAME);
}

void LiveStitcherController::createProject() {
  project.reset(new LiveProjectDefinition());
  LiveStitcherController::connect(project.data(), &LiveProjectDefinition::reqSetAudioDelay, this,
                                  &StitcherController::setAudioDelay);
}

// -------------------------- Orientation ------------------------------------

void LiveStitcherController::selectOrientation() {
  // get the interactive rotation from a stitcher
  const VideoStitch::Quaternion<double> interOrient = stitcher->getRotation();
  stitcher->resetOrientation();
  const VideoStitch::Quaternion<double> currO = project->getPano()->getGlobalOrientation().at(0);
  // now chain the two rotations to get to the final orientation
  const VideoStitch::Quaternion<double> orientation = currO * interOrient;
  // then update the constant orientation and notify the rest of the application
  VideoStitch::Core::QuaternionCurve* oc =
      new VideoStitch::Core::QuaternionCurve(VideoStitch::Core::SphericalSpline::point(0, orientation));
  project->getPano()->replaceGlobalOrientation(oc);
  project->setModified();
  this->saveProject(ProjectFileHandler::getInstance()->getFilename());
  reset();
}

// -------------------------- External Calibration ------------------------------------

void LiveStitcherController::importCalibration(const QString& templateFile) {
  QString errorString;
  VideoStitch::Potential<VideoStitch::Core::PanoDefinition> templatePanoDef =
      this->createAPanoTemplateFromCalibration(templateFile, errorString);
  if (!templatePanoDef.ok()) {
    emit this->notifyCalibrationStatus(StitcherController::tr("Could not apply calibration template"),
                                       templatePanoDef.status());
    return;
  }

  emit this->notifyCalibrationStatus(StitcherController::tr("Successfully imported calibration template"),
                                     templatePanoDef.status());
  project->setPano(templatePanoDef.release());
  updateProjectAfterConfigurationChanged();
}

void LiveStitcherController::applyTemplate(const QString& templateFile) {
  VideoStitch::Potential<VideoStitch::Core::PanoDefinition> panoDefinitionNew =
      this->createAPanoTemplateFromProject(templateFile);
  if (!panoDefinitionNew.ok()) {
    emit this->notifyCalibrationStatus(StitcherController::tr("Could not apply calibration template"),
                                       panoDefinitionNew.status());
    return;
  }

  project->setPano(panoDefinitionNew.release());
  this->saveProject(ProjectFileHandler::getInstance()->getFilename());
  reset();
  emit this->notifyCalibrationStatus(StitcherController::tr("Successfully imported calibration template"),
                                     VideoStitch::Status::OK());
  emit this->projectInitialized(project.data());
}

bool LiveStitcherController::updateProjectAfterConfigurationChanged() {
  QString fileName = ProjectFileHandler::getInstance()->getFilename();
  this->saveProject(fileName);
  closeProject();

  return this->openProject(fileName);
}

void LiveStitcherController::connectWriterEvents(Controller::Output::Writer& writer, const QString& outputSelected) {
  writer.getOutputEventManager().subscribe(
      VideoStitch::Output::OutputEventManager::EventType::Connected,
      [outputSelected, this](const std::string&) { emit notifyOutputConnected(outputSelected); });

  writer.getOutputEventManager().subscribe(
      VideoStitch::Output::OutputEventManager::EventType::Connecting,
      [outputSelected, this](const std::string&) { emit notifyOutputConnecting(outputSelected); });

  writer.getOutputEventManager().subscribe(
      VideoStitch::Output::OutputEventManager::EventType::Disconnected,
      [outputSelected, this](const std::string&) { emit notifyOutputDisconnected(outputSelected); });
}
