// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "liveprojectdefinition.hpp"

#include "libvideostitch-gui/videostitcher/stitchercontroller.hpp"
#include "libvideostitch-gui/widgets/crop/cropshapeeditor.hpp"

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoramaDefinitionUpdater.hpp"

#include <QAtomicInt>
#include <QFuture>

#include <memory>

class LiveStitcherController : public StitcherController {
  Q_OBJECT
 public:
  struct Callback {
    Callback(VideoStitch::Util::OnlineAlgorithm* cb, VideoStitch::Core::AlgorithmOutput::Listener* listener,
             VideoStitch::Util::OpaquePtr** context)
        : callback(cb), listener(listener), context(context) {}
    Callback() : callback(nullptr), listener(nullptr), context(nullptr) {}

    // owned
    VideoStitch::Util::OnlineAlgorithm* callback;

    // not owned
    VideoStitch::Core::AlgorithmOutput::Listener* listener;
    VideoStitch::Util::OpaquePtr** context;
  };

  /**
   * @brief Constructor
   * @param cudaDeviceHandlers All GPUs to be used by this stitcher
   */
  explicit LiveStitcherController(DeviceDefinition& device);
  virtual ~LiveStitcherController();

  virtual void resetProject() override;

  virtual void preCloseProject() override;

  virtual void createNewProject();

  virtual void finishProjectOpening() override;

  void reset() override;

  virtual void onResetRig() override;

  virtual void importCalibration(const QString& templateFile);

  virtual void applyTemplate(const QString& templateFile);

  virtual void clearCalibration();

  /**
   * @brief Activates or deactivate an existing output from UI
   * @param id The output name
   */
  void toggleOutputActivation(const QString& id);

  /**
   * @brief Reset the current panorama and save the file
   * @param panoDef The panorama
   */
  void onResetPanorama(VideoStitch::Core::PanoramaDefinitionUpdater* panoramaUpdater, bool saveProject = true);

  /**
   * @brief Apply the orientation
   */
  void selectOrientation();

  /**
   * @brief Show or hide control points involved in calibration
   * @param draw
   */
  void toggleControlPoints(bool draw);

  /**
   * @brief Close the current project cleaning all its inputs and outputs
   */
  virtual void closeProject() override;

  /**
   * @brief Clear the existing control points
   */
  void resetControlPoints();

  /**
   * @brief Adds a new output to the output list
   * @param output A configuration of the new Output. Takes ownership
   */
  void insertOutput(LiveOutputFactory* output);

  /**
   * @brief Removes an existing output, saves the VAH and reload the project
   * @param id A unique output Id
   */
  void removeOutput(const QString& id);

  /**
   * @brief Disables an output if the output with given id exists
   * @param id A unique output Id
   * @param notifyUI if true - function emits notifyOutputRemoved
   * @param wait if true destroy linked renderer synchronously
   * @returns False if no such output exists. True if output was successfuly disabled
   */
  bool disableOutput(const QString& id, bool notifyUI = true, bool wait = false);

  /**
   * @brief Update an active output if the output with given id exists
   * @param id A unique output Id
   * @returns False if no such active output exists. True if output was successfuly updated
   */
  bool updateOutput(const QString& id);

  /**
   * @brief Disable all active outputs
   * @param notifyUI if true - emits notifyOutputRemoved for eache removed output
   */
  void disableActiveOutputs(bool notifyUI = false);

  /**
   * @brief Applyes exposure compesantion
   * @param callback Algorithm callback
   */
  void compensateExposure(Callback callback);

  /**
   * @brief Clears the existing exposure and saves the project
   */
  void clearExposure();

  /**
   * @brief Applyes calibration
   * @param callback Algorithm callback
   */
  void calibrate(Callback callback);

  /**
   * @brief Asked for processing adaptation
   */
  void onCalibrationAdaptationProcess(Callback callback);

  /**
   * @brief Check if there is at least one output active
   * @return True if is active
   */
  bool activeOutputs() const;

  /**
   * @brief Probe an input file
   * @param id An uniqued identifier (for UI)
   * @param filename The input filename or url
   */
  void testInputs(const int id, const QString filename);

  /**
   * @brief If the project doesn't exist, it is initialized with the video inputs
   * Otherwise, the existing inputs are configured and the resultant file is saved
   */
  void configureInputsWith(const LiveInputList inputs);

  /**
   * @brief Configures the audio input and save the project on disk
   */
  void configureAudioInput(AudioConfiguration audioConfiguration);

  /**
   * @brief An asynchronous method for activating outputs
   * @param outputSelected A unique output Id
   */
  void activateOutputAsync(const QString outputSelected);

  /**
   * @brief Sets the rig configuration for stereoscopic setups. The VAH file is saved and the project restarted
   * @param orientation The cameras orientation
   * @param geometry The rig geometry
   * @param diameter The rig diameter
   * @param ipd The Inter po
   * @param leftInputs A list of inputs for the left eye
   * @param rightInputs A list of inputs for the right eye
   */
  void configureRig(const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                    const VideoStitch::Core::StereoRigDefinition::Geometry geometry, const double diameter,
                    const double ipd, const QVector<int> leftInputs, const QVector<int> rightInputs);

  /**
   * @brief Adds a new audio processor and saves the project
   * @param liveProcessor The audio processor configuration
   */
  void updateAudioProcessor(LiveAudioProcessFactory* liveProcessor);

  /**
   * @brief Removes a given audio processor and saves the project
   * @param name The audio processor name
   */
  void removeAudioProcessor(const QString name);

  /**
   * @brief Sets the audio processor confirguration and saves the project
   * @param liveProcessor The audio processor configuration
   */
  void setAudioProcessorConfiguration(LiveAudioProcessFactory* liveProcessor);

  bool resetAudioPipe();

 signals:
  /**
   * @brief Signal emitted when the output activation success
   * @param id An unique identifier of the output
   */
  void notifyOutputActivated(const QString& id);

  /**
   * @brief Signal emitted when the output writer is created
   * @param id An unique identifier of the output
   */
  void notifyOutputWriterCreated(const QString& id);

  /**
   * @brief Signal emitted when an output is removed (i.e. deactivated)
   * @param id An unique identifier of the output
   */
  void notifyOutputRemoved(const QString& id);

  /**
   * @brief Signal emitted when the input configuration success
   */
  void notifyInputsConfigurationSuccess(const QString& message);

  /**
   * @brief Signal emitted when the input configuration fails
   * @param message A message explaining the failure
   */
  void notifyInputsConfigurationError(const QString& message);

  /**
   * @brief Signal emitted when waiting for output activation
   */
  void notifyOutputTrying();

  /**
   * @brief Signal emitted when we register "connected" event from writer
   * @param id An unique identifier of the output
   */
  void notifyOutputConnected(const QString& outputId);

  /**
   * @brief Signal emitted when we register "connecting" event from writer
   * @param id An unique identifier of the output
   */
  void notifyOutputConnecting(const QString& outputId);

  /**
   * @brief Signal emitted when we register "disconnected" event from writer
   * @param id An unique identifier of the output
   */
  void notifyOutputDisconnected(const QString& outputId);

  void notifyOutputActivationCancelled(const QString id);

  void notifyInputsCropped();

  void notifyRigConfigureSuccess();

  void notifyRigConfigureError();

  void notifyInputTested(const int id, const bool result, qint64 width, qint64 height);

  void notifyAudioInputNotFound(const QString title, const QString message);

 protected:
  /**
   * @brief Get a reference on the current project.
   * To publicly get the project, connect to the signal StitcherController::projectInitialized(ProjectDefinition*)
   */
  virtual const ProjectDefinition* getProjectPtr() const override { return project.data(); }

  /**
   * @brief Get a reference on the current project.
   * To publicly get the project, connect to the signal StitcherController::projectInitialized(ProjectDefinition*)
   */
  virtual ProjectDefinition* getProjectPtr() override { return project.data(); }

  virtual void createProject() override;
  virtual VideoStitch::Input::ReaderFactory* createReaderFactory() const override;
  virtual void applyCrops(const QVector<Crop>& crops, const InputLensClass::LensType lensType) override;

 private:
  bool updateProjectAfterConfigurationChanged();
  void connectWriterEvents(Controller::Output::Writer& writer, const QString& outputSelected);

  QScopedPointer<LiveProjectDefinition> project;

  QAtomicInt exposureToggle;
  QAtomicInt calibrationToggle;

  Callback exposureCallback;
  Callback calibrationCallback;
  QFuture<void> asyncActivation;
  VideoStitch::FrameRate globalFramerate;
};
