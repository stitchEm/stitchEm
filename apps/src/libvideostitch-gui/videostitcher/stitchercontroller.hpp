// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/videostitcher/videostitcher.hpp"
#include "libvideostitch-gui/videostitcher/backendInitializerProgressReporter.hpp"
#include "libvideostitch-gui/videostitcher/audioplayer.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <QReadWriteLock>
#include <memory>
#include <vector>

namespace VideoStitch {
namespace Input {
class ReaderFactory;
}
namespace Util {
class OnlineAlgorithm;
class OpaquePtr;
}  // namespace Util
}  // namespace VideoStitch
class ProjectDefinition;
class SignalCompressionCaps;
class StitcherControllerProgressReporter;
template <typename Controller>
class VideoStitcher;
class YPRSignalCaps;
class QOffscreenSurface;
class QOpenGLContext;

class VS_GUI_EXPORT StitcherController : public QObject {
  Q_OBJECT
 public:
  enum class NextFrameAction {
    None = 0x00,
    Extract = 0x01,
    Stitch = 0x02,
    StitchAndExtract = NextFrameAction::Extract | NextFrameAction::Stitch
  };
  typedef VideoStitch::Core::Controller::DeviceDefinition DeviceDefinition;
  typedef VideoStitch::Core::Controller Controller;

  explicit StitcherController(DeviceDefinition& device);
  virtual ~StitcherController();
  template <typename T>
  void lockedFunction(T&& f) {  // runs the function given as argument and notifies the controller (that should be
                                // waiting for it) once it is done
    std::lock_guard<std::mutex> lk(conditionMutex);
    if (closing) {
      openCondition.notify_all();
      return;
    }
    f();
    actionDone = true;
    openCondition.notify_all();
  }
  /**
   * @brief indicates the start of project closing, should be called from the main thread, and before closeProject is
   * invoked
   */
  void closingProject();
  bool isProjectOpening() const;
  void setProjectOpening(const bool b);
  bool isPlaying() const;
  void play();
  bool pause();                                                // returns true if the controller was playing
  NextFrameAction setNextFrameAction(NextFrameAction action);  // returns old frame action

 signals:
  /**
   * Agnostic "next frame" signal, connect it to one of the "next frame" slots
   */
  void reqNext(SignalCompressionCaps* comp = nullptr);
  /**
   * @brief This signal is emitted when we initialize a new project
   * This signal is the only method that allow the GUI to connect to the project or to keep a reference on it.
   */
  void projectInitialized(ProjectDefinition*);
  /**
   * @brief This signal is emitted when we reset the current project
   * After this signal, the project no longer exist.
   */
  void openFromInputFailed();
  void projectReset();
  void stitcherClosed();
  void reqCleanStitcher();
  void reqResetDimensions(
      unsigned panoWidth, unsigned panoHeight,
      const QStringList& inputNames);  // Sends a reset dimension when a out of memory error has been catched.
  void reqCheckInputs(int customWidth, int customHeight, QMap<int, QString> missingInputs);
  void snapshotPanoramaExported();

  void reqCreateThumbnails(std::vector<std::tuple<VideoStitch::Input::VideoReader::Spec, bool, std::string>> inputs,
                           std::vector<std::shared_ptr<VideoStitch::Core::SourceRenderer>>* renderers);
  void reqCreatePanoView(std::vector<std::shared_ptr<VideoStitch::Core::PanoRenderer>>* renderers);

  void statusMsg(QString msg, int timeout = 0);
  void reqDisplayWarning(QString);
  void reqDisplayError(QString);
  void reqDisableWindow();
  void reqEnableWindow();
  void notifyErrorMessage(const VideoStitch::Status status, bool needToExit) const;
  void notifyInvalidPano() const;
  void notifyOutputError() const;
  void notifyCalibrationStatus(QString message, const VideoStitch::Status status);
  void reqUpdate();
  void inputNumbersToggled(bool toggled);
  void notifyEndOfStream();
  void notifyProjectOpened();

  // Progress related signals
  void actionStarted();
  void actionStep(int progress);
  void actionFinished();
  void actionCancelled();

  void projectLoading();
  void projectLoaded();
  void cancelProjectLoading();
  void projectClosed();

  void notifyBackendCompileProgress(const QString& message, double progress);
  void notifyBackendCompileDone();
  void notifyStitcherReset();

 public slots:
  VideoStitch::FrameRate getFrameRate() const;
  bool hasInputAudio() const;
  bool hasVuMeter(const QString& audioInputId) const;
  std::vector<double> getRMSValues(const QString& audioInputId) const;
  std::vector<double> getPeakValues(const QString& audioInputId) const;

  // -------------------------- Stitching -------------------------------------
 public slots:
  /**
   * @brief Tries to stitch the same frame to a buffer of pixels, it it fails, it reads the frame and stitches it..
   * @param comp
   * @return True if success
   */
  void restitchOnce();

  /**
   * @brief Re-extracts the inputs
   * @param comp
   * @return True if success
   */
  void reextractOnce();

  /**
   * @brief Tries to stitch the same frame to a buffer of pixels, it it fails, it reads the frame and stitches it..
   * @param comp
   * @return True if success
   */
  void restitchAndExtractOnce();

  // -------------------------- Open/Save/Close/Reconfigure -----------------------

  /**
   * @brief Opens a ptv project.
   * @param PTVFile Ptv project to open.
   * @param customWidth Parameter set when the stitcher has previously failed to initialize.
   *                    The panorama width will be overloaded during the initialization and set to this one.
   * @param customHeight Parameter set when the stitcher has previously failed to initialize.
   *                     The panorama width will be overloaded during the initialization and set to this one.
   * @return True if success
   */
  bool openProject(const QString& PTVFile, int customWidth = 0, int customHeight = 0);
  /**
   * @brief closes all objects related to libvideostitch with a thread safety check.
   */
  virtual void closeProject();
  /**
   * @brief Save the current PTV in the user specified file.
   * @param outFile the file to save
   * @param thisProjectCopy if not null, thisProjectCopy will be saved, otherwise, the current PTV will be saved
   * @return True if success
   */
  bool saveProject(const QString& outFile, const VideoStitch::Ptv::Value* thisProjectCopy = nullptr);

  void onActivateAudioPlayback(const bool b);
  virtual void resetProject() = 0;

  void onReset(SignalCompressionCaps* signalCompressor);

  void onResetRig(SignalCompressionCaps* signalCompressor);

  void changePano(VideoStitch::Core::PanoDefinition* panoDef);

  /**
   * @brief Reset the control points and geometries from the current calibration
   */
  void clearCalibration();

  /**
   * @brief Applies a crop to every input
   * @param crops A vector of crop values
   * @param lensType The ProjectDefinition lens type for all the crop values.
   * @note The vector must contain the same amount of elements than the project inputs.
   */
  virtual void applyCrops(const QVector<Crop>& crops, const InputLensClass::LensType lensType);

  // -------------------------- Snapshots ----------------------------------------

  void onSnapshotPanorama(const QString& filename);
  QStringList onSnapshotSources(const QString& directory);

  // -------------------------- Orientation ----------------------------------------

  void rotatePanorama(YPRSignalCaps* rotations, bool restitch);

  // -------------------------- Sphere Scale ----------------------------------------

  void setSphereScale(const double sphereScale, bool restitch);

  // -------------------------- Processors ----------------------------------------

  /**
   * @brief Display the input numbers over the panorama
   * @param draw If true, the input number is displayed over the pano
   */
  void toggleInputNumbers(bool draw);

  void setAudioDelay(int delay);

  // -------------------------- Output Selection -------------------------------------

  virtual void switchOutput(const QString&, bool) {}

  void setAudioInput(const QString& name);

  void forwardBackendProgress(const QString& message, double progress);
  void tryCancelBackendCompile();
  /**
   * @brief Registers a new renderer for an input source. The renderer should carry a unique name.
   * @param renderer Source renderer.
   * @param inputId The input id.
   */
  void registerSourceRender(std::shared_ptr<VideoStitch::Core::SourceRenderer> renderer, const int inputId);

  /**
   * @brief Removes an existent renderer.
   * @param name Unique renderer name.
   * @param inputId The input id.
   */
  void unregisterSourceRender(const QString name, const int inputId);

 protected slots:
  bool unregisterInputExtractor(int inputId, const QString& name);
  void onProjectClosed();
  void onProjectLoaded();
  void onProjectLoading();
  void forwardStitcherError(const VideoStitch::Core::ControllerStatus status, bool needToExit);

 private slots:
  void stitchRepeat(SignalCompressionCaps* comp);
  void extractRepeat(SignalCompressionCaps* comp);
  void stitchAndExtractRepeat(SignalCompressionCaps* comp);

 protected:
  void requireNextFrame();

  /**
   * @brief Stitch a new frame to a buffer of pixels.
   * Note: only use while paused to keep a constant stitch command queue size.
   * Stitch commands should be enqueued with `reqNext` while playing.
   */
  void stitchOnce();

  /**
   * @brief Extract the input frames.
   * Note: only use while paused to keep a constant stitch command queue size.
   * Stitch commands should be enqueued with `reqNext` while playing.
   */
  void extractOnce();

  /**
   * @brief Stitch a new frame to a buffer of pixels. Extract the input frames along.
   * Note: only use while paused to keep a constant stitch command queue size.
   * Stitch commands should be enqueued with `reqNext` while playing.
   */
  void stitchAndExtractOnce();

  /**
   * @brief Get a reference on the current project.
   * To publicly get the project, connect to the signal StitcherController::projectInitialized(ProjectDefinition*)
   */
  virtual const ProjectDefinition* getProjectPtr() const = 0;

  /**
   * @brief Get a reference on the current project.
   * To publicly get the project, connect to the signal StitcherController::projectInitialized(ProjectDefinition*)
   */
  virtual ProjectDefinition* getProjectPtr() = 0;
  virtual void createProject() = 0;

  /**
   * @brief Creates the inputs, the stitcher and the output
   * @param progressReporter Object to report the progression
   * @param frame The starting frame
   * @param customWidth Panorama width
   * @param customHeight Panorama height
   * @return True if success
   */
  bool open(StitcherControllerProgressReporter* progressReporter = nullptr, int frame = 0, int customWidth = 0,
            int customHeight = 0);
  /**
   * @brief Once the project performs checks on the inputs apply the next steps
   */
  virtual bool checkProject() { return true; }

  virtual void finishProjectOpening() {}

  virtual void reset() = 0;

  /**
   * @brief used to perform pre closing operations within the main thread
   */
  virtual void preCloseProject() {}

  virtual void onResetRig() = 0;

  virtual VideoStitch::Input::ReaderFactory* createReaderFactory() const = 0;

  VideoStitch::Potential<VideoStitch::Core::PanoDefinition> createAPanoTemplateFromCalibration(
      QString calibrationFile, QString& errorString) const;
  VideoStitch::Potential<VideoStitch::Core::PanoDefinition> createAPanoTemplateFromProject(QString projectFile) const;

  void initializeStitcherLoggers();

  void deleteStitcherLoggers();

  void delayedUpdate();

  VideoStitcher<Controller>* stitcher;

  Controller::Output* stitchOutput;
  std::shared_ptr<AudioPlayer> audioPlayer;
  std::map<int, VideoStitch::Core::ExtractOutput*> extractsOutputs;
  ActivableAlgorithmOutput algoOutput;

  Controller::DeviceDefinition device;
  Controller* controller;

  // An OpenGL context for allocating the surfaces
  QOpenGLContext* interopCtx;
  QOffscreenSurface* offscreen;

  // Loggers
  std::ostream *logErrStrm, *logWarnStrm, *logInfoStrm, *logVerbStrm, *logDebugStrm;

  // guard the controller
  mutable QReadWriteLock setupLock;

  BackendInitializerProgressReporter* backendInitializerProgressReporter;

 private:
  // signal compressor shared by all possible next frame actions
  // --> changing action type discards old reqNext commands
  std::shared_ptr<SignalCompressionCaps> stitchRepeatCompressor;

  std::atomic<bool> closing;
  std::atomic<bool> actionDone;
  QStateMachine stateMachine;
  std::condition_variable openCondition;  // this condition is used for synchronization between the main Thread and the
                                          // controller thread for project opening.
  std::mutex conditionMutex;  // the mutex for openCondition
  std::atomic<bool> projectOpening;
  std::vector<int> listGpus(const std::vector<VideoStitch::Core::PanoDeviceDefinition>&);

  VideoStitch::Core::PotentialController makeController(ProjectDefinition*, VideoStitch::Input::ReaderFactory*);

  void updateAfterModifyingPrePostProcessor();
  void checkPanoDeprecatedFeatures(const VideoStitch::Ptv::Value& pano);

  mutable std::mutex playingMutex;  // Protects playing and nextFrameAction
  bool playing;
  NextFrameAction nextFrameAction;
  QState* idle;
  QState* loading;
  QState* loaded;
};
