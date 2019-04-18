// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "postprodmutableprojectdefinition.hpp"
#include "widgets/timeline/curvegraphicsitem.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include "libvideostitch-base/lockingproxy.hpp"
#include "libvideostitch/audio.hpp"
#include <QMutex>

class PostProdProjectDefinition;
typedef VideoStitch::Helper::LockingProxy<VideoStitch::Ptv::Value, const PostProdProjectDefinition> PtvValueLocked;

class PostProdProjectDefinition : public ProjectDefinition {
  Q_OBJECT

 public:
  PostProdProjectDefinition();
  virtual ~PostProdProjectDefinition();

  /**
   * @brief Serialize a GUI project definition.
   * @return The serialized PTV object.
   */
  VideoStitch::Ptv::Value* serialize() const;

  /**
   * @brief Returns the first frame for the project file.
   * @return The first frame.
   */
  frameid_t getFirstFrame() const;

  /**
   * @brief Returns the last frame for the project file. When -1 (auto), the value is replaced by the auto-detected
   * value.
   * @return 0 when no project is loaded (check with isInit()), the last frame otherwise.
   */
  frameid_t getLastFrame() const;

  /**
   * @brief Tells whether getLastFrame() returned an inferred value (i.e. last_frame PTV value was -1).
   * @return True if last frame equals to -1.
   */
  bool isLastFrameAuto() const;

  /**
   * @brief Returns the output configuration for the project.
   * @return Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  PtvValueLocked getOutputConfig() const;

  /**
   * @brief Get the list of input offsets
   * @return Frame offset of every input
   */
  std::vector<int> getOffsets() const;

  /**
   * @brief Get the panorama (uncropped) dimensions.
   * @param[out] width The panorama width
   * @param[out] height The panorama height
   */
  void getImageSize(unsigned& width, unsigned& height) const;

  /**
   * @brief Get the crop values in pixel.
   * @param[out] top The top crop value
   * @param[out] bottom The bottom crop value
   * @param[out] left The left crop value
   * @param[out] right The right crop value
   */
  void getCropValues(unsigned& top, unsigned& bottom, unsigned& left, unsigned& right) const;

  /**
   * @brief Get the optimal resolution for the panorama given the resolution of the inputs and the setup.
   * @note The idea is to minimize the distortions at the center of the panorama.
   * @param[out] width The optimal width.
   * @param[out] height The optimal height.
   */
  void getOptimalSize(unsigned& width, unsigned& height) const;

  /**
   * @brief Get the format name from the output configuration.
   * @return The output video format
   */
  QString getOutputVideoFormat() const;

  /**
   * @brief Get the current video codec name
   * @return The output video codec
   */
  QString getOutputVideoCodec() const;

  /**
   * @brief Get the name of the output file from the output configuration of the project.
   * @return The output file name
   */
  QString getOutputFilename() const;

  /**
   * @brief Checks if the output must be processed only in the selected sequence.
   * @return True if the sequence has to be processed.
   */
  bool getProcessSequence() const;

  /**
   * @brief Indicates that the sequence has to be processed instead of all video.
   * @param processSequence True for process the selected sequence.
   */
  void setProcessSequence(const bool processSequence);

  /**
   * @brief Checks if the output has a valid audio configuration
   * @return True if it has a valid audio configuration.
   */
  bool hasAudioConfiguration() const;

  /**
   * @brief Returns the output audio bitrate
   * @return The audio bitrate in Kbp/s if exists. Otherwise, 0
   */
  int getOutputAudioBitrate() const;

  /**
   * @brief Returns the output audio codec name
   * @return The audio codec if any. Otherwise, and empty string.
   */
  const QString getOutputAudioCodec() const;

  /**
   * @brief Returns the output audio sampling format
   * @return The audio sampling format if any. Otherwise, SD_NONE.
   */
  VideoStitch::Audio::SamplingDepth getOutputAudioSamplingFormat() const;

  /**
   * @brief Gets the sampling rate of the audio pipeline.
   * @return The sampling rate in Hz if any.
   */
  int getOutputAudioSamplingRate() const;

  /**
   * @brief Gets the channel layout configuration.
   * @return The channel layout.
   */
  QString getOutputAudioChannels() const;

  /**
   * @brief Loads the project with the a default project ptv passed as agrument
   *        Not all values are set by the ptv, however, these values will be set by the stitcher when loading the
   * project.
   * @param value Default project. If it is loaded from default_project.ptv it must be fed with an input (there are no
   * inputs in the default template).
   * @return True if the operation was successfull.
   */
  bool setDefaultValues(QList<VideoStitch::Ptv::Value*> userInputs);

  /**
   * @brief Add inputs to the project
   * @param inputs A list of input definitions
   */
  void addInputs(QList<VideoStitch::Ptv::Value*> inputs);

  /**
   * @brief Returns the video bitrate value (if any)
   * @return Bitrate value or 0 if does not exists.
   */
  int getOutputVideoBitrate() const;

  /**
   * @brief setOutputConfig writes the @output configuration in the project. Thread-safe.
   * @param The new video configuration<
   */
  void setOutputVideoConfig(VideoStitch::Ptv::Value* oldConfig, VideoStitch::Ptv::Value* newConfig);

  /**
   * @brief Sets the first frame for the project file. Thread-safe.
   * @param The first frame.
   */
  void setFirstFrame(const frameid_t frame);

  /**
   * @brief Sets the last frame for the project file. Thread-safe.
   * @param The last frame.
   */
  void setLastFrame(const frameid_t frame);

  /**
   * @brief Sets the audio output encoding configuration
   * @param codec Audio codec
   * @param bitrate Bitrate in bits/s
   */
  void setOutputAudioConfig(const QString codec, const int bitrate, const QString input);

  /**
   * @brief Sets the output file name
   * @param filename File name (only accepts standard separators)
   */
  void setOutputFilename(const QString filename);

  /**
   * @brief Sets the panorama size
   * @param width Panorama width in pixels
   * @param height Panoramana height in pixels
   */
  void setPanoramaSize(unsigned width, unsigned height);

  /**
   * @brief Removes all the codec configurations from the output
   */
  void removeAudioSource();

  /**
   * @brief Sets the file is going to be splitted in size parts
   * @param size The number of parts.
   */
  void addOutputFileChunkSize(const int size);

  /**
   * @brief Removes the configuration related with the file split
   */
  void removeOutputFileChunkSize();

  /**
   * @brief fixInputPaths normalize paths according to the current OS
   *        Keeps audio_pipe synchronized, rereating it, and saving selected
   *        audio input
   */
  void fixInputPaths();

  void fixMissingInputs(QString& newFolder);
 public slots:
  /**
   * @brief Sets the timeline range
   * @param firstStitchableFrame First frame
   * @param lastStitchableFrame Last frame
   */
  void checkRange(const frameid_t firstStitchableFrame, const frameid_t lastStitchableFrame);

  /**
   * @brief Resets the EV curves
   * @param resetController Reset the stitcher controller if True
   */
  void resetEvCurves(bool resetController = false);

  /**
   * @brief Resets the EV curves in a given range
   * @param startPoint initial frame
   * @param endPoint ending frame
   */
  void resetEvCurvesSequence(const frameid_t startPoint, const frameid_t endPoint);

  /**
   * @brief Resets all the curves: ev, blueCB, redCB, greenCB, orientation and stabilization
   * @param resetController Reset the stitcher controller if True
   */
  void resetCurves(bool resetController = true);

  /**
   * @brief Resets the orientation curve.
   * @param resetController Reset the stitcher controller if True
   */
  void resetOrientationCurve(bool resetController = true);

  /**
   * @brief Resets the stabilization curve.
   */
  void resetStabilizationCurve();

  /**
   * @brief Resets the photometric callibration.
   */
  void resetPhotometricCalibration();

  /**
   * @brief Resets the curve of global YPR and EV.
   */
  void resetCurve(CurveGraphicsItem::Type type, int inputId = -1);

  /**
   * @brief Updates a curve value
   * @param comp Signals emiter
   * @param curve The curve to update
   * @param type Curve type
   * @param inputId Input related to the curve
   */
  void curveChanged(SignalCompressionCaps* comp, VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type,
                    int inputId);

  /**
   * @brief Updates a quaternion curve value
   * @param comp Signals emiter
   * @param curve The curve to update
   * @param type Curve type
   * @param inputId Input related to the curve
   */
  void quaternionCurveChanged(SignalCompressionCaps* comp, VideoStitch::Core::QuaternionCurve* curve,
                              CurveGraphicsItem::Type type, int inputId);

  /**
   * @brief Updates a list of curves
   * @param comp Signals emiter
   * @param curves The curves list to update
   * @param quaternionCurves The quaternion curves list to update
   */
  void curvesChanged(
      SignalCompressionCaps* comp, std::vector<std::pair<VideoStitch::Core::Curve*, CurveGraphicsItem::Type> > curves,
      std::vector<std::pair<VideoStitch::Core::QuaternionCurve*, CurveGraphicsItem::Type> > quaternionCurves);

 signals:
  // the working zone range
  void reqSetWorkingArea(frameid_t firstFrame, frameid_t lastFrame);
  void reqUpdateCurve(VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type, int inputId = -1);
  void reqUpdateQuaternionCurve(VideoStitch::Core::QuaternionCurve* curve, CurveGraphicsItem::Type type,
                                int inputId = -1);
  void reqRefreshCurves();
  void reqRefreshPhotometry();
  void reqMissingInputs(QString& newFolder);
  void reqWarnWrongInputSize(unsigned widthIs, unsigned heightIs, unsigned widthShouldBe, unsigned heightShouldbe);

 private:
  void changeCurve(VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type, int inputId);
  void changeQuaternionCurve(VideoStitch::Core::QuaternionCurve* curve, CurveGraphicsItem::Type type, int inputId);
  void applyCurvesChanges();
  void setOutputConfig(VideoStitch::Ptv::Value* outputConfig);
  void updateAudioPipe(QString& oldSelectedAudio);
  virtual PostProdMutableProjectDefinition* getDelegate() const;
  virtual void createDelegate(const VideoStitch::Ptv::Value& value);
  virtual void destroyDelegate();
  std::unique_ptr<VideoStitch::Ptv::Value> defaultFromInput(VideoStitch::Ptv::Value* input) const;
  PostProdMutableProjectDefinition* delegate;
  friend class VideoStitch::Helper::LockingProxy<VideoStitch::Ptv::Value,
                                                 const PostProdProjectDefinition>;  // PtvValueLocked;
  // takes ownership of curveToReset
  VideoStitch::Core::Curve* resetEvCurve(frameid_t startPoint, frameid_t endPoint, double value,
                                         const VideoStitch::Core::Curve* curveToReset);
};
