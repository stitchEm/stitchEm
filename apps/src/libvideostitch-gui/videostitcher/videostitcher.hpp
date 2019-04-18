// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "projectdefinition.hpp"

#include "libvideostitch-base/lockingproxy.hpp"
#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <atomic>
#include <sstream>
#include <QReadWriteLock>
#include <QThread>
#include <QStringList>

class SignalCompressionCaps;

struct ActivableAlgorithmOutput {
  VideoStitch::Core::AlgorithmOutput* algoOutput;
  std::atomic_flag toggle;
};

/**
 * @brief The VideoStitcher class is a wrapper used to communicate with libvideostitch.
 * There's one VideoStitcher per GPU, with a global, shared StitchOutput to register
 * user callbacks.
 */
class VS_GUI_EXPORT VideoStitcherSignalSlots : public QObject {
  Q_OBJECT;

 public:
  explicit VideoStitcherSignalSlots(QObject* parent) : QObject(parent) {}

 public slots:
  virtual void closeProject() = 0;
  virtual VideoStitch::Status init() = 0;
  virtual void resetMerger() = 0;
  virtual void resetAdvancedBlending() = 0;

  virtual void stitch() = 0;
  virtual void extract() = 0;
  virtual void stitchAndExtract() = 0;

  virtual void restitch() = 0;
  virtual void reextract() = 0;
  virtual void restitchAndExtract() = 0;

  virtual VideoStitch::Quaternion<double> getRotation() const = 0;
  virtual void rotatePanorama(double yaw, double pitch, double roll) = 0;
  virtual void resetOrientation() = 0;
  virtual VideoStitch::Quaternion<double> getCurrentOrientation() const = 0;
  virtual void onSnapshotPanorama(const QString& filename) = 0;
  virtual void onSnapshotSources(std::vector<VideoStitch::Ptv::Value*> outputConfigs) = 0;

 signals:
  void notifyErrorMessage(const VideoStitch::Core::ControllerStatus status, bool needToExit) const;
  void snapshotPanoramaExported();
};

template <typename Controller>
class VS_GUI_EXPORT VideoStitcher : public VideoStitcherSignalSlots {
 public:
  /**
   * @brief Constructor
   * @param cudaDeviceHandler A CUDA handler
   */
  VideoStitcher(QObject*, typename Controller::DeviceDefinition&, ProjectDefinition& project, Controller& controller,
                typename Controller::Output& stitchOutput,
                std::vector<VideoStitch::Core::ExtractOutput*>& extractOutputs, ActivableAlgorithmOutput& algoOutput,
                QReadWriteLock& setupLock);

  /**
   * @brief Virtual destructor
   */
  virtual ~VideoStitcher();

  void closeProject();

  VideoStitch::Status init();

  // ---------------------------- Setup ---------------------------------------

  /**
   * @brief Redo the setup of all mergers
   */
  void resetMerger();

  /**
   * @brief Redo the setup of all advanced blending
   */
  void resetAdvancedBlending();

  // -------------------------- Stitching -------------------------------------

  /**
   * @brief Stitch a new frame to a buffer of pixels.
   */
  void stitch();

  /**
   * @brief Extract the input frames.
   */
  void extract();

  /**
   * @brief Stitch a new frame to a buffer of pixels. Extract the input frames along.
   */
  void stitchAndExtract();

  /**
   * @brief Tries to stitch the same frame to a buffer of pixels, it it fails, it reads the frame and stitches it..
   */
  void restitch();

  /**
   * @brief Re-extracts the inputs
   */
  void reextract();

  /**
   * @brief Tries to stitch the same frame to a buffer of pixels, it it fails, it reads the frame and stitches it..
   */
  void restitchAndExtract();

  // -------------------------- Orientation ------------------------------------

  /**
   * @brief Get the current interactive orientation of the panorama.
   */
  VideoStitch::Quaternion<double> getRotation() const;

  /**
   * @brief Rotates the panorama given yaw, pitch, roll rotations
   */
  void rotatePanorama(double yaw, double pitch, double roll);

  /**
   * @brief Reset the orientation values
   */
  void resetOrientation();

  /**
   * @brief Obtain the current orientation from the stitcher
   * @return A quaternion curve with the orientation values
   */
  VideoStitch::Quaternion<double> getCurrentOrientation() const;

  // -------------------------- Snapshots -------------------------------

  /**
   * @brief Take a snapshot of the current panorama and saves the file in filename
   */
  void onSnapshotPanorama(const QString& filename);

  /**
   * @brief Take a snapshot of each input according to the config
   */
  void onSnapshotSources(std::vector<VideoStitch::Ptv::Value*> outputConfigs);

 protected:
  /**
   * @brief stitchInternal
   * @param readFrame Frame number to extract
   * @param extract Extract outputs
   * @param stitch True if we want to stitch and extract
   */
  void stitchInternal(bool readFrame, bool extract, bool stitch);

  /**
   * @brief Creates an error message from the status and sends a signal to the widgets
   * @param status The status code
   */
  void showError(const VideoStitch::Core::ControllerStatus status);

  typename Controller::DeviceDefinition cudaDeviceDefinition;
  ProjectDefinition& project;
  // Core library objects.
  Controller& controller;
  typename Controller::Output& stitchOutput;
  ActivableAlgorithmOutput& algoOutput;
  std::vector<VideoStitch::Core::ExtractOutput*> extractsOutputs;
  QReadWriteLock& setupLock;
};
