// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "computationwidget.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include "libvideostitch/allocator.hpp"
#include "itoolwidget.hpp"
#include "libvideostitch-gui/utils/inputlensenum.hpp"

#include <memory>

struct Crop;
class QListWidgetItem;
class ProjectDefinition;

namespace Ui {
class CalibrationWidget;
}

class CalibrationWidget : public ComputationWidget, public IToolWidget {
  Q_OBJECT

 public:
  explicit CalibrationWidget(QWidget* const parent = nullptr);
  ~CalibrationWidget();

  void fillRecentCalibrationMenuWith(QList<QAction*> recentCalibrationActions);
  void applyCalibration(VideoStitch::Core::PanoDefinition* panoDef);  // Takes the ownership of the panoDef

 signals:
  void reqImportCalibration();
  void reqApplyCalibration(VideoStitch::Core::PanoDefinition* panoDef = nullptr);
  void reqApplyCrops(const QVector<Crop>& crops, const InputLensClass::LensType lensType);
  void reqResetCalibration();

  /**
   * @brief Sends a signal to seek to the frame given in parameter
   * @param frameToSeekTo Frame to seek to.
   */
  void reqSeek(frameid_t frameToSeekTo);

  // Register and unregister callbacks
  void reqChangeCrop(const unsigned int index, const Crop& crop,
                     const VideoStitch::Core::InputDefinition::Format format, const bool applyToAll);
  void reqRegisterRender(std::shared_ptr<VideoStitch::Core::SourceRenderer> renderer, const int inputId);
  void reqUnregisterRender(const QString name, const int inputId);
  void reqReset();
  void reextract(SignalCompressionCaps* = nullptr);

 public slots:
  virtual void onProjectOpened(ProjectDefinition* project) override;
  virtual void updateSequence(const QString, const QString stop) override;
  virtual void clearProject() override;
  void startComputation(int calibrationOptions);
  void startCalibrationOnList();
  void deshuffleOnlyOnList();
  void deshuffleOnlyOnCurrentFrame();
  void improveCalibrationOnCurrentFrame();
  void generateSyntheticKeypointsStateChanged(int state);
  void calibrationModeChanged(bool automatic);
  void calibrationFrameSelectionModes(bool automatic);

  void refresh(mtime_t frame);

  void addCurrentFrame();
  void removeFrame();
  void clearFrames();
  void seekFrame(QListWidgetItem* item);
  void frameItemSelected();
  void updateRigValues();

 protected:
  virtual QString getAlgorithmName() const override;
  virtual void manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) override;
  virtual void reset() override {}

 private slots:
  void deshuffle();
  void resetCalibration();
  void updateRigRelatedWidgets();
  void applyRigPreset(const QString rig);
  void adjustCrop();

 private:
  enum class CalibrationOption {
    NoOption = 0x00,
    ImproveGeometricCalibrationMode = 0x01,
    ApplyPresetOnly = 0x02,
    DeshuffleOnly = 0x04
  };

  enum class AlgorithmType { Deshuffle, Calibration, None };

  VideoStitch::Status* calibrationComputation(std::shared_ptr<VideoStitch::Ptv::Value> calibrationConfig);
  VideoStitch::Status* deshuffleComputation();
  void addFrame(frameid_t frame);
  void attachListOfFramesToConfig(std::shared_ptr<VideoStitch::Ptv::Value> config,
                                  bool improveGeometricCalibrationMode) const;
  void updateWhenCalibrationChanged(const VideoStitch::Core::PanoDefinition* pano);
  std::shared_ptr<VideoStitch::Ptv::Value> buildCalibrationConfig(int calibrationOptions) const;
  bool addCustomRigToCalibrationConfig(std::shared_ptr<VideoStitch::Ptv::Value> calibrationConfig, double fov,
                                       VideoStitch::Core::InputDefinition::Format lensFormat) const;
  VideoStitch::Core::InputDefinition::Format getFormat() const;
  void showCalibrationModes(bool show);

  QScopedPointer<Ui::CalibrationWidget> ui;
  frameid_t currentFrame;
  QScopedPointer<VideoStitch::Core::PanoDefinition> panoDef;
  QScopedPointer<VideoStitch::Core::PanoDefinition> oldPanoDef;
  AlgorithmType algorithmType;
};
