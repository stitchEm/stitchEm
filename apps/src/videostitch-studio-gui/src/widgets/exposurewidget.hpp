// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef EXPOSUREWIDGET_HPP
#define EXPOSUREWIDGET_HPP

#include "computationwidget.hpp"
#include "itoolwidget.hpp"
#include <memory>

namespace Ui {
class ExposureWidget;
}

class ExposureWidget : public ComputationWidget, public IToolWidget {
  Q_OBJECT

 public:
  explicit ExposureWidget(QWidget* parent = nullptr);
  ~ExposureWidget();

  void resetEvCurves();

 public slots:
  virtual void onProjectOpened(ProjectDefinition* p) override;
  virtual void clearProject() override;
  virtual void updateSequence(const QString, const QString) override;
  void updateToPano(frameid_t newLastStitchableFrame, frameid_t newCurrentFrame);
  void updatePhotometryResults();
  void refresh(mtime_t date);

 signals:
  void reqApplyExposure(VideoStitch::Core::PanoDefinition* panoDef);
  void reqApplyPhotometricCalibration(VideoStitch::Core::PanoDefinition* panoDef);
  void reqResetPhotometricCalibration();
  void reqResetEvCurves(bool resetController);
  void reqResetEvCurvesSequence(frameid_t startPoint, frameid_t endPoint);

 protected:
  virtual QString getAlgorithmName() const override;
  virtual void manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) override;
  virtual void reset() override {}

 private:
  void startExposureCompensation(bool computeCurrentFrame);
  VideoStitch::Status* computeExposureCompensation(std::shared_ptr<VideoStitch::Ptv::Value> config);
  VideoStitch::Status* computePhotometricCalibration(std::shared_ptr<VideoStitch::Ptv::Value> config);

 private slots:
  void startPhotometricCalibration();
  void resetExposureCompensationOnSequence();
  void startExposureCompensationOnSequence();
  void startExposureCompensationHere();

 private:
  enum class AlgorithmType { PhotometricCalibration, ExposureCompensation, None };

  QScopedPointer<Ui::ExposureWidget> ui;
  AlgorithmType algorithmType;
  QScopedPointer<VideoStitch::Core::PanoDefinition> panoDef;
  QScopedPointer<VideoStitch::Core::PanoDefinition> oldPanoDef;
  frameid_t currentFrame;
};

#endif  // EXPOSUREWIDGET_HPP
