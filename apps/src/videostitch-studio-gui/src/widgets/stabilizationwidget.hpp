// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "computationwidget.hpp"
#include "itoolwidget.hpp"
#include <memory>

namespace Ui {
class StabilizationWidget;
}
class QAbstractButton;

class StabilizationWidget : public ComputationWidget, public IToolWidget {
  Q_OBJECT

 public:
  explicit StabilizationWidget(QWidget* parent = nullptr);
  ~StabilizationWidget();

 public slots:
  virtual void onProjectOpened(ProjectDefinition* p) override;
  virtual void updateSequence(const QString start, const QString stop) override;
  virtual void clearProject() override;
  void onProjectOrientable(bool yprModificationsAllowed);
  void toggleOrientationButton();

 signals:
  /**
   * @brief Triggers a signal for applying the stabilization
   */
  void reqApplyStabilization(VideoStitch::Core::PanoDefinition* panoDef);

  /**
   * @brief Triggers a signal for reseting the stabilization
   */
  void reqResetStabilization();

  void reqSetEditOrientationActivated(bool editOrientationActivated, bool restitch);
  void orientationActivated(bool activated);

 protected:
  virtual QString getAlgorithmName() const override;
  virtual void manageComputationResult(bool hasBeenCancelled, VideoStitch::Status* status) override;
  virtual void reset() override;

 private slots:
  void onButtonOrientationToggled(const bool activate);
  void startComputation();
  void onResetStabilizationClicked();

 private:
  VideoStitch::Status* computation(std::shared_ptr<VideoStitch::Ptv::Value> stabilizationConfig);
  QScopedPointer<Ui::StabilizationWidget> ui;
  VideoStitch::Core::PanoDefinition* panoDef;
  VideoStitch::Core::PanoDefinition* oldPanoDef;
  VideoStitch::Util::OpaquePtr* ctx;
};
