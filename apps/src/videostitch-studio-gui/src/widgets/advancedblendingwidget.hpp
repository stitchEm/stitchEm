// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "computationwidget.hpp"
#include "itoolwidget.hpp"

namespace Ui {
class AdvancedBlendingWidget;
}

class SignalCompressionCaps;

class AdvancedBlendingWidget : public ComputationWidget, public IToolWidget {
  Q_OBJECT

 public:
  explicit AdvancedBlendingWidget(QWidget* parent = nullptr);
  ~AdvancedBlendingWidget();

  void changeBlender(const QString& flow, const QString& warper);

 public slots:
  virtual void onProjectOpened(ProjectDefinition* project) override;
  virtual void updateSequence(const QString, const QString) override {}
  virtual void clearProject() override;

 signals:
  void reqApplyAdvancedBlending(VideoStitch::Core::PanoDefinition* panoDef = nullptr);
  void reqResetAdvancedBlending(SignalCompressionCaps* comp = nullptr);

 protected:
  virtual QString getAlgorithmName() const override;
  virtual void manageComputationResult(bool /*hasBeenCancelled*/, VideoStitch::Status* /*status*/) override {}
  virtual void reset() override {}

 private:
  void updateFlowBox(QString currentFlow);
  void updateWarperBox(QString currentWarper, QString currentFlow);

 private slots:
  void createAdvancedBlenderChangedCommand();

 private:
  VideoStitch::Core::PanoDefinition* panoDef;
  VideoStitch::Core::PanoDefinition* oldPanoDef;
  QScopedPointer<Ui::AdvancedBlendingWidget> ui;
};
