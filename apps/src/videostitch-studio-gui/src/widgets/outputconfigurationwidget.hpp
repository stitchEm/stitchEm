// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-base/projection.hpp"
#include "itoolwidget.hpp"

#include <QWidget>

namespace Ui {
class OutputConfigurationWidget;
}
class ProjectDefinition;
class SignalCompressionCaps;

class OutputConfigurationWidget : public QWidget, public IToolWidget {
  Q_OBJECT

 public:
  explicit OutputConfigurationWidget(QWidget* parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
  ~OutputConfigurationWidget();

  void changeBlender(QString merger, int feather);
  void changeProjectionAndFov(VideoStitch::Projection projection, double hfov);
  void changeSphereScale(double sphereScale);

 public slots:
  void clearProject();
  void enabledInputNumberButton();
  void setProject(ProjectDefinition* newProject);
  void updateIncompatibleProjectionWarning();
  void updateSphereScaleAvailability();

 signals:
  void reqResetMerger(SignalCompressionCaps* comp = nullptr);
  void reqSetProjection(VideoStitch::Projection projection, double fov);
  void reqSetSphereScale(const double sphereScale, bool restitch);

 protected:
  virtual void reset() override;

 private:
  static QString getImageMergerDisplayableName(const std::string& id);
  double sphereScaleSliderRepresentationToValue(int sliderVal) const;
  int sphereScaleValueToSliderRepresentation(double sphereScale) const;

 private slots:
  void createBlenderChangedCommand();
  void createHfovChangedCommand();
  void createProjectionChangedCommand();
  void createSphereScaleChangedCommand(double updatedSphereScale);
  void sphereScaleSliderChanged();

 private:
  Q_DISABLE_COPY(OutputConfigurationWidget)
  QScopedPointer<Ui::OutputConfigurationWidget> ui;
  ProjectDefinition* project;
};
