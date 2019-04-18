// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/common.hpp"
#include "libvideostitch-gui/utils/inputlensenum.hpp"

#include "libvideostitch/panoDef.hpp"

#include <QPointer>
#include <QWidget>

#include <memory>

class ProjectDefinition;
namespace Ui {
class RigWidget;
}

class VS_GUI_EXPORT RigWidget : public QWidget {
  Q_OBJECT

 public:
  explicit RigWidget(QWidget* parent = nullptr);
  ~RigWidget();

  static QString customRigName();

  void setProject(ProjectDefinition* p);
  void clearProject();
  void updateWhenCalibrationChanged(const VideoStitch::Core::PanoDefinition* pano);

  ProjectDefinition* getProject() const;
  bool customRigIsSelected() const;
  QString getCurrentRig() const;
  std::unique_ptr<VideoStitch::Ptv::Value> cloneSelectedRigPreset() const;
  InputLensClass::LensType getCurrentLensType() const;
  double getHfov() const;

 public slots:
  void applyRigPreset(QString rig);

 signals:
  /**
   * @brief Sends a signal whenever the rig change
   */
  void currentRigChanged(QString rig);
  /**
   * @brief Sends a signal when a rig preset is selected by the user
   */
  void rigPresetSelected(QString rig);
  void currentLensTypeChanged(VideoStitch::Core::InputDefinition::Format lensType);
  void hfovChanged(double hfov);

 private:
  static VideoStitch::Core::InputDefinition::Format getFormatFrom(QString rig);
  bool presetIsCompatibleWithPano(std::shared_ptr<const VideoStitch::Ptv::Value> rigPresetValue) const;
  void checkNewFormat(VideoStitch::Core::InputDefinition::Format newFormat) const;

 private slots:
  void updateRigRelatedWidgets(QString rig);
  void browseRigPresets();
  void updateRigBox();
  void checkLensType();
  void updateRigNameValue();

 private:
  QScopedPointer<Ui::RigWidget> ui;
  QPointer<ProjectDefinition> project;
};
