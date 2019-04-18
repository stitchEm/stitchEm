// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/utils/inputlensenum.hpp"

#include <QObject>

class OutputControlsPanel;
class ProjectDefinition;
struct Crop;

namespace VideoStitch {
namespace Core {
class PanoDefinition;
}
}  // namespace VideoStitch

class CropInputController : public QObject {
  Q_OBJECT
 public:
  explicit CropInputController(OutputControlsPanel *panel, QObject *const parent = nullptr);

 public slots:
  void onShowCropWindow();
  void setProject(ProjectDefinition *p);
  void onCropApplied();

 signals:
  void reqApplyCrops(const QVector<Crop> &crops, const InputLensClass::LensType lensType);

 private slots:
  void applyCropsAsked(const QVector<Crop> &crops, const InputLensClass::LensType lensType);

 private:
  ProjectDefinition *project;
};
