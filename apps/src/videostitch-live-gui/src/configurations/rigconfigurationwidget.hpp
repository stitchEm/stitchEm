// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef RIGCONFIGURATIONWIDGET_HPP
#define RIGCONFIGURATIONWIDGET_HPP

#include <QWidget>
#include "ui_rigconfigurationwidget.h"
#include "libvideostitch/stereoRigDef.hpp"

class RigConfigurationWidget : public QFrame, public Ui::RigConfigurationWidgetClass {
  Q_OBJECT
 public:
  explicit RigConfigurationWidget(QWidget* const parent = nullptr);
  ~RigConfigurationWidget();
  void loadConfiguration(const QStringList inputNames,
                         const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                         const VideoStitch::Core::StereoRigDefinition::Geometry geometry, const double diameter,
                         const double ipd, const QVector<int> leftInputs, const QVector<int> rightInputs);

 signals:
  void notifyRigConfigured(const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                           const VideoStitch::Core::StereoRigDefinition::Geometry geometry, const double diameter,
                           const double ipd, const QVector<int> leftInputs, const QVector<int> rightInputs);

 public slots:
  void onButtonAcceptClicked();
  void onButtonCircularChecked();
  void onButtonPolygonalChecked();
  void onOrientationChanged(const QString& orientation);

 private:
  void addInputsToList(const QVector<int> left, const QVector<int> right);
  QVector<int> getLeftInputs() const;
  QVector<int> getRightInputs() const;
  QStringList inputNames;
};

#endif  // RIGCONFIGURATIONWIDGET_HPP
