// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef STEREOCONFIGURATIONWIDGET_HPP
#define STEREOCONFIGURATIONWIDGET_HPP

#include <QWidget>
#include "itoolwidget.hpp"

namespace Ui {
class StereoConfigurationWidget;
}

class ProjectDefinition;
class StereoConfigurationWidget : public QWidget, public IToolWidget {
  Q_OBJECT

 public:
  explicit StereoConfigurationWidget(QWidget* const parent = nullptr);
  ~StereoConfigurationWidget();

 public slots:
  void onProjectOpened(ProjectDefinition* project);

 signals:
  void switchOutput(const QString&);
  void ipdParameterChanged(double ipd);

 protected:
  virtual void reset() override {}

 private slots:
  void ipdSliderChanged(int ipd);

 private:
  Ui::StereoConfigurationWidget* ui;
};

#endif  // STEREOCONFIGURATIONWIDGET_HPP
