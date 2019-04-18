// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stereoconfigurationwidget.hpp"
#include "ui_stereoconfigurationwidget.h"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include "libvideostitch-gui/utils/stereooutputenum.hpp"

// TODO: complete this widget with the rest of the stereo configurations
StereoConfigurationWidget::StereoConfigurationWidget(QWidget *parent)
    : QWidget(parent), ui(new Ui::StereoConfigurationWidget) {
  ui->setupUi(this);

  ui->comboProjections->addItems(StereoOutputEnum::getDescriptorsList());
  connect(ui->comboProjections, &QComboBox::currentTextChanged, this, &StereoConfigurationWidget::switchOutput);
  connect(ui->sliderIPD, &QSlider::valueChanged, this, &StereoConfigurationWidget::ipdSliderChanged);
}

StereoConfigurationWidget::~StereoConfigurationWidget() { delete ui; }

void StereoConfigurationWidget::onProjectOpened(ProjectDefinition *project) {
  connect(this, &StereoConfigurationWidget::ipdParameterChanged, project, &ProjectDefinition::setInterPupillaryDistance,
          Qt::UniqueConnection);
}

void StereoConfigurationWidget::ipdSliderChanged(int ipd) { emit ipdParameterChanged(ipd / 100.0); }
