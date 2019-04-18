// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configurationinputprocedural.hpp"
#include "ui_configurationinputprocedural.h"

#include "videostitcher/liveinputprocedural.hpp"
#include "videostitcher/liveprojectdefinition.hpp"

#include <QColorDialog>

ConfigurationInputProcedural::ConfigurationInputProcedural(std::shared_ptr<const LiveInputProcedural> liveInput,
                                                           QWidget *parent)
    : InputConfigurationWidget(parent), ui(new Ui::ConfigurationInputProcedural), templateInput(liveInput) {
  ui->setupUi(this);
  ui->labelTitleProcedurals->setProperty("vs-title1", true);
  ui->verticalLayout->addLayout(buttonsLayout);

  connect(ui->proceduralsNbSpinBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
          &ConfigurationInputProcedural::onConfigurationChanged);
  connect(ui->widthBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
          &ConfigurationInputProcedural::onConfigurationChanged);
  connect(ui->heightBox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
          &ConfigurationInputProcedural::onConfigurationChanged);
}

ConfigurationInputProcedural::~ConfigurationInputProcedural() {}

void ConfigurationInputProcedural::saveData() {
  editedInputs.clear();
  QColor color;
  color.setHsv(0, 255, 255);
  int nbInputs = ui->proceduralsNbSpinBox->value();
  const int hueIncrement = 55;  // VSA-6277: we want the procedural inputs to not change
  for (int index = 0; index < nbInputs; ++index) {
    std::shared_ptr<LiveInputProcedural> input(
        new LiveInputProcedural(Procedural::NameEnum::getDescriptorFromEnum(Procedural::frameNumber)));
    input->setWidth(ui->widthBox->value());
    input->setHeight(ui->heightBox->value());
    input->setColor(color);
    editedInputs.append(input);
    color.setHsv(color.hsvHue() + hueIncrement, 255, 255);
  }
}

void ConfigurationInputProcedural::reactToChangedProject() {
  if (templateInput) {
    // Set the boxes to the input parameters
    int nbInputs = 4;  // Default procedurals nb
    if (projectDefinition->isInit()) {
      nbInputs = projectDefinition->getPanoConst()->numVideoInputs();
    }
    ui->proceduralsNbSpinBox->setValue(nbInputs);
    ui->widthBox->setValue(templateInput->getWidth());
    ui->heightBox->setValue(templateInput->getHeight());
  }
}
