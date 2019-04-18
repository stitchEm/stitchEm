// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rigwidget.hpp"
#include "ui_rigwidget.h"

#include "mainwindow/msgboxhandlerhelper.hpp"
#include "mainwindow/vssettings.hpp"
#include "videostitcher/presetsmanager.hpp"
#include "videostitcher/projectdefinition.hpp"

#include "libvideostitch-gui/utils/inputlensenum.hpp"

#include <QDesktopServices>
#include <QUrl>

RigWidget::RigWidget(QWidget* parent) : QWidget(parent), ui(new Ui::RigWidget), project(nullptr) {
  ui->setupUi(this);
  ui->toolButtonBrowseRigPresets->setProperty("vs-blue-button", true);

  ui->toolButtonBrowseRigPresets->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->comboBoxLensType->addItem(InputLensEnum::getDescriptorFromEnum(InputLensClass::LensType::Rectilinear),
                                int(InputLensClass::LensType::Rectilinear));
  ui->comboBoxLensType->addItem(InputLensEnum::getDescriptorFromEnum(InputLensClass::LensType::FullFrameFisheye),
                                int(InputLensClass::LensType::FullFrameFisheye));
  ui->comboBoxLensType->addItem(InputLensEnum::getDescriptorFromEnum(InputLensClass::LensType::CircularFisheye),
                                int(InputLensClass::LensType::CircularFisheye));
  ui->spinBoxHfov->setValue(PTV_DEFAULT_AUTO_FOV);

  connect(ui->comboBoxRig, &QComboBox::currentTextChanged, this, &RigWidget::updateRigRelatedWidgets);
  connect(ui->toolButtonBrowseRigPresets, &QToolButton::clicked, this, &RigWidget::browseRigPresets);
  connect(ui->comboBoxLensType, &QComboBox::currentTextChanged, this, &RigWidget::checkLensType);
  connect(ui->spinBoxHfov, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this,
          &RigWidget::hfovChanged);
  connect(PresetsManager::getInstance(), &PresetsManager::presetsHasChanged, this, [this](QString presetCategory) {
    if (presetCategory == PresetsManager::rigsCategory()) {
      updateRigBox();
    }
  });
}

RigWidget::~RigWidget() {}

void RigWidget::setProject(ProjectDefinition* p) {
  project = p;
  updateRigBox();
  disconnect(ui->comboBoxRig, &QComboBox::currentTextChanged, this,
             &RigWidget::applyRigPreset);  // The rig should already be applied
  updateRigNameValue();
  connect(ui->comboBoxRig, &QComboBox::currentTextChanged, this, &RigWidget::applyRigPreset);
  updateWhenCalibrationChanged(project->getPanoConst().get());
}

void RigWidget::clearProject() { project = nullptr; }

QString RigWidget::customRigName() {
  //: Custom rig name
  return tr("Custom");
}

VideoStitch::Core::InputDefinition::Format RigWidget::getFormatFrom(QString rig) {
  VideoStitch::Core::InputDefinition::Format newFormat = VideoStitch::Core::InputDefinition::Format::FullFrameFisheye;
  std::shared_ptr<const VideoStitch::Ptv::Value> rigPresetValue =
      PresetsManager::getInstance()->getPresetContent(PresetsManager::rigsCategory(), rig);
  const VideoStitch::Ptv::Value* cameraListValue = rigPresetValue->has("cameras");
  if (cameraListValue && cameraListValue->getType() == VideoStitch::Ptv::Value::LIST) {
    const std::string& newFormatString = cameraListValue->asList().at(0)->has("format")->asString();
    VideoStitch::Core::InputDefinition::getFormatFromName(newFormatString, newFormat);
  }
  return newFormat;
}

bool RigWidget::presetIsCompatibleWithPano(std::shared_ptr<const VideoStitch::Ptv::Value> rigPresetValue) const {
  return project->getPanoConst()->isRigPresetCompatible(rigPresetValue.get());
}

void RigWidget::checkNewFormat(VideoStitch::Core::InputDefinition::Format newFormat) const {
  if (project->hasCroppedArea()) {
    const VideoStitch::Core::InputDefinition::Format oldFormat =
        project->getPanoConst()->getLensFormatFromInputSources();
    if (InputLensClass::getLensTypeFromInputDefinitionFormat(oldFormat) !=
        InputLensClass::getLensTypeFromInputDefinitionFormat(newFormat)) {
      MsgBoxHandler::getInstance()->generic(tr("Inputs' crop values will be lost."), tr("Warning"), WARNING_ICON);
    }
  }
}

void RigWidget::updateWhenCalibrationChanged(const VideoStitch::Core::PanoDefinition* pano) {
  updateRigNameValue();
  ui->comboBoxLensType->setCurrentIndex(ui->comboBoxLensType->findData(
      int(InputLensClass::getLensTypeFromInputDefinitionFormat(pano->getLensFormatFromInputSources()))));
  // The HFOV was already calculated
  if (pano->hasCalibrationControlPoints()) {
    ui->spinBoxHfov->setValue(pano->getHFovFromInputSources());
  } else {
    // Use auto value when there is not calibration
    ui->spinBoxHfov->setValue(PTV_DEFAULT_AUTO_FOV);
  }
}

ProjectDefinition* RigWidget::getProject() const { return project; }

bool RigWidget::customRigIsSelected() const { return ui->comboBoxRig->currentText() == customRigName(); }

QString RigWidget::getCurrentRig() const { return ui->comboBoxRig->currentText(); }

std::unique_ptr<VideoStitch::Ptv::Value> RigWidget::cloneSelectedRigPreset() const {
  if (customRigIsSelected()) {
    return std::unique_ptr<VideoStitch::Ptv::Value>();
  } else {
    return PresetsManager::getInstance()->clonePresetContent(PresetsManager::rigsCategory(),
                                                             ui->comboBoxRig->currentText());
  }
}

InputLensClass::LensType RigWidget::getCurrentLensType() const {
  if (customRigIsSelected()) {
    return InputLensClass::LensType(ui->comboBoxLensType->currentData().toInt());
  } else {
    return InputLensClass::getLensTypeFromInputDefinitionFormat(getFormatFrom(getCurrentRig()));
  }
}

double RigWidget::getHfov() const { return ui->spinBoxHfov->value(); }

void RigWidget::updateRigRelatedWidgets(QString rig) {
  const bool customRig = rig == customRigName();
  ui->labelHfov->setVisible(customRig);
  ui->labelLensType->setVisible(customRig);
  ui->comboBoxLensType->setVisible(customRig);
  ui->spinBoxHfov->setVisible(customRig);
  emit currentRigChanged(rig);
}

void RigWidget::browseRigPresets() {
  QDesktopServices::openUrl(QUrl::fromLocalFile(PresetsManager::getRigPresetsFolder()));
}

void RigWidget::updateRigBox() {
  disconnect(ui->comboBoxRig, &QComboBox::currentTextChanged, this, &RigWidget::applyRigPreset);
  const QString oldText = ui->comboBoxRig->currentText();
  ui->comboBoxRig->clear();
  if (!project) {
    return;
  }

  const auto presetsManager = PresetsManager::getInstance();
  for (QString rigPreset : presetsManager->getPresetNames(PresetsManager::rigsCategory())) {
    auto rigPresetValue = presetsManager->getPresetContent(PresetsManager::rigsCategory(), rigPreset);
    if (presetIsCompatibleWithPano(rigPresetValue)) {
      ui->comboBoxRig->addItem(rigPreset);
    }
  }
  ui->comboBoxRig->addItem(customRigName());
  ui->comboBoxRig->setCurrentText(oldText);
  connect(ui->comboBoxRig, &QComboBox::currentTextChanged, this, &RigWidget::applyRigPreset);
}

void RigWidget::checkLensType() {
  VideoStitch::Core::InputDefinition::Format newFormat = InputLensClass::getInputDefinitionFormatFromLensType(
      static_cast<InputLensClass::LensType>(ui->comboBoxLensType->currentData().toInt()),
      project->getPanoConst()->getLensModelCategoryFromInputSources());
  checkNewFormat(newFormat);
  emit currentLensTypeChanged(newFormat);
}

void RigWidget::updateRigNameValue() {
  const QString rigName = project->getPanoConst()->hasCalibrationRigPresets()
                              ? QString::fromStdString(project->getPanoConst()->getCalibrationRigPresetsName())
                              : customRigName();
  ui->comboBoxRig->setCurrentText(rigName);
}

void RigWidget::applyRigPreset(QString rig) {
  if (rig != customRigName()) {
    checkNewFormat(getFormatFrom(rig));
    emit rigPresetSelected(rig);
  } else {
    checkLensType();
  }
}
