// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputconfigurationwidget.hpp"
#include "ui_outputconfigurationwidget.h"

#include "commands/blenderchangedcommand.hpp"
#include "commands/spherescalechangedcommand.hpp"
#include "commands/projectionchangedcommand.hpp"

#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include "libvideostitch-gui/mainwindow/objectutil.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"

#include "libvideostitch/gpu_device.hpp"

OutputConfigurationWidget::OutputConfigurationWidget(QWidget* parent, Qt::WindowFlags f)
    : QWidget(parent, f), ui(new Ui::OutputConfigurationWidget), project(nullptr) {
  ui->setupUi(this);

  const std::vector<std::string>& availableMergers = VideoStitch::Core::ImageMergerFactory::availableMergers();
  std::vector<std::string> mergers;
  if (VSSettings::getSettings()->getShowExperimentalFeatures()) {
    mergers = availableMergers;
  } else {
    if (std::find(availableMergers.cbegin(), availableMergers.cend(), "gradient") != availableMergers.cend()) {
      mergers.push_back("gradient");
    }
    if (std::find(availableMergers.cbegin(), availableMergers.cend(), "laplacian") != availableMergers.cend()) {
      mergers.push_back("laplacian");
    }
  }
  for (const std::string& merger : mergers) {
    ui->comboBlender->addItem(getImageMergerDisplayableName(merger), QString::fromStdString(merger));
  }

  ui->projectionComboBox->addItems(VideoStitch::guiStringListProjection());
  ui->incompatibleProjectionLabel->hide();
  ui->projectionBox->setVisible(VSSettings::getSettings()->getValue("debug/projection-visibility", false).toBool());

  // hide the advanced calibration parameters in the experimental Studio features
  ui->sphereScaleLabel->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->sphereScaleSpinBox->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->sphereScaleSlider->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->sphereScaleMinLabel->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->sphereScaleMaxLabel->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());

  connect(ui->imageNumberButton, &QPushButton::clicked, [=]() { ui->imageNumberButton->setEnabled(false); });
}

OutputConfigurationWidget::~OutputConfigurationWidget() {}

void OutputConfigurationWidget::changeBlender(QString merger, int feather) {
  ui->comboBlender->blockSignals(true);
  ui->blendingSlider->blockSignals(true);

  // Update GUI
  ui->comboBlender->setCurrentIndex(ui->comboBlender->findData(merger));
  ui->blendingSlider->setValue(feather);

  // Update project and stitcher
  project->changeBlendingParameters(merger, feather);
  emit reqResetMerger();

  ui->comboBlender->blockSignals(false);
  ui->blendingSlider->blockSignals(false);
}

void OutputConfigurationWidget::changeProjectionAndFov(VideoStitch::Projection projection, double hfov) {
  ui->projectionComboBox->blockSignals(true);
  ui->hFovSpinbox->blockSignals(true);

  // Update GUI
  ui->projectionComboBox->setCurrentIndex(static_cast<int>(projection));
  ui->hFovSpinbox->setRange(VideoStitch::getMinimumValueFor(projection), VideoStitch::getMaximumValueFor(projection));
  ui->hFovSpinbox->setValue(hfov);
  updateIncompatibleProjectionWarning();
  // Update project and stitcher
  emit reqSetProjection(projection, hfov);

  ui->projectionComboBox->blockSignals(false);
  ui->hFovSpinbox->blockSignals(false);
}

void OutputConfigurationWidget::changeSphereScale(double sphereScale) {
  ui->sphereScaleSpinBox->blockSignals(true);
  ui->sphereScaleSlider->blockSignals(true);

  // Update GUI
  ui->sphereScaleSpinBox->setValue(sphereScale);
  ui->sphereScaleSlider->setValue(sphereScaleValueToSliderRepresentation(sphereScale));

  // Update project and stitcher
  project->setSphereScale(sphereScale);
  emit reqSetSphereScale(sphereScale, true);

  ui->sphereScaleSpinBox->blockSignals(false);
  ui->sphereScaleSlider->blockSignals(false);
}

void OutputConfigurationWidget::clearProject() {
  if (project) {
    reset();
    disconnect(ui->comboBlender, &QComboBox::currentTextChanged, this,
               &OutputConfigurationWidget::createBlenderChangedCommand);
    disconnect(ui->blendingSlider, &QSlider::valueChanged, this,
               &OutputConfigurationWidget::createBlenderChangedCommand);
    disconnect(ui->sphereScaleSpinBox, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
               this, &OutputConfigurationWidget::createSphereScaleChangedCommand);
    disconnect(ui->sphereScaleSlider, &QSlider::valueChanged, this,
               &OutputConfigurationWidget::sphereScaleSliderChanged);
    disconnect(ui->projectionComboBox, &QComboBox::currentTextChanged, this,
               &OutputConfigurationWidget::createProjectionChangedCommand);
    disconnect(ui->hFovSpinbox, &QDoubleSpinBox::editingFinished, this,
               &OutputConfigurationWidget::createHfovChangedCommand);
    disconnect(ui->imageNumberButton, &QPushButton::clicked, project, &ProjectDefinition::toggleInputNumberDrawing);
  }

  project = nullptr;

  ui->imageNumberButton->setChecked(false);
}

void OutputConfigurationWidget::enabledInputNumberButton() { ui->imageNumberButton->setEnabled(true); }

void OutputConfigurationWidget::setProject(ProjectDefinition* newProject) {
  if (project) {
    disconnect(ui->imageNumberButton, &QPushButton::clicked, project, &ProjectDefinition::toggleInputNumberDrawing);
  }

  project = newProject;

  ui->imageNumberButton->setChecked(project->isDrawingInputNumbers());
  ui->comboBlender->setCurrentIndex(ui->comboBlender->findData(project->getBlender()));
  ui->blendingSlider->setValue(project->getFeather());
  VideoStitch::Projection projection = VideoStitch::mapPTVStringToIndex.value(project->getProjection());
  ui->projectionComboBox->setCurrentIndex(projection);
  ui->hFovSpinbox->setRange(VideoStitch::getMinimumValueFor(projection), VideoStitch::getMaximumValueFor(projection));
  ui->hFovSpinbox->setValue(project->getHFOV());
  const auto minSphereScale = project->computeMinimumSphereScale();
  ui->sphereScaleSpinBox->setMinimum(minSphereScale);
  //: sphere scale slider lower limit label, value displayed in meters
  ui->sphereScaleMinLabel->setText(tr("%0 m").arg(QString::number(minSphereScale, 'f', 2)));
  ui->sphereScaleSpinBox->setValue(project->getSphereScale());
  ui->sphereScaleSlider->setValue(sphereScaleValueToSliderRepresentation(project->getSphereScale()));
  updateSphereScaleAvailability();
  updateIncompatibleProjectionWarning();

  connect(ui->comboBlender, &QComboBox::currentTextChanged, this,
          &OutputConfigurationWidget::createBlenderChangedCommand, Qt::UniqueConnection);
  connect(ui->blendingSlider, &QSlider::valueChanged, this, &OutputConfigurationWidget::createBlenderChangedCommand,
          Qt::UniqueConnection);
  connect(ui->sphereScaleSpinBox, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this,
          &OutputConfigurationWidget::createSphereScaleChangedCommand, Qt::UniqueConnection);
  connect(ui->sphereScaleSlider, &QSlider::valueChanged, this, &OutputConfigurationWidget::sphereScaleSliderChanged,
          Qt::UniqueConnection);
  connect(ui->projectionComboBox, &QComboBox::currentTextChanged, this,
          &OutputConfigurationWidget::createProjectionChangedCommand, Qt::UniqueConnection);
  connect(ui->hFovSpinbox, &QDoubleSpinBox::editingFinished, this, &OutputConfigurationWidget::createHfovChangedCommand,
          Qt::UniqueConnection);
  connect(ui->imageNumberButton, &QPushButton::toggled, project, &ProjectDefinition::toggleInputNumberDrawing,
          Qt::UniqueConnection);
}

void OutputConfigurationWidget::updateIncompatibleProjectionWarning() {
  if (project == nullptr || !project->isInit()) {
    return;
  }
  bool displayWarning = !(project->getPano()->getProjection() == VideoStitch::Core::PanoProjection::Equirectangular &&
                          project->getPano()->getHFOV() == 360. &&
                          double(project->getPano()->getWidth()) / double(project->getPano()->getHeight()) == 2.0);
  ui->incompatibleProjectionLabel->setVisible(displayWarning);
}

void OutputConfigurationWidget::updateSphereScaleAvailability() {
  if (project) {
    ui->sphereScaleSpinBox->setEnabled(project->getPanoConst()->hasTranslations());
    ui->sphereScaleLabel->setEnabled(project->getPanoConst()->hasTranslations());
    ui->sphereScaleSlider->setEnabled(project->getPanoConst()->hasTranslations());
    ui->sphereScaleMinLabel->setEnabled(project->getPanoConst()->hasTranslations());
    ui->sphereScaleMaxLabel->setEnabled(project->getPanoConst()->hasTranslations());
  }
}

void OutputConfigurationWidget::reset() { ui->imageNumberButton->setChecked(false); }

QString OutputConfigurationWidget::getImageMergerDisplayableName(const std::string& id) {
  if (id == "gradient") {
    //: Blender option
    return tr("Linear (faster rendering)");
  } else if (id == "laplacian") {
    //: Blender option
    return tr("Multiband (higher quality)");
  } else {
    return QString::fromStdString(id);
  }
}

double OutputConfigurationWidget::sphereScaleSliderRepresentationToValue(int sliderVal) const {
  const double logMin = std::max(log2(ui->sphereScaleSpinBox->minimum()), -8.);
  const double logMax = log2(ui->sphereScaleSpinBox->maximum());
  const double logRange = logMax - logMin;

  const double normalized = sliderVal / (double)ui->sphereScaleSlider->maximum();
  const double logSphereScale = normalized * logRange + logMin;
  return exp2(logSphereScale);
}

int OutputConfigurationWidget::sphereScaleValueToSliderRepresentation(double sphereScale) const {
  const double logMin = std::max(log2(ui->sphereScaleSpinBox->minimum()), -8.);
  const double logMax = log2(ui->sphereScaleSpinBox->maximum());
  const double logRange = logMax - logMin;

  const double logSphereScale = log2(sphereScale);
  const double normalized = (logSphereScale - logMin) / logRange;
  return int(round(normalized * ui->sphereScaleSlider->maximum()));
}

void OutputConfigurationWidget::createBlenderChangedCommand() {
  if (project && project->isInit()) {
    BlenderChangedCommand* command =
        new BlenderChangedCommand(project->getBlender(), project->getFeather(),
                                  ui->comboBlender->currentData().toString(), ui->blendingSlider->value(), this);
    qApp->findChild<QUndoStack*>()->push(command);
  }
}

void OutputConfigurationWidget::createHfovChangedCommand() {
  if (project && project->isInit() && project->getHFOV() != ui->hFovSpinbox->value()) {
    VideoStitch::Projection projection = VideoStitch::Projection(ui->projectionComboBox->currentIndex());
    ProjectionChangedCommand* command =
        new ProjectionChangedCommand(projection, project->getHFOV(), projection, ui->hFovSpinbox->value(), this);
    qApp->findChild<QUndoStack*>()->push(command);
  }
}

void OutputConfigurationWidget::createProjectionChangedCommand() {
  if (project && project->isInit()) {
    VideoStitch::Projection oldProjection = VideoStitch::mapPTVStringToIndex.value(project->getProjection());
    VideoStitch::Projection newProjection = VideoStitch::Projection(ui->projectionComboBox->currentIndex());
    ProjectionChangedCommand* command = new ProjectionChangedCommand(oldProjection, project->getHFOV(), newProjection,
                                                                     VideoStitch::getDefaultFor(newProjection), this);
    qApp->findChild<QUndoStack*>()->push(command);
  }
}

void OutputConfigurationWidget::createSphereScaleChangedCommand(double newSphereScale) {
  if (project && project->isInit()) {
    SphereScaleChangedCommand* command = new SphereScaleChangedCommand(project->getSphereScale(), newSphereScale, this);
    qApp->findChild<QUndoStack*>()->push(command);
  }
}

void OutputConfigurationWidget::sphereScaleSliderChanged() {
  if (project && project->isInit()) {
    createSphereScaleChangedCommand(sphereScaleSliderRepresentationToValue(ui->sphereScaleSlider->value()));
  }
}
