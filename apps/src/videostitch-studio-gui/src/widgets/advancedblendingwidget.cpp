// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "advancedblendingwidget.hpp"
#include "ui_advancedblendingwidget.h"

#include "commands/advancedblenderchangedcommand.hpp"
#include "videostitcher/postprodprojectdefinition.hpp"

AdvancedBlendingWidget::AdvancedBlendingWidget(QWidget* parent)
    : ComputationWidget(parent), panoDef(nullptr), oldPanoDef(nullptr), ui(new Ui::AdvancedBlendingWidget) {
  ui->setupUi(this);
}

AdvancedBlendingWidget::~AdvancedBlendingWidget() {}

void AdvancedBlendingWidget::onProjectOpened(ProjectDefinition* p) {
  ComputationWidget::onProjectOpened(p);
  if (!project) {
    return;
  }

  updateFlowBox(project->getFlow());
  updateWarperBox(project->getWarper(), project->getFlow());
}

void AdvancedBlendingWidget::clearProject() {
  if (project) {
    disconnect(ui->comboFlow, &QComboBox::currentTextChanged, this,
               &AdvancedBlendingWidget::createAdvancedBlenderChangedCommand);
    disconnect(ui->comboWarper, &QComboBox::currentTextChanged, this,
               &AdvancedBlendingWidget::createAdvancedBlenderChangedCommand);
  }

  ComputationWidget::clearProject();
}

void AdvancedBlendingWidget::createAdvancedBlenderChangedCommand() {
  if (project && project->isInit()) {
    AdvancedBlenderChangedCommand* command = new AdvancedBlenderChangedCommand(
        project->getFlow(), project->getWarper(), ui->comboFlow->currentText(), ui->comboWarper->currentText(), this);
    qApp->findChild<QUndoStack*>()->push(command);
  }
}

void AdvancedBlendingWidget::changeBlender(const QString& flow, const QString& warper) {
  disconnect(ui->comboFlow, &QComboBox::currentTextChanged, this,
             &AdvancedBlendingWidget::createAdvancedBlenderChangedCommand);

  // Update GUI
  const int flowIndex = ui->comboFlow->findText(flow);
  if (flowIndex >= 0) {
    ui->comboFlow->setCurrentIndex(flowIndex);
    updateWarperBox(warper, flow);

    // Update project and stitcher
    project->changeAdvancedBlendingParameters(flow, ui->comboWarper->currentText());
  } else {
    project->changeAdvancedBlendingParameters("", "");
  }

  emit reqResetAdvancedBlending();

  connect(ui->comboFlow, &QComboBox::currentTextChanged, this,
          &AdvancedBlendingWidget::createAdvancedBlenderChangedCommand, Qt::UniqueConnection);
}

QString AdvancedBlendingWidget::getAlgorithmName() const { return tr("Advanced blending"); }

void AdvancedBlendingWidget::updateFlowBox(QString currentFlow) {
  disconnect(ui->comboFlow, &QComboBox::currentTextChanged, this,
             &AdvancedBlendingWidget::createAdvancedBlenderChangedCommand);

  ui->comboFlow->clear();
  for (const std::string& flow : VideoStitch::Core::ImageFlowFactory::availableFlows()) {
    ui->comboFlow->addItem(QString::fromStdString(flow));
  }

  ui->comboFlow->setCurrentText(currentFlow);

  connect(ui->comboFlow, &QComboBox::currentTextChanged, this,
          &AdvancedBlendingWidget::createAdvancedBlenderChangedCommand, Qt::UniqueConnection);
}

void AdvancedBlendingWidget::updateWarperBox(QString currentWarper, QString currentFlow) {
  disconnect(ui->comboWarper, &QComboBox::currentTextChanged, this,
             &AdvancedBlendingWidget::createAdvancedBlenderChangedCommand);

  ui->comboWarper->clear();
  const std::vector<std::string>& warpers =
      VideoStitch::Core::ImageWarperFactory::compatibleWarpers(currentFlow.toStdString());
  for (const std::string& warper : warpers) {
    ui->comboWarper->addItem(QString::fromStdString(warper));
  }

  ui->comboWarper->setCurrentText(currentWarper);

  connect(ui->comboWarper, &QComboBox::currentTextChanged, this,
          &AdvancedBlendingWidget::createAdvancedBlenderChangedCommand, Qt::UniqueConnection);
}
