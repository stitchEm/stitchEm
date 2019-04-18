// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "cropwidget.hpp"
#include "ui_cropwidget.h"

#include "cropinputtab.hpp"
#include "videostitcher/projectdefinition.hpp"
#include "videostitcher/globalcontroller.hpp"

#include <libvideostitch/logging.hpp>

#include <QDesktopWidget>

const QString ALL_CROPS_LABEL(qApp->translate("CropWidget", "Apply input %0 crop to all the inputs"));

CropWidget::CropWidget(QWidget* parent) : CropWidget(nullptr, InputLensClass::defaultValue, 0, parent) {}

CropWidget::CropWidget(ProjectDefinition* p, InputLensClass::LensType t, const int extended, QWidget* parent)
    : QWidget(parent),
      lensType(t),
      ui(new Ui::CropWidget),
      project(p),
      height(getAvailableHeight() - extended),
      blockIndex(-1),
      oldFrameAction(int(StitcherController::NextFrameAction::None)) {
  ui->setupUi(this);
  connect(ui->checkApplyToAll, &QCheckBox::toggled, this, &CropWidget::onCropLocked);
  connect(ui->tabWidgetInputs, &QTabWidget::currentChanged, this, &CropWidget::onTabChanged);
}

CropWidget::~CropWidget() {}

void CropWidget::onCropLocked(const bool block) {
  const int index = ui->tabWidgetInputs->currentIndex();
  QString label;
  if (block) {
    CropInputTab* mainInputTab = getTabWidget(index);
    blockIndex = index;
    connectTo(mainInputTab);
    label = tr("Input %0 crop applied to all the inputs").arg(index + 1);
  } else {
    CropInputTab* mainInputTab = getTabWidget(blockIndex);
    blockIndex = -1;
    disconnectFrom(mainInputTab);
    label = ALL_CROPS_LABEL.arg(ui->tabWidgetInputs->currentIndex() + 1);
  }
  ui->checkApplyToAll->setText(label);
}

void CropWidget::onTabChanged(const int index) {
  if (!ui->checkApplyToAll->isChecked()) {
    ui->checkApplyToAll->setText(ALL_CROPS_LABEL.arg(index + 1));
  }
}

void CropWidget::initializeTabs() {
  for (auto input = 0; input < project->getPano()->numVideoInputs(); ++input) {
    const VideoStitch::Core::InputDefinition& inputDef = project->getPano()->getVideoInput(input);
    Crop initCrop(inputDef.getCropLeft(), inputDef.getCropRight(), inputDef.getCropTop(), inputDef.getCropBottom());
    const QSize videoSize(inputDef.getWidth(), inputDef.getHeight());
    CropInputTab* inputTab = new CropInputTab(input, videoSize, height, initCrop, lensType, this);
    ui->tabWidgetInputs->addTab(inputTab, tr("Input %0").arg(QString::number(input + 1)));
    emit reqRegisterRender(inputTab->getVideoWidget(), input);
  }

  StitcherController* stitcherController = GlobalController::getInstance().getController();
  // We should at least extract
  oldFrameAction = int(stitcherController->setNextFrameAction(StitcherController::NextFrameAction::StitchAndExtract));
  if (!stitcherController->isPlaying()) {
    emit reextract();
  }
}

void CropWidget::deinitializeTabs() {
  if (ui->tabWidgetInputs->count() == 0) {
    // The widget has not been dispalyed yet, nothing to deinitialize
    return;
  }
  for (auto input = 0; input < ui->tabWidgetInputs->count(); ++input) {
    CropInputTab* inputTab = getTabWidget(input);
    emit reqUnregisterRender(inputTab->getReaderName(), input);
  }

  while (ui->tabWidgetInputs->count() != 0) {
    delete ui->tabWidgetInputs->widget(0);
  }
  ui->tabWidgetInputs->clear();

  StitcherController* stitcherController = GlobalController::getInstance().getController();
  if (stitcherController) {
    stitcherController->setNextFrameAction(StitcherController::NextFrameAction(oldFrameAction));
  }
}

void CropWidget::connectTo(CropInputTab* mainInputTab) {
  for (auto tabIndex = 0; tabIndex < ui->tabWidgetInputs->count(); ++tabIndex) {
    CropInputTab* inputTab = getTabWidget(tabIndex);
    if (mainInputTab != inputTab) {
      inputTab->setCrop(mainInputTab->getCrop());
      inputTab->disableCropActions(true);
      connect(mainInputTab, &CropInputTab::cropChanged, inputTab, &CropInputTab::setCrop, Qt::UniqueConnection);
      ui->tabWidgetInputs->setTabIcon(tabIndex, QIcon());
    } else {
      inputTab->disableCropActions(false);
      ui->tabWidgetInputs->setTabIcon(tabIndex, QIcon(":/live/icons/assets/icon/live/lock-closed.png"));
    }
  }
}

void CropWidget::disconnectFrom(CropInputTab* mainInputTab) {
  disconnect(mainInputTab, &CropInputTab::cropChanged, 0, 0);
  for (auto tabIndex = 0; tabIndex < ui->tabWidgetInputs->count(); ++tabIndex) {
    CropInputTab* inputTab = getTabWidget(tabIndex);
    if (mainInputTab != inputTab) {
      inputTab->disableCropActions(false);
    } else {
      ui->tabWidgetInputs->setTabIcon(tabIndex, QIcon());
    }
  }
}

QHBoxLayout* CropWidget::getHorizontalLayout() { return ui->horizontalLayout; }

void CropWidget::applyCrop() {
  QString lensTypeStr = InputLensEnum::getDescriptorFromEnum(lensType);
  VideoStitch::Logger::get(VideoStitch::Logger::Info)
      << "Applying crop with lens type: " << lensTypeStr.toStdString() << std::endl;
  emit reqApplyCrops(getCrops(), lensType);
}

void CropWidget::setDefaultCrop() {
  for (auto tabIndex = 0; tabIndex < ui->tabWidgetInputs->count(); ++tabIndex) {
    CropInputTab* inputTab = getTabWidget(tabIndex);
    inputTab->setDefaultCrop();
  }
}

void CropWidget::setProject(ProjectDefinition* p) { project = p; }

void CropWidget::setLensType(InputLensClass::LensType t) { lensType = t; }

void CropWidget::setWidgetExtension(int extended) { height = getAvailableHeight() - extended; }

int CropWidget::getAvailableHeight() const {
  const QRect screenSize = QApplication::desktop()->availableGeometry(this);
  return int(float(screenSize.height() * (3.f / 4.f)));
}

CropInputTab* CropWidget::getTabWidget(const int index) const {
  return qobject_cast<CropInputTab*>(ui->tabWidgetInputs->widget(index));
}

QVector<Crop> CropWidget::getCrops() const {
  QVector<Crop> crops;
  for (auto tabIndex = 0; tabIndex < ui->tabWidgetInputs->count(); ++tabIndex) {
    int index = tabIndex;
    if (blockIndex >= 0) {
      index = blockIndex;
    }
    const CropInputTab* widget = getTabWidget(index);
    if (widget != nullptr) {
      crops.push_back(widget->getCrop());
    }
  }
  return crops;
}
