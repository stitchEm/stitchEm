// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configpanorama.hpp"
#include "libvideostitch-gui/utils/panolensenum.hpp"
#include "videostitcher/liveoutputfactory.hpp"
#include "videostitcher/liveoutputlist.hpp"
#include "videostitcher/liveprojectdefinition.hpp"

#include "libvideostitch-base/logmanager.hpp"
#include "libvideostitch-base/videowidget.hpp"

#include <QPushButton>

ConfigPanoramaWidget::ConfigPanoramaWidget(QWidget* const parent) : IConfigurationCategory(parent) {
  setupUi(this);
  labelTitlePano->setProperty("vs-title1", true);
  labelMessage->hide();
  verticalLayout->addLayout(buttonsLayout);

  connect(panoSizeSelector, &PanoSizeSelector::sizeChanged, this, &ConfigPanoramaWidget::onConfigurationChanged);
  connect(panoSizeSelector, &PanoSizeSelector::sizeChanged, this, &ConfigPanoramaWidget::checkOutputParameters);
  changeMode(IConfigurationCategory::Mode::Edition);
}

ConfigPanoramaWidget::~ConfigPanoramaWidget() {}

void ConfigPanoramaWidget::recoverPanoFromError() {
  // Go back to the correct values
  panoSizeSelector->setSize(validWidth, validHeight);
  save();
}

void ConfigPanoramaWidget::updateEditability(bool outputIsActivated, bool algorithmIsActivated) {
  panoSizeSelector->setEnabled(!outputIsActivated && !algorithmIsActivated);
  labelMessage->setVisible(outputIsActivated || algorithmIsActivated);
}

void ConfigPanoramaWidget::saveData() {
  if (projectDefinition != nullptr) {
    projectDefinition->updateSize(panoSizeSelector->getWidth(), panoSizeSelector->getHeight());
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        tr("Saving new panorama size: %0x%1").arg(panoSizeSelector->getWidth()).arg(panoSizeSelector->getHeight()));
  }
}

void ConfigPanoramaWidget::restoreData() {
  if (projectDefinition && projectDefinition->isInit()) {
    validWidth = projectDefinition->getPanoConst()->getWidth();
    validHeight = projectDefinition->getPanoConst()->getHeight();
    panoSizeSelector->setSize(projectDefinition->getPanoConst()->getWidth(),
                              projectDefinition->getPanoConst()->getHeight());
    warningLabel->clear();
    warningLabel->hide();
    warningIconLabel->hide();
    labelHFov->setText(QString::number(projectDefinition->getHFOV()));
    const auto format =
        projectDefinition->getPanoConst()->getFormatFromName(projectDefinition->getProjection().toStdString());
    labelProjection->setText(PanoLensEnum::getDescriptorFromEnum(format));
  }
}

void ConfigPanoramaWidget::checkOutputParameters() {
  warningLabel->clear();
  const int oldWidth = projectDefinition->getPanoConst()->getWidth();
  const int oldHeight = projectDefinition->getPanoConst()->getHeight();
  const int newWidth = panoSizeSelector->getWidth();
  const int newHeight = panoSizeSelector->getHeight();
  if (newWidth == oldWidth && newHeight == oldHeight) {
    warningLabel->hide();
    warningIconLabel->hide();
    return;
  }

  QString text;
  LiveOutputList* outputs = projectDefinition->getOutputConfigs();
  for (const LiveOutputFactory* output : outputs->getValues()) {
    LiveOutputFactory::PanoSizeChange change = output->supportPanoSizeChange(newWidth, newHeight);
    Q_ASSERT_X(change != LiveOutputFactory::PanoSizeChange::NotSupported, "ConfigPanoramaWidget::checkOutputParameters",
               "Changing pano size is expected to be supported");
    if (change == LiveOutputFactory::PanoSizeChange::SupportedWithUpdate) {
      if (!text.isEmpty()) {
        text += "\n\n";
      }
      text += output->getPanoSizeChangeDescription(newWidth, newHeight);
    }
  }

  if (!text.isEmpty()) {
    //: Warning when changing panorama size
    text.prepend(
        tr("Changing the panorama size will automatically update the impacted output parameters to these new "
           "values:\n\n"));
    warningLabel->setText(text);
    warningLabel->show();
    warningIconLabel->show();
  }
}

void ConfigPanoramaWidget::reactToChangedProject() { restoreData(); }
