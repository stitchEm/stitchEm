// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "guiconstants.hpp"
#include "sourcestabwidget.hpp"
#include "widgetsmanager.hpp"
#include "livesettings.hpp"
#include "configurations/rigconfigurationwidget.hpp"
#include "generic/backgroundcontainer.hpp"
#include "generic/genericdialog.hpp"
#include "liveprojectdefinition.hpp"

#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/widgets/multivideowidget.hpp"

#include "libvideostitch/stereoRigDef.hpp"

SourcesTabWidget::SourcesTabWidget(QWidget* const parent)
    : QWidget(parent), sourceWidget(new SourceWidget(false, this)), projectDefinition(nullptr) {
  setupUi(this);
  sourceWidget->getMultiVideoWidget().syncOff();
  buttonsBackground->setProperty("vs-button-bar", true);
  buttonSnapshot->setProperty("vs-button-medium", true);
  buttonEditVideo->setProperty("vs-button-medium", true);
  buttonEditAudio->setProperty("vs-button-medium", true);
  buttonEdit3DRig->setProperty("vs-button-medium", true);
  layoutInput->addWidget(sourceWidget);
  activateOptions(false);
  connect(buttonSnapshot, &QPushButton::clicked, this, &SourcesTabWidget::onButtonSnapshotClicked);
  connect(buttonEdit3DRig, &QPushButton::clicked, this, &SourcesTabWidget::onButtonConfigure3DRigClicked);
  StateManager::getInstance()->registerObject(this);

  bool isStereo = VSSettings::getSettings()->getIsStereo();
  buttonEdit3DRig->setVisible(isStereo);
  labelEdit3DRig->setVisible(isStereo);
}

SourcesTabWidget::~SourcesTabWidget() {}

SourceWidget* SourcesTabWidget::getSourcesWidget() const { return sourceWidget; }

void SourcesTabWidget::onFileHasTobeSaved() {
  emit reqSaveProject(ProjectFileHandler::getInstance()->getFilename());
  WidgetsManager::getInstance()->changeTab(GuiEnums::Tab::TabOutPut);
  emit reqStitcherReload();
}

void SourcesTabWidget::restore() {
  if (stackedWidget->currentWidget() != pageInput && stackedWidget->currentWidget() != pageLoading &&
      stackedWidget->currentWidget() != inputTypePage) {
    stackedWidget->currentWidget()->deleteLater();
  }
  stackedWidget->setCurrentWidget(projectDefinition->isInit() ? pageInput : inputTypePage);
  buttonsBackground->setVisible(projectDefinition->isInit());
  buttonsBackground->raise();
}

void SourcesTabWidget::showLoadingWidget() {
  stackedWidget->setCurrentWidget(pageLoading);
  WidgetsManager::getInstance()->showLoadingDialog(tr("Loading inputs..."), loadingWidget);
}

void SourcesTabWidget::showWidgetEdition(IConfigurationCategory* configurationWidget) {
  connect(configurationWidget, &IConfigurationCategory::saved, [=]() {
    configurationWidget->deleteLater();
    showLoadingWidget();
  });
  connect(configurationWidget, &IConfigurationCategory::reqBack, [=]() {
    configurationWidget->deleteLater();
    restore();
  });
  buttonsBackground->setVisible(false);
  stackedWidget->addWidget(configurationWidget);
  stackedWidget->setCurrentWidget(configurationWidget);
}

void SourcesTabWidget::activateOptions(bool activate) {
  for (auto button : buttonsLayout->findChildren<QPushButton*>()) {
    button->setEnabled(activate);
  }
}

void SourcesTabWidget::onButtonSnapshotClicked() {
  WidgetsManager::getInstance()->showSnapshotDialog(pageInput);
  emit notifyTakeSourcesSnapshot(LiveSettings::getLiveSettings()->getSnapshotPath());
}

void SourcesTabWidget::onButtonConfigure3DRigClicked() {
  if (projectDefinition != nullptr) {
    RigConfigurationWidget* rigConfigurationWidget = new RigConfigurationWidget(this);
    connect(rigConfigurationWidget, &RigConfigurationWidget::notifyRigConfigured, this,
            &SourcesTabWidget::notifyRigConfigured);

    if (projectDefinition->hasRigConfiguration()) {
      rigConfigurationWidget->loadConfiguration(
          projectDefinition->getInputNames(), projectDefinition->getStereoRigConst()->getOrientation(),
          projectDefinition->getStereoRigConst()->getGeometry(), projectDefinition->getStereoRigConst()->getDiameter(),
          projectDefinition->getStereoRigConst()->getIPD(),
          QVector<int>::fromStdVector(projectDefinition->getStereoRigConst()->getLeftInputs()),
          QVector<int>::fromStdVector(projectDefinition->getStereoRigConst()->getRightInputs()));
    } else {
      rigConfigurationWidget->loadConfiguration(
          projectDefinition->getInputNames(), VideoStitch::Core::StereoRigDefinition::Orientation::Portrait,
          VideoStitch::Core::StereoRigDefinition::Geometry::Circular, 20.0, 5.5, QVector<int>(), QVector<int>());
    }
    BackgroundContainer* rigConfigContainer =
        new BackgroundContainer(rigConfigurationWidget, tr("Stereoscopic rig configuration:"),
                                WidgetsManager::getInstance()->getMainWindowRef());
    connect(rigConfigurationWidget, &RigConfigurationWidget::notifyRigConfigured, rigConfigContainer,
            &BackgroundContainer::notifyWidgetClosed);
    connect(rigConfigContainer, &BackgroundContainer::notifyWidgetClosed, this, [=]() {
      rigConfigContainer->hide();
      rigConfigurationWidget->hide();
      rigConfigContainer->deleteLater();
      rigConfigurationWidget->deleteLater();
    });
    rigConfigurationWidget->show();
    rigConfigContainer->show();
    rigConfigContainer->raise();
  }
}

void SourcesTabWidget::onRigConfigurationSuccess() {
  GenericDialog* successDialog =
      new GenericDialog(tr("Rig configured"), tr("Rig configuration was successfully modified"),
                        GenericDialog::DialogMode::ACCEPT, WidgetsManager::getInstance()->getMainWindowRef());
  successDialog->show();
}

void SourcesTabWidget::changeState(GUIStateCaps::State state) {
  if (state == GUIStateCaps::State::idle) {
    WidgetsManager::getInstance()->closeLoadingDialog();
    stackedWidget->setCurrentWidget(inputTypePage);
    buttonsBackground->hide();
    activateOptions(false);
  } else if (state == GUIStateCaps::disabled) {
    showLoadingWidget();
    activateOptions(false);
  } else {
    stackedWidget->setCurrentWidget(pageInput);
    WidgetsManager::getInstance()->closeLoadingDialog();
    activateOptions(true);
  }
}

void SourcesTabWidget::setProject(ProjectDefinition* project) {
  projectDefinition = qobject_cast<LiveProjectDefinition*>(project);
  restore();
}

void SourcesTabWidget::clearProject() { projectDefinition = nullptr; }
