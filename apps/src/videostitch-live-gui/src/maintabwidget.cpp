// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "maintabwidget.hpp"
#include "guiconstants.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"

#include <QTabBar>
#include <QLabel>
#include <QHBoxLayout>
#include <QPixmap>
#include "maintabwidget.hpp"
#include "guiconstants.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"

MainTabWidget::MainTabWidget(QWidget* const parent)
    : QTabWidget(parent),
      sourcesTabWidget(nullptr),
      outPutTabWidget(new OutPutTabWidget(this)),
      interactiveTabWidget(new InteractiveTabWidget(this)),
      configurationTabWidget(new ConfigurationTabWidget(this)) {
  tabBar()->setProperty("vs-button-bar", true);
  setTabPosition(West);
  setFocusPolicy(Qt::NoFocus);
  connect(this, &MainTabWidget::currentChanged, this, &MainTabWidget::onTabChanged);
  StateManager::getInstance()->registerObject(this);
}

void MainTabWidget::addMainTabs() {
  if (tabBar()->count() == 0) {
    addMainTab(GuiEnums::Tab::TabSources, sourcesTabWidget);
    addMainTab(GuiEnums::Tab::TabOutPut, outPutTabWidget);
    addMainTab(GuiEnums::Tab::TabInteractive, interactiveTabWidget);
    addMainTab(GuiEnums::Tab::TabConfiguration, configurationTabWidget);
  }
}

void MainTabWidget::addAllTabs(LiveStitcherController* liveVideoStitcher, StitcherController* videoStitcher) {
  connect(outPutTabWidget, &OutPutTabWidget::reqResetPanorama, liveVideoStitcher,
          &LiveStitcherController::onResetPanorama);
  connect(outPutTabWidget, &OutPutTabWidget::notifyTakePanoSnapshot, videoStitcher,
          &StitcherController::onSnapshotPanorama);
  connect(outPutTabWidget, &OutPutTabWidget::reqStitcherReload, videoStitcher, &StitcherController::onReset);
  connect(outPutTabWidget, &OutPutTabWidget::reqSaveProject, videoStitcher, &StitcherController::saveProject);
  connect(videoStitcher, &StitcherController::stitcherClosed, outPutTabWidget->getControlsBar(),
          &OutputControlsPanel::onProjectClosed);
  connect(videoStitcher, &StitcherController::notifyInvalidPano, outPutTabWidget, &OutPutTabWidget::onInvalidPano);

  if (!sourcesTabWidget) {
    sourcesTabWidget = new SourcesTabWidget(this);
  }
  connect(sourcesTabWidget, &SourcesTabWidget::notifyTakeSourcesSnapshot, videoStitcher,
          &StitcherController::onSnapshotSources);
  connect(sourcesTabWidget, &SourcesTabWidget::reqStitcherReload, liveVideoStitcher, &LiveStitcherController::reset);
  connect(sourcesTabWidget, &SourcesTabWidget::reqSaveProject, videoStitcher, &StitcherController::saveProject);
  connect(liveVideoStitcher, &LiveStitcherController::notifyRigConfigureSuccess, sourcesTabWidget,
          &SourcesTabWidget::onRigConfigurationSuccess);
  connect(sourcesTabWidget, &SourcesTabWidget::notifyRigConfigured, liveVideoStitcher,
          &LiveStitcherController::configureRig);

  connect(configurationTabWidget, &ConfigurationTabWidget::reqStitcherReload, videoStitcher,
          &StitcherController::onReset);
  connect(configurationTabWidget, &ConfigurationTabWidget::reqSaveProject, videoStitcher,
          &StitcherController::saveProject);

  addMainTabs();
}

void MainTabWidget::changeCurrentTab(const GuiEnums::Tab tab) { setCurrentIndex(int(tab)); }

void MainTabWidget::changeState(GUIStateCaps::State state) {
  bool isInitialized = state == GUIStateCaps::State::stitch;
  activateTab(GuiEnums::Tab::TabSources, true);
  activateTab(GuiEnums::Tab::TabOutPut, isInitialized);
  activateTab(GuiEnums::Tab::TabInteractive, isInitialized);
  activateTab(GuiEnums::Tab::TabConfiguration, isInitialized);
  if (!isInitialized) {
    setCurrentIndex(int(GuiEnums::Tab::TabSources));
  }
}

void MainTabWidget::addMainTab(const GuiEnums::Tab index, QWidget* widget) {
  QLabel* tabHeader(new QLabel(this));
  addTab(tabHeader, QString());
  tabHeader->setObjectName("tabHeader");
  QHBoxLayout* tabLayout(new QHBoxLayout(tabHeader));
  tabLayout->setSpacing(0);
  tabLayout->setContentsMargins(0, 0, 0, 0);
  tabLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);
  tabLayout->addWidget(widget);

  QLabel* const tabIcon(new QLabel(tabHeader));
  tabIcon->setScaledContents(true);
  tabIcon->setObjectName("tabIcon");

  QLabel* const tabTitle(new QLabel(GuiEnums::getTabName(index)));
  tabTitle->setFocusPolicy(Qt::NoFocus);
  tabTitle->setWordWrap(true);
  tabTitle->setObjectName("tabTitle");
  tabTitle->setAlignment(Qt::AlignHCenter);

  QWidget* const tabBack(new QWidget());
  tabBack->setFixedSize(TAB_SIZE, TAB_SIZE);

  QVBoxLayout* const layout(new QVBoxLayout(tabBack));
  layout->setSpacing(0);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->addWidget(tabIcon);
  layout->addWidget(tabTitle);

  tabBar()->setTabButton(int(index), QTabBar::LeftSide, tabBack);
}

void MainTabWidget::activateTab(const GuiEnums::Tab tab, bool activate) {
  QString iconFileName = GuiEnums::getPixmapPath(tab, activate);
  QString tooltipText;
  if (!activate && tab == GuiEnums::Tab::TabSources) {
    tooltipText = tr("Disable the output or algorithm to access Sources tab");
  }

  tabBar()->tabButton(int(tab), QTabBar::LeftSide)->setEnabled(activate);
  tabBar()->tabButton(int(tab), QTabBar::LeftSide)->setToolTip(tooltipText);
  setTabEnabled(int(tab), activate);
  qobject_cast<QLabel*>(tabBar()->tabButton(int(tab), QTabBar::LeftSide)->layout()->itemAt(0)->widget())
      ->setPixmap(QPixmap(iconFileName));
}

void MainTabWidget::onTabChanged(int tab) {
  if (GuiEnums::Tab(tab) != GuiEnums::Tab::TabSources) {
    sourcesTabWidget->restore();
  }
  if (GuiEnums::Tab(tab) != GuiEnums::Tab::TabOutPut) {
    outPutTabWidget->restore();
  }
  if (GuiEnums::Tab(tab) != GuiEnums::Tab::TabConfiguration) {
    configurationTabWidget->restore();
  }

  if (GuiEnums::Tab(tab) == GuiEnums::Tab::TabOutPut) {
    outPutTabWidget->setOutputWidgetReference(outPutTabWidget->getControlsBar());
  } else if (GuiEnums::Tab(tab) == GuiEnums::Tab::TabInteractive) {
    interactiveTabWidget->setOutputWidgetReference(outPutTabWidget->getControlsBar());
  }
}
