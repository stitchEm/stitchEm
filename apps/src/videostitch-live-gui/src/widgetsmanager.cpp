// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "widgetsmanager.hpp"

#include "guiconstants.hpp"
#include "maintabwidget.hpp"
#include "mainwindow.hpp"
#include "generic/genericloader.hpp"

WidgetsManager::WidgetsManager()
    : QObject(new QObject()),
      loadingDialog(nullptr),
      mainWindowRef(nullptr),
      tabWidget(nullptr),
      snapshotDialog(nullptr),
      snapshotShowTimer(nullptr) {
  obtainMainWindowRef();
  obtainTabsRef();
  connect(mainWindowRef, &MainWindow::notifySizeChanged, this, &WidgetsManager::onMainWindowResized);
}

void WidgetsManager::showLoadingDialog(const QString& message, QWidget* parent) {
  if (loadingDialog == nullptr) {
    parent = parent == nullptr ? mainWindowRef : parent;
    loadingDialog = new GenericLoader(message, parent);
    loadingDialog->show();
    loadingDialog->raise();
    loadingDialog->updateSize(parent->size().width(), parent->size().height());
  }
}

void WidgetsManager::closeLoadingDialog() {
  if (loadingDialog != nullptr) {
    loadingDialog->close();
    loadingDialog = nullptr;
  }
}

void WidgetsManager::showSnapshotDialog(QWidget* parent) {
  if (snapshotDialog == nullptr) {
    parent = parent == nullptr ? mainWindowRef : parent;
    snapshotShowTimer = new QTimer(this);
    connect(snapshotShowTimer, &QTimer::timeout, this, &WidgetsManager::onSnapTimerFinished);
    snapshotDialog = new GenericLoader("Snap!", parent);
    snapshotDialog->show();
    snapshotDialog->raise();
    snapshotDialog->updateSize(parent->size().width(), parent->size().height());
    snapshotShowTimer->start(SNAP_SHOW_DIALOG);
  }
}

void WidgetsManager::changeTab(GuiEnums::Tab tab) {
  if (tabWidget != nullptr) {
    tabWidget->changeCurrentTab(tab);
  }
}

QWidget* WidgetsManager::currentTab() {
  if (tabWidget != nullptr) {
    return tabWidget->currentWidget();
  } else {
    return nullptr;
  }
}

void WidgetsManager::showConfiguration(const ConfigIdentifier config) {
  changeTab(GuiEnums::Tab::TabConfiguration);
  tabWidget->configurationTabWidget->changeConfigurationPage(config);
}

void WidgetsManager::activateSourcesTab(bool active) {
  if (tabWidget != nullptr) tabWidget->activateTab(GuiEnums::Tab::TabSources, active);
}

QWidget* WidgetsManager::getMainWindowRef() const { return mainWindowRef; }

void WidgetsManager::obtainMainWindowRef() {
  for (QWidget* widget : QApplication::topLevelWidgets()) {
    if (widget->objectName() == "MainWindowClass") mainWindowRef = qobject_cast<MainWindow*>(widget);
  }
}

void WidgetsManager::obtainTabsRef() {
  for (QWidget* widget : QApplication::allWidgets()) {
    if (widget->objectName() == "projectTabs") tabWidget = qobject_cast<MainTabWidget*>(widget);
  }
}

ProjectWorkWidget* WidgetsManager::getProjectWorkWidget() const { return mainWindowRef->getProjectWorkWidget(); }

void WidgetsManager::onMainWindowResized(const QSize& size) {
  if (loadingDialog != nullptr) loadingDialog->updateSize(size.width(), size.height());
}

void WidgetsManager::onSnapTimerFinished() {
  if (snapshotDialog != nullptr) {
    snapshotDialog->close();
    delete snapshotShowTimer;
    snapshotDialog = nullptr;
    snapshotShowTimer = nullptr;
  }
}
