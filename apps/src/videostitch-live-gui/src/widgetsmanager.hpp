// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef WIDGETSMANAGER_HPP
#define WIDGETSMANAGER_HPP

#include "libvideostitch-base/singleton.hpp"
#include "configurationtabwidget.hpp"

class GenericLoader;
class MainTabWidget;
class MainWindow;
class ProjectWorkWidget;

class WidgetsManager : public QObject, public Singleton<WidgetsManager> {
  friend class Singleton<WidgetsManager>;
  Q_OBJECT
 public:
  void showLoadingDialog(const QString& message, QWidget* parent = nullptr);

  void showSnapshotDialog(QWidget* parent = nullptr);

  void closeLoadingDialog();

  QWidget* currentTab();
  void changeTab(GuiEnums::Tab tab);

  void showConfiguration(const ConfigIdentifier config);

  void activateSourcesTab(bool active);

  QWidget* getMainWindowRef() const;
  ProjectWorkWidget* getProjectWorkWidget() const;

 private:
  WidgetsManager();
  void obtainMainWindowRef();
  void obtainTabsRef();

  GenericLoader* loadingDialog;

  MainWindow* mainWindowRef;

  MainTabWidget* tabWidget;

  GenericLoader* snapshotDialog;

  QTimer* snapshotShowTimer;

 public slots:
  void onMainWindowResized(const QSize& size);

  void onSnapTimerFinished();
};

#endif  // WIDGETSMANAGER_HPP
