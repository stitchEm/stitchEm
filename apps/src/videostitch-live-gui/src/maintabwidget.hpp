// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "configurationtabwidget.hpp"
#include "interactivetabwidget.hpp"
#include "outputtabwidget.hpp"
#include "sourcestabwidget.hpp"
#include "utils/maintabs.hpp"

#include <QTabWidget>

class MainTabWidget : public QTabWidget, public GUIStateCaps {
  Q_OBJECT
 public:
  explicit MainTabWidget(QWidget* const parent = nullptr);

  void addAllTabs(LiveStitcherController* liveVideoStitcher, StitcherController* videoStitcher);
  void changeCurrentTab(const GuiEnums::Tab tab);
  void activateTab(const GuiEnums::Tab tab, bool activate);

  SourcesTabWidget* sourcesTabWidget;
  OutPutTabWidget* outPutTabWidget;
  InteractiveTabWidget* interactiveTabWidget;
  ConfigurationTabWidget* configurationTabWidget;

 signals:
  void reqChangeState(GUIStateCaps::State s);

 protected slots:
  virtual void changeState(GUIStateCaps::State state);

 private slots:
  void onTabChanged(int tab);

 private:
  void addMainTab(const GuiEnums::Tab index, QWidget* widget);
  void addMainTabs();
};
