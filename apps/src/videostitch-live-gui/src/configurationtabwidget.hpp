// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_configurationtabwidget.h"

#include "generic/generictablewidget.hpp"
#include "configurations/configurationtablewidget.hpp"
#include "configurations/iconfigurationvalue.hpp"
#include "configurations/configoutputswidget.hpp"
#include "utils/maintabs.hpp"

#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"

#include <QMap>
#include <QPair>
#include <QFrame>

typedef QMap<ConfigIdentifier, QWidget*> PageMap;

class ConfigOutputsWidget;

class ConfigurationTabWidget : public QFrame, public Ui::ConfigurationTabWidgetClass {
  Q_OBJECT

 public:
  explicit ConfigurationTabWidget(QWidget* const parent = nullptr);
  ~ConfigurationTabWidget();

  void restore();
  void toggleOutput(const QString& id);
  void changeConfigurationPage(const ConfigIdentifier identifier);
  ConfigOutputsWidget* getConfigOutputs() const;

 private:
  void initializePageMap();
  void addConfigValues();
  void addSingleConfigurationValue(ConfigIdentifier identifier, const QString& title, IConfigurationCategory* widget,
                                   bool addInTable = true);

  GenericTableWidget* tableWidget;
  PageMap pageMap;
  LiveProjectDefinition* projectDefinition;

 public slots:
  void showMainPage();
  void setProject(ProjectDefinition*);
  void clearProject();
  void onFileHasTobeSaved();

 private slots:
  void onConfigItemSelected(int row, int col);

 signals:
  void reqStitcherReload(SignalCompressionCaps* comp = nullptr);
  void reqReloadProject();
  void reqSaveProject(QString, const VideoStitch::Ptv::Value* = nullptr);
  void injectProject(LiveProjectDefinition* project);
  void projectCleared();
  void injectStitcher(LiveVideoStitcher* stitcher);
};
