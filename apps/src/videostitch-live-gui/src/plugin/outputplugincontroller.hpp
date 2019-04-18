// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QObject>
#include "pluginscontroller.hpp"

namespace VideoStitch {
namespace Ptv {
class Value;
}
}  // namespace VideoStitch

class LiveProjectDefinition;
class ProjectDefinition;
class LiveOutputFactory;
class ConfigOutputsWidget;
class BackgroundContainer;
class OutputConfigurationWidget;

class OutputPluginController : public QObject, public PluginsController {
  Q_OBJECT
 public:
  explicit OutputPluginController(ConfigOutputsWidget* widget);

  ~OutputPluginController();

 public slots:
  void setProject(ProjectDefinition* project);
  void onProjectClosed();
  void clearProject();

  void onButtonAddClicked();

  void onCancelAddClicked();

  void onOutputsSelected(const QString displayName, const QString model, const QString pluginType, const bool isUsed);

  void onOutputConfigured();

  void onOutputCancelled();

 private:
  void showErrorDialog(const QString& info);

  void closeOutputContainer();

  void showConfigurationWidget(OutputConfigurationWidget* widget, const QString name);

  ConfigOutputsWidget* configWidget;

  QScopedPointer<BackgroundContainer> currentContainer;
  QScopedPointer<LiveOutputFactory> currentLiveOutput;

  LiveProjectDefinition* projectDefinition;

 signals:
  void reqAddOutput(LiveOutputFactory* output);  // Gives ownership of the output
  void reqUpdateOutput(const QString& id);
  void reqRemoveOutput(const QString& id);
};
