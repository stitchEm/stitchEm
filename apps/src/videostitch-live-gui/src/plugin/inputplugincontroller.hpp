// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "pluginscontroller.hpp"
#include "configurations/iconfigurationvalue.hpp"
#include "videostitcher/audioconfiguration.hpp"
#include "videostitcher/liveinputfactory.hpp"
#include "libvideostitch/plugin.hpp"

#include <memory>

class InputConfigurationWidget;
class SourcesTabWidget;
class ProjectDefinition;
class LiveProjectDefinition;

class InputPluginController : public QObject, public PluginsController {
  Q_OBJECT
 public:
  /**
   * @brief Constructor
   * @param widget The sources tab widget reference
   */
  explicit InputPluginController(SourcesTabWidget* widget);

  /**
   * @brief Destructor
   */
  ~InputPluginController();

 public slots:
  void setProject(ProjectDefinition* project);
  void clearProject();

  void onConfiguringInputsSuccess(const QString& message);
  void onConfiguringInputsError(const QString& message);

 signals:
  void notifyTestActivated(const int id, const QString name);
  void notifyTestResult(const int id, bool success, qint64 width, qint64 height);
  void notifyConfigureInputs(const LiveInputList inputs);
  void notifyConfigureAudioInput(AudioConfiguration audioConfiguration);

 private:
  void showDialog(const QString& message);
  void applyExistingInputTemplate(const QStringList selected,
                                  const VideoStitch::InputFormat::InputFormatEnum selectedType,
                                  const VideoStitch::Ptv::Value* templateInput, LiveInputList& list);
  InputConfigurationWidget* createConfigurationWidget(std::shared_ptr<const LiveInputFactory> liveInputTemplate);
  void showConfiguration(std::shared_ptr<const LiveInputFactory> liveInputTemplate, IConfigurationCategory::Mode mode);

 private slots:
  void displayInputTypeConfiguration(VideoStitch::InputFormat::InputFormatEnum inputType);
  void onButtonEditVideoClicked();
  void onButtonEditAudioClicked();

 private:
  SourcesTabWidget* sourcesWidget;
  LiveProjectDefinition* projectDefinition;
};
