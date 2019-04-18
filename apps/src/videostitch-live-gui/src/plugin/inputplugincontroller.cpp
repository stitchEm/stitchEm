// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputplugincontroller.hpp"

#include "guiconstants.hpp"
#include "sourcestabwidget.hpp"
#include "configurations/ajainputconfiguration.hpp"
#include "configurations/audioinputconfigurationwidget.hpp"
#include "configurations/capturecardinputconfiguration.hpp"
#include "configurations/configurationinputprocedural.hpp"
#include "configurations/configurationinputmedia.hpp"
#include "configurations/inputconfigurationwidget.hpp"
#include "generic/genericdialog.hpp"
#include "utils/pixelformat.hpp"
#include "utils/displaymode.hpp"
#include "videostitcher/liveinputdecklink.hpp"
#include "videostitcher/liveinputmagewell.hpp"
#include "videostitcher/liveinputaja.hpp"
#include "videostitcher/liveinputprocedural.hpp"
#include "videostitcher/liveinputfile.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "multiinputconfiguration.hpp"
#include "libvideostitch-base/common-config.hpp"

#include "libvideostitch/ptv.hpp"

InputPluginController::InputPluginController(SourcesTabWidget* widget)
    : sourcesWidget(widget), projectDefinition(nullptr) {
  initializePlugins();
  qRegisterMetaType<LiveInputList>("LiveInputList");
  qRegisterMetaType<std::shared_ptr<LiveInputFactory> >("std::shared_ptr<LiveInput>");
  qRegisterMetaType<AudioConfiguration>("AudioConfiguration");
  connect(sourcesWidget->buttonEditVideo, &QPushButton::clicked, this,
          &InputPluginController::onButtonEditVideoClicked);
  connect(sourcesWidget->buttonEditAudio, &QPushButton::clicked, this,
          &InputPluginController::onButtonEditAudioClicked);
  connect(sourcesWidget->inputTypePage, &InputTypeWidget::inputTypeSelected, this,
          &InputPluginController::displayInputTypeConfiguration);
}

InputPluginController::~InputPluginController() {}

void InputPluginController::setProject(ProjectDefinition* project) {
  projectDefinition = qobject_cast<LiveProjectDefinition*>(project);
}

void InputPluginController::clearProject() { projectDefinition = nullptr; }

void InputPluginController::showDialog(const QString& message) {
  GenericDialog* errorDialog =
      new GenericDialog(tr("Edit Inputs"), message, GenericDialog::DialogMode::ACCEPT, sourcesWidget);
  errorDialog->show();
}

void InputPluginController::displayInputTypeConfiguration(VideoStitch::InputFormat::InputFormatEnum inputType) {
  std::shared_ptr<const LiveInputFactory> liveInputTemplate(LiveInputFactory::makeLiveInput(inputType, QString()));
  showConfiguration(liveInputTemplate, IConfigurationCategory::Mode::CreationInStack);
}

void InputPluginController::onButtonEditVideoClicked() {
  Q_ASSERT(projectDefinition && projectDefinition->isInit());

  std::shared_ptr<const LiveInputFactory> liveInputTemplate(projectDefinition->retrieveConfigurationInput());
  showConfiguration(liveInputTemplate, IConfigurationCategory::Mode::Edition);
}

void InputPluginController::onButtonEditAudioClicked() {
  Q_ASSERT(projectDefinition && projectDefinition->isInit());

  AudioInputConfigurationWidget* configurationWidget = new AudioInputConfigurationWidget();
  configurationWidget->setPluginsController(this);
  configurationWidget->setProject(projectDefinition);

  connect(configurationWidget, &IConfigurationCategory::saved,
          [=]() { emit notifyConfigureAudioInput(configurationWidget->getAudioConfiguration()); });

  configurationWidget->changeMode(IConfigurationCategory::Mode::Edition);
  sourcesWidget->showWidgetEdition(configurationWidget);
}

void InputPluginController::applyExistingInputTemplate(const QStringList selected,
                                                       const VideoStitch::InputFormat::InputFormatEnum type,
                                                       const VideoStitch::Ptv::Value* templateInput,
                                                       LiveInputList& list) {
  for (const QString& name : selected) {
    std::shared_ptr<LiveInputFactory> liveInput(LiveInputFactory::makeLiveInput(type, templateInput));
    if (type != VideoStitch::InputFormat::InputFormatEnum::PROCEDURAL) {
      liveInput->setName(name);
    }
    list.push_back(liveInput);
  }
}

InputConfigurationWidget* InputPluginController::createConfigurationWidget(
    std::shared_ptr<const LiveInputFactory> liveInputTemplate) {
  InputConfigurationWidget* configurationWidget = nullptr;
  switch (liveInputTemplate->getType()) {
    case VideoStitch::InputFormat::InputFormatEnum::PROCEDURAL: {
      configurationWidget = new ConfigurationInputProcedural(
          std::static_pointer_cast<const LiveInputProcedural>(liveInputTemplate), sourcesWidget);
      break;
    }
    case VideoStitch::InputFormat::InputFormatEnum::MEDIA: {
      configurationWidget =
          new ConfigurationInputMedia(std::static_pointer_cast<const LiveInputFile>(liveInputTemplate), sourcesWidget);
      break;
    }
    case VideoStitch::InputFormat::InputFormatEnum::MAGEWELL:
    case VideoStitch::InputFormat::InputFormatEnum::MAGEWELLPRO:
    case VideoStitch::InputFormat::InputFormatEnum::DECKLINK:
    case VideoStitch::InputFormat::InputFormatEnum::V4L2:
    case VideoStitch::InputFormat::InputFormatEnum::XIMEA: {
      configurationWidget = new CaptureCardInputConfiguration(
          std::static_pointer_cast<const CaptureCardLiveInput>(liveInputTemplate), sourcesWidget);
      break;
    }
    case VideoStitch::InputFormat::InputFormatEnum::AJA: {
      configurationWidget =
          new AjaInputConfiguration(std::static_pointer_cast<const LiveInputAJA>(liveInputTemplate), sourcesWidget);
      break;
    }
    case VideoStitch::InputFormat::InputFormatEnum::NETWORK: {
      MultiInputConfiguration* widget = new MultiInputConfiguration(
          std::static_pointer_cast<const LiveInputStream>(liveInputTemplate), sourcesWidget);
      connect(widget, &MultiInputConfiguration::notifyTestActivated, this, &InputPluginController::notifyTestActivated);
      connect(this, &InputPluginController::notifyTestResult, widget, &MultiInputConfiguration::onInputTestResult);
      configurationWidget = widget;
      break;
    }
    default:
      return nullptr;
  }

  configurationWidget->setPluginsController(this);
  configurationWidget->setProject(projectDefinition);

  connect(configurationWidget, &IConfigurationCategory::saved,
          [=]() { emit notifyConfigureInputs(configurationWidget->getEditedInputs()); });
  return configurationWidget;
}

void InputPluginController::showConfiguration(std::shared_ptr<const LiveInputFactory> liveInputTemplate,
                                              IConfigurationCategory::Mode mode) {
  InputConfigurationWidget* configurationWidget = createConfigurationWidget(liveInputTemplate);
  if (configurationWidget) {
    configurationWidget->changeMode(mode);
    sourcesWidget->showWidgetEdition(configurationWidget);
  }
}

void InputPluginController::onConfiguringInputsSuccess(const QString& message) {
  showDialog(message);
  sourcesWidget->restore();
}

void InputPluginController::onConfiguringInputsError(const QString& message) {
  showDialog(message);
  sourcesWidget->restore();
}
