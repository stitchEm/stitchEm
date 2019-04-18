// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputplugincontroller.hpp"

#include "plugin/newoutputwidget.hpp"
#include "configurations/outputconfigurationwidget.hpp"
#include "configurations/configoutputswidget.hpp"
#include "generic/backgroundcontainer.hpp"
#include "videostitcher/liveoutputsdi.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "libvideostitch/ptv.hpp"
#include "generic/genericdialog.hpp"
#include "widgetsmanager.hpp"
#include "staticoutputs.hpp"
#include "livesettings.hpp"

#include "libvideostitch-base/logmanager.hpp"

OutputPluginController::OutputPluginController(ConfigOutputsWidget* widget)
    : configWidget(widget), currentContainer(nullptr), projectDefinition(nullptr) {
  loadStaticOutputs();
  connect(configWidget->getNewOutputWidget(), &NewOutputWidget::notifyBackClicked, this,
          &OutputPluginController::onCancelAddClicked);
  connect(configWidget->getNewOutputWidget(), &NewOutputWidget::notifyDevicesSelected, this,
          &OutputPluginController::onOutputsSelected);
  connect(configWidget, &ConfigOutputsWidget::notifyStartAddingOutput, this,
          &OutputPluginController::onButtonAddClicked);
  connect(configWidget, &ConfigOutputsWidget::reqRemoveOutput, this, &OutputPluginController::reqRemoveOutput);
  connect(configWidget, &ConfigOutputsWidget::reqChangeOutputConfig, this, &OutputPluginController::reqUpdateOutput);
}

OutputPluginController::~OutputPluginController() {}

void OutputPluginController::setProject(ProjectDefinition* project) {
  projectDefinition = qobject_cast<LiveProjectDefinition*>(project);
  configWidget->setProject(projectDefinition);
}

void OutputPluginController::onProjectClosed() { configWidget->removeOutputDevices(); }

void OutputPluginController::clearProject() { projectDefinition = nullptr; }

void OutputPluginController::showErrorDialog(const QString& info) {
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(info);
  GenericDialog* errorDialog =
      new GenericDialog(tr("Add output error"), info, GenericDialog::DialogMode::ACCEPT, configWidget);
  errorDialog->show();
}

void OutputPluginController::closeOutputContainer() {
  configWidget->showOutputList();
  if (currentContainer) {
    currentContainer->hide();
    delete currentContainer->getContainedWidget();
    currentContainer.reset();
  }
}

void OutputPluginController::showConfigurationWidget(OutputConfigurationWidget* widget, const QString name) {
  Q_ASSERT(widget != nullptr);
  widget->setProject(projectDefinition);
  widget->changeMode(IConfigurationCategory::Mode::CreationInPopup);
  connect(widget, &IConfigurationCategory::saved, this, &OutputPluginController::onOutputConfigured);
  currentContainer.reset(new BackgroundContainer(widget, tr("Create output: %0").arg(name), configWidget));
  connect(currentContainer.data(), &BackgroundContainer::notifyWidgetClosed, widget, &IConfigurationCategory::reqBack);
  connect(widget, &IConfigurationCategory::reqBack, this, &OutputPluginController::onOutputCancelled);
  currentContainer->show();
  currentContainer->raise();
}

void OutputPluginController::onButtonAddClicked() {
  NewOutputWidget* newOutputWidget = configWidget->getNewOutputWidget();
  newOutputWidget->clearDevices();

  // Add a group of static output devices
  DeviceList staticDevices;
  staticDevices.push_back(deviceHDD);
  staticDevices.push_back(deviceRTMP);
  // VSA-6594
#ifdef ENABLE_YOUTUBE_OUTPUT
  staticDevices.push_back(deviceYoutube);
#endif
  staticDevices.push_back(deviceOculus);
#if defined(Q_OS_WIN)
  staticDevices.push_back(deviceSteamVR);
#endif
  for (const VideoStitch::Plugin::DiscoveryDevice& device : staticDevices) {
    newOutputWidget->insertDeviceItem(QString::fromStdString(device.displayName), QString(),
                                      QString::fromStdString(device.name), false);
  }

  // Add devices available for each plugin
  QStringList availablePlugins;
  loadPluginList(availablePlugins);
  for (QString pluginName : availablePlugins) {
    DeviceList availableDevices;
    listOutputDevicesFromPlugin(pluginName, availableDevices);
    for (const VideoStitch::Plugin::DiscoveryDevice& device : availableDevices) {
      // ignore audio devices
      if (device.mediaType != VideoStitch::Plugin::DiscoveryDevice::MediaType::AUDIO) {
        const bool isUsed = projectDefinition->isDeviceInUse(pluginName, QString::fromStdString(device.name));
        newOutputWidget->insertDeviceItem(QString::fromStdString(device.displayName),
                                          QString::fromStdString(device.name), pluginName, isUsed);
      }
    }
  }
}

void OutputPluginController::onCancelAddClicked() { configWidget->showOutputList(); }

void OutputPluginController::onOutputsSelected(const QString displayName, const QString model, const QString pluginType,
                                               const bool isUsed) {
  if (isUsed) {
    // This can trigger other actions like messages or dialogs
    return;
  }

  const VideoStitch::OutputFormat::OutputFormatEnum type = VideoStitch::OutputFormat::getEnumFromString(pluginType);
  currentLiveOutput.reset(LiveOutputFactory::createOutput(VideoStitch::Ptv::Value::emptyObject(), type,
                                                          &projectDefinition->getDelegate()->getPanoDefinition()));
  LiveOutputSDI* sdiLiveOutput = qobject_cast<LiveOutputSDI*>(currentLiveOutput.data());
  if (sdiLiveOutput) {
    sdiLiveOutput->setDeviceName(model);
    sdiLiveOutput->setDeviceDisplayName(displayName);
  }

  OutputConfigurationWidget* widget = currentLiveOutput->createConfigurationWidget(configWidget);
  if (widget) {
    showConfigurationWidget(widget, displayName);
  } else {  // No configuration needed
    if (projectDefinition->canAddOutput(currentLiveOutput.data())) {
      emit reqAddOutput(currentLiveOutput.take());  // We give the ownership of the live output
      closeOutputContainer();
    } else {
      showErrorDialog(
          tr("An output with name: %0 is already in the output list").arg(currentLiveOutput->getIdentifier()));
      currentLiveOutput.reset();
    }
  }
}

void OutputPluginController::onOutputConfigured() {
  if (projectDefinition->canAddOutput(currentLiveOutput.data())) {
    emit reqAddOutput(currentLiveOutput.take());  // We give the ownership of the live output
    closeOutputContainer();
  } else {
    showErrorDialog(
        tr("An output with name: %0 is already in the output list").arg(currentLiveOutput->getIdentifier()));
  }
}

void OutputPluginController::onOutputCancelled() {
  currentLiveOutput.reset();
  closeOutputContainer();
}
