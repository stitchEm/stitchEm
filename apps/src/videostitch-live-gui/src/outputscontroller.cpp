// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputscontroller.hpp"

#include "generic/genericdialog.hpp"
#include "videostitcher/liveoutputlist.hpp"
#include "videostitcher/liveoutputoculus.hpp"
#include "configurationtabwidget.hpp"
#include "widgetsmanager.hpp"
#include "livesettings.hpp"
#include "outputtabwidget.hpp"
#include "topinformationbarwidget.hpp"

#include "libvideostitch-gui/mainwindow/vssettings.hpp"

OutputsController::OutputsController(TopInformationBarWidget* topBar, OutPutTabWidget* outputTab,
                                     ConfigurationTabWidget* configTab)
    : informationBarRef(topBar), outputTabRef(outputTab), configTabRef(configTab), projectDefinition(nullptr) {
  connect(outputTabRef, &OutPutTabWidget::notifyOutputActivated, this,
          &OutputsController::onOutputActivatedFromSideBar);
  connect(configTabRef->getConfigOutputs(), &ConfigOutputsWidget::notifyOutputIdChanged, this,
          &OutputsController::onOutputChanged);
  connect(this, &OutputsController::reqActivateOutput, this, &OutputsController::toggleOutputTimer);
  connect(outputTabRef->getControlsBar(), &OutputControlsPanel::notifyStartAddingOutput,
          configTabRef->getConfigOutputs(), &ConfigOutputsWidget::onButtonAddOutputClicked);
  connect(outputTabRef->getControlsBar(), &OutputControlsPanel::notifyConfigureOutput, configTabRef->getConfigOutputs(),
          &ConfigOutputsWidget::configureOutputById);
}

void OutputsController::setProject(ProjectDefinition* project) {
  projectDefinition = qobject_cast<LiveProjectDefinition*>(project);
}

void OutputsController::onOutputChanged(const QString oldName, const QString newName) {
  outputTabRef->updateOutputId(oldName, newName);
}

void OutputsController::clearProject() {
  QMetaObject::invokeMethod(this, "disableOutputsUi", Qt::QueuedConnection);
  projectDefinition = nullptr;
}

void OutputsController::stopTimers() {
  for (auto timer : startingOutputs.values()) {
    timer->stop();
  }
  startingOutputs.clear();
}

void OutputsController::stopTimer(const QString& id) {
  if (startingOutputs.contains(id)) {
    startingOutputs[id]->stop();
    startingOutputs.remove(id);
  }
}

void OutputsController::disableOutputsUi() {
  stopTimers();

  if (!projectDefinition || !projectDefinition->getOutputConfigs()) {
    return;
  }

  for (const auto& output : projectDefinition->getOutputConfigs()->getValues()) {
    if (output->getOutputState() != LiveOutputFactory::OutputState::DISABLED) {
      onWriterRemoved(output->getIdentifier());
    }
  }
}

void OutputsController::onOutputDisconnected(const QString& outputId) {
  if (!projectDefinition) {
    return;
  }

  auto output = projectDefinition->getOutputById(outputId);
  if (startingOutputs.contains(outputId) || !output ||
      output->getOutputState() == LiveOutputFactory::OutputState::DISABLED) {
    return;
  }

  startingOutputs[outputId].reset(new QTimer());
  startingOutputs[outputId]->start(LiveSettings::getLiveSettings()->getOutputReconnectionTimeout());

  connect(startingOutputs[outputId].get(), &QTimer::timeout, this,
          [this, outputId]() { outputConnectionTimerExpired(outputId, true); });
}

void OutputsController::onOutputConnected(const QString& outputId) {
  if (startingOutputs.contains(outputId)) {
    startingOutputs[outputId]->stop();
    startingOutputs.remove(outputId);
    connectionDialog.reset();
  }
}

void OutputsController::toggleOutputTimer(const QString& outputId) {
  if (startingOutputs.contains(outputId)) {
    startingOutputs[outputId]->stop();
    startingOutputs.remove(outputId);
    return;
  }
  auto output = projectDefinition->getOutputById(outputId);

  if (!output || output->getOutputState() == LiveOutputFactory::OutputState::ENABLED) {
    return;
  }

  startingOutputs[outputId].reset(new QTimer());
  startingOutputs[outputId]->start(LiveSettings::getLiveSettings()->getOutputConnectionTimeout());

  connect(startingOutputs[outputId].get(), &QTimer::timeout, this,
          [this, outputId]() { outputConnectionTimerExpired(outputId); });
}

void OutputsController::outputConnectionTimerExpired(const QString& outputId, bool reconnectionMode) {
  startingOutputs[outputId]->stop();

  auto output = projectDefinition->getOutputById(outputId);
  if (output && (output->getOutputState() != LiveOutputFactory::OutputState::ENABLED || reconnectionMode)) {
    connectionDialog = std::unique_ptr<GenericDialog, std::function<void(GenericDialog*)>>(
        new GenericDialog(
            tr("Output timeout"),
            tr("Output %0 activation failed. Press OK to continue trying, otherwise - press Cancel to stop the output")
                .arg(outputId),
            GenericDialog::DialogMode::ACCEPT_CANCEL, WidgetsManager::getInstance()->getMainWindowRef()),
        [](GenericDialog* dataPtr) { dataPtr->close(); });

    connect(connectionDialog.get(), &GenericDialog::notifyAcceptClicked, this, [this, outputId]() {
      connectionDialog.reset();
      startingOutputs[outputId]->start(LiveSettings::getLiveSettings()->getOutputConnectionTimeout());
    });

    connect(connectionDialog.get(), &GenericDialog::notifyCancelClicked, this, [this, outputId]() {
      emit reqToggleOutput(outputId);

      startingOutputs.remove(outputId);
      configTabRef->toggleOutput(outputId);
      outputTabRef->toggleOutput(outputId);
      outputTabRef->setOutputActionable(outputId, false);
      connectionDialog.reset();
    });

    connectionDialog->show();
    connectionDialog->raise();
  } else {
    // enable onOutputDisconnected timeout after unhandled connection timeout
    startingOutputs.remove(outputId);
  }
}

void OutputsController::onOutputCreated(const QString& id) {
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
      QCoreApplication::translate("OutputsController", "Output activated: %1").arg(id));
  informationBarRef->onActivateOutputs(id);
  informationBarRef->showLoading(false);
  emit outputActivationChanged();
}

void OutputsController::onWriterCreated(const QString& id) { outputTabRef->setOutputActionable(id, true); }

void OutputsController::onWriterRemoved(const QString& id) {
  informationBarRef->onDeactivateOutputs(id);
  informationBarRef->showLoading(false);
  outputTabRef->setOutputActionable(id, true);
  stopTimer(id);
  emit outputActivationChanged();
}

void OutputsController::onOutputActivatedFromSideBar(const QString& id) {
  LiveOutputFactory* output = projectDefinition->getOutputById(id);
  QString message;
  if (output->getOutputState() == LiveOutputFactory::OutputState::DISABLED &&
      !output->checkIfIsActivable(&projectDefinition->getDelegate()->getPanoDefinition(), message)) {
    GenericDialog* errorDialog(new GenericDialog(tr("Output activation"), message, GenericDialog::DialogMode::ACCEPT,
                                                 WidgetsManager::getInstance()->getMainWindowRef()));
    errorDialog->show();
    outputTabRef->toggleOutput(id);
    outputTabRef->setOutputActionable(id, true);
    configTabRef->toggleOutput(id);
    return;
  }

  lastActivated = id;
  configTabRef->toggleOutput(id);
  emit reqActivateOutput(id);
}

void OutputsController::onOutputError() {
  // We can recover from an incorrect behavior
  outputTabRef->toggleOutput(lastActivated);
  configTabRef->toggleOutput(lastActivated);
  informationBarRef->onDeactivateOutputs(lastActivated);
  informationBarRef->showLoading(false);
  //  }
}

void OutputsController::onOutputTryingToActivate() { informationBarRef->showLoading(true); }
