// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configoutputswidget.hpp"

#include "guiconstants.hpp"
#include "outputconfigurationwidget.hpp"
#include "outputdetailwidget.hpp"
#include "generic/genericdialog.hpp"
#include "generic/generictablewidget.hpp"
#include "plugin/newoutputwidget.hpp"
#include "videostitcher/liveoutputfactory.hpp"
#include "videostitcher/liveoutputlist.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "widgetsmanager.hpp"

#include "libvideostitch/ptv.hpp"

ConfigOutputsWidget::ConfigOutputsWidget(QWidget* const parent)
    : IConfigurationCategory(parent),
      outputList(new GenericTableWidget(this)),
      newOutputWidget(new NewOutputWidget(this)),
      buttonAddOutput(nullptr),
      hasEmptyOutputs(true) {
  setupUi(this);
  labelTitle->setProperty("vs-title1", true);
  labelNoOutputsMessage->setProperty("vs-message", true);
  labelNoOutputsMessage->setVisible(false);
  stacked->addWidget(outputList);
  stacked->addWidget(newOutputWidget);
  outputsLayout->addLayout(buttonsLayout);

  // Update output list
  addOutputDevices(QList<LiveOutputFactory*>());
  outputList->setResizable();
  outputList->setProperty("vs-section-container", true);

  connect(outputList, &GenericTableWidget::cellClicked, this, &ConfigOutputsWidget::onItemClicked);

  changeMode(IConfigurationCategory::Mode::View);
}

ConfigOutputsWidget::~ConfigOutputsWidget() {}

void ConfigOutputsWidget::addOutputDevices(QList<LiveOutputFactory*> liveOutputs) {
  removeOutputDevices();

  outputList->initializeTable(INPUT_COLUMNS, liveOutputs.count() + 1);
  QList<QString> outputToRemove;
  if (!liveOutputs.isEmpty()) {
    for (auto liveOutput : liveOutputs) {
      if (liveOutput != nullptr) {
        // VSA-6594
        //////////////////////////////////////////////////////////////////////////////
        if (liveOutput->isSupported()) {
          addSingleConfigurationValue(liveOutput);
        } else {
          if (liveOutput->getOutputTypeDisplayName() == QStringLiteral("YouTube")) {
            GenericDialog* dialog =
                new GenericDialog(tr("Output configuration"),
                                  tr("YouTube is not currently supported, your output has been removed. Please use "
                                     "RTMP to stream to YouTube."),
                                  GenericDialog::DialogMode::ACCEPT, WidgetsManager::getInstance()->getMainWindowRef());
            dialog->show();
          }
          outputToRemove.push_back(liveOutput->getIdentifier());
        }
        /////////////////////////////////////////////////////////////////////////////////
      }
    }
  }

  if (!outputToRemove.isEmpty()) {
    for (auto outputId : outputToRemove) {
      emit reqRemoveOutput(outputId);
    }
  }

  hasEmptyOutputs = liveOutputs.isEmpty();
  labelNoOutputsMessage->setVisible(hasEmptyOutputs);
  createAddButton();
  outputList->finishTable();
}

void ConfigOutputsWidget::showOutputList() {
  stacked->setCurrentWidget(outputList);
  displayConfigInTheView(false);
  labelNoOutputsMessage->setVisible(hasEmptyOutputs);
}

void ConfigOutputsWidget::addSingleConfigurationValue(LiveOutputFactory* output) {
  OutputDetailWidget* item = new OutputDetailWidget(output, this);
  connect(item, &OutputDetailWidget::notifyDeleteOutput, this, &ConfigOutputsWidget::reqRemoveOutput);
  outputList->addElementToTable(item);
}

void ConfigOutputsWidget::createAddButton() {
  if (buttonAddOutput) {
    delete buttonAddOutput;
  }
  buttonAddOutput = new QPushButton(tr("Add a new output"), (newOutputWidget));
  // TODO FIXME: the buttonAddOutput is an item in the outputList, so it will have twice the 4px top/bottom border
  // to fix this, we should remove this button from the list and put it directly in the form
  buttonAddOutput->setProperty("vs-section-add-button", true);
  connect(buttonAddOutput, &QPushButton::clicked, this, &ConfigOutputsWidget::onButtonAddOutputClicked);
  outputList->addElementToTable(buttonAddOutput);
}

OutputDetailWidget* ConfigOutputsWidget::getListItemById(const QString& id) const {
  for (int i = 0; i < outputList->rowCount(); ++i) {
    OutputDetailWidget* widget = qobject_cast<OutputDetailWidget*>(outputList->cellWidget(i, 0));
    if (widget != nullptr && widget->getOutput()->getIdentifier() == id) {
      return widget;
    }
  }
  return nullptr;
}

void ConfigOutputsWidget::onButtonAddOutputClicked() {
  stacked->setCurrentWidget(newOutputWidget);
  displayConfigInTheView(true);
  labelNoOutputsMessage->setVisible(false);
  emit notifyStartAddingOutput();
}

void ConfigOutputsWidget::onItemClicked(int row, int column) {
  OutputDetailWidget* item(qobject_cast<OutputDetailWidget*>(outputList->cellWidget(row, column)));
  if (item != nullptr) {
    updateConfiguration(item->getOutput());
  }
}

void ConfigOutputsWidget::configureOutputById(const QString& id) {
  updateConfiguration(projectDefinition->getOutputById(id));
}

void ConfigOutputsWidget::updateConfiguration(LiveOutputFactory* liveOutput) {
  OutputConfigurationWidget* widget = liveOutput->createConfigurationWidget(this);
  if (!widget) {
    GenericDialog* dialog =
        new GenericDialog(tr("Output configuration"),
                          tr("There is nothing to configure on output %0").arg(liveOutput->getOutputTypeDisplayName()),
                          GenericDialog::DialogMode::ACCEPT, WidgetsManager::getInstance()->getMainWindowRef());
    dialog->show();
    dialog->raise();
    return;
  }

  labelTitle->setText(tr("Edit output %0").arg(liveOutput->getOutputTypeDisplayName()));
  stacked->addWidget(widget);
  stacked->setCurrentWidget(widget);
  displayConfigInTheView(true);

  // The 2 first connections will not be needed when the widget will be displayed in a modal popup (instead of in the
  // stack)
  connect(this, &ConfigOutputsWidget::injectProject, widget, &IConfigurationCategory::setProject);
  connect(this, &ConfigOutputsWidget::projectCleared, widget, &IConfigurationCategory::clearProject);
  connect(widget, &IConfigurationCategory::reqBack, this, &ConfigOutputsWidget::restoreMainPage);
  connect(widget, &OutputConfigurationWidget::saved, this, &ConfigOutputsWidget::saved);
  connect(widget, &OutputConfigurationWidget::reqChangeOutputId, this, &ConfigOutputsWidget::onOutputIdChanged);
  connect(widget, &OutputConfigurationWidget::reqChangeOutputConfig, this, &ConfigOutputsWidget::reqChangeOutputConfig);

  widget->setProject(projectDefinition);
  widget->changeMode(IConfigurationCategory::Mode::Edition);
}

void ConfigOutputsWidget::onOutputIdChanged(const QString& oldId, const QString& newId) {
  emit notifyOutputIdChanged(oldId, newId);
}

void ConfigOutputsWidget::reactToChangedProject() {
  if (projectDefinition->isInit()) {
    addOutputDevices(projectDefinition->getOutputConfigs()->getValues());
  }
  emit injectProject(projectDefinition);
}

void ConfigOutputsWidget::reactToClearedProject() { emit projectCleared(); }

OutputConfigurationWidget* ConfigOutputsWidget::getConfigurationWidgetForId(const QString& outputId) {
  // This will not be needed when the widget will be displayed in a modal popup (instead of in the stack)
  for (int index = 0; index < stacked->count(); ++index) {
    auto widget = qobject_cast<OutputConfigurationWidget*>(stacked->widget(index));
    if (widget && widget->getOutput()->getIdentifier() == outputId) {
      return widget;
    }
  }
  return nullptr;
}

void ConfigOutputsWidget::toggleOutput(const QString& id) {
  auto widget = getConfigurationWidgetForId(id);
  if (widget) {
    widget->toggleWidgetState();
  }
}

void ConfigOutputsWidget::removeOutputDevices() {
  // This will not be needed when the widget will be displayed in a modal popup (instead of in the stack)
  for (int index = 0; index < stacked->count(); ++index) {
    OutputConfigurationWidget* widget = qobject_cast<OutputConfigurationWidget*>(stacked->widget(index));
    if (widget) {
      delete widget;
    }
  }

  outputList->clearElements();
  labelNoOutputsMessage->setVisible(true);
  showOutputList();
}

void ConfigOutputsWidget::restoreMainPage() {
  showOutputList();
  labelTitle->setText(tr("Outputs"));

  for (int index = 0; index < stacked->count();) {
    OutputConfigurationWidget* widget = qobject_cast<OutputConfigurationWidget*>(stacked->widget(index));
    if (widget) {
      // The user stopped the live output edition, the widget is not needed anymore
      stacked->removeWidget(widget);
      delete widget;
    } else {
      ++index;
    }
  }
}

NewOutputWidget* ConfigOutputsWidget::getNewOutputWidget() const { return newOutputWidget; }

void ConfigOutputsWidget::updateEditability(bool outputIsActivated, bool algorithmIsActivated) {
  Q_UNUSED(algorithmIsActivated);
  buttonAddOutput->setEnabled(!outputIsActivated);
  for (int i = 0; i < outputList->rowCount(); ++i) {
    OutputDetailWidget* widget = qobject_cast<OutputDetailWidget*>(outputList->cellWidget(i, 0));
    if (widget != nullptr) {
      widget->allowsRemoving(!outputIsActivated);
    }
  }
}
