// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioprocessorswidget.hpp"
#include "newaudioprocessorwidget.hpp"
#include "audioprocessorconfigurationwidget.hpp"
#include "audioprocessdetailwidget.hpp"
#include "videostitcher/liveaudioprocessfactory.hpp"
#include "videostitcher/liveprojectdefinition.hpp"
#include "guiconstants.hpp"
#include "generic/generictablewidget.hpp"

AudioProcessorsWidget::AudioProcessorsWidget(QWidget* const parent)
    : IConfigurationCategory(parent),
      tableProcessors(new GenericTableWidget(this)),
      newAudioProcessorWidget(new NewAudioProcessorWidget(this)),
      buttonAddProcessor(nullptr) {
  setupUi(this);
  labelTitle->setProperty("vs-title1", true);
  labelNoProcessorsMessage->setProperty("vs-message", true);
  labelNoProcessorsMessage->setVisible(false);
  stacked->addWidget(tableProcessors);
  stacked->addWidget(newAudioProcessorWidget);
  mainLayout->addLayout(buttonsLayout);
  tableProcessors->setResizable();
  tableProcessors->setProperty("vs-section-container", true);
  addProcessors(QList<LiveAudioProcessFactory*>());
  connect(tableProcessors, &GenericTableWidget::cellClicked, this, &AudioProcessorsWidget::onProcessorClicked);
  connect(newAudioProcessorWidget, &NewAudioProcessorWidget::notifyBackClicked, this,
          &AudioProcessorsWidget::onNewProcessorBackClicked);
  connect(newAudioProcessorWidget, &NewAudioProcessorWidget::notifyProcessorSelected, this,
          &AudioProcessorsWidget::onNewProcessorSelected);
  changeMode(IConfigurationCategory::Mode::View);
}

void AudioProcessorsWidget::onProcessorClicked(int row, int col) {
  AudioProcessDetailWidget* item(qobject_cast<AudioProcessDetailWidget*>(tableProcessors->cellWidget(row, col)));
  if (item != nullptr) {
    LiveAudioProcessFactory* liveProcessor = item->getProcessor();
    if (projectDefinition) {
      projectDefinition->updateAudioProcessor(liveProcessor);
    }
    AudioProcessorConfigurationWidget* widget = liveProcessor->createConfigurationWidget(this);
    if (widget) {
      showConfigurationWidget(widget);
    }
  }
}

void AudioProcessorsWidget::reactToChangedProject() {
  if (projectDefinition->isInit()) {
    newAudioProcessorWidget->clearDevices();
    for (const auto& processor : LiveAudioProcessFactory::getAvailableProcessors()) {
      newAudioProcessorWidget->insertProcessorItem(processor.first, processor.second, false);
    }
    addProcessors(projectDefinition->getAudioProcessors());
  }
}

void AudioProcessorsWidget::reactToClearedProject() { newAudioProcessorWidget->clearDevices(); }

void AudioProcessorsWidget::onButtonAddProcessorClicked() {
  stacked->setCurrentWidget(newAudioProcessorWidget);
  displayConfigInTheView(true);
  labelNoProcessorsMessage->setVisible(false);
}

void AudioProcessorsWidget::onNewProcessorBackClicked() {
  stacked->setCurrentWidget(tableProcessors);
  changeMode(IConfigurationCategory::Mode::View);
  emit saved();
  reactToChangedProject();
}

void AudioProcessorsWidget::onNewProcessorSelected(const QString displayName, const QString type, const bool isUsed) {
  Q_UNUSED(isUsed)
  Q_UNUSED(displayName)
  const VideoStitch::AudioProcessors::ProcessorEnum enumType = VideoStitch::AudioProcessors::getEnumFromString(type);
  QString inputName = QStringLiteral("");
  if (projectDefinition && projectDefinition->getAudioInputNames().count() > 0) {
    inputName = projectDefinition->getAudioInputNames().first();
  }
  auto defaultConfig = LiveAudioProcessFactory::createDefaultConfig(inputName, enumType);
  LiveAudioProcessFactory* liveProcess = LiveAudioProcessFactory::createProcessor(defaultConfig.get(), enumType);
  AudioProcessorConfigurationWidget* widget = liveProcess->createConfigurationWidget(this);
  if (widget) {
    showConfigurationWidget(widget);
    emit notifyEditProcessor(widget->getConfiguration());
  }
}

void AudioProcessorsWidget::onProcessorWidgetFinished() {
  onNewProcessorBackClicked();
  reactToChangedProject();
}

void AudioProcessorsWidget::addProcessors(QList<LiveAudioProcessFactory*> liveAudioProcessors) {
  removeProcessors();
  tableProcessors->initializeTable(INPUT_COLUMNS, liveAudioProcessors.count() + 1);
  for (auto liveProcessor : liveAudioProcessors) {
    addSingleProcessor(liveProcessor);
  }
  labelNoProcessorsMessage->setVisible(liveAudioProcessors.isEmpty());
  createAddButton();
  tableProcessors->finishTable();
}

void AudioProcessorsWidget::createAddButton() {
  if (buttonAddProcessor) {
    delete buttonAddProcessor;
  }
  buttonAddProcessor = new QPushButton(tr("Add a new audio processor"), (this));
  buttonAddProcessor->setProperty("vs-section-add-button", true);
  connect(buttonAddProcessor, &QPushButton::clicked, this, &AudioProcessorsWidget::onButtonAddProcessorClicked);
  tableProcessors->addElementToTable(buttonAddProcessor);
}

void AudioProcessorsWidget::showConfigurationWidget(AudioProcessorConfigurationWidget* widget) {
  widget->changeMode(IConfigurationCategory::Mode::View);
  stacked->addWidget(widget);
  stacked->setCurrentWidget(widget);
  connect(widget, &AudioProcessorConfigurationWidget::reqBack, this, &AudioProcessorsWidget::onProcessorWidgetFinished);
  connect(widget, &AudioProcessorConfigurationWidget::notifyConfigurationChanged, this,
          &AudioProcessorsWidget::notifyEditProcessor);
  displayConfigInTheView(true);
}

void AudioProcessorsWidget::removeProcessors() {
  for (int index = 0; index < stacked->count(); ++index) {
    AudioProcessorConfigurationWidget* widget =
        qobject_cast<AudioProcessorConfigurationWidget*>(stacked->widget(index));
    if (widget) {
      delete widget;
    }
  }
  tableProcessors->clearElements();
  labelNoProcessorsMessage->setVisible(true);
}

void AudioProcessorsWidget::addSingleProcessor(LiveAudioProcessFactory* processor) {
  AudioProcessDetailWidget* item = new AudioProcessDetailWidget(processor, this);
  connect(item, &AudioProcessDetailWidget::notifyDeleteProcessor, this, &AudioProcessorsWidget::notifyRemoveProcessor);
  tableProcessors->addElementToTable(item);
}
