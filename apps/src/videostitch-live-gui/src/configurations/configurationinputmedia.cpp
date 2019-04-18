// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "configurationinputmedia.hpp"
#include "ui_configurationinputmedia.h"

#include "videostitcher/liveprojectdefinition.hpp"
#include "videostitcher/liveinputfile.hpp"
#include "libvideostitch-gui/utils/inputformat.hpp"

#include <QFileDialog>

ConfigurationInputMedia::ConfigurationInputMedia(std::shared_ptr<const LiveInputFile> liveInput, QWidget* parent)
    : InputConfigurationWidget(parent), ui(new Ui::ConfigurationInputMedia), templateInput(liveInput) {
  ui->setupUi(this);
  ui->labelTitleMedia->setProperty("vs-title1", true);
  ui->verticalLayout->addLayout(buttonsLayout);
  ui->buttonClearFiles->setDisabled(true);
  ui->stackFiles->setCurrentWidget(ui->pageEmpty);
  connect(ui->buttonAddMedia, &QPushButton::clicked, this, &ConfigurationInputMedia::onSelectFilesClicked);
  connect(ui->buttonClearFiles, &QPushButton::clicked, this, &ConfigurationInputMedia::onFilesCleared);
}

ConfigurationInputMedia::~ConfigurationInputMedia() {}

void ConfigurationInputMedia::saveData() {
  const auto nbInputs = ui->listFiles->count();
  for (auto index = 0; index < nbInputs; ++index) {
    std::shared_ptr<LiveInputFile> input(new LiveInputFile(ui->listFiles->item(index)->text()));
    input->setWidth(ui->spinWidth->value());
    input->setHeight(ui->spinHeight->value());
    input->setHasAudio(false);
    editedInputs.append(input);
  }
}

void ConfigurationInputMedia::reactToChangedProject() {
  if (templateInput && projectDefinition->isInit()) {
    for (const VideoStitch::Core::InputDefinition& input : projectDefinition->getPanoConst()->getVideoInputs()) {
      ui->listFiles->addItem(QString::fromStdString(input.getDisplayName()));
    }
    ui->spinWidth->setValue(templateInput->getWidth());
    ui->spinHeight->setValue(templateInput->getHeight());
  }
  ui->stackFiles->setCurrentWidget(ui->pageFiles);
  // TODO determine inputs size before saving like in the RTSP
}

bool ConfigurationInputMedia::hasValidConfiguration() const { return ui->listFiles->count() > 0; }

void ConfigurationInputMedia::onSelectFilesClicked() {
  QStringList files = QFileDialog::getOpenFileNames(this, tr("Open media"), "",
                                                    QString("%0 %1;;%2 %3;;%4 (*)")
                                                        .arg(tr("Videos"))
                                                        .arg(VideoStitch::InputFormat::VIDEO_FORMATS)
                                                        .arg(tr("Images"))
                                                        .arg(VideoStitch::InputFormat::IMAGE_FORMATS)
                                                        .arg(tr("All files")));
  ui->buttonClearFiles->setDisabled(files.isEmpty());
  ui->stackFiles->setCurrentWidget(files.isEmpty() ? ui->pageEmpty : ui->pageFiles);
  if (files.isEmpty()) {
    return;
  }
  ui->listFiles->addItems(files);
  onConfigurationChanged();
}

void ConfigurationInputMedia::onFilesCleared() {
  ui->listFiles->clear();
  ui->buttonClearFiles->setDisabled(true);
  ui->stackFiles->setCurrentWidget(ui->pageEmpty);
  onConfigurationChanged();
}
