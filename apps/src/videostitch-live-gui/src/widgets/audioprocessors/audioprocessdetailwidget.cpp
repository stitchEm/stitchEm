// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioprocessdetailwidget.hpp"

#include "videostitcher/liveaudioprocessfactory.hpp"

AudioProcessDetailWidget::AudioProcessDetailWidget(LiveAudioProcessFactory* processor, QWidget* parent)
    : QWidget(parent), processor(processor), ui(new Ui::AudioProcessDetailWidgetClass()) {
  ui->setupUi(this);
  ui->frame->setProperty("vs-section-contained-widget", true);
  ui->buttonDelete->setProperty("vs-section-delete-button", true);
  ui->labelName->setText(VideoStitch::AudioProcessors::getDisplayNameFromEnum(processor->getType()));
  ui->labelIcon->setPixmap(processor->getIcon());
  ui->labelDetailedName->setText(QString("(%0)").arg(processor->getDescription()));
  ui->labelIcon->setScaledContents(true);
  connect(ui->buttonDelete, &QPushButton::clicked, this, &AudioProcessDetailWidget::onDeleteClicked);
}

AudioProcessDetailWidget::~AudioProcessDetailWidget() { delete ui; }

LiveAudioProcessFactory* AudioProcessDetailWidget::getProcessor() const { return processor; }

void AudioProcessDetailWidget::onDeleteClicked() {
  emit notifyDeleteProcessor(VideoStitch::AudioProcessors::getStringFromEnum(processor->getType()));
}
