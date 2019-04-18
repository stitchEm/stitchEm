// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "newaudioprocessorwidget.hpp"
#include "guiconstants.hpp"

static const int DATA_TYPE_OUTPUT(100);
static const int DATA_USED_OUTPUT(102);

NewAudioProcessorWidget::NewAudioProcessorWidget(QWidget* const parent) : QWidget(parent) {
  setupUi(this);
  listConnections->setProperty("vs-section-container", true);
  buttonBack->setProperty("vs-button-medium", true);
  connect(listConnections, &QListWidget::itemClicked, this, &NewAudioProcessorWidget::onItemClicked);
  connect(buttonBack, &QPushButton::clicked, this, &NewAudioProcessorWidget::notifyBackClicked);
}

void NewAudioProcessorWidget::insertProcessorItem(const QString displayName, const QString type, const bool isUsed) {
  const QString extra = isUsed ? tr(" (Already in use)") : QString();
  QListWidgetItem* item = new QListWidgetItem(displayName + extra, listConnections);
  item->setSizeHint(QSize(listConnections->width(), ITEM_HEIGHT));
  item->setData(DATA_TYPE_OUTPUT, type);
  item->setData(DATA_USED_OUTPUT, isUsed);
  if (isUsed) {
    item->setFlags(item->flags() & ~Qt::ItemIsEnabled & ~Qt::ItemIsSelectable);
  }
  listConnections->setFocusPolicy(Qt::NoFocus);
  listConnections->addItem(item);
}

void NewAudioProcessorWidget::clearDevices() { listConnections->clear(); }

void NewAudioProcessorWidget::onItemClicked(QListWidgetItem* item) {
  emit notifyProcessorSelected(item->text(), item->data(DATA_TYPE_OUTPUT).toString(),
                               item->data(DATA_USED_OUTPUT).toBool());
}
