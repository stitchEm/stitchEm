// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "newoutputwidget.hpp"
#include "guiconstants.hpp"

static const int DATA_TYPE_OUTPUT(100);
static const int DATA_MODEL_OUTPUT(101);
static const int DATA_USED_OUTPUT(102);

NewOutputWidget::NewOutputWidget(QWidget* const parent) : QWidget(parent) {
  setupUi(this);
  listConnections->setProperty("vs-section-container", true);
  buttonBack->setProperty("vs-button-medium", true);
  connect(listConnections, &QListWidget::itemClicked, this, &NewOutputWidget::onItemClicked);
  connect(buttonBack, &QPushButton::clicked, this, &NewOutputWidget::notifyBackClicked);
}

void NewOutputWidget::insertDeviceItem(const QString displayName, const QString model, const QString pluginType,
                                       const bool isUsed) {
  const QString extra = isUsed ? tr(" (Already in use)") : QString();
  QListWidgetItem* item = new QListWidgetItem(displayName + extra, listConnections);
  item->setSizeHint(QSize(listConnections->width(), ITEM_HEIGHT));
  item->setData(DATA_TYPE_OUTPUT, pluginType);
  item->setData(DATA_MODEL_OUTPUT, model);
  item->setData(DATA_USED_OUTPUT, isUsed);
  if (isUsed) {
    item->setFlags(item->flags() & ~Qt::ItemIsEnabled & ~Qt::ItemIsSelectable);
  }
  listConnections->setFocusPolicy(Qt::NoFocus);
  listConnections->addItem(item);
}

void NewOutputWidget::clearDevices() { listConnections->clear(); }

void NewOutputWidget::onItemClicked(QListWidgetItem* item) {
  emit notifyDevicesSelected(item->text(), item->data(DATA_MODEL_OUTPUT).toString(),
                             item->data(DATA_TYPE_OUTPUT).toString(), item->data(DATA_USED_OUTPUT).toBool());
}
