// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QKeyEvent>
#include "backgroundcontainer.hpp"
#include "ui_backgroundcontainer.h"
#include "guiconstants.hpp"

BackgroundContainer::BackgroundContainer(QWidget* contained, const QString& title, QWidget* const parent,
                                         bool closeButton)
    : QFrame(parent), ui(new Ui::BackgroundContainer), containedWidget(contained) {
  ui->setupUi(this);
  ui->labelContainerTitle->setProperty("vs-title2", true);
  ui->centralLayout->addWidget(containedWidget);
  ui->labelContainerTitle->setText(title);
  setFixedSize(parent->size());
  connect(ui->buttonCloseWidget, &QPushButton::clicked, this, &BackgroundContainer::notifyWidgetClosed);
  ui->buttonCloseWidget->setVisible(closeButton);
  parent->installEventFilter(this);
}

BackgroundContainer::~BackgroundContainer() {}

QWidget* BackgroundContainer::getContainedWidget() const { return containedWidget; }

bool BackgroundContainer::eventFilter(QObject* object, QEvent* event) {
  if (event->type() == QEvent::Resize) {
    setFixedSize(parentWidget()->size());
  } else if (event->type() == QEvent::KeyPress) {
    QKeyEvent* key = static_cast<QKeyEvent*>(event);
    if (key->key() == Qt::Key_Escape) {
      close();
    }
  }
  return QWidget::eventFilter(object, event);
}
