// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QGraphicsEffect>
#include <QLabel>
#include <QMovie>
#include <QVBoxLayout>
#include "genericloader.hpp"
#include "guiconstants.hpp"
#include "animations/fadeanimation.hpp"

static const QString& LOADING_ICON(":/live/icons/assets/icon/live/loading.gif");

GenericLoader::GenericLoader(const QString& message, QWidget* const parent)
    : QFrame(parent),
      movieLoading(new QMovie(LOADING_ICON)),
      labelLoading(new QLabel(this)),
      labelMessage(new QLabel(message, this)),
      labelPercentage(new QLabel(this)) {
  labelLoading->setAlignment(Qt::AlignCenter);
  labelLoading->setAlignment(Qt::AlignHCenter);
  labelLoading->setObjectName("labelLoading");
  labelLoading->setMovie(movieLoading.data());
  labelLoading->setAttribute(Qt::WA_TranslucentBackground);

  labelMessage->setAlignment(Qt::AlignCenter);
  labelMessage->setAlignment(Qt::AlignHCenter);
  labelMessage->setObjectName("labelMessage");

  labelPercentage->setAlignment(Qt::AlignCenter);
  labelPercentage->setAlignment(Qt::AlignHCenter);
  labelPercentage->setObjectName("labelPercentage");

  QVBoxLayout* layoutVerticalCentral = new QVBoxLayout();
  layoutVerticalCentral->addWidget(labelLoading);
  layoutVerticalCentral->addWidget(labelMessage);
  layoutVerticalCentral->addWidget(labelPercentage);

  QHBoxLayout* layoutHorizontal = new QHBoxLayout();
  layoutHorizontal->addSpacerItem(
      new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));  // Transfer the ownership
  layoutHorizontal->addLayout(layoutVerticalCentral);                      // Transfer the ownership
  layoutHorizontal->addSpacerItem(
      new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed));  // Transfer the ownership

  QVBoxLayout* layoutVertical = new QVBoxLayout();
  layoutVertical->addSpacerItem(
      new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));  // Transfer the ownership
  layoutVertical->addLayout(layoutHorizontal);                             // Transfer the ownership
  layoutVertical->addSpacerItem(
      new QSpacerItem(0, 0, QSizePolicy::Fixed, QSizePolicy::Expanding));  // Transfer the ownership

  setLayout(layoutVertical);  // Transfer the ownership
  movieLoading->start();
  parent->installEventFilter(this);
}

GenericLoader::~GenericLoader() {}

void GenericLoader::updateSize(unsigned int width, unsigned int height) { setFixedSize(width, height); }

bool GenericLoader::eventFilter(QObject* object, QEvent* event) {
  if (event->type() == QEvent::Resize) {
    updateSize(parentWidget()->width(), parentWidget()->height());
  }
  return QWidget::eventFilter(object, event);
}

void GenericLoader::showEvent(QShowEvent*) {
  raise();
  show();
}
