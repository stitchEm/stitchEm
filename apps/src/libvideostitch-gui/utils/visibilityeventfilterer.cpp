// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "visibilityeventfilterer.hpp"

#include <QEvent>

static const QList<QEvent::Type> watchedEventTypes = {QEvent::Show, QEvent::Hide, QEvent::ActionChanged};

VisibilityEventFilterer::VisibilityEventFilterer(QObject* watchedObject, std::function<bool()> filterCondition,
                                                 QObject* parent)
    : QObject(parent), filterCondition(filterCondition) {
  watchedObject->setProperty("visible", !filterCondition());
  watchedObject->installEventFilter(this);
}

bool VisibilityEventFilterer::eventFilter(QObject* watchedObject, QEvent* event) {
  if (watchedEventTypes.contains(event->type()) && filterCondition()) {
    watchedObject->setProperty("visible", QVariant(false));
    return true;
  }
  return QObject::eventFilter(watchedObject, event);
}
