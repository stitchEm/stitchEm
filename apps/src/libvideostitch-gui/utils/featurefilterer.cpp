// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "featurefilterer.hpp"

#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"

// To ensure the correct behaviour, we need 2 things:
// - first, the watched objects should be updated when the condition changed (in the reimplemented method
// connectProjectConditionToUpdate)
// - second, we should prevent other code to enable the watched objects if the condition is not filled (the eventFilter
// method)

FeatureFilterer::FeatureFilterer(QObject* parent) : QObject(parent), project(nullptr) {}

void FeatureFilterer::setProject(ProjectDefinition* newProject) {
  project = newProject;
  if (project) {
    connectProjectConditionToUpdate();
  }
  updateWatchedObjects(disableFeaturesCondition());
}

void FeatureFilterer::watch(QObject* watchedObject, PropertyToWatch propertyToWatch) {
  watchedObjects[watchedObject] = propertyToWatch;
  watchedObject->setProperty(getPropertyName(propertyToWatch), QVariant(!disableFeaturesCondition()));
  // 'this' should live in the main thread for the event filter feature to work
  watchedObject->installEventFilter(this);
}

bool FeatureFilterer::eventFilter(QObject* watchedObject, QEvent* event) {
  PropertyToWatch property = watchedObjects.value(watchedObject);
  if (getEventTypes(property).contains(event->type())) {
    auto propertyName = getPropertyName(property);
    if (watchedObject->property(propertyName).toBool() && disableFeaturesCondition()) {
      watchedObject->setProperty(propertyName, QVariant(false));
      return true;
    }
  }
  return QObject::eventFilter(watchedObject, event);
}

void FeatureFilterer::updateWatchedObjects(bool disableFeatures) {
  for (auto it = watchedObjects.constBegin(); it != watchedObjects.constEnd(); ++it) {
    it.key()->setProperty(getPropertyName(it.value()), QVariant(!disableFeatures));
  }
}

const char* FeatureFilterer::getPropertyName(FeatureFilterer::PropertyToWatch property) {
  switch (property) {
    case PropertyToWatch::enabled:
      return "enabled";
    case PropertyToWatch::visible:
      return "visible";
  }
  return "";
}

QVector<QEvent::Type> FeatureFilterer::getEventTypes(FeatureFilterer::PropertyToWatch property) {
  switch (property) {
    case PropertyToWatch::enabled:
      return QVector<QEvent::Type>() << QEvent::EnabledChange << QEvent::ActionChanged;
    case PropertyToWatch::visible:
      return QVector<QEvent::Type>() << QEvent::Show << QEvent::Hide << QEvent::ActionChanged;
  }
  return QVector<QEvent::Type>();
}
