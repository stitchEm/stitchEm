// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QEvent>
#include <QObject>
#include <QPointer>

class ProjectDefinition;

/**
 * @brief This class allows to disable|hide features that are not compatible with some projects.
 */
class VS_GUI_EXPORT FeatureFilterer : public QObject {
  Q_OBJECT

 public:
  enum class PropertyToWatch { enabled, visible };

  explicit FeatureFilterer(QObject* parent = nullptr);
  ~FeatureFilterer() = default;

  void setProject(ProjectDefinition* newProject);
  void watch(QObject* watchedObject, PropertyToWatch propertyToWatch = PropertyToWatch::enabled);
  virtual bool eventFilter(QObject* watchedObject, QEvent* event);

 protected:
  virtual void connectProjectConditionToUpdate() = 0;
  virtual bool disableFeaturesCondition() const = 0;

 protected slots:
  void updateWatchedObjects(bool disableFeatures);

 protected:
  QPointer<ProjectDefinition> project;

 private:
  Q_DISABLE_COPY(FeatureFilterer)
  static const char* getPropertyName(PropertyToWatch property);
  static QVector<QEvent::Type> getEventTypes(PropertyToWatch property);

  QHash<QObject*, PropertyToWatch> watchedObjects;
};
