// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "onevisualinputfilterer.hpp"

#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"

#include <QCoreApplication>
#include <QThread>

OneVisualInputFilterer* OneVisualInputFilterer::getInstance() {
  Q_ASSERT_X(QThread::currentThread() == qApp->thread(), "OneVisualInputFilterer",
             "OneVisualInputFilterer must be created in the GUI thread in order to ensure it to work.");
  OneVisualInputFilterer* instance = qApp->findChild<OneVisualInputFilterer*>();
  if (!instance) {
    instance = new OneVisualInputFilterer();
  }
  return instance;
}

void OneVisualInputFilterer::connectProjectConditionToUpdate() {
  connect(project, &ProjectDefinition::severalVisualInputsHasChanged, this,
          &OneVisualInputFilterer::internalUpdateWatchedObjects, Qt::UniqueConnection);
}

bool OneVisualInputFilterer::disableFeaturesCondition() const {
  return !project || !project->isInit() || !project->hasSeveralVisualInputs();
}

void OneVisualInputFilterer::internalUpdateWatchedObjects(bool hasSeveralVisualInputs) {
  updateWatchedObjects(!hasSeveralVisualInputs);
}

OneVisualInputFilterer::OneVisualInputFilterer() : FeatureFilterer(qApp) {}
