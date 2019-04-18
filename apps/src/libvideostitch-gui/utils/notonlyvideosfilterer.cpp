// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "notonlyvideosfilterer.hpp"

#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"

#include <QCoreApplication>
#include <QThread>

NotOnlyVideosFilterer* NotOnlyVideosFilterer::getInstance() {
  Q_ASSERT_X(QThread::currentThread() == qApp->thread(), "NotOnlyVideosFilterer",
             "NotOnlyVideosFilterer must be created in the GUI thread in order to ensure it to work.");
  NotOnlyVideosFilterer* instance = qApp->findChild<NotOnlyVideosFilterer*>();
  if (!instance) {
    instance = new NotOnlyVideosFilterer();
  }
  return instance;
}

void NotOnlyVideosFilterer::connectProjectConditionToUpdate() {
  connect(project, &ProjectDefinition::severalVideosOnlyHasChanged, this,
          &NotOnlyVideosFilterer::internalUpdateWatchedObjects, Qt::UniqueConnection);
}

bool NotOnlyVideosFilterer::disableFeaturesCondition() const {
  return !project || !project->isInit() || !project->hasSeveralVideos();
}

void NotOnlyVideosFilterer::internalUpdateWatchedObjects(bool hasSeveralVideos) {
  updateWatchedObjects(!hasSeveralVideos);
}

NotOnlyVideosFilterer::NotOnlyVideosFilterer() : FeatureFilterer(qApp) {}
