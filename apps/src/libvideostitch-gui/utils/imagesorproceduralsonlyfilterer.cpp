// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "imagesorproceduralsonlyfilterer.hpp"

#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"

#include <QCoreApplication>
#include <QThread>

ImagesOrProceduralsOnlyFilterer* ImagesOrProceduralsOnlyFilterer::getInstance() {
  Q_ASSERT_X(QThread::currentThread() == qApp->thread(), "ImagesOrProceduralsOnlyFilterer",
             "ImagesOrProceduralsOnlyFilterer must be created in the GUI thread in order to ensure it to work.");
  ImagesOrProceduralsOnlyFilterer* instance = qApp->findChild<ImagesOrProceduralsOnlyFilterer*>();
  if (!instance) {
    instance = new ImagesOrProceduralsOnlyFilterer();
  }
  return instance;
}

void ImagesOrProceduralsOnlyFilterer::connectProjectConditionToUpdate() {
  connect(project, &ProjectDefinition::imagesOrProceduralsOnlyHasChanged, this,
          &ImagesOrProceduralsOnlyFilterer::updateWatchedObjects, Qt::UniqueConnection);
}

bool ImagesOrProceduralsOnlyFilterer::disableFeaturesCondition() const {
  return !project || !project->isInit() || project->hasImagesOrProceduralsOnly();
}

ImagesOrProceduralsOnlyFilterer::ImagesOrProceduralsOnlyFilterer() : FeatureFilterer(qApp) {}
