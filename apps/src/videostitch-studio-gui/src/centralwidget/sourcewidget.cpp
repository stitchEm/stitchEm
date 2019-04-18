// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sourcewidget.hpp"
#include "ui_sourcewidget.h"

#include "libvideostitch-gui/widgets/vsgraphics.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include "libvideostitch-gui/utils/sourcewidgetlayoututil.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"

#include "libvideostitch-base/frame.hpp"

#include "libvideostitch/panoDef.hpp"

#include "videostitcher/globalpostprodcontroller.hpp"

#include <QDropEvent>

SourceWidget::SourceWidget(QWidget* parent)
    : IFreezableWidget("Source", parent),
      ui(new Ui::SourceWidget),
      thumbnailWidget(new CompositeWidget(this)),
      view(gridView),
      listMapper(new QSignalMapper(this)),
      indexToDisplayInMain(0),
      mainView(new VSGraphicsView(this)),
      scene(new VSGraphicsScene(mainView)),
      item(new QGraphicsPixmapItem()),
      project(nullptr) {
  ui->setupUi(this);
  setAcceptDrops(true);
  thumbnailWidget->setObjectName("thumbnailWidget");
  ui->listSplitter->insertWidget(0, thumbnailWidget);

  QGridLayout* layout = new QGridLayout(ui->widgetMain);
  layout->setVerticalSpacing(1);
  layout->setHorizontalSpacing(1);
  layout->setMargin(0);
  ui->widgetMain->setLayout(layout);

  connect(listMapper, static_cast<void (QSignalMapper::*)(int)>(&QSignalMapper::mapped), this,
          &SourceWidget::changeMainThumbnail);

  QList<int> sizes;
  sizes << width() / 5 << width() - width() / 5;
  ui->listSplitter->setSizes(sizes);
  scene->addItem(item);
  scene->setBackgroundBrush(VSGraphicsScene::backgroundBrush);
  mainView->setObjectName("mainView");
  mainView->setScene(scene);
  mainView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  mainView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  mainView->fitInView(item, Qt::KeepAspectRatio);
#ifdef HIDE_MASK_FEATURE
  ui->maskButton->hide();
#endif
  connect(ui->pushButtonViewSelector, &QPushButton::clicked, this, &SourceWidget::onButtonViewModeClicked);

  // TODO FIXME: the list view doesn't work for the moment.
  // It can be fixed with QOpenGLWidget but this introduces performance regression.
  // See VSA-3916 for more details
  ui->pushButtonViewSelector->hide();
}

SourceWidget::~SourceWidget() {
  delete item;
  delete ui;
}

void SourceWidget::activate() {
  for (ThumbnailWidget* thumbnail : thumbnails) {
    thumbnail->activate();
  }
  IFreezableWidget::activate();
}

void SourceWidget::deactivate() {
  IFreezableWidget::deactivate();
  for (ThumbnailWidget* thumbnail : thumbnails) {
    thumbnail->deactivate();
  }
}

// ----------------------- Open/Close Project ---------------------------------

void SourceWidget::createThumbnails(
    std::vector<std::tuple<VideoStitch::Input::VideoReader::Spec, bool, std::string, QtDeviceWriter*> > inputs) {
  // remove the previous thumbnails
  closeThumbnails();

  // add all the new ones
  int index = 0;
  for (const auto& input : inputs) {
    VideoStitch::Input::VideoReader::Spec spec = std::get<0>(input);
    const bool enabled = std::get<1>(input);
    const QString filename = QString::fromStdString(std::get<2>(input));
    ThumbnailWidget* widget =
        new ThumbnailWidget(index, spec.width, spec.height, spec.frameRate, spec.frameNum, filename, enabled, this);
    widget->setObjectName(filename + QString("Thumbnail"));

    thumbnails.push_back(widget);
    listMapper->setMapping(widget, thumbnails.size() - 1);

    if (index == 0) {
      connect(widget, &ThumbnailWidget::forwardFrame, this, &SourceWidget::displayMainThumbnail);
      connect(widget, &ThumbnailWidget::notifyUploaderError, this, &SourceWidget::onUploaderError);
    }

    connect(widget, &ThumbnailWidget::reqUpdate, this, &SourceWidget::onReqUpdateThumbnail);
    connect(widget, &ThumbnailWidget::mouseClicked, listMapper,
            static_cast<void (QSignalMapper::*)(void)>(&QSignalMapper::map));
    connect(widget, &ThumbnailWidget::mouseClicked, this, &SourceWidget::updateListView);
    connect(widget, &ThumbnailWidget::forwardFrame, this, &SourceWidget::glViewReady);

    widget->setDeviceWriter(std::get<3>(input));
    ++index;
  }

  switchView(view);

  if (isLowPerformance && IFreezableWidget::isActive) {
    activate();
  }
}

void SourceWidget::closeThumbnails() {
  if (ui->widgetMain->layout()) {
    while (ui->widgetMain->layout()->takeAt(0)) {
    }
  }
  thumbnailWidget->cleanup();
  qDeleteLaterAll(thumbnails);
  thumbnails.clear();
}

void SourceWidget::setProject(ProjectDefinition* p) {
  project = p;
  connect(project,
          static_cast<void (ProjectDefinition::*)(int, unsigned char*, int, int)>(&ProjectDefinition::reqUpdateMask),
          this, static_cast<void (SourceWidget::*)(int, unsigned char*, int, int)>(&SourceWidget::updateMask),
          Qt::UniqueConnection);

  connect(project, static_cast<void (ProjectDefinition::*)(int, QImage*)>(&ProjectDefinition::reqUpdateMask), this,
          static_cast<void (SourceWidget::*)(int, QImage*)>(&SourceWidget::updateMask), Qt::UniqueConnection);

  connect(this, &SourceWidget::reqReplaceReader, project, &ProjectDefinition::replaceInput, Qt::UniqueConnection);
  connect(this, &SourceWidget::reqEnableReader, project, &ProjectDefinition::enableInput, Qt::UniqueConnection);
  connect(this, &SourceWidget::reqUpdateMasks, project, &ProjectDefinition::updateMasks, Qt::UniqueConnection);
}

void SourceWidget::clearProject() {
  project = nullptr;
  closeThumbnails();
}

void SourceWidget::forceEnable(int inputIndex) {
  if (inputIndex < 0 || inputIndex >= thumbnails.size()) {
    return;
  }

  thumbnails.at(inputIndex)->forceEnable(true);
}

void SourceWidget::updateGlViews() {
  for (ThumbnailWidget* thumbnail : thumbnails) {
    thumbnail->view->update();
  }
}

// ----------------------- View Selection ---------------------------------

void SourceWidget::onButtonViewModeClicked(bool checked) {
  if (!checked) {
    switchView(SourceView(gridView));
    ui->pushButtonViewSelector->setText(tr("List View"));
  } else {
    switchView(SourceView(listView));
    ui->pushButtonViewSelector->setText(tr("Grid View"));
  }
}

void SourceWidget::switchView(SourceView v) {
  QGridLayout* layout = qobject_cast<QGridLayout*>(ui->widgetMain->layout());
  thumbnailWidget->cleanup();
  if (ui->widgetMain->layout()) {
    while (ui->widgetMain->layout()->takeAt(0) != nullptr) {
    }
  }
  foreach (ThumbnailWidget* widget, thumbnails) {
    connect(widget, &ThumbnailWidget::removeInput, this, &SourceWidget::removeThumbnail, Qt::UniqueConnection);
    connect(widget, &ThumbnailWidget::replaceInput, this, &SourceWidget::replaceThumbnail, Qt::UniqueConnection);
    connect(widget, &ThumbnailWidget::enableInput, this, &SourceWidget::enableThumbnail, Qt::UniqueConnection);
  }

  switch (v) {
    case listView: {
      layout->addWidget(mainView, 0, 0);
      mainView->show();
      thumbnailWidget->show();

      for (int index = 0; index < thumbnails.count(); ++index) {
        thumbnailWidget->addWidget(thumbnails.at(index));
      }
      break;
    }
    case gridView: {
      mainView->hide();
      thumbnailWidget->hide();

      const int numberOfColumns = SourceWidgetLayoutUtil::getColumnsNumber(thumbnails.count());
      Q_ASSERT(numberOfColumns > 0);
      for (int index = 0; index < thumbnails.count(); ++index) {
        ThumbnailWidget* widget = thumbnails.at(index);
        layout->addWidget(widget, SourceWidgetLayoutUtil::getItemLine(index, numberOfColumns),
                          SourceWidgetLayoutUtil::getItemColumn(index, numberOfColumns));
      }
      break;
    }
    default:
      Q_ASSERT(0);
  }
  view = v;
}

int SourceWidget::numEnabledThumbnails() {
  // check there's at least one enabled input
  int ret = 0;

  for (auto thumbnail : thumbnails) {
    if (thumbnail->enableBox->isChecked()) ret++;
  }

  return ret;
}

void SourceWidget::changeState(GUIStateCaps::State s) {
  setEnabled(s == GUIStateCaps::stitch);
#ifndef HIDE_MASK_FEATURE
  if (s == GUIStateCaps::stitch) {
    ui->maskButton->show();
  }
#endif
}

// In List view, the current thumbnail
void SourceWidget::changeMainThumbnail(int /*index*/) {
  // XXX TODO FIXME

  /*

  // disconnect the previous
  disconnect(&thumbnails[indexToDisplayInMain]->view->getUploader(), SIGNAL(forwardFrame(std::shared_ptr<Frame>)),
             this, SLOT(displayMainThumbnail(std::shared_ptr<Frame>)));
  indexToDisplayInMain = index;
  // connect the next
  connect(&thumbnails[indexToDisplayInMain]->view->getUploader(), SIGNAL(forwardFrame(std::shared_ptr<Frame>)),
          this, SLOT(displayMainThumbnail(std::shared_ptr<Frame>)));

          */
}

// Enable/Replace/Remove user actions
void SourceWidget::replaceThumbnail(QString newFile) {
  ThumbnailWidget* w = qobject_cast<ThumbnailWidget*>(sender());
  emit reqChangeState(GUIStateCaps::disabled);
  emit reqReplaceReader(thumbnails.indexOf(w), newFile);
}

void SourceWidget::enableThumbnail(bool enable) {
  ThumbnailWidget* w = qobject_cast<ThumbnailWidget*>(sender());
  // check there's at least one enabled input
  if (numEnabledThumbnails() < 1) {
    w->forceEnable(true);
    MsgBoxHandler::getInstance()->generic(tr("At least one input must be enabled"), tr("Warning"), WARNING_ICON);
  } else {
    emit reqChangeState(GUIStateCaps::disabled);
    emit reqEnableReader(thumbnails.indexOf(w), enable);
  }
}

void SourceWidget::removeThumbnail() {
  ThumbnailWidget* w = qobject_cast<ThumbnailWidget*>(sender());
  // check there's at least one enabled input
  if (numEnabledThumbnails() == 1 && w->enableBox->isChecked()) {
    w->activate();
    MsgBoxHandler::getInstance()->generic(tr("At least one input must be enabled"), tr("Warning"), WARNING_ICON);
  } else {
    emit reqChangeState(GUIStateCaps::disabled);
    emit reqRemoveReader(thumbnails.indexOf(w));
  }
}

// ------------------------------ Masks management -------------------------------------

void SourceWidget::maskToggled(bool toggled) {
  if (toggled) {
    emit reqUpdateMasks();
  }
}

void SourceWidget::updateMask(int index, unsigned char* maskData, int width, int height) {
  if (index < (int)thumbnails.size()) {
    thumbnails[index]->setMask(maskData, width, height);
  }
}

// We don't need to delete the mask since the QPixmap generated from the image will take the ownership
void SourceWidget::updateMask(int index, QImage* mask) {
  if (index < (int)thumbnails.size()) {
    thumbnails[index]->setMask(mask->bits(), mask->width(), mask->height());
  }
}

void SourceWidget::updateListView() {}

void SourceWidget::onUploaderError(const VideoStitch::Status errorStatus, bool needToExit) {
  if (errorStatus.hasUnderlyingCause(VideoStitch::ErrType::OutOfResources) && project != nullptr) {
    emit reqResetDimensions(project->getPanoConst()->getWidth(), project->getPanoConst()->getHeight(),
                            project->getInputNames());
  } else {
    emit notifyUploadError(errorStatus, needToExit);
  }
}

void SourceWidget::onReqUpdateThumbnail() {
  if (!GlobalController::getInstance().getController()->getClock().isActive()) {
    emit reqReextract();
  }
}

// ------------------------------- Extract loop ----------------------------------------

void SourceWidget::displayMainThumbnail(Frame* frame) {
  if (view == listView) {
    frame->readLock();
    QImage img((unsigned char*)frame->buffer(), frame->getWidth(), frame->getHeight(), frame->getWidth() * 4,
               QImage::Format_RGBA8888);
    item->setPixmap(QPixmap::fromImage(img));
    frame->unlock();
    mainView->fitInView(item, Qt::KeepAspectRatio);
  }
  for (auto thumb : thumbnails) {
    thumb->view->update();
  }
  frame->readLock();
  mtime_t date = frame->getDate();
  frame->unlock();
  emit refresh(date);
}
