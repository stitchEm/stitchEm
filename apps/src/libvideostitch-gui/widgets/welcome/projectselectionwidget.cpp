// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "projectselectionwidget.hpp"
#include "ui_projectselectionwidget.h"
#include "mainwindow/vssettings.hpp"
#include "libvideostitch-base/file.hpp"

#include <QFileInfo>
#include <QDropEvent>
#include <QMimeData>
#include <QPair>
#include <QDesktopServices>

static const int FULL_PATH(1);
static const int ITEM_HEIGHT(30);
static const int LIST_MARGIN(20);

ProjectSelectionWidget::ProjectSelectionWidget(QWidget *parent) : QFrame(parent), ui(new Ui::ProjectSelectionWidget) {
  ui->setupUi(this);
  connect(ui->buttonOpenProject, &QPushButton::clicked, this, &ProjectSelectionWidget::notifyProjectOpened);
  connect(ui->buttonNewProject, &QPushButton::clicked, this, &ProjectSelectionWidget::notifyNewProject);
  connect(ui->listRecentlyOpened, &QListWidget::clicked, this, &ProjectSelectionWidget::onProjectSelected);
  ui->dropArea->setAcceptDrops(true);
  ui->dropArea->installEventFilter(this);
  ui->labelDropIcon->hide();
  ui->labelDropHere->hide();
  ui->buttonNewProject->setProperty("vs-button-action", true);
  ui->buttonOpenProject->setProperty("vs-button-action", true);
  const QString supported = ui->labelDragDropHere->text();
  QString formats;
  if (QCoreApplication::applicationName() == VIDEOSTITCH_STUDIO_APP_NAME) {
    formats = tr("media, PTV or PTVB");
  } else {
    formats = tr("VAH");
  }
  ui->labelDragDropHere->setText(supported.arg(formats));
  loadRecentProjects();
}

ProjectSelectionWidget::~ProjectSelectionWidget() { delete ui; }

void ProjectSelectionWidget::onContentUpdated() { loadRecentProjects(); }

void ProjectSelectionWidget::onProjectSelected(const QModelIndex &index) {
  emit notifyProjectSelected(index.data(FULL_PATH).toString());
}

void ProjectSelectionWidget::onSampleSelected(const QModelIndex &index) {
  QDesktopServices::openUrl(index.data(FULL_PATH).toString());
}

bool ProjectSelectionWidget::eventFilter(QObject *watched, QEvent *event) {
  if (watched == ui->dropArea) {
    if (event->type() == QEvent::Type::DragEnter) {
      event->setAccepted(true);
      ui->labelDropIcon->show();
      ui->labelDragIcon->hide();
      ui->labelDropHere->show();
      ui->labelDragDropHere->hide();
    } else if (event->type() == QEvent::Type::DragLeave) {
      ui->labelDropIcon->hide();
      ui->labelDragIcon->show();
      ui->labelDropHere->hide();
      ui->labelDragDropHere->show();
    } else if (event->type() == QEvent::Type::Drop) {
      QDropEvent *dropEvent = static_cast<QDropEvent *>(event);
      emit notifyFilesDropped(dropEvent);
      ui->labelDropIcon->hide();
      ui->labelDragIcon->show();
      ui->labelDropHere->hide();
      ui->labelDragDropHere->show();
    }
  }
  return QObject::eventFilter(watched, event);
}

void ProjectSelectionWidget::loadRecentProjects() {
  ui->listRecentlyOpened->clear();
  const QStringList files = VSSettings::getSettings()->getRecentFileList();
  foreach (auto name, files) {
    if (QFileInfo(name).exists()) {
      const QString text = tr("%0. %1").arg(ui->listRecentlyOpened->count() + 1).arg(File::strippedName(name));
      QListWidgetItem *item = new QListWidgetItem(text, ui->listRecentlyOpened);
      item->setTextAlignment(Qt::AlignVCenter);
      item->setData(FULL_PATH, name);
      item->setSizeHint(QSize(ui->listRecentlyOpened->width() - 10, ITEM_HEIGHT));
      ui->listRecentlyOpened->addItem(item);
    }
  }
  ui->labelRecentTitle->setVisible(!files.isEmpty());
  ui->listRecentlyOpened->setVisible(!files.isEmpty());
  ui->listRecentlyOpened->adjustSize();
  ui->listRecentlyOpened->setFixedHeight(ui->listRecentlyOpened->count() * ITEM_HEIGHT + LIST_MARGIN);
}
