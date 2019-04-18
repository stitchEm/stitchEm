// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "tasktable.hpp"
#include "logdialog.hpp"

#include "libvideostitch-base/logmanager.hpp"
#include "autoelidelabel.hpp"

#include <QResizeEvent>
#include <QAbstractButton>
#include <QMessageBox>
#include <QProgressBar>
#include <QHeaderView>
#include <QAction>
#include <QMenu>

TaskList::TaskList(QObject *parent) : QObject(parent) { beingStitched = -1; }

TaskList::TaskList(const TaskList &copy) : QObject(copy.parent()), QList<BatcherTask *>() {
  beingStitched = copy.beingStitched;
  qCopy(copy.begin(), copy.begin(), begin());
}

bool TaskList::isRunning() const { return beingStitched != -1; }

void TaskList::signalFinished() {
  beingStitched = -1;
  emit finished();
}

void TaskList::stitchNext() {
  if (beingStitched != -1 && at(beingStitched)->isRunning()) {
    disconnect(at(beingStitched), SIGNAL(finished()), this, SLOT(stitchNext()));
    at(beingStitched)->kill();
    signalFinished();
    return;
  }
  if (isEmpty()) {
    return;
  }
  ++beingStitched;
  if (beingStitched > 0) {
    disconnect(at(beingStitched - 1), SIGNAL(finished()), this, SLOT(stitchNext()));
  }
  if (beingStitched >= size()) {
    signalFinished();
    return;
  }
  connect(at(beingStitched), SIGNAL(finished()), this, SLOT(stitchNext()));
  at(beingStitched)->start();
}

void TaskList::removeItem(BatcherTask *item) {
  removeAll(item);
  if (item->isRunning()) {
    beingStitched--;
    item->kill();
  }
  delete item;
}

TaskTable::TaskTable(QWidget *parent) : QTableWidget(parent), vsWindow(nullptr), selectedRow(-1), state(Idle) {
  setColumnCount(BatcherTask::getTaskModel().size());
  setRowCount(0);
  setShowGrid(false);
  setHorizontalHeaderLabels(BatcherTask::getTaskModel());
  setEditTriggers(QAbstractItemView::NoEditTriggers);
  connect(&tasks, SIGNAL(finished()), this, SLOT(taskListFinished()));
  connect(this, SIGNAL(currentCellChanged(int, int, int, int)), this, SLOT(onCurrentCellChanged(int, int, int, int)));
  verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
  setGridStyle(Qt::NoPen);
}

void TaskTable::addTask(const QString &ptvLocation) {
  if (taskNames.contains(ptvLocation)) {
    raise();
    activateWindow();
    QMessageBox::warning(this, tr("Couldn't add duplicate project."),
                         tr("Couldn't add duplicate project %0.").arg(ptvLocation));
    return;
  }
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QString("Adding %0 to batch list").arg(ptvLocation));
  BatcherTask *task = new BatcherTask(this);
  AutoElideLabel *nameLabel = new AutoElideLabel(Qt::ElideLeft, this);
  nameLabel->setText(ptvLocation);
  QProgressBar *bar = new QProgressBar(this);

  task->setTaskProgress(bar);
  task->setPtvName(nameLabel);

  setRowCount(rowCount() + 1);
  int index = rowCount() - 1;
  setCellWidget(index, 0, nameLabel);
  setCellWidget(index, 1, bar);
  tasks.push_back(task);
  task->setID(BatcherTask::lastID++);
  task->setDevices(selectedDevices);
  updateColumnSize(width());
  taskNames << ptvLocation;
}

TaskTable::State TaskTable::getState() const { return state; }

BatcherTask *TaskTable::getTaskAt(int row) { return (row >= 0 && tasks.size() > row) ? tasks[row] : NULL; }

void TaskTable::removeSelected() {
  if (selectedRow >= 0 && selectedRow < rowCount()) {
    removeTaskAt(selectedRow);
  }
}

void TaskTable::startStitching() {
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Starting to stitch the batch list.");
  state = RUNNING;
  emit processing();
  tasks.stitchNext();
}

void TaskTable::onCurrentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn) {
  Q_UNUSED(currentColumn)
  Q_UNUSED(previousRow)
  Q_UNUSED(previousColumn)
  selectedRow = currentRow;
}

void TaskTable::updateColumnSize(int width) {
  for (int i = 0; i < columnCount(); i++) {
    setColumnWidth(i, width / columnCount());
  }
}

void TaskTable::showEvent(QShowEvent *) { updateColumnSize(width()); }

void TaskTable::resizeEvent(QResizeEvent *event) { updateColumnSize(event->size().width()); }

void TaskTable::mousePressEvent(QMouseEvent *event) {
  QTableWidget::mousePressEvent(event);
  if (event->button() == Qt::RightButton) {
    BatcherTask *task = getTaskAt(rowAt(event->pos().y()));

    if (task) {
      QMenu menu;
      QAction *removeTask = menu.addAction(tr("Remove task"));
      QAction *openWithVS = menu.addAction(tr("Open With %0").arg(VIDEOSTITCH_STUDIO_APP_NAME));
      QAction *resetTask = (task->isRunning()) ? NULL : menu.addAction(tr("Reset task"));
      QAction *openLog = menu.addAction(tr("Open Log"));

      QAction *selectedAction = menu.exec(event->globalPos());
      if (selectedAction == removeTask) {
        removeTaskAt(rowAt(event->pos().y()));
      } else if (selectedAction == openWithVS) {
        emit reqOpenVS(task->getPtvName());
        VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
            QString("Asking %0 to open %1").arg(VIDEOSTITCH_STUDIO_APP_NAME).arg(task->getPtvName()));
      } else if (selectedAction == resetTask) {
        task->resetState();
        VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
            QString("Resetting task state %0 at index %1.")
                .arg(task->getPtvName(), QString::number(rowAt(event->pos().y()))));
      } else if (selectedAction == openLog) {
        LogDialog dialog(task->getLog(), task->getPtvName(), this);
        connect(task, SIGNAL(newLogLine(QString)), &dialog, SLOT(appendLogLine(QString)));
        dialog.exec();
        VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
            QString("Opening log of %0.").arg(task->getPtvName()));
      }
    }
  }
}

void TaskTable::mouseDoubleClickEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    BatcherTask *task = getTaskAt(rowAt(event->pos().y()));
    if (task) {
      if (task->getState() == BatcherTask::Error || task->getState() == BatcherTask::Finished) {
        LogDialog dialog(task->getLog(), task->getPtvName(), this);
        dialog.exec();
      }
    }
  }
}

void TaskTable::updateGeometries() {
  QTableWidget::updateGeometries();
  QAbstractButton *cornerButton = findChild<QAbstractButton *>();
  if (cornerButton) {
    cornerButton->hide();
  }
}

void TaskTable::removeTaskAt(int row) {
  if (row < 0) {
    return;
  }

  removeRow(row);
  BatcherTask *task = tasks.at(row);
  taskNames.remove(task->getPtvName());
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
      QString("Removing %0 from batch list").arg(task->getPtvName()));
  tasks.removeItem(task);
  if (selectedRow >= rowCount()) {
    selectedRow = rowCount() - 1;
  }
  emit removedTask();
}

void TaskTable::removeAll() {
  while (!tasks.isEmpty()) {
    removeTaskAt(0);
  }
}

void TaskTable::taskListFinished() {
  state = Idle;
  emit finished();
}

void TaskTable::onDeviceSelectionChanged(QList<int> devices) {
  selectedDevices = devices;
  for (auto i = 0; i < tasks.size(); ++i) {
    tasks.at(i)->setDevices(selectedDevices);
  }
}
