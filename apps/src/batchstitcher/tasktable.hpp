// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef TASKTABLE_HPP
#define TASKTABLE_HPP

#include <QTableWidget>
#include "batchertask.hpp"

class TaskList : public QObject, public QList<BatcherTask *> {
  Q_OBJECT
 public:
  explicit TaskList(QObject *parent = 0);
  TaskList(const TaskList &copy);
  bool isRunning() const;
 public slots:
  void stitchNext();
  void removeItem(BatcherTask *item);
  void signalFinished();
 signals:
  void finished();

 private:
  int beingStitched;
};

class TaskTable : public QTableWidget {
  Q_OBJECT
 public:
  typedef enum State { Idle, RUNNING } State;

  explicit TaskTable(QWidget *parent = nullptr);
  void addTask(const QString &ptvLocation);
  BatcherTask *getTaskAt(int row);
  void removeSelected();
  State getState() const;
 signals:
  void processing();
  void finished();
  void reqOpenVS(QString file);
  void removedTask();
 public slots:

  void startStitching();
  void removeTaskAt(int row);
  void removeAll();
  void taskListFinished();
  void onDeviceSelectionChanged(QList<int> devices);
 private slots:
  void onCurrentCellChanged(int currentRow, int currentColumn, int previousRow, int previousColumn);

 private:
  void updateColumnSize(int width);
  void showEvent(QShowEvent *);
  void resizeEvent(QResizeEvent *event);
  void mousePressEvent(QMouseEvent *event);
  void mouseDoubleClickEvent(QMouseEvent *event);
  void updateGeometries();

  TaskList tasks;
  QProcess *vsWindow;
  QStringList cudaDevices;
  QList<int> selectedDevices;
  int selectedRow;
  State state;
  QSet<QString> taskNames;
};

#endif  // TASKTABLE_HPP
