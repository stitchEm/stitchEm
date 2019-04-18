// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef BATCHERTASK_HPP
#define BATCHERTASK_HPP

#include <QObject>
#include <QProcess>

class QProgressBar;
class VSCommandProcess;
class AutoElideLabel;

class BatcherTask : public QObject {
  Q_OBJECT

 public:
  enum State { Processing, Idle, Finished, Canceled, Error };
  typedef State State;

  static int lastID;
  explicit BatcherTask(QObject *parent = nullptr);

  static QStringList getTaskModel() { return QStringList() << tr("Project") << tr("Progress"); }

  bool operator<(const BatcherTask &toCompare);
  bool operator>(const BatcherTask &toCompare);

  void setPtvName(AutoElideLabel *name);
  void setTaskProgress(QProgressBar *taskProgress);

  void setID(int id);
  QString getLog() const;
  QString getPtvName() const;

  int getID() const;
  State getState() const;
  void resetState();
  bool isRunning() const;
 signals:
  void finished();
  void newLogLine(const QString &line);
 public slots:
  void start();
  void kill();
  void setDevices(const QList<int> &deviceIds);

 private slots:
  /**
   * @brief Slot called when the process state changes
   * @param newState
   */
  void processStateChanged(QProcess::ProcessState newState);

  void processFinished(int exitCode, QProcess::ExitStatus status);
  void logMessage(const QString &logLine);
  void setProgress(int current);
  void setState(State newState);

 private:
  void setErrorStyle(bool error);
  QString getStringFromState();
  void RetrieveFrameBounds(int &firstFrame, int &lastFrame) const;

  AutoElideLabel *ptvName;
  QProgressBar *taskProgress;
  VSCommandProcess *process;
  State state;
  int id;
  QString log;
  int firstFrame;
  int lastFrame;
  QList<int> devices;
};

#endif  // BATCHERTASK_HPP
