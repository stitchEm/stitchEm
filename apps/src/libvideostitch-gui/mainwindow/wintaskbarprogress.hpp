// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>
#ifdef Q_OS_WIN
#include <ShObjIdl.h>
#endif
class VS_GUI_EXPORT WinTaskbarProgress : public QObject {
  Q_OBJECT
#ifdef Q_OS_WIN
 public:
  explicit WinTaskbarProgress(QWidget* parent);
  virtual ~WinTaskbarProgress();
  bool init();
  void cleanup();

 public slots:
  void setProgressState(TBPFLAG flag);
  void setProgressValue(ULONGLONG ullCompleted, ULONGLONG ullTotal);

 private:
  ITaskbarList3* taskBarListHandler;
  HWND nativeHandle;
#endif
};
