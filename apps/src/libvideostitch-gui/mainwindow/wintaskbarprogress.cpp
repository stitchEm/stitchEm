// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "wintaskbarprogress.hpp"
#ifdef Q_OS_WIN
WinTaskbarProgress::WinTaskbarProgress(QWidget *parent)
    : QObject(parent), taskBarListHandler(nullptr), nativeHandle(HWND(parent->winId())) {}

WinTaskbarProgress::~WinTaskbarProgress() { cleanup(); }

void WinTaskbarProgress::setProgressState(TBPFLAG flag) {
  if (taskBarListHandler) {
    taskBarListHandler->SetProgressState(nativeHandle, flag);
  }
}

void WinTaskbarProgress::setProgressValue(ULONGLONG ullCompleted, ULONGLONG ullTotal) {
  if (taskBarListHandler) {
    taskBarListHandler->SetProgressValue(nativeHandle, ullCompleted, ullTotal);
  }
}

bool WinTaskbarProgress::init() {
  if (taskBarListHandler) {
    return true;
  }

  CoInitialize(NULL);

  CoCreateInstance(CLSID_TaskbarList, NULL, CLSCTX_INPROC_SERVER, IID_ITaskbarList3, (void **)&taskBarListHandler);

  if (taskBarListHandler) {
    return true;
  }

  CoUninitialize();
  return false;
}

void WinTaskbarProgress::cleanup() {
  if (taskBarListHandler) {
    taskBarListHandler->Release();
    CoUninitialize();
  }
}

#endif
