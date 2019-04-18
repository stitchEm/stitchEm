// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "processutility.hpp"

int ProcessUtility::getProcessByName(QString processName) {
  int processID = -1;
#ifdef Q_OS_WIN
  PROCESSENTRY32 entry;
  entry.dwSize = sizeof(PROCESSENTRY32);
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, NULL);
  if (Process32First(snapshot, &entry)) {
    while (Process32Next(snapshot, &entry)) {
      if ((QString::fromWCharArray(entry.szExeFile) == processName)) {
        processID = entry.th32ProcessID;
      }
    }
  }
#elif defined(Q_OS_LINUX)
  // Open the /proc directory
  QDir processDir = QDir("/proc");

  // Enumerate all entries in directory until process found
  QStringList processes = processDir.entryList();
  foreach (QString process, processes) {
    bool procIDFound;
    int id = process.toInt(&procIDFound);
    if (!procIDFound) {
      continue;
    }
    QString cmdPath = processDir.path() + QDir::separator() + process + QDir::separator() + "cmdline";
    QFile cmdFile(cmdPath);
    cmdFile.open(QIODevice::ReadOnly | QIODevice::Text);
    QString processCommand = cmdFile.readAll();
    QFileInfo info(processCommand);
    if (info.baseName() == processName) {
      processID = id;
      break;
    }
  }
#elif defined(Q_OS_MAC)
  char path[PROC_PIDPATHINFO_MAXSIZE];
  int numberOfProcesses = proc_listpids(PROC_ALL_PIDS, 0, NULL, 0);
  pid_t *pids = new pid_t[numberOfProcesses];
  bzero(pids, numberOfProcesses);
  proc_listpids(PROC_ALL_PIDS, 0, pids, sizeof(pid_t) * numberOfProcesses);
  for (int i = 0; i < numberOfProcesses; i++) {
    if (pids[i] != 0) {
      proc_pidpath(pids[i], path, sizeof(path));
      QFileInfo info(path);
      if (info.baseName() == processName) {
        processID = pids[i];
        break;
      }
    }
  }

  delete[] pids;
#endif
  return processID;
}
