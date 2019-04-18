// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "file.hpp"

#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QProcess>
#include <QRegExpValidator>
#include <QUrl>

File::Type File::getTypeFromFile(const QString& file) {
  QString extension = file.split(QDir::separator()).last();
  extension = extension.split(".").last();
  extension = extension.toLower();
  if (extension == "ptv" || extension == "ptvb") {
    return File::PTV;

  } else if (extension == "vah") {
    return File::VAH;

  } else if ((extension == "pto") || (extension == "pts")) {
    return File::CALIBRATION;

  } else if ((extension == "mp4") || (extension == "mov") || (extension == "mpg") || (extension == "avi") ||
             (extension == "mkv") || (extension == "mp2") || (extension == "3gp") || (extension == "m4v") ||
             (extension == "mpeg") || (extension == "ogv") || (extension == "ogg") || (extension == "wmv")) {
    return File::VIDEO;

  } else if ((extension == "tiff") || (extension == "tif") || (extension == "jpeg") || (extension == "jpg") ||
             (extension == "png")) {
    return File::STILL_IMAGE;

  } else {
    return File::UNKNOWN;
  }
}

QDir File::getFirstCommonDirectory(QList<QDir> directories) {
  QDir commonDirectory;
  int numberOfDirectories = 0;
  foreach (QDir dir, directories) {
    QString path = dir.path();
    if (numberOfDirectories <= path.split(QDir::separator()).size()) {
      numberOfDirectories = std::max(numberOfDirectories, path.split(QDir::separator()).size());
      commonDirectory = dir;
    }
  }
  bool commonDirFound = false;
  while (numberOfDirectories > 0 && !commonDirFound) {
    commonDirFound = true;
    for (int i = 0; i < directories.size(); i++) {
      commonDirFound = commonDirFound && directories[i] == commonDirectory;
      if ((int)directories[i].path().split(QDir::separator()).size() == numberOfDirectories) {
        directories[i].cdUp();
      }
    }

    if (!commonDirFound) {
      commonDirectory.cdUp();
      numberOfDirectories--;
    }
  }

  return commonDirectory;
}

QString File::suffixIfAlreadyExists(const QString& basename, const QString& extension) {
  if (!QFile::exists(basename + QString(".") + extension)) {
    return basename;
  }

  QString finalName = basename;
  QRegExpValidator validator(QRegExp("^.*\\(?\\d+\\)+$"));
  int i = 1, pos = 0;

  if (validator.validate(finalName, pos) != QRegExpValidator::Acceptable) {
    finalName.append("(1)");  // there is no (x), add one
  }
  QString placeholder;
  while (QFile::exists(finalName + QString(".") + extension) && i < 10) {
    placeholder = QString("(%0)").arg(i);
    finalName.replace(finalName.lastIndexOf(placeholder), placeholder.size(), QString("(%0)").arg(i + 1));
    i++;
  }
  return finalName;
}

QString File::getAppDataFolder() {
#ifdef Q_OS_WIN
  QString path = qgetenv("PROGRAMDATA");
  return path + QDir::separator() + QCoreApplication::applicationName() + QDir::separator();
#else
  return QDir::homePath() + QDir::separator() + "." + QCoreApplication::applicationName() + QDir::separator();
#endif
}

QString File::getVSDataFolder() {
#ifdef Q_OS_WIN
  QString path = qgetenv("PROGRAMDATA");
  return path + QDir::separator() + QCoreApplication::organizationName() + QDir::separator();
#else
  return QDir::homePath() + QDir::separator() + "." + QCoreApplication::organizationName() + QDir::separator();
#endif
}

QString File::getDocumentsLocation() {
  return QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation).first();
}

void File::showInShellExporer(const QString& pathIn) {
  if (pathIn.isEmpty()) {
    QDesktopServices::openUrl(QUrl(QDir::currentPath()));
  }
  if (!QFile::exists(pathIn)) {
    QDesktopServices::openUrl(QUrl(QFileInfo(pathIn).absolutePath()));
    return;
  }
  // Mac, Windows support folder or file.
#ifdef Q_OS_WIN
  const QString explorer = QString(qgetenv("WINDIR")) + QString(QDir::separator()) + "explorer.exe";

  QString param;
  if (!QFileInfo(pathIn).isDir()) param = QLatin1String("/select,");
  param += QDir::toNativeSeparators(pathIn);
  QProcess::startDetached(explorer, QStringList(param));
#elif defined(Q_OS_MAC)
  QStringList scriptArgs;
  scriptArgs << QLatin1String("-e")
             << QString::fromLatin1("tell application \"Finder\" to reveal POSIX file \"%0\"").arg(pathIn);
  QProcess::execute(QLatin1String("/usr/bin/osascript"), scriptArgs);
  scriptArgs.clear();
  scriptArgs << QLatin1String("-e") << QLatin1String("tell application \"Finder\" to activate");
  QProcess::execute("/usr/bin/osascript", scriptArgs);
#else
  QDesktopServices::openUrl(QUrl(pathIn));
#endif
}

QString File::strippedName(const QString& fullFileName) { return QFileInfo(fullFileName).fileName(); }

bool File::fileExists(const QString& filename) { return QFile::exists(QFileInfo(filename).absoluteFilePath()); }

QString File::normalizePath(const QString& filename) {
  QString normalizedFilename;

  if (filename.startsWith("//") || filename.startsWith("file://")) {
    // url, all backward slashes
    normalizedFilename = filename.mid(filename.indexOf("//") + 2);
    normalizedFilename.replace("\\", "/");
    normalizedFilename.replace("//", "/");
    normalizedFilename = filename.left(filename.indexOf("//") + 2) + normalizedFilename;
  } else {
    normalizedFilename = filename;
#ifdef Q_OS_WIN
    normalizedFilename.replace("//", "/");
    normalizedFilename.replace("/", "\\");
#else
    normalizedFilename.replace("\\", "/");
    normalizedFilename.replace("//", "/");
#endif
  }
  return normalizedFilename;
}
