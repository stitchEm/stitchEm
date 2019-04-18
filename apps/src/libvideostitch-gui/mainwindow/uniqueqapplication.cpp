// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/config.hpp"

#include "uniqueqapplication.hpp"
#include "vslocalserver.hpp"
#include "vslocalsocket.hpp"
#include "vssettings.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include <QFileOpenEvent>
#include <QFile>
#include <QDir>
#include <QLocalSocket>
#include <QStringList>
#include <QProcess>
#include <QSharedMemory>
#include <QThread>
#include <QTranslator>

#include <cassert>

#ifdef _MSC_VER
#include <Windows.h>
#endif

class UniqueQApplication::Impl {
 public:
  QSharedMemory sharedMem;
  bool isUnique;
  QLocalSocket* socket;
  Packet packetToSend;
  QString key;
#ifdef _MSC_VER
  HANDLE uniqueVSMutex;  // used for the Windows installer - VSA-1525
#endif
  bool restartApp;
};

UniqueQApplication::UniqueQApplication(int& argc, char* argv[], QString applicationKey)
    : QApplication(argc, argv), impl(new Impl()) {
  impl->isUnique = false;
  impl->socket = new QLocalSocket(this);
  impl->key = applicationKey;
  impl->restartApp = false;
  setUpLocalServer();
}

UniqueQApplication::~UniqueQApplication() {
  cleanup();
#ifdef _MSC_VER
  if (impl->uniqueVSMutex) {
    ReleaseMutex(impl->uniqueVSMutex);
  }
#endif
  if (impl->restartApp) {
    QProcess proc;
    proc.startDetached(QApplication::applicationFilePath(), QStringList());
  }
}

UniqueQApplication* UniqueQApplication::instance() { return qobject_cast<UniqueQApplication*>(qApp); }

void UniqueQApplication::initializeOrganization() {
  QCoreApplication::setOrganizationName(VIDEOSTITCH_ORG_NAME);
  QCoreApplication::setOrganizationDomain(VIDEOSTITCH_ORG_DOMAIN);
}

void UniqueQApplication::cleanup() {
  // close the logManager in its own thread
  QMetaObject::invokeMethod(VideoStitch::Helper::LogManager::getInstance(), "close", Qt::BlockingQueuedConnection);

  QThread* loggerThread = findChild<QThread*>("logger.thread", Qt::FindDirectChildrenOnly);
  if (loggerThread) {
    loggerThread->quit();
    loggerThread->wait(100);
  }
  QThread* serverThread = findChild<QThread*>("local.server.thread", Qt::FindDirectChildrenOnly);
  if (serverThread) {
    serverThread->quit();
    serverThread->wait(100);
  }
  impl->sharedMem.detach();
}

void UniqueQApplication::loadStylesheetFile(QString stylesheetPath, QString stylesheetVariablesPath,
                                            QString commonStylesheetVariablesPath) {
  if (!QFile::exists(stylesheetPath)) {
    qDebug() << QString("Style %1 not found.").arg(stylesheetPath);
    return;
  }
  QFile stylesheetFile(stylesheetPath);
  if (!stylesheetFile.open(QFile::ReadOnly)) {
    qDebug() << QString("Could not open stylesheet %1.").arg(stylesheetPath);
    return;
  }
  QString stylesheetContent = QLatin1String(stylesheetFile.readAll());

  auto applyStylesheetVariables = [&stylesheetContent](QString variablesPath) {
    if (variablesPath.isEmpty()) {
      return;
    }
    if (!QFile::exists(variablesPath)) {
      qWarning() << QString("Style variables %1 not found.").arg(variablesPath);
    }

    QSettings variablesFile(variablesPath, QSettings::IniFormat);
    QStringList keys = variablesFile.childKeys();
    for (QString key : keys) {
      QString value = variablesFile.value(key).toString();
      for (QString otherKey : keys) {
        QString otherValue = variablesFile.value(otherKey).toString();
        otherValue.replace(key, value);
        variablesFile.setValue(otherKey, otherValue);
      }
    }
    for (QString key : variablesFile.childKeys()) {
      stylesheetContent.replace(key, variablesFile.value(key).toString());
    }
  };

  applyStylesheetVariables(stylesheetVariablesPath);
  applyStylesheetVariables(commonStylesheetVariablesPath);
  setStyleSheet(styleSheet() + stylesheetContent);
}

void UniqueQApplication::loadTranslationFiles() {
  QString language = VSSettings::getSettings()->getLanguage();
  QDir dir(applicationDirPath() + QString("/translations"));
  QString nameFilter = QString("*_%0.qm").arg(language);
  QStringList translationFiles = dir.entryList(QStringList() << nameFilter, QDir::Files);
  if (translationFiles.isEmpty() && language != "en") {
    nameFilter = QString("*_en.qm");
    translationFiles = dir.entryList(QStringList() << nameFilter, QDir::Files);
  }

  auto logManager = VideoStitch::Helper::LogManager::getInstance();
  logManager->writeToLogFile(QString("Found %0 translation files").arg(translationFiles.count()));

  for (QString filePath : translationFiles) {
    QTranslator* translator = new QTranslator(this);
    bool loaded = translator->load(filePath, dir.absolutePath());
    if (loaded) {
      if (installTranslator(translator)) {
        logManager->writeToLogFile(QString("Translation file %0 loaded and installed").arg(filePath));
      } else {
        logManager->writeToLogFile(QString("Translation file %0 loaded but failed to install").arg(filePath));
      }
    } else {
      delete translator;
      logManager->writeToLogFile(QString("Translation file %0 failed to load").arg(filePath));
    }
  }
}

QString UniqueQApplication::getYoutubeUrl() const { return property("youtube-url").toString(); }

QString UniqueQApplication::getTutorialUrl() const { return property("tutorial-url").toString(); }

void UniqueQApplication::setYoutubeUrl(QString url) { setProperty("youtube-url", url); }

void UniqueQApplication::setTutorialUrl(QString url) { setProperty("tutorial-url", url); }

bool UniqueQApplication::uniqueInstance() const { return impl->isUnique && impl->sharedMem.isAttached(); }

void UniqueQApplication::setUpLogger() {
  QThread* loggerThread = new QThread(this);
  loggerThread->setObjectName("logger.thread");
  loggerThread->start();

  VideoStitch::Helper::LogManager* logManager = VideoStitch::Helper::LogManager::getInstance();
  logManager->moveToThread(loggerThread);
  logManager->setUpLogger();
}

void UniqueQApplication::incomingClient(VSLocalSocket* instance) {
  connect(instance, &VSLocalSocket::messageReceived, this, &UniqueQApplication::receiveMessage);
}

void UniqueQApplication::receiveMessage(const Packet& packet) { emit messageAvailable(packet); }

void UniqueQApplication::connectAndSend(const Packet& pack, QString host, bool andDie) {
  impl->packetToSend = pack;
  connect(impl->socket, SIGNAL(connected()), this, SLOT(sendPacket()));
  if (andDie) {
    connect(impl->socket, SIGNAL(bytesWritten(qint64)), this, SLOT(quit()));
  } else {
    connect(impl->socket, SIGNAL(bytesWritten(qint64)), this, SLOT(closeSocket()));
  }
  impl->socket->connectToServer(host, QIODevice::WriteOnly);
}

void UniqueQApplication::sendPacket() {
  QByteArray packet;
  QDataStream out(&packet, QIODevice::WriteOnly);
  out << impl->packetToSend;
  impl->socket->write(packet);
  impl->socket->flush();
}

bool UniqueQApplication::event(QEvent* event) {
  switch (event->type()) {
    case QEvent::FileOpen:
      emit reqOpenFile(QStringList() << static_cast<QFileOpenEvent*>(event)->file());
      return true;
    default:
      return QApplication::event(event);
  }
}

void UniqueQApplication::setUpLocalServer() {
  QThread* serverThread = new QThread(this);
  serverThread->setObjectName("local.server.thread");

  QScopedPointer<VSLocalServer> localServer(new VSLocalServer());
  connect(serverThread, &QThread::finished, localServer.data(), &VSLocalServer::deleteLater);
  connect(localServer.data(), &VSLocalServer::newClient, this, &UniqueQApplication::incomingClient,
          Qt::BlockingQueuedConnection);

  qRegisterMetaType<Packet>("Packet");

  impl->sharedMem.setKey(impl->key);

  if (impl->sharedMem.attach()) {
#ifndef Q_OS_WIN
    impl->isUnique = localServer->tryToStartServer(impl->key);
    if (impl->isUnique) {
      localServer->moveToThread(serverThread);
      localServer.take();
      serverThread->start();
    }
#else
    impl->isUnique = false;
#endif
  } else {
    impl->isUnique = true;
    if (!impl->sharedMem.create(1)) {
      return;
    }

#ifdef _MSC_VER
    QString stringToConvert("VideoStitch-" + impl->key);
    impl->uniqueVSMutex = CreateMutex(NULL, true, (const wchar_t*)stringToConvert.utf16());
    if ((impl->uniqueVSMutex == NULL) && GetLastError() != ERROR_ALREADY_EXISTS) {
      assert(0);
    }
#endif
    localServer->tryToStartServer(impl->key);
    localServer->moveToThread(serverThread);
    localServer.take();
    serverThread->start();
  }
}

void UniqueQApplication::closeSocket() { impl->socket->disconnectFromServer(); }

void UniqueQApplication::restartApplication() {
  impl->restartApp = true;
  quit();
}
