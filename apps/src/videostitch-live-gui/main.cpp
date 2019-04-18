// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "src/mainwindow.hpp"

#include <QApplication>
#include <QSurfaceFormat>

#include "libvideostitch-gui/mainwindow/uniqueqapplication.hpp"
#include "libvideostitch-gui/mainwindow/signalhandler.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/vslocalserver.hpp"
#include "libvideostitch-gui/utils/gpuhelper.hpp"

#include "libvideostitch-base/file.hpp"
#include "libvideostitch-base/logmanager.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#include "libvideostitch/logging.hpp"
#include "version.hpp"

#include "livesettings.hpp"
#include "guiconstants.hpp"

#include <memory>
#include <sstream>

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

// Sets Up the unique application instance
static void setUpApplication(UniqueQApplication* application);

// Connects the main window with the unique application
static void setUpMainWindow(UniqueQApplication* application, MainWindow* mainWindow);

// Check if Vahana is already running
static bool isRunningInstance(UniqueQApplication* application, int argc);

int main(int argc, char* argv[]) {
  // First, set organization and application names
  UniqueQApplication::initializeOrganization();
  QCoreApplication::setApplicationVersion(APPS_VIDEOSTITCH_VERSION);
  QCoreApplication::setApplicationName(VAHANA_VR_APP_NAME);
  QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
  QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

#ifdef Q_OS_WIN
  AllowSetForegroundWindow(ASFW_ANY);
#endif

  QScopedPointer<UniqueQApplication> liveApplication(new UniqueQApplication(argc, argv, VSLIVEKEY));
  liveApplication->setUpLogger();
  liveApplication->setYoutubeUrl(VIDEOSTITCH_YOUTUBE_VAHANA_URL);
  liveApplication->setTutorialUrl(VIDEOSTITCH_YOUTUBE_VAHANA_URL);

  LiveSettings::createLiveSettings();
  setUpApplication(liveApplication.data());

  // Exit if the application is not unique
  if (isRunningInstance(liveApplication.data(), argc)) {
    return 0;
  }

  {
    const auto bestFramework = VideoStitch::Discovery::getBestFramework(VideoStitch::Discovery::Framework::CUDA);
    if ((!VideoStitch::GPU::checkGPUFrameworkAvailable(bestFramework) ||
         !VideoStitch::BackendLibHelper::selectBackend(bestFramework))) {
      return 0;
    }
  }

  // DON'T USE LIBVIDEOSTITCH BEFORE THIS LINE
  // Reason: libvideostitch is delay loaded on Windows. Before to load it (thus to use any symbol of it),
  // we should check that we have a usable GPU (this is the job of checkGPUFrameworkAvailable)

  VideoStitch::Logger::setLevel(VideoStitch::Logger::LogLevel(LiveSettings::getLiveSettings()->getLogLevel()));

  QScopedPointer<MainWindow> mainWindow(new MainWindow());
  setUpMainWindow(liveApplication.data(), mainWindow.data());

  int result = liveApplication->exec();
  return result;
}

static void setUpApplication(UniqueQApplication* application) {
  // Load stylesheets vs_common always goes first
  application->loadStylesheetFile(QSS_FILE_COMMON, QSS_VARIABLE_FILE, QSS_COMMON_VARIABLE_FILE);
  application->loadStylesheetFile(QSS_FILE_VAHANA, QSS_VARIABLE_FILE, QSS_COMMON_VARIABLE_FILE);
  application->loadTranslationFiles();
  SignalHandler::setupHandlers();
}

static void setUpMainWindow(UniqueQApplication* application, MainWindow* mainWindow) {
  QObject::connect(application, &UniqueQApplication::messageAvailable, mainWindow, &MainWindow::processMessage);
  QObject::connect(application, &UniqueQApplication::reqOpenFile, mainWindow, &MainWindow::onFileOpened);
  QObject::connect(mainWindow, &MainWindow::reqSendMessage, application, &UniqueQApplication::connectAndSend);
  QObject::connect(mainWindow, &MainWindow::reqRestartApplication, application,
                   &UniqueQApplication::restartApplication);
  mainWindow->onFileOpened(QStringList() << application->arguments().value(1));
}

static bool isRunningInstance(UniqueQApplication* application, int argc) {
  if (!application->uniqueInstance()) {
    if (argc == 1) {
      application->connectAndSend(Packet(Packet::WAKEUP), VSLIVEKEY);
    } else {
      QStringList arguments = application->arguments();
      arguments.pop_front();
      const QString args = arguments.join("");
      application->connectAndSend(Packet(Packet::OPEN_FILES, args.size(), args.toLocal8Bit()), VSLIVEKEY);
    }
    return true;
  }
  return false;
}
