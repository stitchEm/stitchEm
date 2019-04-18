// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mainwindow/mainwindow.hpp"
#include "mainwindow/postprodsettings.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/signalhandler.hpp"
#include "libvideostitch-gui/mainwindow/vslocalserver.hpp"
#include "libvideostitch-gui/utils/gpuhelper.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#include <QGLFormat>

#ifdef Q_OS_WIN
#define PLATFORM_SUFFIX "-win"
#elif defined(Q_OS_LINUX)
#define PLATFORM_SUFFIX "-linux"
#elif defined(Q_OS_MAC)
#define PLATFORM_SUFFIX "-mac"
#endif

#define MONTHS_TO_BETA_EXPIRATION 3
static const QString STYLE_VARIABLES_FILE(":/assets/qss/style_variables.ini");
static const QString COMMON_STYLE_VARIABLES_FILE(":/style/common_style_variables.ini");

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
FARPROC WINAPI delayFailureHook(unsigned dliNotify, PDelayLoadInfo pdli) {
  switch (dliNotify) {
    case dliFailLoadLib:
      throw std::runtime_error(std::string(pdli->szDll) + " could not be loaded");
    case dliFailGetProc:
      throw std::runtime_error("could not find procedure in module " + std::string(pdli->szDll));
    default:
      // this hook function should be called only with dliFailLoadLib or dliFailGetProc
      assert(false);
      throw std::runtime_error("Unknown error code");
      return 0;
  };
}
#endif  // DELAY_LOAD_ENABLED

#include "libvideostitch-gui/mainwindow/versionHelper.hpp"
void showNoDeviceWarning() {
  QString message =
      QCoreApplication::translate("GPU device check", "No CUDA nor OpenCL capable GPU detected on your system.");
  MsgBoxHandler::getInstance()->genericSync(
      message, QCoreApplication::translate("GPU device check", "No compatible device found"), CRITICAL_ERROR_ICON);
}

bool checkOtherInstance(UniqueQApplication& app, int argc) {
  if (!app.uniqueInstance()) {
    if (argc == 1) {
      app.connectAndSend(Packet(Packet::WAKEUP), VSSTUDIOKEY);
    } else {
      QStringList arguments = app.arguments();
      arguments.pop_front();
      QString args = arguments.join("");
      app.connectAndSend(Packet(Packet::OPEN_FILES, args.size(), args.toLocal8Bit()), VSSTUDIOKEY);
    }

#ifdef Q_OS_LINUX
    qDebug() << "This is not a unique instance of VideoStitch\n"
                "To remove the shared memory segment on Linux and Mac, please type the following command:\n"
                "SEM_ID=$( ipcs -m | awk \'{if($5==\"1\")  print $1 }\') && ipcrm -M $SEM_ID";
#endif

    return true;
  }

  return false;
}

int main(int argc, char* argv[]) {
  // First, set organization and application names
  UniqueQApplication::initializeOrganization();
  QCoreApplication::setApplicationVersion(APPS_VIDEOSTITCH_VERSION);
  QCoreApplication::setApplicationName(VIDEOSTITCH_STUDIO_APP_NAME);

#ifdef Q_OS_WIN
  AllowSetForegroundWindow(ASFW_ANY);
#endif

  QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
  QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

  QSurfaceFormat fmt;
  // https://www.khronos.org/opengl/wiki/Core_And_Compatibility_in_Contexts
  // we use the Fixed Function Pipeline to display video frames,
  // so we have to stick to 2.1 for Mac OS X
  fmt.setVersion(2, 1);
  fmt.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
  QSurfaceFormat::setDefaultFormat(fmt);

  std::unique_ptr<UniqueQApplication> app(new UniqueQApplication(argc, argv, VSSTUDIOKEY));

#ifdef Q_OS_LINUX
  std::setlocale(LC_NUMERIC, "C");
#endif

  if (!app) {
    std::cerr << "Unable to create the application... Check you have enough memory available." << std::endl;
    return 1;
  }

  SignalHandler::setupHandlers();
  app->setUpLogger();
  app->setYoutubeUrl(VIDEOSTITCH_YOUTUBE_STUDIO_URL);
  app->setTutorialUrl(VIDEOSTITCH_YOUTUBE_STUDIO_URL);
  PostProdSettings::createPostProdSettings();
  // Load stylesheets vs_common always goes first
  app->loadStylesheetFile(":/style/vs_common.qss", STYLE_VARIABLES_FILE, COMMON_STYLE_VARIABLES_FILE);
  app->loadStylesheetFile(":/videostitch-studio.qss", STYLE_VARIABLES_FILE, COMMON_STYLE_VARIABLES_FILE);
  app->loadTranslationFiles();

  VideoStitch::Discovery::Framework selectedFramework = VideoStitch::Discovery::Framework::Unknown;

  int mainDevice = PostProdSettings::getPostProdSettings()->getMainDevice();
  VideoStitch::Discovery::DeviceProperties mainDeviceProp;
  if (!VideoStitch::Discovery::getDeviceProperties(mainDevice, mainDeviceProp) ||
      !VideoStitch::BackendLibHelper::isBackendAvailable(mainDeviceProp.supportedFramework)) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        QStringLiteral("Selected device is not available anymore, selecting default device"));
    selectedFramework = VideoStitch::BackendLibHelper::getBestFrameworkAndBackend();
  } else {
    selectedFramework = mainDeviceProp.supportedFramework;
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QString("Selected device %0").arg(mainDevice));
  }

  bool needToRestart = false;
  if (VideoStitch::BackendLibHelper::selectBackend(selectedFramework, &needToRestart)) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        QString::fromStdString(VideoStitch::Discovery::getFrameworkName(selectedFramework)) +
        QStringLiteral(" backend selected for stitching"));
  } else {
    Discovery::Framework bestFramework = Discovery::getBestFramework();
    VideoStitch::GPU::checkGPUFrameworkAvailable(bestFramework);
    return 0;
  }

  // DON'T USE LIBVIDEOSTITCH BEFORE THIS LINE
  // Reason: libvideostitch is delay loaded on Windows. Before loading it (thus to use any symbol of it),
  // we should check that we have a usable GPU (this is the job of checkGPUFrameworkAvailable)
#ifdef DELAY_LOAD_ENABLED
  // failure hook, to prevent crash on delay load failures
  PfnDliHook oldFailureHook = __pfnDliFailureHook2;
  __pfnDliFailureHook2 = &delayFailureHook;
  try {
#endif
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        "Using " + QString::fromStdString(VideoStitch::Discovery::getFrameworkName(VideoStitch::GPU::getFramework())) +
        " backend for stitching");
#ifdef DELAY_LOAD_ENABLED
  } catch (std::exception& e) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        QStringLiteral("Error loading backend runtime library: ") + QString::fromUtf8(e.what()));
  }
  __pfnDliFailureHook2 = oldFailureHook;
#endif

  if (checkOtherInstance(*app, argc)) {
    return 0;
  }

  if (needToRestart) {
    MsgBoxHandler::getInstance()->genericSync(
        QCoreApplication::translate("GPU setup", "%0 needs to be restarted to work on your machine.")
            .arg(QCoreApplication::applicationName()),
        QCoreApplication::translate("GPU setup", "Setup completed"), INFORMATION_ICON);
#ifdef __APPLE__
    VideoStitch::BackendLibHelper::forceUpdateSymlink();
#endif
    return 0;
  }

  std::unique_ptr<MainWindow> wnd(new MainWindow());
  if (!wnd) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        "Unable to create the main window for the application. Check you have enough memory available.");
    return 1;
  }
  QObject::connect(app.get(), &UniqueQApplication::messageAvailable, wnd.get(), &MainWindow::processMessage);
  QObject::connect(app.get(), &UniqueQApplication::reqOpenFile, wnd.get(),
                   [&](const QStringList& files) { wnd->openFile(files); });
  QObject::connect(wnd.get(), &MainWindow::reqSendMessage, app.get(), &UniqueQApplication::connectAndSend);
  QObject::connect(wnd.get(), &MainWindow::reqRestartApplication, app.get(), &UniqueQApplication::restartApplication);
  wnd->openFile(QStringList() << app->arguments().value(1));
  return app->exec();
}
