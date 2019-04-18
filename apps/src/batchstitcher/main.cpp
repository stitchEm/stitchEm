// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/config.hpp"

#include "batchwindow.hpp"

#include "libvideostitch-gui/mainwindow/signalhandler.hpp"
#include "libvideostitch-gui/mainwindow/uniqueqapplication.hpp"
#include "libvideostitch-gui/mainwindow/vslocalserver.hpp"

#include "libvideostitch-base/file.hpp"
#include "libvideostitch-base/logmanager.hpp"
#include "libgpudiscovery/delayLoad.hpp"

#include <memory>

#ifdef Q_OS_WIN
#define PLATFORM_SUFFIX "-win"
#elif defined(Q_OS_LINUX)
#define PLATFORM_SUFFIX "-linux"
#elif defined(Q_OS_MAC)
#define PLATFORM_SUFFIX "-mac"
#endif

static const QString STYLE_VARIABLES_FILE(":/assets/qss/style_variables.ini");
static const QString COMMON_STYLE_VARIABLES_FILE(":/style/common_style_variables.ini");

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

int main(int argc, char *argv[]) {
  // First, set organization and application names
  UniqueQApplication::initializeOrganization();
  QCoreApplication::setApplicationName(VIDEOSTITCH_BATCH_STITCHER_APP_NAME);

  // select best framework backend
  if (!VideoStitch::BackendLibHelper::selectBackend(VideoStitch::BackendLibHelper::getBestFrameworkAndBackend())) {
    std::cerr << "Unable to load "
              << VideoStitch::Discovery::getFrameworkName(VideoStitch::Discovery::getBestFramework()) << " backend."
              << std::endl;
    return 1;
  }

  UniqueQApplication app(argc, argv, VSBATCHKEY);
  SignalHandler::setupHandlers();

  app.setUpLogger();

  if (!app.uniqueInstance()) {
    if (argc == 1) {
      app.connectAndSend(Packet(Packet::WAKEUP), VSBATCHKEY);
    } else {
      QStringList arguments = app.arguments();
      arguments.pop_front();
      QString args;
      for (int i = 0; i < arguments.size(); i++) {
        args += arguments[i];
      }
      app.connectAndSend(Packet(Packet::OPEN_FILES, args.size(), args.toLocal8Bit()), VSBATCHKEY);
    }
  }
  app.loadStylesheetFile(":/style/vs_common.qss", STYLE_VARIABLES_FILE, COMMON_STYLE_VARIABLES_FILE);
  app.loadStylesheetFile(":/resources/batchstitcher.qss", STYLE_VARIABLES_FILE, COMMON_STYLE_VARIABLES_FILE);
  /*app.loadStylesheetFile(QString(":/assets/qss/vs_darkgray%0.qss").arg(PLATFORM_SUFFIX), STYLE_VARIABLES_FILE,
   * COMMON_STYLE_VARIABLES_FILE);*/
  QString fileToOpen;
  if (argc >= 1) {
    fileToOpen = app.arguments().value(1);
  }

  BatchWindow w(fileToOpen);
  if (app.uniqueInstance()) {
    w.show();
  }
  QObject::connect(&app, SIGNAL(messageAvailable(Packet)), &w, SLOT(processMessage(Packet)));
  return app.exec();
}
