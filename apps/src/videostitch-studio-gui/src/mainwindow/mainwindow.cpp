// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mainwindow.hpp"
#include "ui_mainwindow.h"

#include "crasher.hpp"
#include "shortcutmanager.hpp"
#include "centralwidget/outputtabwidget.hpp"
#include "dialogs/extractdialog.hpp"
#include "dialogs/shortcutdialog.hpp"
#include "dialogs/jumpdialog.hpp"
#include "dialogs/preferencesdialog.hpp"
#include "mainwindow/postprodsettings.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/dialogs/resetdimensionsdialog.hpp"
#include "libvideostitch-gui/mainwindow/objectutil.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/timeconverter.hpp"
#include "libvideostitch-gui/mainwindow/ui_header/progressreporterwrapper.hpp"
#include "libvideostitch-gui/utils/imagesorproceduralsonlyfilterer.hpp"
#include "libvideostitch-gui/utils/notonlyvideosfilterer.hpp"
#include "libvideostitch-gui/utils/onevisualinputfilterer.hpp"
#include "libvideostitch-gui/utils/pluginshelpers.hpp"
#include "libvideostitch-gui/utils/widgetshelper.hpp"
#include "libvideostitch-gui/utils/inputformat.hpp"
#ifdef WITH_AUDIO_PLAYER
#include "libvideostitch-gui/videostitcher/audioplayer.hpp"
#endif
#include "libvideostitch-gui/widgets/aboutwindow.hpp"
#include "libvideostitch-gui/widgets/multivideowidget.hpp"
#include "libvideostitch-gui/videostitcher/presetsmanager.hpp"

#include "libvideostitch-base/yprsignalcaps.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/opengl.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/logging.hpp"
#include "version.hpp"

#include <QDesktopServices>
#include <QDesktopWidget>
#include <QFileDialog>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QProgressBar>
#include <QPushButton>
#include <QShortcut>
#include <QToolBar>
#include <QUndoStack>

#ifdef Q_OS_MAC
#include <ApplicationServices/ApplicationServices.h>
#endif

#include <fstream>
#include <vector>

static const size_t BytesInMB = 1024 * 1024;
static const int64_t MINIMUM_RESOLUTION_HEIGHT(360);

int getOpenGLDevice() {
  QOpenGLContext ctx;
  ctx.create();
  QOffscreenSurface surface;
  surface.create();
  ctx.makeCurrent(&surface);
  std::vector<int> glDevices = VideoStitch::getGLDevices();
  ctx.doneCurrent();
  return glDevices[0];
}

MainWindow::MainWindow()
    : StitchingWindow(),
      ui(new Ui::MainWindow),
      menuTabs(new QTabBar(this)),
      referenceOnProject(nullptr),
#ifdef WITH_AUDIO_PLAYER
      audioPlayer(new AudioPlayer(this)),
#endif
      state(GUIStateCaps::idle),
      stitchedSizeLabel(new QLabel(this)),
      usedGPUMem(0),
      totalGPUMem(0),
      showState(Normal),
      isMaxWindow(false) {

  MsgBoxHandler::getInstance();
  PresetsManager::getInstance();
  GlobalPostProdController::getInstance();

  QMainWindow::setWindowTitle(QCoreApplication::applicationName());
  setWindowIcon(QIcon(QPixmap(":/assets/logo/VideoStitch_Logo.png")));
  menuTabs->setObjectName("mainTabs");
  ui->setupUi(this);
  registerMetaTypes();
  buildWindowMenu();
  configureDockWidgets();
  buildStatusBar();
  buildMenuTab();
  initializeUndoFramework();
  // Controller and devices
  displayGpuNames();
  configureController();
  configureOculusDisplay();

  ShortcutManager::createInstance(this);

  // Load the Studio plugins
  VideoStitch::Plugin::loadPlugins(VideoStitch::Plugin::getCorePluginFolderPath().toStdString());
  auto path = VideoStitch::Plugin::getGpuCorePluginFolderPath();
  if (!path.isEmpty()) {
    VideoStitch::Plugin::loadPlugins(path.toStdString());
  }

  // Qt signals/slots, sorted by sender
  connectStitcherController();
  connectMainWindow();
  connectWidgets();

  connect(&gpuInfoUpdater, &GPUInfoUpdater::reqUpdateGPUInfo, this, &MainWindow::displayGPUInfo);

  // Calibration history
  loadFileHistory();
  loadCalibrationHistory();

  StateManager::getInstance()->registerObject(this);

#ifdef Q_OS_WIN
  progressThumbnail = new WinTaskbarProgress(this);
#endif

  gpuInfoUpdater.start();
  emit reqForceGPUInfoRefresh();

  const VSSettings* settings = VSSettings::getSettings();
  if (settings->contains("geometry")) {
    restoreGeometry(settings->getValue("geometry").toByteArray());
#ifdef Q_OS_OSX
    // TODO: Workaround for bug QTBUG-41679
    if (isFullScreen()) {
      setNormal();
    }
#endif
    show();
  } else {
    showMaximized();
  }

  QList<QShortcut*> shortcuts = findChildren<QShortcut*>();
  foreach (QShortcut* shortcut, shortcuts) { shortcut->setContext(Qt::ApplicationShortcut); }

  setWindowTitle();
  setTimeline();

  changeState(GUIStateCaps::idle);
}

MainWindow::~MainWindow() {
  // disconnect all except the &videoWidget which shares a mutex with videoStitcher.
  // this avoids objects to make some actions while being destroyed.
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  disconnect(videoStitcher, &StitcherController::projectInitialized, this, &MainWindow::onProjectInitialized);
  GlobalController::getInstance().deleteController();
  delete ui;
  MsgBoxHandler::getInstance()->destroy();
  ProjectFileHandler::getInstance()->destroy();
  ShortcutManager::getInstance()->destroy();
}

void MainWindow::loadFileHistory() {
  int nbRecentFiles = VSSettings::getSettings()->getRecentFileNumber();
  for (int i = 0; i < nbRecentFiles; ++i) {
    recentFilesActs.append(new QAction(this));
    connect(recentFilesActs[i], &QAction::triggered, this, &MainWindow::openRecentProject);
    ui->menuFile->insertAction(ui->actionExit, recentFilesActs[i]);
    recentFilesActs[i]->setVisible(false);
  }
  ui->menuFile->insertSeparator(ui->actionExit);
  if (!recentFilesActs.isEmpty()) {
    recentFilesActs.first()->setShortcut(QKeySequence("F5"));
  }

  updateRecentFileActions();
}

void MainWindow::loadCalibrationHistory() {
  int nbRecentFiles = VSSettings::getSettings()->getRecentFileNumber();
  for (int i = 0; i < nbRecentFiles; ++i) {
    recentCalibrationActs.append(new QAction(this));
    connect(recentCalibrationActs[i], &QAction::triggered, this, &MainWindow::openRecentCalibration);
    recentCalibrationActs[i]->setVisible(false);
  }
  if (!recentCalibrationActs.isEmpty()) {
    addAction(recentCalibrationActs.first());  // To be usable, the action should be in an available widget
    recentCalibrationActs.first()->setShortcut(QKeySequence("F4"));
  }

  updateRecentCalibrationActions();
}

void MainWindow::changeState(GUIStateCaps::State s) {
  state = s;
  switch (state) {
    case GUIStateCaps::disabled:
    case GUIStateCaps::frozen:
      break;
    case GUIStateCaps::idle:
      // change all GUI before destroying the backend
      setWindowTitle(QCoreApplication::applicationName());
      switchTab(CentralStackedWidget::VSTabWidget::welcome);
      ui->synchronizationWidget->setDisabled(true);
      ui->seekBarDock->setVisible(false);
      ui->seekBarDock->setFloating(false);
      ui->seekbar->setEnabled(false);
      ui->renderedFrame->setText("");
      hideCommandDocks();
      disableMenu(true);
      closeProject();
      updatePanoDimensions();
      setWindowModified(false);
      menuTabs->hide();
#ifdef Q_OS_MAC
      macxToolBar->hide();
#endif
      break;
    case GUIStateCaps::stitch: {
      if (referenceOnProject && referenceOnProject->getPanoConst().get()) {
        updatePanoDimensions(referenceOnProject->getPanoConst()->getWidth(),
                             referenceOnProject->getPanoConst()->getHeight());
      }
      menuTabs->show();
#ifdef Q_OS_MAC
      macxToolBar->show();
#endif
      break;
    }
    default:
      Q_ASSERT(0);
      VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QString("Error: unknown state: %0").arg(state));
      return;
  }

  const bool notIdleState = state != GUIStateCaps::idle;
  ui->actionToggle_Fullscreen->setEnabled(notIdleState);

  const bool notDisabledState = state != GUIStateCaps::disabled;
  ui->actionOpenPTV->setEnabled(notDisabledState);
  ui->actionOpen_Media->setEnabled(notDisabledState);

  const bool stitchState = state == GUIStateCaps::stitch;
  ui->actionExtract_stills->setEnabled(stitchState);
  ui->actionExtract_stills_to->setEnabled(stitchState);
  ui->actionSave_Project_As->setEnabled(stitchState);
  ui->actionSave_ptv->setEnabled(stitchState);
  ui->actionApply_Calibration->setEnabled(stitchState);
  ui->actionOpen_Working_Directory->setEnabled(stitchState);
  ui->actionNew_Calibration->setEnabled(stitchState);
  ui->actionExtract_output_to->setEnabled(stitchState);
  menuTabs->setEnabled(stitchState);
  disableMenu(!stitchState);
  // setAcceptDrops(notDisabledState);

  ShortcutManager::getInstance()->toggleConnections(notDisabledState);
  VideoStitch::Helper::toggleConnect(notDisabledState, ui->actionNew, SIGNAL(triggered()), this,
                                     SLOT(on_actionNew_triggered()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(notDisabledState, ui->actionNew_Calibration, SIGNAL(triggered()), this,
                                     SLOT(on_actionNew_Calibration_triggered()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(notDisabledState, ui->actionOpenPTV, SIGNAL(triggered()), this,
                                     SLOT(on_actionOpenPTV_triggered()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(notDisabledState, ui->actionSave_ptv, SIGNAL(triggered()), this,
                                     SLOT(on_actionSave_ptv_triggered()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(notDisabledState, ui->actionSave_Project_As, SIGNAL(triggered()), this,
                                     SLOT(on_actionSave_Project_As_triggered()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(notDisabledState, ui->actionApply_Calibration, SIGNAL(triggered()), this,
                                     SLOT(openCalibration()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(notDisabledState, ui->actionExtract_stills, SIGNAL(triggered()), this,
                                     SLOT(on_actionExtract_stills_triggered()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(notDisabledState, ui->actionExtract_stills_to, SIGNAL(triggered()), this,
                                     SLOT(on_actionExtract_stills_to_triggered()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(notDisabledState, ui->actionOpen_Working_Directory, SIGNAL(triggered()), this,
                                     SLOT(on_actionOpen_Working_Directory_triggered()), Qt::UniqueConnection);
}

void MainWindow::changeStateToStitch() { emit reqChangeState(GUIStateCaps::stitch); }

void MainWindow::jumpShortcutCalled() {
  if ((state != GUIStateCaps::stitch) || (menuTabs->currentIndex() == CentralStackedWidget::VSTabWidget::process)) {
    return;
  }
  JumpDialog dialog(ui->seekbar->getMinimumFrame(), ui->seekbar->getMaximumFrame(), this);
  connect(&dialog, &JumpDialog::reqSeek, ui->seekbar, &SeekBar::setValue);
  dialog.exec();
}

void MainWindow::tabShortcutCalled(int index) {
  if (state == GUIStateCaps::disabled || state == GUIStateCaps::idle) {
    return;
  }
  if (index == CentralStackedWidget::VSTabWidget::interactive &&
      referenceOnProject->getProjection() != "equirectangular") {
    return;
  }

  switchTab(index);
}

// File->Exit
void MainWindow::on_actionExit_triggered() { close(); }

// Help->About
void MainWindow::on_actionAbout_triggered() {
  AboutWidget about(VideoStitch::Helper::AppsInfo(QCoreApplication::applicationVersion()).toString(), this);
  QBoxLayout* layout = new QVBoxLayout();
  layout->addWidget(&about);
  QDialog* dialog = new QDialog();
  dialog->setWindowFlags(dialog->windowFlags() & (~Qt::WindowContextHelpButtonHint));
  dialog->setFixedSize(dialog->size());
  dialog->setLayout(layout);
  dialog->exec();
  layout->deleteLater();
  dialog->deleteLater();
}

// Help->Shortcut
void MainWindow::on_actionShortcut_triggered() {
  ShortcutDialog s(this);
  s.exec();
}

// Edit->Preferences
void MainWindow::on_actionPreferences_triggered() {
  PostProdSettings* postProdSettings = PostProdSettings::getPostProdSettings();
  QString language = postProdSettings->getLanguage();
  QVector<int> deviceIds = getCurrentDevices();
  PreferencesDialog preferencesWindow(this, deviceIds, language);

  if (preferencesWindow.exec() == QDialog::Rejected) {
    return;
  }
  postProdSettings->setDevices(deviceIds);
  postProdSettings->setLanguage(language);
}

// Help->support
void MainWindow::on_actionSupport_triggered() {
  QDesktopServices::openUrl(QUrl(VIDEOSTITCH_SUPPORT_URL, QUrl::TolerantMode));
}

// File -> Open working directory
void MainWindow::on_actionOpen_Working_Directory_triggered() {
  QDesktopServices::openUrl(QUrl::fromLocalFile(QDir::currentPath()));
}

//-------------------- GUI Event management -------------------------

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
  if (watched == menuTabs && event->type() == QEvent::Wheel) {
    return true;
  }

#ifdef Q_OS_MAC
  // TODO: Workaround for bug QTBUG-41679
  if (event->type() == QEvent::WindowStateChange) {
    if (this->windowState() & Qt::WindowFullScreen) {
      macxToolBar->hide();
      macxToolBar->show();
    }
  }
#endif

  return QWidget::eventFilter(watched, event);
}

void MainWindow::closeEvent(QCloseEvent* event) {
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  videoStitcher->closingProject();
  pause();
  if (saveModifiedPTV(true)) {
    event->accept();
  } else {
    event->ignore();
    return;
  }
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Closing " + QCoreApplication::applicationName());

#if defined(OCULUS_DISPLAY) && defined(Q_OS_WIN)
  oculusRenderWindow->stop();
  oculusRenderWindow->close();
#endif
  emit reqCloseProjectBlocking();
  stopThreads();

  if (showState == PreviewFullscreen) {
    on_actionToggle_Fullscreen_triggered();
  }

  VSSettings::getSettings()->setValue("geometry", saveGeometry());
}

void MainWindow::dropEvent(QDropEvent* e) {
  if (!e->mimeData()->hasUrls()) {
    return;
  }
  QList<QUrl> urlList = e->mimeData()->urls();
  if (!urlList.size()) return;
  QStringList files;
  QList<QDir> directories;

  foreach (QUrl u, urlList) {
    const QString path = u.toLocalFile();
    if (QFileInfo(path).isFile()) {
      files << path;
    } else {
      directories << QDir(path);
    }
  }

  while (directories.size()) {
    QDir directory = directories.first();
    QStringList entries = directory.entryList(QStringList(), QDir::Files | QDir::NoDotAndDotDot, QDir::Name);
    foreach (QString entry, entries) {
      QString entryPath = directory.absolutePath() + entry;
      if (File::getTypeFromFile(entry) == File::VIDEO && QFileInfo(entryPath).isFile()) {
        files << entryPath;
      }
    }
    directories.pop_front();
  }
  openFile(files);
  activateWindow();
}

void MainWindow::dragMoveEvent(QDragMoveEvent* e) { e->accept(); }

void MainWindow::dragEnterEvent(QDragEnterEvent* e) { e->acceptProposedAction(); }

//---------------------- Project management -------------------------

void MainWindow::onProjectInitialized(ProjectDefinition* project) {
  referenceOnProject = project;

  // Connect to the project
  connect(project, &PostProdProjectDefinition::reqDisplayWarning, this, &MainWindow::warningBox, Qt::UniqueConnection);
  connect(project, &PostProdProjectDefinition::hasBeenModified, this, &MainWindow::setWindowModified,
          Qt::UniqueConnection);
  connect(this, SIGNAL(reqSetWorkingArea(frameid_t, frameid_t)), project,
          SIGNAL(reqSetWorkingArea(frameid_t, frameid_t)), Qt::UniqueConnection);

  // Update the project in the child widgets
  ui->exposureWidget->onProjectOpened(referenceOnProject);
  ui->synchronizationWidget->onProjectOpened(referenceOnProject);
  ui->calibrationWidget->onProjectOpened(referenceOnProject);
  ui->blendingMaskWidget->onProjectOpened(referenceOnProject);
  ui->advancedBlendingWidget->onProjectOpened(referenceOnProject);
  ui->stabilizationWidget->onProjectOpened(referenceOnProject);
  ui->stereoWidget->onProjectOpened(referenceOnProject);
  ui->seekbar->setProject(referenceOnProject);
  ui->centralStackedWidget->getProcessTabWidget()->onProjectOpened(referenceOnProject);
  ui->centralStackedWidget->getSourceTabWidget()->setProject(referenceOnProject);
  ui->outputConfigurationWidget->setProject(referenceOnProject);

  // Various updates
  ImagesOrProceduralsOnlyFilterer::getInstance()->setProject(referenceOnProject);
  NotOnlyVideosFilterer::getInstance()->setProject(referenceOnProject);
  OneVisualInputFilterer::getInstance()->setProject(referenceOnProject);
  emit reqForceGPUInfoRefresh();
  setWindowTitle();
  ui->seekBarDock->setVisible(menuTabs->currentIndex() != CentralStackedWidget::process);
  setWindowModified(referenceOnProject->hasLocalModifications());
  updateInteractiveViewVisibility(VideoStitch::mapPTVStringToIndex.value(referenceOnProject->getProjection()),
                                  referenceOnProject->getHFOV());

  // This is THE place where the GUIStateCaps::stitch state is activated.
  // We want to be sure that we change state after every widget has the current project
  emit reqChangeState(GUIStateCaps::stitch);
}

void MainWindow::clearProject() {
  referenceOnProject = nullptr;

  // Update the project in the singleton
  ImagesOrProceduralsOnlyFilterer::getInstance()->setProject(nullptr);
  NotOnlyVideosFilterer::getInstance()->setProject(nullptr);
  OneVisualInputFilterer::getInstance()->setProject(nullptr);
  ProjectFileHandler::getInstance()->resetFilename();

  // Update the project in the child widgets
  ui->exposureWidget->clearProject();
  ui->synchronizationWidget->clearProject();
  ui->calibrationWidget->clearProject();
  ui->blendingMaskWidget->clearProject();
  ui->advancedBlendingWidget->clearProject();
  ui->stabilizationWidget->clearProject();
  ui->seekbar->clearProject();
  ui->centralStackedWidget->getProcessTabWidget()->clearProject();
  ui->centralStackedWidget->getSourceTabWidget()->clearProject();
  ui->outputConfigurationWidget->clearProject();
  ui->centralStackedWidget->getWelcomeTabWidget()->updateContent();

  // Various updates
  setWindowTitle();
}

void MainWindow::cleanStitcher() {
  emit reqForceGPUInfoRefresh();
  ui->seekBarDock->setVisible(menuTabs->currentIndex() != CentralStackedWidget::process);
  ui->seekbar->cleanStitcher();
  emit reqChangeState(GUIStateCaps::idle);
}

void MainWindow::switchFromProcessTabToSourceTab(bool shouldSwitch) {
  if (shouldSwitch && ui->centralStackedWidget->activeTab() == CentralStackedWidget::process) {
    switchTab(CentralStackedWidget::source);
  }
}

void MainWindow::switchToOutputTab(bool shouldSwitch) {
  if (shouldSwitch && ui->centralStackedWidget->activeTab() != CentralStackedWidget::output) {
    switchTab(CentralStackedWidget::output);
  }
}

void MainWindow::switchToSourceTab(bool shouldSwitch) {
  if (shouldSwitch && ui->centralStackedWidget->activeTab() != CentralStackedWidget::source) {
    switchTab(CentralStackedWidget::source);
  }
}

void MainWindow::switchToOutputOrInteractiveTab(bool shouldSwitch) {
  if (shouldSwitch && ui->centralStackedWidget->activeTab() != CentralStackedWidget::output &&
      ui->centralStackedWidget->activeTab() != CentralStackedWidget::interactive) {
    switchTab(CentralStackedWidget::output);
  }
}

void MainWindow::showOutputOrInteractiveTab() { switchToOutputOrInteractiveTab(true); }

void MainWindow::closeProject() {
  setWindowModified(false);
  updateRecentFileActions();
  ui->renderedFrame->clear();
  framerate.stop();
  ExtractDialog::resetLastCalibrationDirectory();
  emit reqCloseProject();
  ui->seekbar->reset();
  QUndoStack* undoStack = qApp->findChild<QUndoStack*>();
  if (undoStack) {
    undoStack->clear();
  }
}

void MainWindow::buildWindowMenu() {
  commands << ui->synchronizationWidgetDock;
  commands << ui->calibrationWidgetDock;
  commands << ui->blendingMaskWidgetDock;
  commands << ui->advancedBlendingWidgetDock;
  commands << ui->colorCorrectionWidgetDock;
  commands << ui->stabilizationWidgetDock;
  commands << ui->outputConfigDockWidget;
  commands << ui->stereoWidgetDock;
  utils << ui->logWidgetDock;
  utils << ui->commandWidgetDock;
  // Check all docks (except for the timeline) are in the Window menu.
  Q_ASSERT_X(getDocks().size() - 1 == commands.size() + utils.size(), "MainWindow",
             "Not all the dock widgets are added into the menu");
  setDocks(commands);
  ui->menuWindow->addSeparator();
  setDocks(utils);

#ifndef Q_OS_OSX
  // TODO: Workaround for bug QTBUG-41679
  ui->menuWindow->addSeparator();
  ui->menuWindow->addAction(ui->actionToggle_Fullscreen);
#endif

  // Group1: Synchro | Calib | Mask | Advanced | Expo | Stab | Output | Stereo
  tabify(commands);
  // Group2: Log | History
  tabify(utils);
}

void MainWindow::configureDockWidgets() {
  hideUtilsDocks();
  NotOnlyVideosFilterer::getInstance()->watch(ui->synchronizationWidgetDock);
  NotOnlyVideosFilterer::getInstance()->watch(ui->synchronizationWidgetDock->toggleViewAction());
  OneVisualInputFilterer::getInstance()->watch(ui->colorCorrectionWidgetDock);
  OneVisualInputFilterer::getInstance()->watch(ui->colorCorrectionWidgetDock->toggleViewAction());

  // Only visible when Experimental features are enabled
  ui->blendingMaskWidgetDock->toggleViewAction()->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->advancedBlendingWidgetDock->toggleViewAction()->setVisible(
      VSSettings::getSettings()->getShowExperimentalFeatures());

  // Only visible when Stereo settings are enabled
  ui->stereoWidgetDock->toggleViewAction()->setVisible(VSSettings::getSettings()->getIsStereo());

  // Configure associated tabs
  connect(ui->synchronizationWidgetDock->toggleViewAction(), &QAction::triggered, this,
          &MainWindow::switchFromProcessTabToSourceTab);
  connect(ui->calibrationWidgetDock->toggleViewAction(), &QAction::triggered, this,
          &MainWindow::switchToOutputOrInteractiveTab);
  connect(ui->blendingMaskWidgetDock->toggleViewAction(), &QAction::triggered, this,
          &MainWindow::switchToOutputOrInteractiveTab);
  connect(ui->advancedBlendingWidgetDock->toggleViewAction(), &QAction::triggered, this,
          &MainWindow::switchToOutputOrInteractiveTab);
  connect(ui->colorCorrectionWidgetDock->toggleViewAction(), &QAction::triggered, this,
          &MainWindow::switchToOutputOrInteractiveTab);
  connect(ui->stabilizationWidgetDock->toggleViewAction(), &QAction::triggered, this, &MainWindow::switchToOutputTab);
  connect(ui->outputConfigDockWidget->toggleViewAction(), &QAction::triggered, this,
          &MainWindow::switchToOutputOrInteractiveTab);
  connect(ui->stereoWidgetDock->toggleViewAction(), &QAction::triggered, this,
          &MainWindow::switchToOutputOrInteractiveTab);
}

void MainWindow::buildStatusBar() {
  QProgressBar* progressBar = new QProgressBar(statusBar());
  progressBar->hide();
  statusBar()->addPermanentWidget(progressBar);
  statusBar()->addPermanentWidget(stitchedSizeLabel);
  ui->gpuInfoLabel->setText("");
  statusBar()->addPermanentWidget(ui->gpuInfoLabel);
  ui->graphicsCardLabel->setText("");
  statusBar()->addPermanentWidget(ui->graphicsCardLabel);
  statusBar()->addPermanentWidget(ui->renderedFrame);
}

void MainWindow::buildMenuTab() {
  setTabPosition(Qt::AllDockWidgetAreas, QTabWidget::North);
  //: Tab name in the main window
  menuTabs->addTab(tr("Source"));
  //: Tab name in the main window
  menuTabs->addTab(tr("Output"));
  //: Tab name in the main window
  menuTabs->addTab(tr("Interactive"));
  //: Tab name in the main window
  menuTabs->addTab(tr("Process"));

#ifdef Q_OS_WIN
  ui->menuBar->setCornerWidget(menuTabs);
#else
  // menu tab: special treatment for macosx because we can't have tabs in the menubar.
  // the menu tab will be stored in a tool bar
  macxToolBar = new QToolBar(this);
  macxToolBar->setContentsMargins(0, 0, 10, 0);
  macxToolBar->setMovable(false);

  // create a spacer to align them on the right
  QWidget* spacer = new QWidget(this);
  spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  macxToolBar->addWidget(spacer);
  macxToolBar->addWidget(menuTabs);
  addToolBar(macxToolBar);
  macxToolBar->hide();
#endif
  menuTabs->installEventFilter(this);

#ifdef Q_OS_MAC
  this->installEventFilter(this);
#endif

  setMenuBar(ui->menuBar);
}

void MainWindow::displayGpuNames() { ui->graphicsCardLabel->setText(getGpuNames()); }

void MainWindow::configureController() {
  GlobalPostProdController::getInstance().createController(getCurrentDevices()[0]);
  controllerThread.setObjectName("Controller Thread");
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  PostProdStitcherController* postProdVideoStitcher = GlobalPostProdController::getInstance().getController();

  videoStitcher->moveToThread(&controllerThread);
  postProdVideoStitcher->moveToThread(&controllerThread);

  controllerThread.start();
}

void MainWindow::configureOculusDisplay() {
#if defined(OCULUS_DISPLAY) && defined(Q_OS_WIN)
  oculusRenderWindow = new StitcherRiftWindow(VSSettings::getSettings()->isStereo());
  if (!oculusRenderWindow->start()) {
    delete oculusRenderWindow;
    oculusRenderWindow = nullptr;
  }

  if (oculusRenderWindow) {
    StitcherController* videoStitcher = GlobalController::getInstance().getController();
    connect(videoStitcher, &StitcherController::oculusStereoVideoOutputCreated, oculusRenderWindow,
            &StitcherRiftWindow::connectToStereoVideoOutput);
    connect(videoStitcher, &StitcherController::oculusVideoOutputCreated, oculusRenderWindow,
            &StitcherRiftWindow::connectToVideoOutput);
  }
#endif
}

void MainWindow::registerMetaTypes() {
  qRegisterMetaType<frameid_t>("frameid_t");
  qRegisterMetaType<size_t>("mtime_t");
  qRegisterMetaType<GUIStateCaps::State>("GUIStateCaps::State");
  qRegisterMetaType<QList<QUrl>>("QList<QUrl>");
  qRegisterMetaType<std::vector<std::string>>("std::vector<std::string>");
  qRegisterMetaType<std::string>("std::string");
  qRegisterMetaType<CurveGraphicsItem::Type>("CurveGraphicsItem::Type");
  qRegisterMetaType<size_t>("size_t");
  qRegisterMetaType<VideoStitch::Status>("VideoStitch::Status");
  qRegisterMetaType<std::vector<std::pair<VideoStitch::Core::Curve*, CurveGraphicsItem::Type>>>(
      "std::vector<std::pair<VideoStitch::Core::Curve*,CurveGraphicsItem::Type> >");
  qRegisterMetaType<std::vector<std::pair<VideoStitch::Core::QuaternionCurve*, CurveGraphicsItem::Type>>>(
      "std::vector<std::pair<VideoStitch::Core::QuaternionCurve*,CurveGraphicsItem::Type> >");
  qRegisterMetaType<YPRSignalCaps*>("YPRSignalCaps*");
  qRegisterMetaType<QList<int>>("QList<int>");
  qRegisterMetaType<VideoStitch::Projection>("VideoStitch::Projection");
  qRegisterMetaType<QMap<int, QString>>("QMap<int,QString>");
  qRegisterMetaType<SignalCompressionCaps*>("SignalCompressionCaps*");
  qRegisterMetaType<std::vector<VideoStitch::Ptv::Value*>>("std::vector<VideoStitch::Ptv::Value*>");
  qRegisterMetaType<Crop>("Crop");
  qRegisterMetaType<QVector<Crop>>("QVector<Crop>");
  qRegisterMetaType<InputLensClass::LensType>("InputLensClass::LensType");
  qRegisterMetaType<VideoStitch::Quaternion<double>>("VideoStitch::Quaternion<double>");
  qRegisterMetaType<frameid_t>("frameid_t");
  qRegisterMetaType<VideoStitch::Core::PanoDefinition*>("VideoStitch::Core::PanoDefinition*");
  qRegisterMetaType<VideoStitch::Core::ControllerStatus>("VideoStitch::Core::ControllerStatus");
  qRegisterMetaType<std::shared_ptr<VideoStitch::Core::SourceRenderer>>(
      "std::shared_ptr<VideoStitch::Core::SourceRenderer>");
  qRegisterMetaType<QDropEvent*>("QDropEvent*");
  qRegisterMetaType<std::vector<std::tuple<VideoStitch::Input::VideoReader::Spec, bool, std::string>>>(
      "std::vector<std::tuple<VideoStitch::Input::VideoReader::Spec, bool, std::string> >");
}

void MainWindow::connectStitcherController() {
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  PostProdStitcherController* postProdVideoStitcher = GlobalPostProdController::getInstance().getController();
  DeviceVideoWidget& videoWidget = ui->centralStackedWidget->getOutputTabWidget()->getVideoWidget();
  InteractiveTabWidget* interactiveTab = ui->centralStackedWidget->getInteractiveTabWidget();
  OutputTabWidget* outputWidget = ui->centralStackedWidget->getOutputTabWidget();
  SourceWidget* sourceWidget = ui->centralStackedWidget->getSourceTabWidget();

#ifdef WITH_AUDIO_PLAYER
  connect(videoStitcher, &StitcherController::panoramaAudioOutputCreated, audioPlayer,
          &AudioPlayer::connectToAudioOutput);
#endif
  connect(videoStitcher, &StitcherController::statusMsg, statusBar(), &QStatusBar::showMessage);
  connect(videoStitcher, &StitcherController::reqCleanStitcher, this, &MainWindow::cleanStitcher);
  connect(videoStitcher, &StitcherController::reqDisplayWarning, this, &MainWindow::warningBox);
  connect(videoStitcher, &StitcherController::reqDisplayError, this, &MainWindow::errorBox);
  connect(videoStitcher, &StitcherController::reqResetDimensions, this, &MainWindow::resetDimensions);
  connect(videoStitcher, &StitcherController::projectReset, this, &MainWindow::clearProject,
          Qt::BlockingQueuedConnection);
  connect(videoStitcher, &StitcherController::notifyErrorMessage, this, &MainWindow::onGenericErrorMessage);
  connect(videoStitcher, &StitcherController::notifyEndOfStream, this, &MainWindow::onEndOfStreamReached);
  connect(videoStitcher, &StitcherController::snapshotPanoramaExported, this, &MainWindow::changeStateToStitch);
  connect(videoStitcher, &StitcherController::projectInitialized, this, &MainWindow::onProjectInitialized);
  connect(videoStitcher, &StitcherController::actionStarted, this, &MainWindow::startProgress);
  connect(videoStitcher, &StitcherController::actionStep, this, &MainWindow::updateProgress);
  connect(videoStitcher, &StitcherController::actionFinished, this, &MainWindow::finishProgress);
  connect(videoStitcher, &StitcherController::actionCancelled, this, &MainWindow::cancelProgress);
  connect(videoStitcher, &StitcherController::inputNumbersToggled, ui->outputConfigurationWidget,
          &OutputConfigurationWidget::enabledInputNumberButton);
  connect(videoStitcher, &StitcherController::reqCreateThumbnails, sourceWidget, &SourceWidget::createThumbnails);
  connect(videoStitcher, &StitcherController::reqCreatePanoView, this, &MainWindow::registerRenderer);

  connect(videoStitcher, &StitcherController::openFromInputFailed, this, &MainWindow::createNewProject);

  connect(videoStitcher, &StitcherController::reqUpdate, &outputWidget->getVideoWidget(),
          static_cast<void (DeviceVideoWidget::*)()>(&DeviceVideoWidget::update));
  connect(videoStitcher, &StitcherController::reqUpdate, interactiveTab->getInteractiveWidget(),
          static_cast<void (DeviceInteractiveWidget::*)()>(&DeviceInteractiveWidget::update));
  connect(videoStitcher, &StitcherController::reqUpdate, &sourceWidget->getMultiVideoWidget(),
          static_cast<void (MultiVideoWidget::*)()>(&MultiVideoWidget::update));
  connect(videoStitcher, &StitcherController::notifyProjectOpened, this, &MainWindow::onPtvOpened);
  connect(videoStitcher, &StitcherController::notifyBackendCompileProgress, this, &MainWindow::onKernelProgressChanged);
  connect(videoStitcher, &StitcherController::notifyBackendCompileDone, this, &MainWindow::onKernelCompileDone);
  connect(videoStitcher, &StitcherController::notifyStitcherReset, this, &MainWindow::onStitcherReset);
  connect(videoStitcher, &StitcherController::reqDisableWindow, this, &MainWindow::disableWindow);
  connect(videoStitcher, &StitcherController::reqEnableWindow, this, &MainWindow::enableWindow);

  connect(postProdVideoStitcher, &PostProdStitcherController::reqWarnWrongInputSize, this,
          &MainWindow::warnWrongInputSize);
  connect(postProdVideoStitcher, &PostProdStitcherController::reqCheckInputsDialog, this,
          &MainWindow::checkInputsDialog, Qt::BlockingQueuedConnection);
  connect(postProdVideoStitcher, &PostProdStitcherController::mergerNotSupportedByGpu, this,
          &MainWindow::manageMergerNotSupportedByGpu, Qt::BlockingQueuedConnection);
  connect(postProdVideoStitcher, &PostProdStitcherController::calibrationApplied, this,
          &MainWindow::showOutputOrInteractiveTab);
  connect(postProdVideoStitcher, &PostProdStitcherController::calibrationApplied, ui->outputConfigurationWidget,
          &OutputConfigurationWidget::updateSphereScaleAvailability);
  connect(postProdVideoStitcher, &PostProdStitcherController::calibrationApplied, ui->calibrationWidget,
          &CalibrationWidget::updateRigValues);
  connect(postProdVideoStitcher, &PostProdStitcherController::blendingMaskApplied, this,
          &MainWindow::showOutputOrInteractiveTab);
  connect(postProdVideoStitcher, &PostProdStitcherController::advancedBlendingApplied, this,
          &MainWindow::showOutputOrInteractiveTab);
  connect(postProdVideoStitcher, &PostProdStitcherController::exposureApplied, this,
          &MainWindow::showOutputOrInteractiveTab);
  connect(postProdVideoStitcher, &PostProdStitcherController::resetMergerApplied, this,
          &MainWindow::showOutputOrInteractiveTab);
  connect(postProdVideoStitcher, &PostProdStitcherController::resetAdvancedBlendingApplied, this,
          &MainWindow::showOutputOrInteractiveTab);
  connect(postProdVideoStitcher, &PostProdStitcherController::statusBarUpdate, this, &MainWindow::updateStatusBar);
  connect(postProdVideoStitcher, &PostProdStitcherController::projectOrientable, ui->seekbar->getCurvesTreeWidget(),
          &CurvesTreeWidget::onProjectOrientable);
  connect(postProdVideoStitcher, &PostProdStitcherController::reqUpdatePhotometry, ui->exposureWidget,
          &ExposureWidget::updatePhotometryResults);
  connect(postProdVideoStitcher, &PostProdStitcherController::projectOrientable, ui->stabilizationWidget,
          &StabilizationWidget::onProjectOrientable);
  connect(postProdVideoStitcher, &PostProdStitcherController::reqChangeProjection, &videoWidget,
          &DeviceVideoWidget::setProjection);
  connect(postProdVideoStitcher, &PostProdStitcherController::reqChangeProjection, this,
          &MainWindow::updateInteractiveViewVisibility);
  connect(postProdVideoStitcher, &PostProdStitcherController::panoChanged, ui->seekbar, &SeekBar::updateToPano);
  connect(postProdVideoStitcher, &PostProdStitcherController::notifyInputsOpened, this, &MainWindow::onInputsOpened);
}

void MainWindow::connectMainWindow() {
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  PostProdStitcherController* postProdVideoStitcher = GlobalPostProdController::getInstance().getController();
  SourceWidget* sourceWidget = ui->centralStackedWidget->getSourceTabWidget();

  connect(this, &MainWindow::reqOpenProject, videoStitcher, &StitcherController::openProject);
  connect(this, &MainWindow::reqResetProject, videoStitcher, &StitcherController::resetProject);
  connect(this, &MainWindow::reqCloseProject, videoStitcher, &StitcherController::closeProject);
  connect(this, &MainWindow::reqCloseProjectBlocking, videoStitcher, &StitcherController::closeProject,
          Qt::BlockingQueuedConnection);

  connect(this, &MainWindow::reqRestitch, videoStitcher, &StitcherController::restitchOnce);
  connect(this, &MainWindow::reqReextract, videoStitcher, &StitcherController::reextractOnce);

  connect(this, &MainWindow::reqSnapshotPanorama, videoStitcher, &StitcherController::onSnapshotPanorama);
  connect(this, &MainWindow::reqRotatePanorama, videoStitcher, &StitcherController::rotatePanorama);
  connect(this, &MainWindow::reqCancelKernelCompile, videoStitcher, &StitcherController::tryCancelBackendCompile,
          Qt::DirectConnection);

  connect(this, &MainWindow::reqSeekNStitch, postProdVideoStitcher, &PostProdStitcherController::stitch);
  connect(this, &MainWindow::reqSeekNExtract, postProdVideoStitcher, &PostProdStitcherController::extract);
  connect(this, &MainWindow::reqReset, postProdVideoStitcher, &PostProdStitcherController::reset);
  connect(this, &MainWindow::reqOpenInputs, postProdVideoStitcher, &PostProdStitcherController::openInputs,
          Qt::UniqueConnection);
  connect(this, &MainWindow::reqApplyTemplate, postProdVideoStitcher, &PostProdStitcherController::importTemplate,
          Qt::UniqueConnection);
  connect(this, &MainWindow::reqImportCalibration, postProdVideoStitcher,
          &PostProdStitcherController::importCalibration, Qt::UniqueConnection);
  connect(this, &MainWindow::reqSnapshotSources, postProdVideoStitcher, &PostProdStitcherController::snapshotSources);

  connect(this, &MainWindow::reqCloseProject, sourceWidget, &SourceWidget::clearThumbnails);
  connect(this, &MainWindow::reqCloseProject, &ui->centralStackedWidget->getOutputTabWidget()->getVideoWidget(),
          &DeviceVideoWidget::onCloseProject);
  connect(this, &MainWindow::reqCloseProject,
          ui->centralStackedWidget->getInteractiveTabWidget()->getInteractiveWidget(),
          &DeviceInteractiveWidget::onCloseProject);
  connect(this, &MainWindow::reqForceGPUInfoRefresh, &gpuInfoUpdater, &GPUInfoUpdater::refreshTick,
          Qt::DirectConnection);

  connect(ui->actionOpen_Media, &QAction::triggered, this, &MainWindow::openMedia);
}

void MainWindow::connectWidgets() {
  connectProcessWidget();

  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  PostProdStitcherController* postProdVideoStitcher = GlobalPostProdController::getInstance().getController();
  DeviceVideoWidget& videoWidget = ui->centralStackedWidget->getOutputTabWidget()->getVideoWidget();
  SourceWidget* sourceWidget = ui->centralStackedWidget->getSourceTabWidget();
  InteractiveTabWidget* interactiveTabWidget = ui->centralStackedWidget->getInteractiveTabWidget();
  ProcessTabWidget* processTabWidget = ui->centralStackedWidget->getProcessTabWidget();
  OutputTabWidget* outputWidget = ui->centralStackedWidget->getOutputTabWidget();
  WelcomeScreenWidget* welcomeWidget = ui->centralStackedWidget->getWelcomeTabWidget();

  // Connect welcome tab events
  // TODO: FI-681 replace the call to the media manager when it's ready
  connect(welcomeWidget, &WelcomeScreenWidget::notifyNewProject, this, &MainWindow::openMedia);
  connect(welcomeWidget, &WelcomeScreenWidget::notifyProjectOpened, this, &MainWindow::on_actionOpenPTV_triggered);
  connect(welcomeWidget, &WelcomeScreenWidget::notifyFilesDropped, this, &MainWindow::dropInput);
  connect(welcomeWidget, &WelcomeScreenWidget::notifyProjectSelected,
          [=](const QString file) { openFile(QStringList() << file); });

  // notify a new panorama is up
  connect(&videoWidget, &DeviceVideoWidget::gotFrame, this, &MainWindow::refresh);
  connect(outputWidget, &OutputTabWidget::notifyUploadError, this, &MainWindow::onGenericErrorMessage);
  connect(outputWidget, &OutputTabWidget::reqResetDimensions, this, &MainWindow::resetDimensions);
  connect(outputWidget, &OutputTabWidget::reqRestitch, videoStitcher, &StitcherController::restitchOnce);

  // notify interactive view is ready
  connect(interactiveTabWidget, &InteractiveTabWidget::reqRestitch, videoStitcher, &StitcherController::restitchOnce);

  // a set of frames has been extracted:
  connect(&sourceWidget->getMultiVideoWidget(), &MultiVideoWidget::gotFrame, this, &MainWindow::refresh);
  connect(sourceWidget, &SourceWidget::sendDropEvent, this, &MainWindow::dropInput);
  connect(sourceWidget, &SourceWidget::reqResetDimensions, this, &MainWindow::resetDimensions);
  connect(sourceWidget, &SourceWidget::reqReextract, videoStitcher, &StitcherController::reextractOnce);

  // Connect timeline events
  connect(ui->seekbar, &SeekBar::reqPlay, this, &MainWindow::play);
  connect(ui->seekbar, &SeekBar::reqPause, this, &MainWindow::pause);
  connect(ui->seekbar, &SeekBar::reqSeek, this, &MainWindow::seekFrame);
  connect(ui->seekbar, &SeekBar::reqUpdateSequence, ui->synchronizationWidget, &SynchronizationWidget::updateSequence);
  connect(ui->seekbar, &SeekBar::reqUpdateSequence, ui->stabilizationWidget, &StabilizationWidget::updateSequence);
  connect(ui->seekbar, &SeekBar::reqUpdateSequence, ui->calibrationWidget, &CalibrationWidget::updateSequence);
  connect(ui->seekbar, &SeekBar::reqUpdateSequence, ui->exposureWidget, &ExposureWidget::updateSequence);
  connect(ui->seekbar, &SeekBar::reqUpdateSequence, processTabWidget, &ProcessTabWidget::reqUpdateSequence);

  connect(menuTabs, &QTabBar::currentChanged, this, &MainWindow::switchTab);
  connect(ui->stereoWidget, &StereoConfigurationWidget::switchOutput, postProdVideoStitcher,
          &PostProdStitcherController::switchOutputAndRestitch);
  connect(&videoWidget, &VideoWidget::applyOrientation, postProdVideoStitcher,
          &PostProdStitcherController::selectOrientation);
  connect(&videoWidget, &VideoWidget::rotatePanorama,
          [=](YPRSignalCaps* rotations) { emit reqRotatePanorama(rotations, true); });

  // Connect stabilization widget
  connect(ui->stabilizationWidget, &StabilizationWidget::reqPause, this, &MainWindow::pause);
  connect(ui->stabilizationWidget, &StabilizationWidget::reqApplyStabilization, postProdVideoStitcher,
          &PostProdStitcherController::applyStabilization);

  connect(ui->stabilizationWidget, &StabilizationWidget::reqSetEditOrientationActivated, &videoWidget,
          &VideoWidget::setEditOrientationActivated);
  connect(ui->stabilizationWidget, &StabilizationWidget::reqSetEditOrientationActivated, this,
          &MainWindow::switchToOutputTab);

  connect(ui->exposureWidget, &ExposureWidget::reqPause, this, &MainWindow::pause);
  connect(ui->exposureWidget, &ExposureWidget::reqApplyExposure, postProdVideoStitcher,
          &PostProdStitcherController::applyExposure);
  connect(ui->exposureWidget, &ExposureWidget::reqApplyPhotometricCalibration, postProdVideoStitcher,
          &PostProdStitcherController::applyPhotometricCalibration);

  // Connect synchro widget
  connect(ui->synchronizationWidget, &SynchronizationWidget::reqPause, this, &MainWindow::pause);
  connect(ui->synchronizationWidget, &SynchronizationWidget::reqTogglePlay, this, &MainWindow::togglePlayOrPause);
  connect(ui->synchronizationWidget, &SynchronizationWidget::reqApplySynchronization, postProdVideoStitcher,
          &PostProdStitcherController::applySynchronization);

  // Connect calibration widget
  connect(ui->calibrationWidget, &CalibrationWidget::reqSeek, ui->seekbar, &SeekBar::setValue);
  connect(ui->calibrationWidget, &CalibrationWidget::reqPause, this, &MainWindow::pause);
  connect(ui->calibrationWidget, &CalibrationWidget::reextract, this, &MainWindow::reqReextract);
  connect(ui->calibrationWidget, &CalibrationWidget::reqApplyCalibration, postProdVideoStitcher,
          &PostProdStitcherController::applyCalibration);
  connect(ui->calibrationWidget, &CalibrationWidget::reqImportCalibration,
          [=]() { ui->actionApply_Calibration->triggered(); });
  connect(ui->calibrationWidget, &CalibrationWidget::reqApplyCrops, postProdVideoStitcher,
          &PostProdStitcherController::applyCrops);
  connect(ui->calibrationWidget, &CalibrationWidget::reqRegisterRender, videoStitcher,
          &StitcherController::registerSourceRender);
  connect(ui->calibrationWidget, &CalibrationWidget::reqUnregisterRender, videoStitcher,
          &StitcherController::unregisterSourceRender, Qt::BlockingQueuedConnection);

  // Connect blending mask widget
  connect(ui->blendingMaskWidget, &BlendingMaskWidget::reqPause, this, &MainWindow::pause);
  connect(ui->blendingMaskWidget, &BlendingMaskWidget::reqApplyBlendingMask, postProdVideoStitcher,
          &PostProdStitcherController::applyBlendingMask);
  connect(ui->blendingMaskWidget, &BlendingMaskWidget::reqSeek, ui->seekbar, &SeekBar::setValue);

  // Connect advandec blending widget
  connect(ui->advancedBlendingWidget, &AdvancedBlendingWidget::reqApplyAdvancedBlending, postProdVideoStitcher,
          &PostProdStitcherController::applyAdvancedBlending);
  connect(ui->advancedBlendingWidget, &AdvancedBlendingWidget::reqResetAdvancedBlending, postProdVideoStitcher,
          &PostProdStitcherController::resetAdvancedBlending);

  // Connect output config widget
  connect(ui->outputConfigurationWidget, &OutputConfigurationWidget::reqResetMerger, postProdVideoStitcher,
          &PostProdStitcherController::resetMerger);
  connect(ui->outputConfigurationWidget, &OutputConfigurationWidget::reqSetProjection, postProdVideoStitcher,
          &PostProdStitcherController::setProjection);
  connect(ui->outputConfigurationWidget, &OutputConfigurationWidget::reqSetSphereScale, videoStitcher,
          &StitcherController::setSphereScale);
}

void MainWindow::connectProcessWidget() {
  ProcessTabWidget* processTabWidget = ui->centralStackedWidget->getProcessTabWidget();
  StitcherController* videoStitcher = GlobalController::getInstance().getController();

  connect(processTabWidget, &ProcessTabWidget::reqSendToBatch, this, &MainWindow::sendToBatch);
  connect(processTabWidget, &ProcessTabWidget::panoSizeChanged, this, &MainWindow::updatePanoDimensions);
  connect(processTabWidget, &ProcessTabWidget::panoSizeChanged, ui->outputConfigurationWidget,
          &OutputConfigurationWidget::updateIncompatibleProjectionWarning);
  connect(processTabWidget, &ProcessTabWidget::reqSavePtv, this, &MainWindow::on_actionSave_ptv_triggered);
  connect(processTabWidget, &ProcessTabWidget::reqChangeAudioInput, videoStitcher, &StitcherController::setAudioInput);
  connect(processTabWidget, &ProcessTabWidget::reqReset, videoStitcher, &StitcherController::onReset);
#ifdef Q_OS_WIN
  connect(processTabWidget, &ProcessTabWidget::reqStopProgress, this, &MainWindow::stopWinProgress);
  connect(processTabWidget, &ProcessTabWidget::reqChangeProgressState, this, &MainWindow::changeWinProgressState);
  connect(processTabWidget, &ProcessTabWidget::reqChangeProgressValue, this, &MainWindow::changeWinProgressValue);
  connect(processTabWidget, &ProcessTabWidget::reqStartProgress, this, &MainWindow::startWinProgress);
#endif
}

void MainWindow::stopThreads() {
  // quit controller thread
  controllerThread.quit();
  controllerThread.wait();

  // quit gpuInfoUpdater
  gpuInfoUpdater.quit();
  gpuInfoUpdater.wait();
}

Docks MainWindow::getDocks() const {
  Docks docks;
  for (auto object : findChildren<QDockWidget*>()) {
    docks << object;
  }
  return docks;
}

void MainWindow::openRecentProject() {
  const QAction* action = qobject_cast<const QAction*>(sender());
  if (action) {
    QString recentFileName = action->data().toString();
    if (!QFile::exists(recentFileName)) {
      MsgBoxHandler::getInstance()->generic(tr("Error: couldn't open the file %0.").arg(recentFileName),
                                            tr("File not found"), CRITICAL_ERROR_ICON);
      return;
    }
    openFile(QStringList() << recentFileName);
  }
}

//------------------------- Tab management ------------------------

int MainWindow::tabCount() const { return menuTabs->count(); }

void MainWindow::switchTab(int tab) {
  // The tab can be already selected since QTabBar/QTabWidget don't allow to select no tab
  // so we update menuTabs here and we must never switch tab on it directly
  disconnect(menuTabs, &QTabBar::currentChanged, this, &MainWindow::switchTab);
  menuTabs->setCurrentIndex(tab);
  connect(menuTabs, &QTabBar::currentChanged, this, &MainWindow::switchTab);

  // First, update depending of the previous tab
  if (ui->centralStackedWidget->activeTab() == CentralStackedWidget::process ||
      ui->centralStackedWidget->activeTab() == CentralStackedWidget::welcome) {
    restoreOpenedDocks();
  }

  // Then update depending of the current tab
  ui->seekBarDock->setVisible(tab != CentralStackedWidget::process);
  ui->seekbar->setEnabled(tab != CentralStackedWidget::welcome);

  switch (tab) {
    case CentralStackedWidget::welcome:
      ui->centralStackedWidget->activate(CentralStackedWidget::welcome);
      ui->centralStackedWidget->getSourceTabWidget()->repaint();  // see VSA-356
      ui->seekBarDock->setVisible(false);
      break;
    case CentralStackedWidget::source:
      ui->centralStackedWidget->activate(CentralStackedWidget::source);

      ui->seekbar->setEnabled(true);
      ui->seekBarDock->setVisible(true);
      restoreOpenedDocks();
      //      if (!GlobalController::getInstance().getController()->getClock().isActive()) {
      //        emit reqReextract();
      //      }
      break;
    case CentralStackedWidget::output:
      QMetaObject::invokeMethod(GlobalPostProdController::getInstance().getController(), "ensureProjectionIsValid",
                                Qt::AutoConnection);
      ui->centralStackedWidget->activate(CentralStackedWidget::output);

      ui->seekbar->setEnabled(true);
      ui->seekBarDock->setVisible(true);
      restoreOpenedDocks();
      //      if (!GlobalController::getInstance().getController()->getClock().isActive()) {
      //        emit reqRestitch();
      //      }

      break;
    case CentralStackedWidget::interactive:
      ui->centralStackedWidget->activate(CentralStackedWidget::interactive);
      break;

    case CentralStackedWidget::process:
      pause();
      ui->centralStackedWidget->activate(CentralStackedWidget::process);
      saveOpenedDocks();
      hideCommandDocks();
      break;
    default:
      Q_ASSERT(0);
      break;
  }

  updateClockForTab(tab);
}

//----------------------- Video Player --------------------

void MainWindow::play() {
  if (ui->seekbar->getCurrentFrameFromCursorPosition() < ui->seekbar->getMaximumFrame()) {
    GlobalController::getInstance().getController()->play();
    ui->seekbar->play();
    framerate.start();
  } else {
    ui->seekbar->pause();
  }
}

void MainWindow::pause() {
  GlobalController::getInstance().getController()->pause();
  ui->seekbar->pause();
}

void MainWindow::togglePlayOrPause() {
  if (ui->centralStackedWidget->allowsPlayback()) {
    if (ui->seekbar->isEnabled()) {
      if (GlobalController::getInstance().getController()->isPlaying()) {
        pause();
      } else {
        play();
      }
    }
  }
}

void MainWindow::seekFrame(SignalCompressionCaps* caps, frameid_t frame) {
  switch (ui->centralStackedWidget->activeTab()) {
    case CentralStackedWidget::source:
    case CentralStackedWidget::process:
      emit reqSeekNExtract(frame, caps);
      break;
    case CentralStackedWidget::output:
    case CentralStackedWidget::interactive:
      emit reqSeekNStitch(frame, caps);
      break;
    case CentralStackedWidget::welcome:
    default:
      return;
  }
}

// this one is called back each time a virtual frame (pano frame and/or input frames) is ready on the GPU to be
// displayed
void MainWindow::refresh(mtime_t date) {
  ui->seekbar->refresh(date);
  ui->exposureWidget->refresh(date);
  ui->calibrationWidget->refresh(date);
  ui->blendingMaskWidget->refresh(date);
  if (GlobalController::getInstance().getInstance().getController()->isPlaying()) {
    framerate.tick();
    QString message = tr("Rendered frame ") + TimeConverter::dateToTimeDisplay(date);
    float frrate = framerate.getFramerate();
    if (frrate > 0) {
      message += " @ " + QString::number(frrate, 'g', 3) + "fps";
    }
    ui->renderedFrame->setText(message);
  }
}

void MainWindow::selectAllVideo() {
  if (ui->seekbar->isEnabled()) {
    emit reqSetWorkingArea(0, GlobalPostProdController::getInstance().getController()->getLastStitchableFrame());
  }
}

//------------------------ Open / Close ------------------------

// File->New... This closes the project
void MainWindow::on_actionNew_triggered() {
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QString("Create a new project"));
  pause();
  framerate.stop();
  if (!saveModifiedPTV(true)) {
    return;
  }
  emit reqResetProject();
  emit reqChangeState(GUIStateCaps::idle);
}

// File->Open Project
void MainWindow::on_actionOpenPTV_triggered() {
  QString file =
      QFileDialog::getOpenFileName(this, tr("Open project"), ProjectFileHandler::getInstance()->getWorkingDirectory(),
                                   tr("%0 Project (*.ptvb *.ptv)").arg(QCoreApplication::applicationName()));
  openFile(QStringList() << file);
}

// File->Open media
void MainWindow::openMedia() {
  QStringList files =
      QFileDialog::getOpenFileNames(this, tr("Open media"), ProjectFileHandler::getInstance()->getWorkingDirectory(),
                                    QString("%0 %1;;%2 %3;;%4 (*)")
                                        .arg(tr("Videos"))
                                        .arg(VideoStitch::InputFormat::VIDEO_FORMATS)
                                        .arg(tr("Images"))
                                        .arg(VideoStitch::InputFormat::IMAGE_FORMATS)
                                        .arg(tr("All Files")));
  openFile(files);
}

void MainWindow::checkInputsDialog(QString& newFolder) {
  if (MsgBoxHandler::getInstance()->genericSync(
          tr("%0 can't find some of the input videos. Do you want to look for these in another folder?")
              .arg(QCoreApplication::applicationName()),
          tr("Missing inputs"), WARNING_ICON, QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
    newFolder = QFileDialog::getExistingDirectory(this);
  }
}

void MainWindow::manageMergerNotSupportedByGpu(const std::string& merger, bool& fallBackToLinearBlending) {
  int result =
      MsgBoxHandler::getInstance()->genericSync(tr("The project image merger '%0' is not supported by %1 with your "
                                                   "GPU. Do you want to fall back to linear blending?"
                                                   "\nOtherwise, the project will be closed.")
                                                    .arg(QString::fromStdString(merger))
                                                    .arg(QCoreApplication::applicationName()),
                                                tr("Warning"), WARNING_ICON, QMessageBox::Yes | QMessageBox::No);
  fallBackToLinearBlending = (result == QMessageBox::Yes);
}

// ------------------------------- Calibration Menu -----------------------------------

// Calibration / new calibration
void MainWindow::on_actionNew_Calibration_triggered() {
  if (state == GUIStateCaps::idle) {
    return;
  }
  QString path = QFileDialog::getExistingDirectory(this, tr("Select the destination directory"), QDir::currentPath());
  if (path.isEmpty() || path.isNull()) {
    return;
  }
  emit reqSnapshotSources(path, true);
}

// Calibration->Extract stills to
void MainWindow::on_actionExtract_stills_to_triggered() { this->snapshotSources(true); }

// Calibration->Extract stills
void MainWindow::on_actionExtract_stills_triggered() { this->snapshotSources(false); }

// Calibration->Apply calibration
void MainWindow::openCalibration(QString templateFile) {
  if (state == GUIStateCaps::idle) {
    return;
  }

  QString file = templateFile;
  if (file.isEmpty()) {
    file = QFileDialog::getOpenFileName(
        this, tr("Apply calibration"), "",
        (tr("All supported templates ") + "(*.ptv *.ptvb *.pto *.pts);;" + tr("PTGui or Hugin Calibration ") +
         "(*.pto *.pts);;" + tr("%0 Project ").arg(QCoreApplication::applicationName()) + "(*.ptv *.ptvb)"));
  }
  if (file.isEmpty()) {
    return;
  }

  updateRecentCalibration(file);
  if (state == GUIStateCaps::stitch) {
    switch (File::getTypeFromFile(file)) {
      case File::CALIBRATION:
        emit reqImportCalibration(file);
        break;
      case File::PTV:
        emit reqApplyTemplate(file);
        break;
      default:
        Q_ASSERT(0);
    }
  }
}

void MainWindow::startProgress() {
  QProgressBar* progressBar = statusBar()->findChild<QProgressBar*>();
  progressBar->setValue(0);
  progressBar->show();
}

void MainWindow::updateProgress(int progress) {
  QProgressBar* progressBar = statusBar()->findChild<QProgressBar*>();
  progressBar->setValue(progress);
  if (progress != progressBar->maximum()) {
    progressBar->show();
  }
}

void MainWindow::finishProgress() {
  QProgressBar* progressBar = statusBar()->findChild<QProgressBar*>();
  progressBar->setValue(progressBar->maximum());
  QTimer::singleShot(1000, progressBar, SLOT(hide()));
}

void MainWindow::cancelProgress() {
  QProgressBar* progressBar = statusBar()->findChild<QProgressBar*>();
  progressBar->hide();
}

void MainWindow::updateInteractiveViewVisibility(Projection proj, double) {
  menuTabs->setTabEnabled(CentralStackedWidget::interactive, proj == Projection::equirectangular ||
                                                                 proj == Projection::cubemap ||
                                                                 proj == Projection::equiangular_cubemap);
}

void MainWindow::onPtvOpened() {
  if (ui->centralStackedWidget->activeTab() != CentralStackedWidget::output) {
    switchTab(CentralStackedWidget::output);
  }
}

void MainWindow::onInputsOpened() {
  if (ui->centralStackedWidget->activeTab() != CentralStackedWidget::source) {
    switchTab(CentralStackedWidget::source);
  }
}

void MainWindow::onStitcherReset() {
  framerate.restart();
  ui->renderedFrame->clear();
}

void MainWindow::openFile(QStringList files, int customWidth, int customHeight, bool showWarning) {
  ui->renderedFrame->setText("");
  if (files.isEmpty() || files.first().isEmpty()) {
    return;
  }
  pause();

  if (!ProjectFileHandler::getInstance()->getFilename().isEmpty() && !saveModifiedPTV(true)) {
    return;
  }
  // set the file path as the working dir (see #88)
  QFileInfo fInfo(files.first());
  if (!fInfo.isReadable() || !fInfo.isWritable()) {
    if (showWarning) {
      MsgBoxHandler::getInstance()->generic(
          tr("The file %0 doesn't exist or you don't have the right permissions.").arg(files.first()), "Warning",
          WARNING_ICON);
    }
    return;
  }

  const File::Type type = File::getTypeFromFile(files.first());
  if (type != File::CALIBRATION) {
    bool res = QDir::setCurrent(fInfo.absolutePath());
    Q_UNUSED(res)
    Q_ASSERT(res);
  }
  switch (type) {
    case File::PTV:
      VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QString("Open PTV: %0").arg(files.first()));
      closeProject();
      updateRecentFile(files.first());
      ProjectFileHandler::getInstance()->setFilename(files.first());
      emit reqChangeState(GUIStateCaps::disabled);
      emit reqOpenProject(files.first(), customWidth, customHeight);
      break;
    case File::CALIBRATION:
      if (referenceOnProject) {
        // a project is loaded, apply a calibration file
        disconnect(referenceOnProject, SIGNAL(hasBeenModified(bool)), this,
                   SLOT(setWindowModified(
                       bool)));  // FIXME : this disconnect isn't called with the action Calibration/Apply calibration
        openCalibration(files.first());
      } else {
        // no project is loaded, show an error message
        warningBox(tr("Please load a project or input videos before applying a calibration template."));
      }
      break;
    case File::VIDEO:
    case File::STILL_IMAGE: {
      QList<QUrl> urls;
      foreach (const QString file, files) {
        if (File::VIDEO == File::getTypeFromFile(file) || File::STILL_IMAGE == File::getTypeFromFile(file)) {
          urls << QUrl(file);
          VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(QString("Open file: %0").arg(file));
        }
      }
      emit reqChangeState(GUIStateCaps::disabled);
      qSort(urls.begin(), urls.end(), qLess<QUrl>());
      emit reqOpenInputs(urls, customWidth, customHeight);
      break;
    }
    case File::UNKNOWN:
    default:
      warningBox(tr("This file format is not supported"));
      break;
  }
}

//---------------------- Save ------------------------------------

// File->Save
bool MainWindow::on_actionSave_ptv_triggered() {
  if (ProjectFileHandler::getInstance()->getFilename().isEmpty()) {
    return on_actionSave_Project_As_triggered();
  }

  if (referenceOnProject->hasFileFormatChanged()) {
    QMessageBox msgBox(
        QMessageBox::Warning, tr("Warning"),
        tr("%0 project format has been updated."
           " If you overwrite your current project file you might not be able to open it in previous versions of %0."
           " Please use the 'Save as' option and pick a different file name if you want to keep both project versions.")
            .arg(QCoreApplication::applicationName()),
        QMessageBox::Save | QMessageBox::SaveAll | QMessageBox::Cancel, this);
    msgBox.button(QMessageBox::Save)->setText(tr("Save anyway"));
    msgBox.button(QMessageBox::SaveAll)->setText(tr("Save as"));  // Use the "Save all" button as a "Save as"
    int result = msgBox.exec();
    if (result == QMessageBox::SaveAll) {
      return on_actionSave_Project_As_triggered();
    } else if (result == QMessageBox::Cancel) {
      return false;
    }
  }
  return GlobalController::getInstance().getController()->saveProject(ProjectFileHandler::getInstance()->getFilename());
}

// File->Save As
bool MainWindow::on_actionSave_Project_As_triggered() {
  if (!saveAs()) {
    return false;
  }
  updateRecentFile(ProjectFileHandler::getInstance()->getFilename());
  QDir::setCurrent(QFileInfo(ProjectFileHandler::getInstance()->getFilename()).absolutePath());
  referenceOnProject->setModified(false);
  setWindowModified(false);
  setWindowTitle();
  return true;
}

bool MainWindow::saveAs() {
  QString selectedFilter;
  QString file =
      QFileDialog::getSaveFileName(this, tr("Save project as"), "",
                                   tr("Binary project") + " (*.ptvb);;" + tr("Project") + " (*.ptv)", &selectedFilter);
  if (file.isEmpty()) {
    return false;
  }
  if (!file.endsWith(".ptv", Qt::CaseInsensitive) && !file.endsWith(".ptvb", Qt::CaseInsensitive)) {
    if (selectedFilter.contains("ptvb")) {
      file.append(".ptvb");
    } else {
      file.append(".ptv");
    }
  }

  const QFileInfo fileInfo(file);
  if (fileInfo.exists() && !fileInfo.isWritable()) {
    errorBox(tr("File '%0' could not be opened for writing. 'Save as' cancelled.").arg(file));
    return false;
  }

  const QDir indir = QFileInfo(ProjectFileHandler::getInstance()->getFilename()).absolutePath();
  const QDir outdir = QFileInfo(file).absolutePath();
  std::unique_ptr<VideoStitch::Ptv::Value> root;

  // keep relative paths, if folder was changed
  if (indir != outdir) {
    if (referenceOnProject && referenceOnProject->getPanoConst().get()) {
      for (int inputIndex = 0; inputIndex < referenceOnProject->getPanoConst()->numInputs(); inputIndex++) {
        VideoStitch::Core::InputDefinition& input = referenceOnProject->getPano()->getInput(inputIndex);
        const std::string& path = input.getDisplayName();

        // check that the file exists (it will prevent renaming procedural readers) and that the path is relative
        if (QFileInfo(path.c_str()).exists() && QDir::isRelativePath(path.c_str())) {
          const QString relativePath = outdir.relativeFilePath(indir.filePath(path.c_str()));
          input.setFilename(relativePath.toStdString().c_str());
        }
      }
    }

    root.reset(referenceOnProject->serialize());
  }

  ProjectFileHandler::getInstance()->setFilename(file);
  GlobalController::getInstance().getController()->saveProject(file, root.get());
  return true;
}

bool MainWindow::saveModifiedPTV(bool showDiscard) {
  // we also should consider unapplied modification if we are still on the process tab
  if (referenceOnProject && referenceOnProject->hasLocalModifications()) {
    QFlags<QMessageBox::StandardButton> buttons = (!showDiscard)
                                                      ? QMessageBox::Save | QMessageBox::Cancel
                                                      : QMessageBox::Save | QMessageBox::Cancel | QMessageBox::Discard;
    int ret = MsgBoxHandler::getInstance()->genericSync(
        tr("Your project contains modifications. Do you want to save your changes?"),
        QCoreApplication::applicationName(), QUESTION_ICON, buttons);

    switch (ret) {
      case QMessageBox::Save:
        return on_actionSave_ptv_triggered();
      case QMessageBox::Discard:
        return true;
      case QMessageBox::Cancel:
        return false;
      default:
        Q_ASSERT(0);
        return false;
    }
  }
  return true;
}

//---------------------- Snapshot orchestration ------------------

void MainWindow::snapshotSources(bool selectFolder) {
  QString pathToExtract = (ExtractDialog::getLastCalibrationDirectory().isEmpty())
                              ? QDir::currentPath()
                              : ExtractDialog::getLastCalibrationDirectory();
  if (selectFolder) {
    ExtractDialog d(this, pathToExtract);
    if (d.exec() == QDialog::Rejected) {
      return;
    }
  }
  emit reqSnapshotSources(pathToExtract, false);
}

void MainWindow::registerRenderer(std::vector<std::shared_ptr<Core::PanoRenderer>>* renderers) {
  StitcherController* videoStitcher = GlobalController::getInstance().getController();
  DeviceVideoWidget* videoWidget = &ui->centralStackedWidget->getOutputTabWidget()->getVideoWidget();
  DeviceInteractiveWidget* interactiveWidget =
      ui->centralStackedWidget->getInteractiveTabWidget()->getInteractiveWidget();
  videoStitcher->lockedFunction([videoWidget, interactiveWidget, renderers]() {
    videoWidget->registerRenderer(renderers);
    interactiveWidget->registerRenderer(renderers);
  });
}

void MainWindow::createNewProject() {
  pause();
  emit reqResetProject();
  emit reqChangeState(GUIStateCaps::idle);
}

void MainWindow::on_actionExtract_output_to_triggered() {
  const QString file = QFileDialog::getSaveFileName(
      this, tr("Select the image path"), QDir::currentPath(),
      tr("JPEG image") + "(*.jpg);;" + tr("TIFF image") + "(*.tif);;" + tr("PNG image") + "(*.png)");
  if (file.isEmpty()) {
    return;
  }
  emit reqChangeState(GUIStateCaps::disabled);
  emit reqSnapshotPanorama(file);
}

//---------------- Fullscreen ------------------

void MainWindow::on_actionToggle_Fullscreen_triggered() {
  if (state == GUIStateCaps::idle) {
    return;
  }
  switch (showState) {
    case Normal:
#ifndef Q_OS_OSX
      // TODO: Workaround for bug QTBUG-41679
      if (ui->centralStackedWidget->activeTab() != CentralStackedWidget::VSTabWidget::welcome) {
        isMaxWindow = isMaximized();
        setFullscreen();
      }
#endif
      break;
    case Fullscreen:
      if (ui->centralStackedWidget->activeTab() == CentralStackedWidget::VSTabWidget::output ||
          ui->centralStackedWidget->activeTab() == CentralStackedWidget::VSTabWidget::interactive) {
        setPreviewFullscreen();
      } else {
        setNormal();
      }
      break;
    case PreviewFullscreen:
      setNormal();
      fullscreenActivate(true);
      break;
  }
  ShortcutManager::getInstance()->toggleFullscreenConnect(showState == PreviewFullscreen);
}

QKeySequence MainWindow::getFullscreenShortcut() const { return ui->actionToggle_Fullscreen->shortcut(); }

SeekBar* MainWindow::getSeekBar() const { return ui->seekbar; }

void MainWindow::setFullscreen() {
  showState = Fullscreen;
  showFullScreen();
}

void MainWindow::setNormal() {
  showState = Normal;
  if (isMaxWindow)
    showMaximized();
  else
    showNormal();
}

void MainWindow::setPreviewFullscreen() {
  showState = PreviewFullscreen;
  fullscreenActivate(false);
}

void MainWindow::fullscreenActivate(bool activate) {
  ui->seekBarDock->setVisible(activate);
  ui->statusBar->setVisible(activate);
  ui->menuBar->setVisible(activate);
  ui->centralStackedWidget->setPreviewFullScreen(activate);
}

void MainWindow::setTimeline() {
  ui->seekbar->setTimeline();
  ui->seekBarDock->style()->unpolish(ui->seekBarDock);
  ui->seekBarDock->style()->polish(ui->seekBarDock);
  /**
   * This block is needed since the QDockWidget size policies are a pain to manage when the docks are
   * painful to resize when they are docked.
   */
  update();
}

void MainWindow::warningBox(QString str) { MsgBoxHandler::getInstance()->generic(str, tr("Warning"), WARNING_ICON); }

void MainWindow::errorBox(const QString str) {
  MsgBoxHandler::getInstance()->generic(str, tr("Critical error"), CRITICAL_ERROR_ICON);
}

void MainWindow::warnWrongInputSize(unsigned widthIs, unsigned heightIs, unsigned widthShouldBe,
                                    unsigned heightShouldbe) {
  MsgBoxHandler::getInstance()->generic(
      tr("The new input size is %0 x %1. The expected input size for this project is %2 x %3.")
          .arg(QString::number(widthIs), QString::number(heightIs), QString::number(widthShouldBe),
               QString::number(heightShouldbe)),
      tr("Wrong input size"), WARNING_ICON);
}

void MainWindow::openRecentCalibration() {
  const QAction* action = qobject_cast<const QAction*>(sender());
  if (action) {
    QString recentFileName = action->data().toString();
    if (!QFile::exists(recentFileName)) {
      MsgBoxHandler::getInstance()->generic(tr("Error: couldn't open the file %0.").arg(recentFileName),
                                            tr("File not found"), CRITICAL_ERROR_ICON);
      return;
    }
    openCalibration(recentFileName);
  }
}

void MainWindow::onEndOfStreamReached() { ui->seekbar->setValue(ui->seekbar->getCurrentFrameFromCursorPosition() - 1); }

void MainWindow::onGenericErrorMessage(const VideoStitch::Status status, bool needToClose) {
  MsgBoxHandlerHelper::genericErrorMessage(status);
  if (needToClose) {
    createNewProject();
  }
}

void MainWindow::updateRecentFile(const QString& newestFilename) {
  VSSettings::getSettings()->addRecentFile(newestFilename);
  updateRecentFileActions();
}

void MainWindow::updateRecentFileActions() {
  const QStringList files = VSSettings::getSettings()->getRecentFileList();
  int index = 0;
  foreach (QString name, files) {
    if (QFileInfo(name).exists()) {
      const QString text = tr("&%0. %1").arg(index + 1).arg(File::strippedName(name));
      recentFilesActs[index]->setText(text);
      recentFilesActs[index]->setData(name);
      recentFilesActs[index]->setVisible(true);
      ++index;
    }
  }
}

void MainWindow::updateRecentCalibration(const QString& newCalibration) {
  PostProdSettings::getPostProdSettings()->addRecentCalibration(newCalibration);
  updateRecentCalibrationActions();
}

void MainWindow::updateRecentCalibrationActions() {
  QStringList files = PostProdSettings::getPostProdSettings()->getRecentCalibrationList();
  for (int i = 0; i < files.size(); ++i) {
    //: Recent calibration template. %0 is the index, %1 is the template name
    QString text = tr("%0. %1").arg(i + 1).arg(File::strippedName(files[i]));
    recentCalibrationActs[i]->setText(text);
    recentCalibrationActs[i]->setData(files[i]);
    recentCalibrationActs[i]->setVisible(true);
  }
  ui->calibrationWidget->fillRecentCalibrationMenuWith(recentCalibrationActs);
}

void MainWindow::processMessage(const Packet& packet) {
  switch (packet.getType()) {
    case Packet::WAKEUP:
      VideoStitch::WidgetsHelpers::bringToForeground(this);
      break;
    case Packet::OPEN_FILES: {
      if (state == GUIStateCaps::disabled) {
        return;
      }
      VideoStitch::WidgetsHelpers::bringToForeground(this);
      QString argString = QString::fromLatin1(packet.getPayload());
      QStringList args = QStringList() << argString;
      openFile(args);
      break;
    }
    default:
      break;
  }
}

void MainWindow::displayGPUInfo(size_t usedBytes, size_t totalBytes, QList<size_t> usedBytesByDevices) {
  usedGPUMem = usedBytes;
  totalGPUMem = totalBytes;

  // update interface label
  ui->gpuInfoLabel->setText(getGpuInfo(usedBytesByDevices));

  VideoStitch::Logger::get(VideoStitch::Logger::Debug)
      << ui->gpuInfoLabel->text().toStdString() << " | " << ui->graphicsCardLabel->text().toStdString()
      << " | total: " << usedGPUMem / BytesInMB << "/" << totalGPUMem / BytesInMB << std::endl;
}

void MainWindow::crashSlot() {
  Crasher crasher;
  crasher.crash();
}

void MainWindow::resetDimensions(unsigned panoramaWidth, unsigned panoramaHeight, const QStringList& inputNames) {
  static bool errorAlreadyManaged = false;
  if (!errorAlreadyManaged) {
    errorAlreadyManaged = true;
    uniqueResetDimensions(panoramaWidth, panoramaHeight, inputNames);
    errorAlreadyManaged = false;
  }
}

void MainWindow::uniqueResetDimensions(unsigned panoramaWidth, unsigned panoramaHeight, const QStringList& inputNames) {
  emit reqForceGPUInfoRefresh();

  // this line does not catch the driver loss on MacOS, seems to do it on Windows
  if (totalGPUMem == 0 && usedGPUMem != 0) {
    //: Error message when the GPU driver is lost
    auto result = MsgBoxHandler::getInstance()->genericSync(
        tr("Critical error: the GPU driver has been lost.\n"
           "Unless you restart the application, you won't be able to load any project.\n"
           "Do you want to restart %0?")
            .arg(QCoreApplication::applicationName()),
        //: Error window title when the GPU driver is lost
        tr("Critical error: GPU driver lost"), CRITICAL_ERROR_ICON, QMessageBox::Yes | QMessageBox::No);
    if (result == QMessageBox::Yes) {
      emit reqRestartApplication();
    } else {
      qApp->quit();
    }
    return;
  }

  if (panoramaHeight / 2 > MINIMUM_RESOLUTION_HEIGHT) {
    ResetDimensionsDialog dialog(File::strippedName(ProjectFileHandler::getInstance()->getFilename()), panoramaWidth,
                                 panoramaHeight, this);
    if (dialog.exec()) {
      panoramaWidth = dialog.getNewPanoWidth();
      panoramaHeight = dialog.getNewPanoHeight();
      // if there's a project file, reload it with new panorama size
      // if not, create a new empty project and reload current input files
      if (!ProjectFileHandler::getInstance()->getFilename().isEmpty()) {
        openFile(QStringList() << ProjectFileHandler::getInstance()->getFilename(), panoramaWidth, panoramaHeight);
      } else {
        // create new project
        createNewProject();
        openFile(inputNames, panoramaWidth, panoramaHeight);
      }
      return;
    }
  } else {
    //: Error message when the GPU doesn't manage to create a pano with the minimum resolution
    MsgBoxHandler::getInstance()->genericSync(
        tr("Your GPU is out of memory. Please try to close other applications and try again.\n"
           "Your computer might not be powerful enough to handle this project."),
        tr("Critical error"), CRITICAL_ERROR_ICON);
  }

  // If nothing succeed
  if (saveModifiedPTV(true)) {
    createNewProject();
  } else {
    // If the user cancelled the save or if there was an error, we restart this method
    uniqueResetDimensions(panoramaWidth, panoramaHeight, inputNames);
  }
}

void MainWindow::updatePanoDimensions(unsigned int newPanoWidth, unsigned int newPanoHeight) {
  if (newPanoWidth != 0 && newPanoHeight != 0) {
    stitchedSizeLabel->setText(
        tr("Stitched size: %0x%1").arg(QString::number(newPanoWidth), QString::number(newPanoHeight)));
  } else {
    stitchedSizeLabel->setText(tr("No stitcher loaded "));
  }
}

void MainWindow::dropInput(QDropEvent* e) { dropEvent(e); }

//-------------------- Taskbar progression thumbnail -------------
#ifdef Q_OS_WIN
void MainWindow::startWinProgress() { progressThumbnail->init(); }

void MainWindow::stopWinProgress() { progressThumbnail->setProgressState(TBPF_NOPROGRESS); }

void MainWindow::changeWinProgressValue(quint64 current, quint64 total) {
  progressThumbnail->setProgressValue(current, total);
}

void MainWindow::changeWinProgressState(TBPFLAG state) { progressThumbnail->setProgressState(state); }
#endif

void MainWindow::setWindowTitle(const QString titleToSet) {
  QString finalTitle;
  if (titleToSet.isEmpty()) {
    if (referenceOnProject && referenceOnProject->isInit()) {
      QString fileName = File::strippedName(ProjectFileHandler::getInstance()->getFilename());
      if (fileName.isEmpty()) {
        //: Default window title when creating a new project
        fileName = tr("Untitled project");
      }
      finalTitle = fileName + QString("[*] - ") + QCoreApplication::applicationName();
    } else {
      finalTitle = QCoreApplication::applicationName();
    }
  } else {
    finalTitle = titleToSet;
  }

  QMainWindow::setWindowTitle(finalTitle);
}

void MainWindow::updateStatusBar(QString message) { ui->statusBar->showMessage(message); }

//---------------------- Batch Stitcher ------------------

void MainWindow::sendToBatch(bool saveACopy) {
  QString fileToSend;
  if (saveACopy) {
    const QString file = ProjectFileHandler::getInstance()->getFilename();
    bool ret = saveAs();
    fileToSend = ProjectFileHandler::getInstance()->getFilename();
    ProjectFileHandler::getInstance()->setFilename(file);
    if (!ret) {
      return;
    }
  } else {
    fileToSend = ProjectFileHandler::getInstance()->getFilename();
  }
  if (!saveModifiedPTV(true)) {
    return;
  }
  startBatchStitcher(QStringList() << fileToSend);
}

bool MainWindow::startBatchStitcher(QStringList params) {
  QString workingDirectory = QDir::toNativeSeparators(QApplication::applicationDirPath() + QDir::separator());
  QString executable = "batchstitcher";
#ifdef Q_OS_WIN
  executable += ".exe";
#endif
  QString program = workingDirectory + executable;
  return QProcess::startDetached(program, params, workingDirectory);
}

void MainWindow::initializeUndoFramework() {
  // Create the stack (owned by the app)
  QUndoStack* undoStack = new QUndoStack(qApp);

  // Create undo and redo actions
  QAction* undoAction = undoStack->createUndoAction(this);
  ui->menuEdit->addAction(undoAction);
  undoAction->setShortcuts(QKeySequence::Undo);

  QAction* redoAction = undoStack->createRedoAction(this);
  ui->menuEdit->addAction(redoAction);
  redoAction->setShortcuts(QKeySequence::Redo);

  // Configure the undo view
  ui->undoView->setStack(undoStack);
  ui->undoView->setEmptyLabel(tr("No change applied"));
}

void MainWindow::disableMenu(const bool disable) {
  for (auto dock : commands) {
    auto action = dock->toggleViewAction();
    if (ui->menuWindow->actions().contains(action)) {
      action->setDisabled(disable);
    }
  }
}

void MainWindow::disableWindow() { setEnabled(false); }

void MainWindow::enableWindow() { setEnabled(true); }

void MainWindow::hideCommandDocks() {
  for (auto dock : commands) {
    dock->hide();
  }
}

void MainWindow::hideUtilsDocks() {
  for (auto dock : utils) {
    dock->hide();
  }
}

void MainWindow::saveOpenedDocks() {
  opened.clear();
  for (auto dock : commands) {
    if (dock->isVisible()) {
      opened.append(dock);
    }
  }
}

void MainWindow::restoreOpenedDocks() {
  for (auto dock : opened) {
    dock->show();
  }
}

void MainWindow::setDocks(const Docks docks) {
  const auto size = QDesktopWidget().size();
  for (QDockWidget* dock : docks) {
    ui->menuWindow->addAction(dock->toggleViewAction());
    dock->setMaximumSize(size);
    dock->setTitleBarWidget(0);
    connect(dock->toggleViewAction(), &QAction::triggered, [=](bool toggled) {
      if (toggled) {
        dock->raise();
      }
    });
    connect(dock, &QDockWidget::visibilityChanged, [=](bool visible) {
      // Closed, perform a reset of the widget
      if (!visible && !dock->toggleViewAction()->isChecked()) {
        IToolWidget* widget = dynamic_cast<IToolWidget*>(dock->widget());
        if (widget != nullptr) {
          widget->reset();
        }
      }
    });
  }
}

void MainWindow::tabify(const Docks docks) {
  for (auto i = 0; i < docks.size() - 1; ++i) {
    tabifyDockWidget(docks.at(i), docks.at(i + 1));
  }
}

void MainWindow::updateClockForTab(int tab) {
  StitcherController* stitcherController = GlobalController::getInstance().getController();

  switch (tab) {
    case CentralStackedWidget::source:
      stitcherController->setNextFrameAction(StitcherController::NextFrameAction::Extract);
      if (!stitcherController->isPlaying()) {
        emit reqReextract();
      }
      break;
    case CentralStackedWidget::output:
    case CentralStackedWidget::interactive:
      stitcherController->setNextFrameAction(StitcherController::NextFrameAction::Stitch);
      if (!stitcherController->isPlaying()) {
        emit reqRestitch();
      }
      break;
    case CentralStackedWidget::process:
    case CentralStackedWidget::welcome:
    default:
      stitcherController->pause();
      stitcherController->setNextFrameAction(StitcherController::NextFrameAction::None);
      break;
  }
}
