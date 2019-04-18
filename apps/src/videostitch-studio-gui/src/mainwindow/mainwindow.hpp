// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videostitcher/globalpostprodcontroller.hpp"

#include "libvideostitch-gui/caps/guistatecaps.hpp"
#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"
#include "libvideostitch-gui/mainwindow/gpuinfoupdater.hpp"
#include "libvideostitch-gui/mainwindow/frameratecompute.hpp"
#include "libvideostitch-gui/mainwindow/stitchingwindow.hpp"
#include "libvideostitch-gui/mainwindow/wintaskbarprogress.hpp"
#include "libvideostitch-gui/mainwindow/packet.hpp"
#ifdef Q_OS_WIN
#include "libvideostitch-gui/mainwindow/stitcheroculuswindow.hpp"
#endif

#include "libvideostitch/controller.hpp"

#include <QPointer>
#include <QTabBar>
#include <QUrl>

#ifdef Q_OS_WIN
#define MENU_TAB QTabBar*
#else
#define MENU_TAB QTabWidget*
#endif

#ifdef WITH_AUDIO_PLAYER
class AudioPlayer;
#endif
class ProjectDefinition;
class SignalCompressionCaps;
class QLabel;
class QAction;
class SeekBar;

typedef QVector<QDockWidget*> Docks;

namespace Ui {
class MainWindow;
}

class MainWindow : public StitchingWindow, public GUIStateCaps {
  Q_OBJECT

  // used by squish
  Q_PROPERTY(bool isProjectLoaded READ isProjectLoaded)
  Q_PROPERTY(bool isProjectClosed READ isProjectClosed)
  Q_PROPERTY(bool isProjectModified READ isProjectModified)
  Q_PROPERTY(QString gpuFramework READ gpuFramework)
  Q_PROPERTY(int numberOfGPU READ numberOfGPUs)

 public:
  enum ShowState { Normal, Fullscreen, PreviewFullscreen };

  /**
   * @brief MainWindow The main windows of the application.
   */
  explicit MainWindow();
  ~MainWindow();

  /**
   * @brief Gets the number of tab used in menuTabs.
   * @return The number of tab used in menuTabs (menuTabs->count())
   */
  int tabCount() const;
  QKeySequence getFullscreenShortcut() const;
  SeekBar* getSeekBar() const;
  void initializeUndoFramework();

  inline bool isProjectLoaded() const { return state == GUIStateCaps::stitch; }
  inline bool isProjectClosed() const { return state == GUIStateCaps::idle; }
  inline bool isProjectModified() const {
    return !referenceOnProject.isNull() && referenceOnProject->hasLocalModifications();
  }
  inline QString gpuFramework() const {
    return QString::fromStdString(VideoStitch::Discovery::getFrameworkName(VideoStitch::GPU::getFramework()));
  }
  inline int numberOfGPUs() const { return VideoStitch::Discovery::getNumberOfDevices(); }

 public slots:
  /**
   * File handling
   */
  void openFile(const QStringList files, int customWidth = 0, int customHeight = 0, bool showWarning = true);

  /**
   *  @brief Processes the messages sent by the application.
   */
  void processMessage(const Packet& packet);

 protected:
  virtual bool eventFilter(QObject* watched, QEvent* event) override;
  /**
   * @brief closeEvent() is an overloaded function to allow MainWindow to take actions and ask the user before the
   * application closes.
   */
  virtual void closeEvent(QCloseEvent* event) override;

  /**
   * Drag and drop support.
   */
  virtual void dropEvent(QDropEvent* e) override;
  virtual void dragMoveEvent(QDragMoveEvent* e) override;
  virtual void dragEnterEvent(QDragEnterEvent* e) override;

 signals:
  /**
   * @brief Requests the State Manager to initiate a specific state transition.
   * @param s is the requested state.
   */
  void reqChangeState(GUIStateCaps::State s) override;

  /**
   * VideoStitcher interface.
   */
  void reqCloseProject();
  void reqCloseProjectBlocking();
  void reqOpenProject(const QString inFile, const int customWidth, const int customHeight);
  void reqResetProject();
  void reqStitch(SignalCompressionCaps* = nullptr);

  void reqRestitch(SignalCompressionCaps* = nullptr);
  void reqReextract(SignalCompressionCaps* = nullptr);

  void reqSeekNExtract(frameid_t, SignalCompressionCaps* = nullptr);
  void reqSeekNStitch(frameid_t, SignalCompressionCaps* = nullptr);
  void reqReset();

  /**
   * @brief Requests the stitcher to apply a template passed in parameter. The second parameter is optional, it is the
   * output file to write the "templated" file.
   * @param templateFile Ptv/pto/pts which will be used as a template.
   * @param outputPtv Output file to write into.
   */
  void reqApplyTemplate(const QString);
  void reqImportCalibration(const QString);

  /**
   * @brief Sends a signal to the GPUInfoUpdater to force it to refresh.
   */
  void reqForceGPUInfoRefresh();
  void reqSendMessage(Packet packet, QString host, bool andDie = false);
  void reqRestartApplication();
  void reqOpenInputs(QList<QUrl> urls, int customWidth, int customHeight);
  void reqSnapshotSources(const QString file, const bool);
  void reqSnapshotPanorama(const QString file);
  void reqRotatePanorama(YPRSignalCaps* rotations, bool restitch);
  void reqSetWorkingArea(frameid_t, frameid_t);

 private slots:
  /**
   * @brief Changes the widget's stats to the given state.
   * @param s State you want to switch to.
   */
  void changeState(GUIStateCaps::State s) override;
  void changeStateToStitch();
  void onProjectInitialized(ProjectDefinition* project);
  void clearProject();
  void cleanStitcher();

  /**
   * Opens a box if the working PTV has local modifications (against the last saved file).
   * Returns false if the user canceled the action.
   */
  bool saveModifiedPTV(bool showDiscard = false);

  /**
   * @brief The server calls play() before being called on timeEvent().
   */
  void play();

  /**
   * @brief The server calls pause() to prepare the end of timeEvent() calls.
   */
  void pause();

  /**
   * @brief refresh is a callback after timeEvent() sent actions.
   * @param FrameNum the latest frame beging processed.
   */
  void refresh(mtime_t date);

  /**
   * @brief Toggles play() or pause() slots.
   */
  void togglePlayOrPause();

  /**
   * Slot receiving timeEvent()
   */
  void seekFrame(SignalCompressionCaps*, frameid_t);

  /**
   * @brief Updates the main window components (including menuTabs)
   */
  void switchTab(const int);
  void switchFromProcessTabToSourceTab(bool shouldSwitch);
  void switchToOutputTab(bool shouldSwitch = true);
  void switchToSourceTab(bool shouldSwitch = true);
  void switchToOutputOrInteractiveTab(bool shouldSwitch);
  void showOutputOrInteractiveTab();

  /**
   * Error helpers.
   */
  void errorBox(const QString message);
  void warningBox(const QString message);

  /**
   * File menu.
   */
  void on_actionNew_triggered();
  void on_actionOpenPTV_triggered();
  void openMedia();
  void on_actionOpen_Working_Directory_triggered();
  bool on_actionSave_ptv_triggered();
  bool on_actionSave_Project_As_triggered();
  void on_actionExit_triggered();

  /**
   * Edit menu.
   */
  void on_actionPreferences_triggered();

  /**
   * Calibration Menu
   */
  void openCalibration(const QString templateFile = QString());
  void on_actionNew_Calibration_triggered();
  void on_actionExtract_stills_to_triggered();
  void on_actionExtract_stills_triggered();

  void createNewProject();
  void registerRenderer(std::vector<std::shared_ptr<VideoStitch::Core::PanoRenderer>>* renderers);

  /**
   * Help Menu
   */
  void on_actionAbout_triggered();
  void on_actionSupport_triggered();
  void on_actionShortcut_triggered();

  /**
   * File history handling
   */
  void openRecentProject();
  void openRecentCalibration();

  /**
   * VideoStitcher message handler
   */
  void onGenericErrorMessage(Status status, bool needToClose);

  /**
   * End of file handler
   */
  void onEndOfStreamReached();

  /**
   * @brief GPU properties handling
   *        This slot may be called every second to update informations in the statusbar.
   */
  void displayGPUInfo(size_t usedBytes, size_t totalBytes, QList<size_t> usedBytesByDevices);

  /**
   * Shortcut callbacks
   */
  void jumpShortcutCalled();
  void selectAllVideo();
  void crashSlot();
  void snapshotSources(bool selectFolder);

  /**
   * @brief Callback used when a shortcut dedicated to the tabs has been called.
   * @param index Switches to the tab index (if it is possible).
   */
  void tabShortcutCalled(int index);
  void warnWrongInputSize(unsigned widthIs, unsigned heightIs, unsigned widthShouldBe, unsigned heightShouldbe);
  void resetDimensions(unsigned panoramaWidth, unsigned panoramaHeight, const QStringList& inputNames);
  void uniqueResetDimensions(unsigned panoramaWidth, unsigned panoramaHeight, const QStringList& inputNames);
  void updatePanoDimensions(unsigned int newPanoWidth = 0, unsigned int newPanoHeight = 0);
  void dropInput(QDropEvent* e);

#ifdef Q_OS_WIN
  void startWinProgress();
  void stopWinProgress();
  void changeWinProgressValue(quint64 current, quint64 total);
  void changeWinProgressState(TBPFLAG state);
#endif

  void sendToBatch(bool saveACopy);
  void updateStatusBar(QString message);
  void checkInputsDialog(QString& newFolder);
  void manageMergerNotSupportedByGpu(const std::string& merger, bool& fallBackToLinearBlending);

  /**
   * @brief Overload of the legacy setWindowTitle.
   *        Using a default empty parameters just asks to update the title (for example, concatenate the asterisk).
   *        Else it sets the title (and formats it depending of the current state of the UI).
   * @param title Title you want to set.
   */
  void setWindowTitle(const QString title = QString());
  void on_actionExtract_output_to_triggered();
  void on_actionToggle_Fullscreen_triggered();
  void setFullscreen();
  void setNormal();
  void setPreviewFullscreen();
  void fullscreenActivate(bool activate);

  // Progress related slots
  void startProgress();
  void updateProgress(int progress);
  void finishProgress();
  void cancelProgress();

  /**
   * @brief updateInteractiveViewVisibility
   *        update the visibility of the interactive tab view
   *        when the projection changes
   * @param proj the new projection
   * @param fov the new fov
   */
  void updateInteractiveViewVisibility(VideoStitch::Projection proj, double);

  // stitcher controller related slots
  void onPtvOpened();
  void onInputsOpened();
  void onStitcherReset();

 private:
  /**
   * @brief closeProject clears all VideoStitcher, FrameExtractor and local value relative to a previously opened
   * content.
   */
  void closeProject();

  void buildWindowMenu();
  void configureDockWidgets();
  void buildStatusBar();
  void buildMenuTab();
  void displayGpuNames();
  void configureController();
  void configureOculusDisplay();
  void registerMetaTypes();
  void connectStitcherController();
  void connectMainWindow();
  void connectWidgets();
  void connectProcessWidget();
  void stopThreads();
  Docks getDocks() const;

  /**
   * File history handling
   */
  void updateRecentFile(const QString& newestFilename);
  void updateRecentFileActions();
  void updateRecentCalibration(const QString& newCalibration);
  void updateRecentCalibrationActions();
  void loadFileHistory();
  void loadCalibrationHistory();
  void setTimeline();
  bool saveAs();
  bool startBatchStitcher(QStringList params);
  void disableMenu(const bool disable);
  void disableWindow();
  void enableWindow();
  void hideCommandDocks();
  void hideUtilsDocks();
  void saveOpenedDocks();
  void restoreOpenedDocks();
  void setDocks(const Docks docks);
  void updateClockForTab(int tab);
  void tabify(const Docks docks);

  /**
   * UI.
   */
  Ui::MainWindow* ui;
  QTabBar* menuTabs;

#ifndef Q_OS_WIN
  QToolBar* macxToolBar;
#endif

  QPointer<ProjectDefinition> referenceOnProject;
#ifdef WITH_AUDIO_PLAYER
  AudioPlayer* audioPlayer;
#endif
  QThread controllerThread;

  /**
   * @brief The current state of MainWindow.
   */
  GUIStateCaps::State state;

  /**
   * Recent Files Management
   */
  QList<QAction*> recentFilesActs;
  QList<QAction*> recentCalibrationActs;

  /**
   * Updates the GPU info statusbar //TODO: shortcut handler
   */
  GPUInfoUpdater gpuInfoUpdater;
  QLabel* stitchedSizeLabel;

#ifdef Q_OS_WIN
  WinTaskbarProgress* progressThumbnail;
#endif
  FramerateCompute framerate;

  size_t usedGPUMem;
  size_t totalGPUMem;
  ShowState showState;
  bool isMaxWindow;
  Docks commands;
  Docks utils;
  Docks opened;

#if defined(OCULUS_DISPLAY) && defined(Q_OS_WIN)
  StitcherRiftWindow* oculusRenderWindow;
#endif
};
