// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "shortcutmanager.hpp"

#include "mainwindow.hpp"

#include "src/widgets/seekbar.hpp"
#include "src/widgets/stabilizationwidget.hpp"

#include "libvideostitch-gui/mainwindow/objectutil.hpp"
#include "libvideostitch-gui/utils/imagesorproceduralsonlyfilterer.hpp"

#include <QShortcut>
#include <QSignalMapper>

ShortcutManager* ShortcutManager::createInstance(MainWindow* parent) {
  if (!_singleton) {
    _singleton = new ShortcutManager(parent);
  }
  return (static_cast<ShortcutManager*>(_singleton));
}

ShortcutManager* ShortcutManager::getInstance() { return (static_cast<ShortcutManager*>(_singleton)); }

void ShortcutManager::toggleConnections(bool connect) {
  VideoStitch::Helper::toggleConnect(connect, spaceShortcut, SIGNAL(activated()), mainWindow, SLOT(togglePlayOrPause()),
                                     Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, jumpShortcut, SIGNAL(activated()), mainWindow, SLOT(jumpShortcutCalled()),
                                     Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, selectAllShortcut, SIGNAL(activated()), mainWindow,
                                     SLOT(selectAllVideo()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, crashShortcut, SIGNAL(activated()), mainWindow, SLOT(crashSlot()),
                                     Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, restartShortcut, SIGNAL(activated()), mainWindow,
                                     SIGNAL(reqRestartApplication()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, setFirstShortcut, SIGNAL(activated()), mainWindow->getSeekBar(),
                                     SLOT(on_toStartButton_clicked()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, setLastShortcut, SIGNAL(activated()), mainWindow->getSeekBar(),
                                     SLOT(on_toStopButton_clicked()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, leftShortcut, SIGNAL(activated()), mainWindow->getSeekBar(),
                                     SLOT(leftShortcutCalled()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, rightShortcut, SIGNAL(activated()), mainWindow->getSeekBar(),
                                     SLOT(rightShortcutCalled()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, keyframeShortcut, SIGNAL(activated()), mainWindow->getSeekBar(),
                                     SIGNAL(reqAddKeyframe()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, nextKeyFrameShortcut, SIGNAL(activated()), mainWindow->getSeekBar(),
                                     SLOT(nextKeyFrameShortcutCalled()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, prevKeyFrameShortcut, SIGNAL(activated()), mainWindow->getSeekBar(),
                                     SLOT(prevKeyFrameShortcutCalled()), Qt::UniqueConnection);
  VideoStitch::Helper::toggleConnect(connect, toggleOrientationModeShortcut, SIGNAL(activated()),
                                     mainWindow->findChild<StabilizationWidget*>(), SLOT(toggleOrientationButton()),
                                     Qt::UniqueConnection);
  nextKeyFrameShortcut->setContext(Qt::ApplicationShortcut);
  prevKeyFrameShortcut->setContext(Qt::ApplicationShortcut);

  for (int i = 0; i < mainWindow->tabCount(); ++i) {
    VideoStitch::Helper::toggleConnect(connect, tabShortcuts.at(i), SIGNAL(activated()), tabShortcutsMapper,
                                       SLOT(map()), Qt::UniqueConnection);
  }
  VideoStitch::Helper::toggleConnect(connect, tabShortcutsMapper, SIGNAL(mapped(int)), mainWindow,
                                     SLOT(tabShortcutCalled(int)), Qt::UniqueConnection);
}

void ShortcutManager::setWhatThisForAll() {
  spaceShortcut->setWhatsThis(tr("Play / Pause"));
  jumpShortcut->setWhatsThis(tr("Jump to frame X"));
  selectAllShortcut->setWhatsThis(tr("Select all the video as working area"));
  setFirstShortcut->setWhatsThis(tr("Set the first frame of your working area"));
  setLastShortcut->setWhatsThis(tr("Set the last frame of your working area"));
  restartShortcut->setWhatsThis(tr("Restart the software"));
  leftShortcut->setWhatsThis(tr("Previous frame"));
  rightShortcut->setWhatsThis(tr("Next frame"));
  keyframeShortcut->setWhatsThis(tr("Add a keyframe at the current frame"));
  prevKeyFrameShortcut->setWhatsThis(tr("Previous keyframe"));
  nextKeyFrameShortcut->setWhatsThis(tr("Next keyframe"));
  toggleOrientationModeShortcut->setWhatsThis(tr("Set 'Edit orientation' mode"));
  tabShortcuts[0]->setWhatsThis(tr("Source panel"));
  tabShortcuts[1]->setWhatsThis(tr("Output panel"));
  tabShortcuts[2]->setWhatsThis(tr("Interactive panel"));
  tabShortcuts[3]->setWhatsThis(tr("Process panel"));
}

void ShortcutManager::toggleFullscreenConnect(bool connectToMainWindow) {
  delete fullscreenShortcut;
  // Disable unnesesary shortcuts for fullscreen and let F and Space shortcuts
  toggleConnections(!connectToMainWindow);
  if (connectToMainWindow) {
    fullscreenShortcut = new QShortcut(QKeySequence(fullscreenSequence), mainWindow);
    connect(fullscreenShortcut, SIGNAL(activated()), mainWindow, SLOT(on_actionToggle_Fullscreen_triggered()),
            Qt::UniqueConnection);
    fullscreenShortcut->setContext(Qt::ApplicationShortcut);
    VideoStitch::Helper::toggleConnect(true, spaceShortcut, SIGNAL(activated()), mainWindow, SLOT(togglePlayOrPause()),
                                       Qt::UniqueConnection);
  } else {
    fullscreenShortcut = nullptr;
  }
}

ShortcutManager::ShortcutManager(MainWindow* parent)
    : QObject(parent),
      spaceShortcut(new QShortcut(Qt::Key_Space, parent)),
      jumpShortcut(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_J), parent)),
      selectAllShortcut(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_A), parent)),
      setFirstShortcut(new QShortcut(QKeySequence(Qt::SHIFT + Qt::Key_Home), parent)),
      setLastShortcut(new QShortcut(QKeySequence(Qt::SHIFT + Qt::Key_End), parent)),
      crashShortcut(new QShortcut(QKeySequence(Qt::SHIFT + Qt::CTRL + Qt::ALT + Qt::Key_H), parent)),
      restartShortcut(new QShortcut(QKeySequence(Qt::SHIFT + Qt::CTRL + Qt::ALT + Qt::Key_Backspace), parent)),
      leftShortcut(new QShortcut(QKeySequence(Qt::Key_Left), parent)),
      rightShortcut(new QShortcut(QKeySequence(Qt::Key_Right), parent)),
      keyframeShortcut(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_K), parent)),
      nextKeyFrameShortcut(new QShortcut(Qt::Key_K, parent)),
      prevKeyFrameShortcut(new QShortcut(Qt::Key_J, parent)),
      toggleOrientationModeShortcut(new QShortcut(QKeySequence(Qt::Key_O), parent)),
      tabShortcutsMapper(new QSignalMapper(parent)),
      mainWindow(parent),
      fullscreenShortcut(nullptr),
      fullscreenSequence(mainWindow->getFullscreenShortcut()) {
  for (auto i = 0; i < mainWindow->tabCount(); ++i) {
    const QString controlSequence = "Ctrl+" + QString::number(i + 1);
    tabShortcuts.append(new QShortcut(QKeySequence(controlSequence), parent));
    tabShortcutsMapper->setMapping(tabShortcuts[i], i);
  }
  toggleConnections(true);
  setWhatThisForAll();

  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(spaceShortcut);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(jumpShortcut);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(selectAllShortcut);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(setFirstShortcut);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(setLastShortcut);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(leftShortcut);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(rightShortcut);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(keyframeShortcut);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(nextKeyFrameShortcut);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(prevKeyFrameShortcut);
}
