// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SHORTCUTMANAGER_HPP
#define SHORTCUTMANAGER_HPP

#include "libvideostitch-base/singleton.hpp"

#include <QKeySequence>

class MainWindow;
class QShortcut;
class QSignalMapper;

/**
 * @brief Manages the global shortcuts that are used by the application. You must call getInstance() when the parent
 *        has initialized what's used in the shortcut manager (so after the constructor, not while the parent is being
 * constructed). The shortcut manager will plug its shortcuts to the "parent" widget passed in parameter (so, the
 * MainWindow for us).
 */
class ShortcutManager : public QObject, public Singleton<ShortcutManager> {
  Q_OBJECT
  friend class Singleton<ShortcutManager>;

 public:
  /**
   * @brief Overload of getInstance function to satisfy the need of a parameter.
   * @param parent MainWindow which will be used to connect the shortcuts
   * @return Static instance of the shortcut manager.
   */
  static ShortcutManager* createInstance(MainWindow* parent);
  /**
   * @brief Overload of getInstance function.
   * @return Static instance of the shortcut manager.
   */
  static ShortcutManager* getInstance();

  void toggleConnections(bool connect);

  void setWhatThisForAll();

  /**
   * @brief Toggles the fullscreen specific shortcut when the GLWidget is in fullscreen.
   * @param connectToMainWindow True = create and connect the specific shortcut, false = destroy it.
   */
  void toggleFullscreenConnect(bool connectToMainWindow);

  QKeySequence getFullscreenShortcut() const { return fullscreenSequence; }

 private:
  /**
   * @brief Initializes the shortcut manager and connects the shortcuts the the MainWindow (parent).
   * @param Pointer to the main window to connect with the shortcuts.
   */
  explicit ShortcutManager(MainWindow* parent);
  ~ShortcutManager() {}

 private:
  QShortcut* spaceShortcut;
  QShortcut* jumpShortcut;
  QShortcut* selectAllShortcut;
  QShortcut* setFirstShortcut;
  QShortcut* setLastShortcut;
  QShortcut* crashShortcut;
  QShortcut* restartShortcut;
  QShortcut* leftShortcut;
  QShortcut* rightShortcut;
  QShortcut* keyframeShortcut;
  QShortcut* nextKeyFrameShortcut;
  QShortcut* prevKeyFrameShortcut;
  QList<QShortcut*> tabShortcuts;
  QShortcut* toggleOrientationModeShortcut;
  QShortcut* newTemplateShortcut;
  QShortcut* applyTemplateShortcut;
  QSignalMapper* tabShortcutsMapper;
  MainWindow* mainWindow;
  /**
   * @brief Shortcut used when the GLWidget is in fullscreen.
   *        When the GLWidget is in fullscreen, the actionToggle_fullscreen becomes unaccessible.
   *        So, this action and its shortcut must be replaced by a dedicated shortcut.
   */
  QShortcut* fullscreenShortcut;
  QKeySequence fullscreenSequence;
};

#endif  // SHORTCUTMANAGER_HPP
