// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/caps/guistatecaps.hpp"

#include <QObject>
#include <atomic>

class VS_GUI_EXPORT StateManager : public QObject {
  Q_OBJECT

 public:
  static StateManager *getInstance();

  /**
   * @brief registerObject registers an object to the State Manager.
   * @param qObj is the object to connect to the State Manager.
   */
  void registerObject(QObject *qObj);

  /**
   * @brief getCurrentState returns current state.
   */
  inline GUIStateCaps::State getCurrentState() const { return currentState; }
 signals:
  /**
   * @brief Informs that the state has changed
   * @param s is new state.
   */
  void stateChanged(GUIStateCaps::State s);

 private slots:
  /**
   * @brief Changes the widget's states to the given state.
   * @param s State you want to switch to.
   */
  void changeState(GUIStateCaps::State s);

 private:
  explicit StateManager(QObject *parent = nullptr);

 private:
  GUIStateCaps::State currentState;
};
