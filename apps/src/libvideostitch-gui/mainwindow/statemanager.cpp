// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "statemanager.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include <QCoreApplication>

StateManager::StateManager(QObject* parent) : QObject(parent), currentState(GUIStateCaps::idle) {}

StateManager* StateManager::getInstance() {
  StateManager* stateManager = qApp->findChild<StateManager*>();
  if (stateManager) {
    return stateManager;
  } else {
    return new StateManager(qApp);
  }
}

void StateManager::registerObject(QObject* qObj) {
  Q_ASSERT(qObj && (qObj != this));
  connect(this, SIGNAL(stateChanged(GUIStateCaps::State)), qObj, SLOT(changeState(GUIStateCaps::State)));
  connect(qObj, SIGNAL(reqChangeState(GUIStateCaps::State)), this, SLOT(changeState(GUIStateCaps::State)));
}

void StateManager::changeState(GUIStateCaps::State s) {
  if (s == currentState) {
    return;
  }

  QString message = QString("StateManager::changeState, state = %0").arg(s);
  if (sender()) {
    message += QString(", sender = %0").arg(sender()->metaObject()->className());
  }
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(message);

  currentState = s;
  emit stateChanged(s);
}
