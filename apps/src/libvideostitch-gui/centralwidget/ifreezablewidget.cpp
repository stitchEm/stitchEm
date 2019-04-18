// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ifreezablewidget.hpp"

#include <libvideostitch/logging.hpp>
#include "libvideostitch-gui/mainwindow/statemanager.hpp"

IFreezableWidget::IFreezableWidget(const QString& name, QWidget* parent)
    : QWidget(parent), isActive(false), isLowPerformance(false), name(name) {
  initializeStateMachine();
  StateManager::getInstance()->registerObject(this);
}

void IFreezableWidget::activate() {
  VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "Showing " << name.toStdString() << std::endl;
  isActive = true;
  updateOnState();
  show();
}

void IFreezableWidget::deactivate() {
  VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "Hiding " << name.toStdString() << std::endl;
  if (isLowPerformance) {
    emit reqFreeze();
    isActive = false;
  }
  hide();
}

void IFreezableWidget::setIsLowPerformance(bool isLowPerformance) {
  IFreezableWidget::isLowPerformance = isLowPerformance;
  // immediately deactivate widget
  if (isLowPerformance && !isActive) {
    deactivate();
  }
}

void IFreezableWidget::changeState(GUIStateCaps::State) {
  if (!isLowPerformance || isActive) {
    updateOnState();
  }
}

void IFreezableWidget::updateOnState() {
  switch (StateManager::getInstance()->getCurrentState()) {
    case GUIStateCaps::stitch:
      emit reqUnfreeze();
      break;
    case GUIStateCaps::idle:
    case GUIStateCaps::disabled:
      break;
    case GUIStateCaps::frozen:
      emit reqFreeze();
      break;
    default:
      Q_ASSERT(0);
      return;
  }
  setEnabled(StateManager::getInstance()->getCurrentState() == GUIStateCaps::stitch);
}

void IFreezableWidget::unload() {
  clearScreenshot();
  disconnectFromDeviceWriter();
}

void IFreezableWidget::onStateUnloadedEntered() {
  VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "-> Unloaded @ " << name.toStdString() << std::endl;
  unload();
}

void IFreezableWidget::onStateFrozenEntered() {
  VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "-> Frozen @ " << name.toStdString() << std::endl;
  freeze();
}

void IFreezableWidget::onStateWaitForGLEntered() {
  VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "-> WaitForGL @ " << name.toStdString() << std::endl;
  unfreeze();
}

void IFreezableWidget::onStateNormalEntered() {
  VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "-> Normal @ " << name.toStdString() << std::endl;
  showGLView();
}

void IFreezableWidget::initializeStateMachine() {
  QState* stateNormal = new QState();

  connect(stateNormal, &QState::entered, this, &IFreezableWidget::onStateNormalEntered);

  stateMachine.addState(stateNormal);
  stateMachine.setInitialState(stateNormal);
  stateMachine.start();
}

//////// disconnect device writer //////////////

void IFreezableWidget::disconnectFromDeviceWriter() {
  foreach (auto var, connections) { QObject::disconnect(var); }
  connections.clear();
}
