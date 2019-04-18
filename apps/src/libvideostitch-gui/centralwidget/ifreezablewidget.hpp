// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/caps/guistatecaps.hpp"

#include <QWidget>
#include <QPointer>
#include <QStateMachine>

#include <atomic>

class VS_GUI_EXPORT IFreezableWidget : public QWidget, protected GUIStateCaps {
  Q_OBJECT
 public:
  explicit IFreezableWidget(const QString& name, QWidget* parent = nullptr);

  /**
   * @brief Activate the widget.
   */
  virtual void activate();

  /**
   * @brief Deactivate the widget, freezing and hiding it.
   */
  virtual void deactivate();

  /**
   * @brief if active, opengl is unloaded when the widget is hidden (not active).
   */
  void setIsLowPerformance(bool isLowPerformance);

 signals:
  void reqPreviousState();
  void reqChangeState(GUIStateCaps::State) override;

  void reqFreeze();
  void reqUnfreeze();
  void glViewReady();
  void unloaded();

 public slots:
  /**
   * @brief Changes the widget's stats to the given state.
   * @param s State you want to switch to.
   */
  virtual void changeState(GUIStateCaps::State s) override;

  /**
   * @brief Set the widget's device writer.
   * @param writer Device writer.
   */
  //  void setDeviceWriter(QtDeviceWriter* writer);

 protected slots:
  void disconnectFromDeviceWriter();
  virtual void clearScreenshot() = 0;

 protected:
  virtual void updateOnState();
  virtual void unload();
  virtual void freeze() = 0;
  virtual void unfreeze() = 0;
  virtual void showGLView() = 0;
  virtual void connectToDeviceWriter() = 0;

  std::atomic<bool> isActive;
  std::atomic<bool> isLowPerformance;
  //  QPointer<QtDeviceWriter> deviceWriter;
  QList<QMetaObject::Connection> connections;

 private slots:
  void onStateUnloadedEntered();

  void onStateFrozenEntered();

  void onStateWaitForGLEntered();

  void onStateNormalEntered();

 private:
  void initializeStateMachine();

  QStateMachine stateMachine;
  QString name;
};
