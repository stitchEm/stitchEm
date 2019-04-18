// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "curvestreewidget.hpp"
#include "timeline/curvegraphicsitem.hpp"

#include "libvideostitch-gui/widgets/stylablewidget.hpp"

#include <QPointer>

class ProjectDefinition;

namespace Ui {
class SeekBar;
}

/**
 * @brief The SeekBar class is a wrapper that wraps the Timeline which is used as a seekbar.
 *        It also contains a set of controls like a play/pause button, buttons to set the first/last frame
 *        and displays to show the current/first/last frames.
 */
class SeekBar : public QWidget, public GUIStateCaps {
  Q_OBJECT
  Q_MAKE_STYLABLE

 public:
  explicit SeekBar(QWidget* parent = nullptr);
  ~SeekBar();
  /**
   * @brief Gets the slider lower and upper boundary values.
   * @param min is the lowest value of the slider.
   * @param max is the highest value of the slider.
   */
  void setTimeline();
  frameid_t getCurrentFrameFromCursorPosition() const;
  frameid_t getMinimumFrame() const;
  frameid_t getMaximumFrame() const;
  CurvesTreeWidget* getCurvesTreeWidget();
  void reset();

 signals:
  /**
   * @brief Requests the State Manager to initiate a specific state transition.
   * @param s is the requested state.
   */
  void reqChangeState(GUIStateCaps::State s) override;

  void reqPlay();
  void reqPause();
  void reqSeek(SignalCompressionCaps* caps, frameid_t targetFrame);
  void valueChanged(int frame);
  void reqCurveChanged(SignalCompressionCaps* comp, VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type,
                       int inputId);
  void reqQuaternionCurveChanged(SignalCompressionCaps* comp, VideoStitch::Core::QuaternionCurve* curve,
                                 CurveGraphicsItem::Type type, int inputId);
  // TODO: offset inputs (not necessarily here depending on the design of the UI - I'm not sure who is supposed to have
  // the 'right' to change the PanoDefinition.).
  // - Modify pano input framerate
  // - call controller->resetPano()
  // - controller->seekFrame() to current frame.
  void reqAddKeyframe();
  /**
   * @brief Sends a signal to the stitcher to ask it to reset all the curves.
   */
  void reqResetCurves();
  void reqResetCurve(CurveGraphicsItem::Type type, int inputId = -1);
  void reqRefreshCurves();
  void reqUpdateSequence(const QString start, const QString stop);

 public slots:
  void pause();
  void play();
  void seek(frameid_t date);

  /**
   * @brief Changes the widget's stats to the given state.
   * @param s State you want to switch to.
   */
  virtual void changeState(GUIStateCaps::State s) override;

  void setProject(ProjectDefinition* p);
  void clearProject();

  void cleanStitcher();

  void refresh(mtime_t date);
  void updateToPano(frameid_t lastStitchableFrame, frameid_t currentFrame);

  void on_playButton_clicked(bool checked);
  void setValue(int val);

  void on_toStartButton_clicked();
  void on_toStopButton_clicked();

  void startTimeEdited(frameid_t frame);
  void stopTimeEdited(frameid_t frame);
  void leftShortcutCalled();
  void rightShortcutCalled();
  void setWorkingArea(frameid_t firstFr, frameid_t lastFr);
  void nextKeyFrameShortcutCalled();
  void prevKeyFrameShortcutCalled();

 private:
  void initialize();
  void enableInternal();
  void refreshTimeWidgets();
  void setCurrentFrameLabel(int frame);
  void updateSequence();

 private slots:
  void updateCurves();

 private:
  Ui::SeekBar* ui;
  qint64 firstBoundFrame, lastBoundFrame;
  std::shared_ptr<SignalCompressionCaps> comp;
  GUIStateCaps::State state;
  QPointer<PostProdProjectDefinition> project;
};
