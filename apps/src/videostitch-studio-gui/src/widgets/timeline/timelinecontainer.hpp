// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>

namespace Ui {
class TimelineContainer;
}
class TimelineTicks;
class Timeline;

/**
 * @brief The TimelineContainer class is a widget that contains Extended Timeline related widgets:
 *        _ the extended timeline itself
 *        _ the zoom controls
 *        _ the widget that displays the playhead and the ticks.
 *
 *        It also synchronizes the drawings of internal widgets to avoid blinking with the playhead.
 */
class TimelineContainer : public QWidget {
  Q_OBJECT
 public:
  explicit TimelineContainer(QWidget *parent = nullptr);
  ~TimelineContainer();
  Timeline *getTimeline() const;
  TimelineTicks *getTimelineTicks() const;
  void enableInternal();
  virtual QSize sizeHint() const override;
  bool allowsKeyFrameNavigation() const;

 private slots:
  /**
   * @brief Updates the timeline's minimum and maximal zoom using the actual timeline's size.
   * @param maxZoomX Maximum horizontal zoom.
   */
  void updateZoomSliders(double maxZoomX);
  void enableKeyFrameNavigation(bool enable);
  void on_plusButton_clicked();
  void on_minusButton_clicked();
  void on_prevKeyFrameButton_clicked();
  void on_nextKeyFrameButton_clicked();

 protected:
  virtual bool eventFilter(QObject *watched, QEvent *event) override;

 private:
  void connectSliders(bool state = true);
  void disconnectSliders();

  Ui::TimelineContainer *ui;
};
