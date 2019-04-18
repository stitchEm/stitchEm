// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"
#include "centralwidget/processtab/iprocesswidget.hpp"

#include <atomic>

class Timeline;

class TimelineTicks : public IProcessWidget {
  Q_OBJECT
  Q_PROPERTY(QColor tickColor READ getTickColor WRITE setTickColor DESIGNABLE true)
  Q_PROPERTY(QColor boundHandleColor READ getBoundHandleColor WRITE setBoundHandleColor DESIGNABLE true)
  Q_PROPERTY(int handleSize READ getHandleSize WRITE setHandleSize DESIGNABLE true)
  Q_PROPERTY(int boundHandleSize READ getBoundHandleSize WRITE setBoundHandleSize DESIGNABLE true)
  Q_PROPERTY(int tickLength READ getTickLength WRITE setTickLength DESIGNABLE true)
  Q_PROPERTY(int subTickLength READ getSubTickLength WRITE setSubTickLength DESIGNABLE true)
  Q_PROPERTY(int frameLabelMargin READ getFrameLabelMargin WRITE setFrameLabelMargin DESIGNABLE true)
 public:
  explicit TimelineTicks(QWidget* parent = nullptr);
  void setTimeline(Timeline* timeline);

  /**
   * Properties
   */
  /**
   * @brief Sets the tick color in the timeline bar
   * @param color Color of the ticks.
   */
  void setTickColor(QColor color);
  QColor getTickColor() const;
  /**
   * @brief Sets the bound handle color in the timeline bar
   * @param color Color of the ticks.
   */
  void setBoundHandleColor(QColor color);
  QColor getBoundHandleColor() const;
  /**
   * @brief Sets the main tick length
   * @param length Length of the main tick in pixels.
   */
  void setTickLength(int length);
  int getTickLength() const;
  /**
   * @brief Sets the sub tick length.
   * @param length Length of the subticks in pixels.
   */
  void setSubTickLength(int length);
  int getSubTickLength() const;
  /**
   * @brief Sets the margin between the bottom of the main ticks and label.
   * @param margin Margin in pixels between the main ticks and the frame label.
   */
  void setFrameLabelMargin(int margin);
  int getFrameLabelMargin() const;
  /**
   * @brief Sets the minimal margin between two frame labels.
   * @param margin Minimum margin between two frame labels. in pixels.
   */
  void setTickLabelMargin(int margin);
  int getTickLabelMargin() const;
  /**
   * @brief Sets the handle's size in pixels.
   * @param size Handle size.
   */
  void setHandleSize(int size);
  int getHandleSize() const;
  void setBoundHandleSize(int size);
  int getBoundHandleSize() const;
  void moveCursorFromPosition(int position);

 signals:
  void lowerBoundHandlePositionChanged(frameid_t frame);
  void upperBoundHandlePositionChanged(frameid_t frame);

 protected:
  virtual void paintEvent(QPaintEvent* event) override;
  virtual void mousePressEvent(QMouseEvent* event) override;
  virtual void mouseMoveEvent(QMouseEvent* event) override;
  virtual void mouseReleaseEvent(QMouseEvent* event) override;
  virtual void mouseDoubleClickEvent(QMouseEvent* event) override;

 private:
  void paintHandle(QPainter& painter, int position, int origin, int size, const QColor color);
  bool isMouseCloseToHandle(int mousePosition, int handlePosition) const;
  frameid_t mapFromPosition(int position) const;
  void drawTicks(QPainter& painter);
  void drawTickLineAndLabel(QPainter& painter, const frameid_t frame, VideoStitch::FrameRate frameRate);
  void drawSubTickLine(QPainter& painter, const frameid_t frame);
  void drawSequenceHandler(const int position, QPainter& painter);
  void drawPlayerHandler(QPainter& painter);
  void drawBackground(QPainter& painter);
  void computeSteps(int& mainStep, int& subStep) const;
  VideoStitch::FrameRate getFrameRateToUse() const;
  static QVector<int> computeFrameScales(int upperFramerate);

  enum class MouseMovingType { NoMoving, MainCursorMoving, LowerBoundMoving, UpperBoundMoving };

  QColor tickColor;
  QColor boundHandleColor;
  int handleSize;
  int boundHandleSize;
  int tickLength;
  int subTickLength;
  int frameLabelMargin;
  Timeline* timeline;
  MouseMovingType movingType;
};
