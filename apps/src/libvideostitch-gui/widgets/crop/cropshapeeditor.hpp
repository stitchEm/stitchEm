// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>
#include <QtConcurrent>
#include <QPen>
#include <memory>
#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/widgets/crop/crop.hpp"

/**
 * @brief Class for drawing and modifying the crop shape over the input frame
 */
class VS_GUI_EXPORT CropShapeEditor : public QWidget {
  Q_OBJECT
  Q_PROPERTY(QColor lineColor READ getLineColor WRITE setLineColor DESIGNABLE true)
  Q_PROPERTY(QColor fillColor READ getFillColor WRITE setFillColor DESIGNABLE true)
  Q_PROPERTY(QColor disableColor READ getDisableColor WRITE setDisableColor DESIGNABLE true)

 public:
  /**
   * @brief Constructor
   * @param thumbSize The window size
   * @param frmSize The real input frame size
   * @param initCrop Initial crop values
   * @param parent Parent widget
   */
  explicit CropShapeEditor(const QSize thumbSize, const QSize frmSize, const Crop& initCrop,
                           QWidget* const parent = nullptr);

  /**
   * @brief Destructor
   */
  ~CropShapeEditor();

  /**
   * @brief Gets the current crop from the drawed image
   * @return A crop
   */
  virtual const Crop getCrop() const;

  /**
   * @brief Sets the crop values to be drawn in the paint event
   * @param crop
   */
  void setCrop(const Crop& crop);

  /**
   * @brief Check whether an auto- or manual-crop is valid.
   * @return True if it's valid.
   */
  virtual bool isValidCrop() const;

  /**
   * @brief Disable the crop edition.
   * @param disable True for disable it.
   */
  void disableEdition(const bool disable);

  /**
   * @brief Draw a crop shape
   * @param painter A Painter reference
   */
  virtual void drawCropShape(QPainter& painter) = 0;

  // Color properties helpers
  void setLineColor(QColor color);
  QColor getLineColor() const;
  void setFillColor(QColor color);
  QColor getFillColor() const;
  void setDisableColor(QColor color);
  QColor getDisableColor() const;

 public slots:
  virtual void setDefaultCrop() = 0;

  /**
   * @brief Sets a default crop shape filling the thumbnail size
   */
  void onResetToDefault();

  /**
   * @brief check if the autocrop function is supported
   * @return
   */
  virtual bool isAutoCropSupported() const;

 signals:
  /**
   * @brief Signal triggered when a new change on the crop shape is performed
   * @param crop A new crop value
   */
  void notifyCropSet(const Crop& crop);

 protected:
  static const unsigned int PEN_THICK = 2;
  static const unsigned int SEL_OFFSET = 3;
  // Event methods override
  virtual void paintEvent(QPaintEvent* event) override;
  virtual void wheelEvent(QWheelEvent* event) override;
  virtual void showEvent(QShowEvent* event) override;

  // Auxiliary methods
  virtual const QRectF getTopArea(const QRectF rectangle) const = 0;
  virtual const QRectF getBottomArea(const QRectF rectangle) const = 0;
  virtual const QRectF getLeftArea(const QRectF rectangle) const = 0;
  virtual const QRectF getRightArea(const QRectF rectangle) const = 0;
  virtual const QRectF getTopLeftCorner(const QRectF rectangle) const = 0;
  virtual const QRectF getTopRightCorner(const QRectF rectangle) const = 0;
  virtual const QRectF getBottomLeftCorner(const QRectF rectangle) const = 0;
  virtual const QRectF getBottomRightCorner(const QRectF rectangle) const = 0;
  const QRectF cropToShape(const Crop& crop) const;
  const QRectF getCentralArea(const QRectF rectangle) const;
  float getRatio() const;

  /**
   * @brief Draws a cross in the center of the shape
   * @param painter A Painter reference
   */
  void drawCenterCross(QPainter& painter);

  enum class ModificationMode {
    NoModification,
    ResizeFromTop,
    ResizeFromBottom,
    ResizeFromLeft,
    ResizeFromRight,
    ResizeFromTopLeft,
    ResizeFromTopRight,
    ResizeFromBottomLeft,
    ResizeFromBottomRight,
    Move
  };

  QSize frameSize;
  QSize thumbnailSize;
  QRectF shape;
  QColor lineColor;
  QColor fillColor;
  QColor disableColor;
  QColor currentLine;
  QColor currentFill;
  QPen border;
  QPointF distanceToCenter;
  const float opacity;
  ModificationMode modificationMode;
  bool ignoreEvent;
};
