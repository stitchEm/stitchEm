// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef VSGRAPHICS_HPP
#define VSGRAPHICS_HPP

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsSceneDragDropEvent>

/**
 * @brief Interface for QGraphicsScene. This interface should be used to overload the Drag&Drop function.
 *        Indeed that function is disabled by default.
 */
class VS_GUI_EXPORT VSGraphicsScene : public QGraphicsScene {
  Q_OBJECT

 public:
  explicit VSGraphicsScene(QObject *parent = 0);
  virtual ~VSGraphicsScene() {}

  static const QColor backgroundColor;
  static const QBrush backgroundBrush;

 private slots:
  /**
   * @brief Slot called when a drag event enters the widget.
   * @param event Drag event you want that will be processed.
   */
  void dragEnterEvent(QGraphicsSceneDragDropEvent *event);
  void dropEvent(QGraphicsSceneDragDropEvent *event);
};

/**
 * @brief Interface for QGraphicsView. This interface should be used when you need to overload the show and resize
 * event. It allows the Widget to display correctly its scene when the widget is shown or resized. It changes some
 * default optimizations to speedup up the rendering (lowering a bit the display quality).
 */
class VS_GUI_EXPORT VSGraphicsView : public QGraphicsView {
  Q_OBJECT

 public:
  explicit VSGraphicsView(QWidget *parent = 0);
  virtual ~VSGraphicsView() {}

 private slots:
  /**
   * @brief Slot called when the widget is shown.
   * @param event Event you want to process.
   */
  void showEvent(QShowEvent *event);
  /**
   * @brief Slot called when the widget is being resized.
   * @param event Resize event you may process.
   */
  void resizeEvent(QResizeEvent *event);
};

#endif  // VSGRAPHICS_HPP
