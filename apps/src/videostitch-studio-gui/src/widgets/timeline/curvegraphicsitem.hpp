// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CURVEGRAPHICSITEMGROUP_HPP
#define CURVEGRAPHICSITEMGROUP_HPP

#include <QGraphicsItem>
#include <QGraphicsPathItem>

#include <memory>
#include <vector>

#include "libvideostitch/curves.hpp"

#define HANDLE_SIZE 6
class Timeline;
class SignalCompressionCaps;

/**
 * A QGraphicsItem that can draw a VS curve.
 */
class CurveGraphicsItem : public QObject, public QGraphicsItem {
  Q_OBJECT
  Q_INTERFACES(QGraphicsItem)
 public:
  class HandleBase;
  class Handle;
  class ConstantCurveHandle;

  typedef std::vector<HandleBase*> HandleBaseVector;
  typedef HandleBaseVector::iterator HandleBaseVectorIterator;
  typedef HandleBaseVector::reverse_iterator HandleBaseVectorRevIterator;

  /**
   * The type of curve that is represented.
   */
  enum Type {
    GlobalOrientation,
    GlobalExposure,
    GlobalBlueCorrection,
    GlobalRedCorrection,
    InputExposure,
    BlueCorrection,
    RedCorrection,
    Stabilization,
    StabilizationYaw,
    StabilizationPitch,
    StabilizationRoll,
    Unknown
  };

  /**
   * Creates a graphics item to display and manipulate @a curve. Ownership of @a curve is retained by the caller, but
   * all updates will be done directly on this object, so it must remain in scope for the lifetime of *this. Updates are
   * not thread safe, so @a curve must not be modified concurrently.
   * @a minValue and @a maxValue are the min/max values for the curve values. The curve will be scaled to fill the
   * timeline vertically.
   */
  CurveGraphicsItem(Timeline* view, VideoStitch::Core::Curve& curve, Type type, double minValue, double maxValue,
                    int inputId, const QColor& color);

  ~CurveGraphicsItem();

  QRectF boundingRect() const;

  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);

  /**
   * Adds a keyframe at the given frame.
   */
  void addKeyframe(int frame);

  /**
   * Removes the given keyframe.
   */
  void removeKeyframe(int frame);
  /**
   * @brief Adds a frame to the "removal List". Once you've added all the frames you wanna remove, just call
   * "processRemovalList"
   * @param frame Frame you want to add to the removal List.
   */
  void addToRemovalList(int frame);

  void processRemovalList();

  /**
   * @brief Returns the previous key frame on the curve (if there is one)
   * @param currentFrame The current frame on the timeline
   * @return the keyframe or zero if is not posible to found the previous one
   */
  int getPrevKeyFrame(int currentFrame);

  /**
   * @brief Returns the next key frame on the curve (if there is one)
   * @param currentFrame The current frame on the timeline
   * @return the keyframe or zero if is not posible to found the next one
   */
  int getNextKeyFrame(int currentFrame);

  const Timeline* view() const { return parentView; }

  /**
   * Signals the outside world that the curve has changed.
   */
  void signalChanged();

  /**
   * Converts from curve to scene value.
   */
  double toSceneV(double v) const;

  /**
   * Converts from scene to curve value.
   */
  double fromSceneV(double v) const;

  Type getType() const;

  int getInputId() const;

  bool hasKeyframe() const;

  QGraphicsPathItem* getPathItem() { return &pathItem; }

 signals:
  /**
   * A signal that handles curves changes. @a curve becomes the property of the recipient, and must be deleted by him.
   * The fully qualified name is required here because the signals system uses string matches and does not know that
   * CurveGraphicsItem::Type and Type are the same thing.
   */
  void changed(SignalCompressionCaps* comp, VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type, int inputId);

 private slots:
  /**
   * Rebuilds the path.
   */
  void rebuildPath();

 private:
  /**
   * Rebuilds the handles.
   */
  void rebuildHandles();
  /**
   * @brief Exponential base 10 method
   * @param x
   * @return exp(x*log(10.0));
   */
  double expBase10(double x) const;
  /**
   * @brief Converts an input number to a log scaled output
   * @param x
   * @return log10(x)
   */
  double toLogScale(double x) const;
  /**
   * @brief Converts an log scaled input to an unscaled output.
   * @param y
   * @return expBase10(y)
   */
  double fromLogScale(double y) const;

  /**
   * @brief Returns the frame where the iterator or reverse iterator is placed
   * @param it Iterator of handle Base
   * @return The frame number or zero in any other case
   */
  template <typename T>
  int getFrameFromIterator(T iterator) const;

  const static QBrush unselectedBrush;
  const static QBrush selectedBrush;
  const static QBrush nonselectableBrush;
  QList<int> removalList;

  Timeline* timeline;
  VideoStitch::Core::Curve& curve;
  const Type type;

  // Graphical objects.
  QGraphicsPathItem pathItem;  // Path used to draw the curve.
  HandleBaseVector handles;
  ConstantCurveHandle* constantHandle;
  const Timeline* const parentView;
  std::shared_ptr<SignalCompressionCaps> comp;
  const double minValue;
  const double maxValue;
  const int inputId;
};

#endif  // CURVEGRAPHICSITEMGROUP_HPP
