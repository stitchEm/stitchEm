// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QObject>
#include <QtTest>

#include "libvideostitch-gui/widgets/crop/cropcircleeditor.hpp"
#include "libvideostitch-gui/widgets/crop/croprectangleeditor.hpp"

class VideostitchTestInputCrop : public QObject {
  Q_OBJECT

 public:
  VideostitchTestInputCrop();

 private:
  Crop createRectangleCropEditor(Crop initCrop) const;

  // Compares the initial and resultant crop. They should be equal after any change
  void compareCrops(const Crop init, const Crop result);

 private Q_SLOTS:
  // All zero
  void testZeroValue();
  // Only ood values
  void testOddValues();
  // Only pair values
  void testPairValues();
  // Negative left top corner
  void testNegativeValues();
  // Only positive values
  void testPositiveValues();
  // The rectangle crop should be equal to the frame size
  void testRestoreRectangleCrop();
  // The circular crop should be equal to the frame size
  void testRestoreCircularCrop();
};

VideostitchTestInputCrop::VideostitchTestInputCrop() {}

Crop VideostitchTestInputCrop::createRectangleCropEditor(Crop initCrop) const {
  const QSize frame(1920, 960);
  const QSize thumbnail(800, 600);
  CropRectangleEditor cropShapeEditor(thumbnail, frame, initCrop);
  cropShapeEditor.setCrop(initCrop);
  return cropShapeEditor.getCrop();
}

void VideostitchTestInputCrop::testZeroValue() {
  const Crop init(0, 0, 0, 0);
  const Crop result = createRectangleCropEditor(init);
  compareCrops(init, result);
}

void VideostitchTestInputCrop::testOddValues() {
  const Crop init(3, 1001, 9, 913);
  const Crop result = createRectangleCropEditor(init);
  compareCrops(init, result);
}

void VideostitchTestInputCrop::testPairValues() {
  const Crop init(12, 1212, 8, 856);
  const Crop result = createRectangleCropEditor(init);
  compareCrops(init, result);
}

void VideostitchTestInputCrop::testNegativeValues() {
  const Crop init(-5, 1900, -9, 800);
  const Crop result = createRectangleCropEditor(init);
  compareCrops(init, result);
}

void VideostitchTestInputCrop::testPositiveValues() {
  const Crop init(50, 1550, 50, 750);
  const Crop result = createRectangleCropEditor(init);
  compareCrops(init, result);
}

void VideostitchTestInputCrop::testRestoreRectangleCrop() {
  const QSize screenSize(1920, 1080);
  const QSize frame(1920, 960);
  const Crop initCrop(12, 1212, 8, 856);
  // This is the same way to calculate the crop dialog window size
  const float newWidth = screenSize.width() / 2.f;
  const QSize thumbnail = QSize(int(newWidth), int((newWidth * (float)frame.height() / frame.width())));
  ///////////////////////////////////////////////////////////////////////////
  CropRectangleEditor cropShapeEditor(thumbnail, frame, initCrop);
  cropShapeEditor.setCrop(initCrop);
  cropShapeEditor.setDefaultCrop();
  const Crop result = cropShapeEditor.getCrop();
  QCOMPARE(result.crop_left, 0);
  QCOMPARE(result.crop_top, 0);
  QCOMPARE(result.crop_bottom, frame.height());
  QCOMPARE(result.crop_right, frame.width());
}

void VideostitchTestInputCrop::testRestoreCircularCrop() {
  const QSize screenSize(1920, 1080);
  const QSize frame(1920, 960);
  const Crop initCrop(12, 1212, 8, 856);
  // This is the same way to calculate the crop dialog window size
  const float newWidth = screenSize.width() / 2.f;
  const QSize thumbnail = QSize(int(newWidth), int((newWidth * (float)frame.height() / frame.width())));
  ///////////////////////////////////////////////////////////////////////////
  CropCircleEditor cropShapeEditor(thumbnail, frame, initCrop);
  cropShapeEditor.setCrop(initCrop);
  cropShapeEditor.setDefaultCrop();
  const Crop result = cropShapeEditor.getCrop();
  const int left = frame.width() / 2 - frame.height() / 2;
  QCOMPARE(result.crop_left, left);
  QCOMPARE(result.crop_top, 0);
  QCOMPARE(result.crop_bottom, frame.height());
  QCOMPARE(result.crop_right, left + frame.height());
}

void VideostitchTestInputCrop::compareCrops(const Crop init, const Crop result) {
  QCOMPARE(init.crop_left, result.crop_left);
  QCOMPARE(init.crop_top, result.crop_top);
  QCOMPARE(init.crop_bottom, result.crop_bottom);
  QCOMPARE(init.crop_right, result.crop_right);
}

QTEST_MAIN(VideostitchTestInputCrop)

#include "cropInputTest.moc"
