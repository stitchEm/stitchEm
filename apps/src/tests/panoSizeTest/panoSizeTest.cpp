// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QtTest>
#include "libvideostitch-gui/utils/panoutilities.hpp"

class VideoStitchPanoSizeTest : public QObject {
  Q_OBJECT

 public:
  VideoStitchPanoSizeTest();

 private Q_SLOTS:
  void testPanoSizeFromWidth();
  void testPanoSizeFromHeight();
  void testPanoSizeFromWidth_data();
  void testPanoSizeFromHeight_data();
};

VideoStitchPanoSizeTest::VideoStitchPanoSizeTest() {}

void VideoStitchPanoSizeTest::testPanoSizeFromWidth() {
  QFETCH(int, width);
  QFETCH(int, expectedWidth);
  QFETCH(int, expectedHeight);
  const VideoStitch::Util::PanoSize calculated = VideoStitch::Util::calculateSizeFromWidth(width);
  QCOMPARE(calculated.width, expectedWidth);
  QCOMPARE(calculated.height, expectedHeight);
}

void VideoStitchPanoSizeTest::testPanoSizeFromWidth_data() {
  QTest::addColumn<int>("width");
  QTest::addColumn<int>("expectedWidth");
  QTest::addColumn<int>("expectedHeight");

  QTest::newRow("Width 1920") << 1920 << 1920 << 960;
  QTest::newRow("Width 1024") << 1024 << 1024 << 512;
  QTest::newRow("Width 2048") << 2048 << 2048 << 1024;
  QTest::newRow("Width 101") << 101 << 100 << 50;
  QTest::newRow("Width 50") << 50 << 52 << 26;
  QTest::newRow("Width 333") << 333 << 332 << 166;
  QTest::newRow("Width 3558") << 3558 << 3560 << 1780;
  QTest::newRow("Width 21") << 21 << 20 << 10;
  QTest::newRow("Width 0") << 0 << 4 << 2;
  QTest::newRow("Width -1") << -1 << 4 << 2;
}

void VideoStitchPanoSizeTest::testPanoSizeFromHeight() {
  QFETCH(int, height);
  QFETCH(int, expectedWidth);
  QFETCH(int, expectedHeight);
  const VideoStitch::Util::PanoSize calculated = VideoStitch::Util::calculateSizeFromHeight(height);
  QCOMPARE(calculated.width, expectedWidth);
  QCOMPARE(calculated.height, expectedHeight);
}

void VideoStitchPanoSizeTest::testPanoSizeFromHeight_data() {
  QTest::addColumn<int>("height");
  QTest::addColumn<int>("expectedWidth");
  QTest::addColumn<int>("expectedHeight");

  QTest::newRow("Height 960") << 960 << 1920 << 960;
  QTest::newRow("Height 512") << 1024 << 2048 << 1024;
  QTest::newRow("Height 1024") << 2048 << 4096 << 2048;
  QTest::newRow("Height 101") << 101 << 204 << 102;
  QTest::newRow("Height 51") << 25 << 52 << 26;
  QTest::newRow("Height 333") << 166 << 332 << 166;
  QTest::newRow("Height 3558") << 1777 << 3556 << 1778;
  QTest::newRow("Height 21") << 21 << 44 << 22;
  QTest::newRow("Height 0") << 0 << 4 << 2;
  QTest::newRow("Height -1") << -1 << 4 << 2;
}

QTEST_APPLESS_MAIN(VideoStitchPanoSizeTest)
#include "panoSizeTest.moc"
