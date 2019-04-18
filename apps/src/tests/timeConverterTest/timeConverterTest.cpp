// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QtTest>
#include "libvideostitch-gui/mainwindow/timeconverter.hpp"

class VideoStitchTimeConverterTest : public QObject {
  Q_OBJECT

 public:
  VideoStitchTimeConverterTest();

 private Q_SLOTS:
  void testTimeConverterFrameString();
  void testTimeConverterLongerThanHour();
  void testTimeConverterInvalidTime();
  void testTimeConverterFrameString_data();
  void testTimeConverterLongerThanHour_data();
  void testTimeConverterInvalidTime_data();

 private:
  VideoStitch::FrameRate frameRate = {100, 1};
};

VideoStitchTimeConverterTest::VideoStitchTimeConverterTest() {}

void VideoStitchTimeConverterTest::testTimeConverterFrameString() {
  QFETCH(frameid_t, frame);
  QFETCH(QString, expectedString);

  QString s = TimeConverter::frameToTimeDisplay(frame, frameRate);

  QCOMPARE(s, expectedString);

  frameid_t f = TimeConverter::timeDisplayToFrame(expectedString, frameRate);
  QCOMPARE(f, frame);
}

void VideoStitchTimeConverterTest::testTimeConverterLongerThanHour() {
  QFETCH(frameid_t, fps);
  QFETCH(bool, expected);
  bool b = TimeConverter::isLongerThanAnHour(fps, frameRate);
  QCOMPARE(b, expected);
}

void VideoStitchTimeConverterTest::testTimeConverterInvalidTime() {
  QFETCH(QString, string);
  QFETCH(frameid_t, expectedFrame);

  frameid_t f = TimeConverter::timeDisplayToFrame(string, frameRate);
  QCOMPARE(f, expectedFrame);
}

void VideoStitchTimeConverterTest::testTimeConverterFrameString_data() {
  QTest::addColumn<frameid_t>("frame");
  QTest::addColumn<QString>("expectedString");

  QTest::newRow("frame 0  @100fps") << 0 << "00:00:00";
  QTest::newRow("frame 10 @100fps") << 10 << "00:00:10";
  QTest::newRow("frame 101 @100fps") << 101 << "00:01:01";
  QTest::newRow("frame 2000 @100fps") << 2000 << "00:20:00";
}

void VideoStitchTimeConverterTest::testTimeConverterLongerThanHour_data() {
  QTest::addColumn<frameid_t>("fps");
  QTest::addColumn<bool>("expected");

  QTest::newRow("frame 10 @100fps") << 10 << false;
  QTest::newRow("frame 2000000 @100fps") << 2000000 << true;  // 5:33:20:00
  QTest::newRow("frame 359999 @100fps") << 359999 << false;   // 59:59:99
  QTest::newRow("frame 360000 @100fps") << 360000 << true;    // 1:00:00:00
}

void VideoStitchTimeConverterTest::testTimeConverterInvalidTime_data() {
  QTest::addColumn<QString>("string");
  QTest::addColumn<frameid_t>("expectedFrame");

  QTest::newRow("Input   :   :   @100fps") << ": : :" << 0;      // 00:00:00
  QTest::newRow("Input 20:   :   @100fps") << "20::" << 120000;  // 20:00:00
  QTest::newRow("Input   :20 :   @100fps") << ":20:" << 2000;    // 00:20:00
  QTest::newRow("Input   :   :40 @100fps") << "::40" << 40;      // 00:00:40
}

QTEST_APPLESS_MAIN(VideoStitchTimeConverterTest)
#include "timeConverterTest.moc"
