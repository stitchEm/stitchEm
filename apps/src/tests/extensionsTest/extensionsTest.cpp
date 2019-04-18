// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QString>
#include <QtTest>
#include <QCoreApplication>
#include "libvideostitch-gui/centralwidget/formats/extensionhandlers/extensionhandler.hpp"

class VideostitchTestExtensions : public QObject {
  Q_OBJECT

 public:
  VideostitchTestExtensions();

 private:
  void runTest();

 private Q_SLOTS:

  void testExtensionHandlerMp4();
  void testExtensionHandlerMp4_data();

  void testExtensionHandlerJpeg();
  void testExtensionHandlerJpeg_data();

  void testExtensionHandlerTiff();
  void testExtensionHandlerTiff_data();

  void testExtensionHandlerPam();
  void testExtensionHandlerPam_data();

  void testExtensionHandlerPpm();
  void testExtensionHandlerPpm_data();

  void testExtensionHandlerRaw();
  void testExtensionHandlerRaw_data();

  void testExtensionHandlerYuv();
  void testExtensionHandlerYuv_data();
};

VideostitchTestExtensions::VideostitchTestExtensions() {}

void VideostitchTestExtensions::runTest() {
  QFETCH(QString, input);
  QFETCH(QString, format);
  QFETCH(QString, result);
  ExtensionHandler h;
  h.init();
  QCOMPARE(h.stripBasename(input, format), result);
  QCOMPARE(h.handle(result, format), input);
}

void VideostitchTestExtensions::testExtensionHandlerMp4() { runTest(); }

void VideostitchTestExtensions::testExtensionHandlerMp4_data() {
  QTest::addColumn<QString>("input");
  QTest::addColumn<QString>("format");
  QTest::addColumn<QString>("result");

  QTest::newRow("mp4 lower") << "output.mp4"
                             << "mp4"
                             << "output";
  QTest::newRow("mp4 higher") << "output.mp4"
                              << "mp4"
                              << "output";
  QTest::newRow("mp4 mixed") << "output.mp4"
                             << "mp4"
                             << "output";
  QTest::newRow("mp4 rand") << "y1ZukndLj5.mp4"
                            << "mp4"
                            << "y1ZukndLj5";
}

void VideostitchTestExtensions::testExtensionHandlerJpeg() { runTest(); }

void VideostitchTestExtensions::testExtensionHandlerJpeg_data() {
  QTest::addColumn<QString>("input");
  QTest::addColumn<QString>("format");
  QTest::addColumn<QString>("result");

  QTest::newRow("jpg lower") << "output-%frame%.jpg"
                             << "jpg"
                             << "output";
  QTest::newRow("jpg higher") << "output-%frame%.jpg"
                              << "jpg"
                              << "output";
  QTest::newRow("jpg mixed") << "output-%frame%.jpg"
                             << "jpg"
                             << "output";
  QTest::newRow("jpg rand") << "y1ZukndLj5-%frame%.jpg"
                            << "jpg"
                            << "y1ZukndLj5";
}

void VideostitchTestExtensions::testExtensionHandlerTiff() { runTest(); }

void VideostitchTestExtensions::testExtensionHandlerTiff_data() {
  QTest::addColumn<QString>("input");
  QTest::addColumn<QString>("format");
  QTest::addColumn<QString>("result");

  QTest::newRow("tiff lower") << "output-%frame%.tif"
                              << "tif"
                              << "output";
  QTest::newRow("tiff higher") << "output-%frame%.tif"
                               << "tif"
                               << "output";
  QTest::newRow("tiff mixed") << "output-%frame%.tif"
                              << "tif"
                              << "output";
  QTest::newRow("tiff rand") << "y1ZukndLj5-%frame%.tif"
                             << "tif"
                             << "y1ZukndLj5";
}

void VideostitchTestExtensions::testExtensionHandlerPpm() { runTest(); }

void VideostitchTestExtensions::testExtensionHandlerPpm_data() {
  QTest::addColumn<QString>("input");
  QTest::addColumn<QString>("format");
  QTest::addColumn<QString>("result");

  QTest::newRow("ppm lower") << "output-%frame%.ppm"
                             << "ppm"
                             << "output";
  QTest::newRow("ppm higher") << "output-%frame%.ppm"
                              << "ppm"
                              << "output";
  QTest::newRow("ppm mixed") << "output-%frame%.ppm"
                             << "ppm"
                             << "output";
  QTest::newRow("ppm rand") << "y1ZukndLj5-%frame%.ppm"
                            << "ppm"
                            << "y1ZukndLj5";
}

void VideostitchTestExtensions::testExtensionHandlerPam() { runTest(); }

void VideostitchTestExtensions::testExtensionHandlerPam_data() {
  QTest::addColumn<QString>("input");
  QTest::addColumn<QString>("format");
  QTest::addColumn<QString>("result");

  QTest::newRow("pam lower") << "output-%frame%.pam"
                             << "pam"
                             << "output";
  QTest::newRow("pam higher") << "output-%frame%.pam"
                              << "pam"
                              << "output";
  QTest::newRow("pam mixed") << "output-%frame%.pam"
                             << "pam"
                             << "output";
  QTest::newRow("pam rand") << "y1ZukndLj5-%frame%.pam"
                            << "pam"
                            << "y1ZukndLj5";
}

void VideostitchTestExtensions::testExtensionHandlerRaw() { runTest(); }

void VideostitchTestExtensions::testExtensionHandlerRaw_data() {
  QTest::addColumn<QString>("input");
  QTest::addColumn<QString>("format");
  QTest::addColumn<QString>("result");

  QTest::newRow("raw lower") << "output.abgr"
                             << "raw"
                             << "output";
  QTest::newRow("raw higher") << "output.abgr"
                              << "raw"
                              << "output";
  QTest::newRow("raw mixed") << "output.abgr"
                             << "raw"
                             << "output";
  QTest::newRow("raw rand") << "y1ZukndLj5.abgr"
                            << "raw"
                            << "y1ZukndLj5";
}

void VideostitchTestExtensions::testExtensionHandlerYuv() { runTest(); }

void VideostitchTestExtensions::testExtensionHandlerYuv_data() {
  QTest::addColumn<QString>("input");
  QTest::addColumn<QString>("format");
  QTest::addColumn<QString>("result");

  QTest::newRow("raw lower") << "output-%frame%y/u/v.ppm"
                             << "yuv420p"
                             << "output";
  QTest::newRow("raw higher") << "output-%frame%y/u/v.ppm"
                              << "yuv420p"
                              << "output";
  QTest::newRow("raw mixed") << "output-%frame%y/u/v.ppm"
                             << "yuv420p"
                             << "output";
  QTest::newRow("raw rand") << "y1ZukndLj5-%frame%y/u/v.ppm"
                            << "yuv420p"
                            << "y1ZukndLj5";
}

QTEST_APPLESS_MAIN(VideostitchTestExtensions)

#include "extensionsTest.moc"
