// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "testsettings.hpp"
#include "testwindow.hpp"

#include "libvideostitch-base/common-config.hpp"

#include "libvideostitch/gpu_device.hpp"
#include "libgpudiscovery/genericDeviceInfo.hpp"

#include <QObject>
#include <QtTest>

class VideostitchTestStitchingWindow : public QObject {
  Q_OBJECT

 public:
  VideostitchTestStitchingWindow();

 private slots:
  void init();
  void testStitchingWindow();

 private:
  QScopedPointer<TestSettings> testSettings;
};

VideostitchTestStitchingWindow::VideostitchTestStitchingWindow() {
  QCoreApplication::setOrganizationName(VIDEOSTITCH_ORG_NAME);
  QCoreApplication::setOrganizationDomain(VIDEOSTITCH_ORG_DOMAIN);
  QCoreApplication::setApplicationName("VideostitchTestStitchingWindow");
  testSettings.reset(new TestSettings());
  testSettings->setParent(qApp);
}

void VideostitchTestStitchingWindow::init() {
  QVector<int> fakeDevices;
  int nb = VideoStitch::Discovery::getNumberOfDevices();
  for (int index = 0; index < nb + 2; ++index) {  // Add 2 fake devices
    fakeDevices.append(index);
  }
  VSSettings::getSettings()->setDevices(fakeDevices);
}

void VideostitchTestStitchingWindow::testStitchingWindow() {
  TestWindow window;

  QVector<int> validDevices = VSSettings::getSettings()->getDevices();
  int devCount = VideoStitch::Discovery::getNumberOfDevices();
  QVERIFY(validDevices.size() <= devCount);
  QVERIFY(validDevices.size() > 0 || devCount == 0);
  for (int device : validDevices) {
    VideoStitch::Discovery::DeviceProperties prop;
    QVERIFY(VideoStitch::Discovery::getDeviceProperties(device, prop));
  }
}

QTEST_MAIN(VideostitchTestStitchingWindow)

#include "stitchingWindowTest.moc"
