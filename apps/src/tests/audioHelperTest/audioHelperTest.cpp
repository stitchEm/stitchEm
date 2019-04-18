// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QtTest>
#include "libvideostitch-gui/utils/audiohelpers.hpp"

Q_DECLARE_METATYPE(VideoStitch::Audio::SamplingRate)

class VideoStitchAudioHelperTest : public QObject {
  Q_OBJECT

 public:
  VideoStitchAudioHelperTest();

 private Q_SLOTS:
  void testSamplingRateIntroString();
  void testSamplingRateIntroString_data();
};

VideoStitchAudioHelperTest::VideoStitchAudioHelperTest() {}

void VideoStitchAudioHelperTest::testSamplingRateIntroString() {
  QFETCH(VideoStitch::Audio::SamplingRate, rate);
  QFETCH(QString, expectedString);
  const QString s = VideoStitch::AudioHelpers::getSampleRateString(rate);
  QCOMPARE(s, expectedString);
}

void VideoStitchAudioHelperTest::testSamplingRateIntroString_data() {
  QTest::addColumn<VideoStitch::Audio::SamplingRate>("rate");
  QTest::addColumn<QString>("expectedString");
  QTest::newRow("SR 44100") << VideoStitch::Audio::SamplingRate::SR_44100 << "44100 Hz";
  QTest::newRow("SR 192000") << VideoStitch::Audio::SamplingRate::SR_192000 << "192000 Hz";
  QTest::newRow("SR NONE") << VideoStitch::Audio::SamplingRate::SR_NONE << "No sampling";
  QTest::newRow("SR invalid") << VideoStitch::Audio::SamplingRate(100) << "No sampling";
}

QTEST_APPLESS_MAIN(VideoStitchAudioHelperTest)
#include "audioHelperTest.moc"
