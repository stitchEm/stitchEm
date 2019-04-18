// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QtTest>

#include "libvideostitch-base/file.hpp"

#include "libgpudiscovery/delayLoad.hpp"

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/status.hpp"

#include <memory>

#ifdef DELAY_LOAD_ENABLED
SET_DELAY_LOAD_HOOK
#endif  // DELAY_LOAD_ENABLED

class VideoStitchNormalizeInput : public QObject {
  Q_OBJECT

 public:
  VideoStitchNormalizeInput();

 private Q_SLOTS:
  void testNormalizeInputs();
  void testNormalizeInputs_data();

 private:
  QString dataFolder;
};

VideoStitchNormalizeInput::VideoStitchNormalizeInput() { dataFolder = QString("./data") + QDir::separator(); }

void VideoStitchNormalizeInput::testNormalizeInputs() {
  QFETCH(QString, inputPtv);

  QString inputPtvFile = dataFolder + inputPtv;

  qDebug() << "Checking " << inputPtv;

  const VideoStitch::Discovery::Framework& selectedFramework =
      VideoStitch::BackendLibHelper::getBestFrameworkAndBackend();
  QVERIFY2(VideoStitch::BackendLibHelper::selectBackend(selectedFramework), "No backend available");

  VideoStitch::Potential<VideoStitch::Ptv::Parser> parser(VideoStitch::Ptv::Parser::create());
  QVERIFY(parser.ok());
  QVERIFY(parser->parse(inputPtvFile.toStdString()));

  std::unique_ptr<VideoStitch::Ptv::Value> val(parser->getRoot().clone());
  std::unique_ptr<VideoStitch::Core::InputDefinition> idef(VideoStitch::Core::InputDefinition::create(*val.get()));
  QVERIFY(idef != NULL);
  QVERIFY(idef->getReaderConfig().getType());

  const QString& fileName = QString::fromStdString(idef->getReaderConfig().asString());
  const QString& normalizedPath = File::normalizePath(fileName);
  int backSlashCount = normalizedPath.count(QLatin1Char('\\'));
  int forwardSlashCount = normalizedPath.count(QLatin1Char('/'));

  // verify it only contains forwardslashes or backslashes
  QVERIFY(backSlashCount == 0 || forwardSlashCount == 0);
}

void VideoStitchNormalizeInput::testNormalizeInputs_data() {
  QTest::addColumn<QString>("inputPtv");

  QTest::newRow("Url malformatted") << "norminput_malformattedurl.ptv";
  QTest::newRow("Url well formatted") << "norminput_wellformattedurl.ptv";
  QTest::newRow("Absolute local path") << "norminput_localabspath.ptv";
  QTest::newRow("Relative local path") << "norminput_locarelpath.ptv";
  QTest::newRow("UTF-8 path") << "norminput_utf8.ptv";
}

QTEST_APPLESS_MAIN(VideoStitchNormalizeInput)
#include "normalizeInputTest.moc"
