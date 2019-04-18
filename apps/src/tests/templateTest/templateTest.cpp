// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QString>
#include <QtTest>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <QCoreApplication>
#include "libvideostitch-gui/base/ptvMerger.hpp"

class VideoStitchTemplateTest : public QObject {
  Q_OBJECT

 public:
  VideoStitchTemplateTest();

 private Q_SLOTS:

  void testTemplate();
  void testTemplate_data();

 private:
  QString dataFolder;
};

VideoStitchTemplateTest::VideoStitchTemplateTest() { dataFolder = QString("./data") + QDir::separator(); }

void VideoStitchTemplateTest::testTemplate() {
  QFETCH(QString, templatePtv);
  QFETCH(QString, targetPtv);
  QFETCH(QString, resultPtv);
  QFETCH(QString, expectedPtv);
  VideoStitch::Helper::PtvMerger::saveMergedPtv((dataFolder + targetPtv).toStdString(),
                                                (dataFolder + templatePtv).toStdString(),
                                                (dataFolder + resultPtv).toStdString());

  QFile expectedPtvFile((dataFolder + expectedPtv));
  QFile resultPtvFile((dataFolder + expectedPtv));
  QString expectedContent;
  QString resultContent;

  QVERIFY(expectedPtvFile.open(QIODevice::ReadOnly | QIODevice::Text));
  expectedContent = QString(expectedPtvFile.readAll());

  QVERIFY(resultPtvFile.open(QIODevice::ReadOnly | QIODevice::Text));
  resultContent = QString(resultPtvFile.readAll());

  qDebug() << expectedContent;
  qDebug() << resultContent;
  QCOMPARE(expectedContent, resultContent);
}

void VideoStitchTemplateTest::testTemplate_data() {
  QTest::addColumn<QString>("templatePtv");
  QTest::addColumn<QString>("targetPtv");
  QTest::addColumn<QString>("resultPtv");
  QTest::addColumn<QString>("expectedPtv");

  QTest::newRow("basic object") << "basic_template.ptv"
                                << "basic_target.ptv"
                                << "basic_result.ptv"
                                << "basic_expectedresult.ptv";
  QTest::newRow("empty target") << "basic_template.ptv"
                                << "empty_target.ptv"
                                << "emptytarget_result.ptv"
                                << "basic_template.ptv";
  QTest::newRow("empty template") << "empty_template.ptv"
                                  << "basic_target.ptv"
                                  << "emptytemplate_result.ptv"
                                  << "basic_target.ptv";
  QTest::newRow("empty") << "empty_template.ptv"
                         << "empty_target.ptv"
                         << "empty_result.ptv"
                         << "empty_target.ptv";
  QTest::newRow("basic overwrite") << "overwrite_template.ptv"
                                   << "overwrite_target.ptv"
                                   << "overwrite_result.ptv"
                                   << "overwrite_template.ptv";
  QTest::newRow("basic list") << "list_template.ptv"
                              << "list_target.ptv"
                              << "list_result.ptv"
                              << "list_expectedresult.ptv";
  QTest::newRow("basic list reverted") << "listreverted_template.ptv"
                                       << "listreverted_target.ptv"
                                       << "list_result.ptv"
                                       << "listreverted_expectedresult.ptv";
}

QTEST_APPLESS_MAIN(VideoStitchTemplateTest)
#include "templateTest.moc"
