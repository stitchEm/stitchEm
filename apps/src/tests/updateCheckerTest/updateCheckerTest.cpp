// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QString>
#include <QtTest>
#include "libvideostitch-gui/mainwindow/versionHelper.hpp"

class UpdateCheckerTest : public QObject {
  Q_OBJECT

 private Q_SLOTS:
  void checkUpdate();
  void checkUpdate_data();

  void checkAppInfo();
  void checkAppInfo_data();

  void versionToString();
  void versionToString_data();

  void appInfoToString();
  void appInfoToString_data();
};

void UpdateCheckerTest::checkUpdate() {
  QFETCH(QString, currentVer);
  QFETCH(QString, remoteVer);
  QFETCH(bool, isOlder);
  VideoStitch::Helper::AppsVersion version1(currentVer);
  VideoStitch::Helper::AppsVersion version2(remoteVer);
  QCOMPARE(version1 < version2, isOlder);
}

void UpdateCheckerTest::checkUpdate_data() {
  QTest::addColumn<QString>("currentVer");
  QTest::addColumn<QString>("remoteVer");
  QTest::addColumn<bool>("isOlder");

  QTest::newRow("Major version superior") << "v2.0.0"
                                          << "v1.0.0" << false;
  QTest::newRow("Medium version superior") << "v1.2.0"
                                           << "v1.1.0" << false;
  QTest::newRow("Minor version superior") << "v0.0.2"
                                          << "v0.0.1" << false;

  QTest::newRow("Major version inferior") << "v1.0.0"
                                          << "v2.0.0" << true;
  QTest::newRow("Medium version inferior") << "v1.1.0"
                                           << "v1.2.0" << true;
  QTest::newRow("Minor version inferior") << "v0.0.1"
                                          << "v0.0.2" << true;

  QTest::newRow("Major & Medium version inferior") << "v1.1.0"
                                                   << "v2.2.0" << true;
  QTest::newRow("Major & Minor inferior") << "v1.0.1"
                                          << "v2.0.2" << true;
  QTest::newRow("All inferior") << "v0.0.1"
                                << "v2.2.2" << true;

  QTest::newRow("Major & Medium version superior") << "v2.2.0"
                                                   << "v1.1.0" << false;
  QTest::newRow("Major & Minor superior") << "v2.0.2"
                                          << "v1.0.1" << false;
  QTest::newRow("All superior") << "v2.2.2"
                                << "v0.0.1" << false;
  QTest::newRow("All equal") << "v2.2.2"
                             << "v2.2.2" << false;

  QTest::newRow("Two digits") << "v12.0.0"
                              << "v9.0.0" << false;

  QTest::newRow("Major & Medium version inferior without v") << "1.1.0"
                                                             << "2.2.0" << true;
  QTest::newRow("Major & Minor inferior without v") << "1.0.1"
                                                    << "2.0.2" << true;
  QTest::newRow("All inferior without v") << "0.0.1"
                                          << "2.2.2" << true;

  QTest::newRow("With channel") << "1.1.0.alpha1"
                                << "1.1.0.alpha1" << false;
  QTest::newRow("With superior channel") << "1.1.0.beta1"
                                         << "1.1.0.alpha1" << false;
  QTest::newRow("With inferior channel") << "1.1.0.alpha1"
                                         << "1.1.0.beta1" << true;
  QTest::newRow("With inferior channel") << "1.1.0.beta1"
                                         << "1.1.0.rc1" << true;
  QTest::newRow("Without superior channel (alpha)") << "1.1.0"
                                                    << "1.1.0.alpha1" << false;
  QTest::newRow("Without superior channel (beta)") << "1.1.0"
                                                   << "1.1.0.beta1" << false;
  QTest::newRow("Without inferior channel (beta)") << "1.1.0.beta1"
                                                   << "1.1.0" << true;
  QTest::newRow("With superior channel") << "1.1.0.beta1"
                                         << "1.1.0.beta2" << true;
  QTest::newRow("Without inferior channel") << "1.1.0.beta2"
                                            << "1.1.0.beta1" << false;
  QTest::newRow("Without channel + inferior minor") << "v1.2.0"
                                                    << "v1.3.0.beta1" << true;
  QTest::newRow("With channel + superior minor") << "v1.3.0.beta1"
                                                 << "v1.2.0" << false;
  QTest::newRow("With case sensitivity") << "v1.1.0.alpha1"
                                         << "v1.1.0.RC1" << true;
}

void UpdateCheckerTest::checkAppInfo() {
  QFETCH(QString, appInfoString);
  QFETCH(bool, isDevVersion);
  VideoStitch::Helper::AppsInfo appInfo(appInfoString);
  QCOMPARE(appInfo.isDevVersion(), isDevVersion);
}

void UpdateCheckerTest::checkAppInfo_data() {
  QTest::addColumn<QString>("appInfoString");
  QTest::addColumn<bool>("isDevVersion");

  QTest::newRow("Dev version, two - in branch name")
      << "AppName-v1.1.5-16246-g42c3f6ce8a-some-branch-name.2017-01-02" << true;
  QTest::newRow("Dev version, one - in branch name")
      << "AppName-v1.1.5-16246-g42c3f6ce8a-somebranch-name.2017-01-02" << true;
  QTest::newRow("Dev version, no -") << "VahanaVR-v1.1.5-16246-g42c3f6ce8a-somebranchname.2017-01-02" << true;

  QTest::newRow("Public version, VahanaVR") << "VahanaVR-v1.1.5-stable-vahanaVR.2017-01-02" << false;
  QTest::newRow("Public version, Studio") << "VahanaVR-v1.1.5-stable-studio.2017-01-02" << false;
}

void UpdateCheckerTest::versionToString() {
  QFETCH(QString, versionString);
  QFETCH(QString, expectedString);
  VideoStitch::Helper::AppsVersion version(versionString);
  QCOMPARE(version.toString(), expectedString);
}

void UpdateCheckerTest::versionToString_data() {
  QTest::addColumn<QString>("versionString");
  QTest::addColumn<QString>("expectedString");

  QTest::newRow("Release tag") << "v1.0.0"
                               << "v1.0.0";
  QTest::newRow("alpha1 tag") << "v1.0.0.alpha1"
                              << "v1.0.0.alpha1";
  QTest::newRow("beta1 tag") << "v1.0.0.beta1"
                             << "v1.0.0.beta1";
  QTest::newRow("rc1 tag") << "v1.0.0.rc1"
                           << "v1.0.0.RC1";
  QTest::newRow("Upper case tag") << "v1.0.0.BETA1"
                                  << "v1.0.0.beta1";
  QTest::newRow("dev tag") << "v1.0.0.beta1-53-dummyhash123"
                           << "v1.0.0.beta1";
  QTest::newRow("With product name") << "VahanaVR-v1.0.0.beta1-53"
                                     << "v1.0.0.beta1";
  QTest::newRow("With product name and date") << "VahanaVR-v1.0.0.beta1-53-dummyhash123-branchname.2017-01-02"
                                              << "v1.0.0.beta1";
}

void UpdateCheckerTest::appInfoToString() {
  QFETCH(QString, appInfoString);
  QFETCH(QString, expectedString);
  VideoStitch::Helper::AppsInfo appInfo(appInfoString);
  QCOMPARE(appInfo.toString(), expectedString);
}

void UpdateCheckerTest::appInfoToString_data() {
  QTest::addColumn<QString>("appInfoString");
  QTest::addColumn<QString>("expectedString");

  QTest::newRow("Dev tag") << "VahanaVR-v1.0.0.beta1-53-dummyhash123-branchname.2017-01-02"
                           << "VahanaVR-v1.0.0.beta1-dummyhash123-branchname";
  QTest::newRow("Dev tag, multi -") << "VahanaVR-v1.0.0.beta1-53-dummyhash123-some-branch-name.2017-01-02"
                                    << "VahanaVR-v1.0.0.beta1-dummyhash123-some-branch-name";
  QTest::newRow("Public beta1 tag") << "VahanaVR-v1.0.0.beta1-stable-vahanaVR.2017-01-02"
                                    << "VahanaVR-v1.0.0.beta1";
  QTest::newRow("Public rc1 tag") << "VahanaVR-v1.0.0.rc1-stable-vahanaVR.2017-01-02"
                                  << "VahanaVR-v1.0.0.RC1";
  QTest::newRow("Public release tag") << "VahanaVR-v1.0.0-stable-vahanaVR.2017-01-02"
                                      << "VahanaVR-v1.0.0";
}

QTEST_APPLESS_MAIN(UpdateCheckerTest)
#include "updateCheckerTest.moc"
