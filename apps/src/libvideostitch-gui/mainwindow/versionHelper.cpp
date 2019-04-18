// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "versionHelper.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include <QRegularExpression>

namespace VideoStitch {
namespace Helper {

static const QString STABLE_VAHANAVR("stable-vahanaVR");
static const QString STABLE_STUDIO("stable-studio");

QString getStringFromChannel(Channel channel) {
  switch (channel) {
    case Channel::Unknown:
      return QStringLiteral("unknown");
    case Channel::Alpha:
      return QStringLiteral("alpha");
    case Channel::Beta:
      return QStringLiteral("beta");
    case Channel::ReleaseCandidate:
      return QStringLiteral("RC");
    case Channel::Stable:
      return QString();
  }

  return getStringFromChannel(Channel::Unknown);
}

Channel getChannelFromString(const QString& str) {
  if (str.compare(getStringFromChannel(Channel::Alpha), Qt::CaseInsensitive) == 0) {
    return Channel::Alpha;
  } else if (str.compare(getStringFromChannel(Channel::Beta), Qt::CaseInsensitive) == 0) {
    return Channel::Beta;
  } else if (str.compare(getStringFromChannel(Channel::ReleaseCandidate), Qt::CaseInsensitive) == 0) {
    return Channel::ReleaseCandidate;
  } else if (str.compare(getStringFromChannel(Channel::Stable), Qt::CaseInsensitive) == 0) {
    return Channel::Stable;
  } else {
    return Channel::Unknown;
  }
}

bool getAppsVersionComponents(const QString& versionString, int& major, int& minor, int& patch, Channel& channel,
                              int& channelVersion) {
  // APP VERSION TEMPLATE
  // v.<MAJOR>.<MINOR>.<PATCH>.<CHANNEL><CHANNEL_REV>
  // example v1.2.3.beta4
  QRegularExpression versionExpression(".*v?\\d+\\.\\d+\\.\\d+\\.?(alpha|beta|RC)?\\d*");
  if (!versionExpression.match(versionString).hasMatch()) {
    LogManager::getInstance()->writeToLogFile("Error: the current version doesn't respect the version template");
    return false;
  }

  QStringList components = versionString.split(".");
  bool ok = false;
  QString majorString = components.takeFirst();
  major = majorString.replace(QRegExp(".*v"), "").toInt(&ok);
  if (!ok) {
    LogManager::getInstance()->writeToLogFile(
        QString("Error while converting major version number to integer: %1").arg(majorString));
  }
  const QString minorString = components.takeFirst();
  minor = minorString.toInt(&ok);
  if (!ok) {
    LogManager::getInstance()->writeToLogFile(
        QString("Error while converting minor version number to integer: %1").arg(minorString));
  }
  const QString patchString = components.takeFirst();
  patch = patchString.left(patchString.indexOf("-")).toInt(&ok);
  if (!ok) {
    LogManager::getInstance()->writeToLogFile(
        QString("Error while converting patch number to integer: %1").arg(patchString));
  }

  if (components.size() > 0) {
    QString chan = components.takeFirst();
    chan = chan.left(chan.indexOf("-"));
    QString channelString;
    QString chanRev;
    foreach (QChar c, chan) {
      if (c.isDigit()) {
        chanRev.append(c);
      } else {
        channelString.append(c);
      }
    }
    channel = getChannelFromString(channelString);
    channelVersion = chanRev.toInt();
  } else {
    channel = Channel::Stable;
    channelVersion = -1;
  }

  return true;
}

bool AppsVersion::parseVersionString(const QString& stringVersion) {
  return getAppsVersionComponents(stringVersion, majorVersion, minorVersion, patchVersion, channel, channelVersion);
}

AppsVersion::AppsVersion(const QString& stringVersion)
    : majorVersion(0), minorVersion(0), patchVersion(0), channel(Channel::Stable), channelVersion(-1) {
  if (!stringVersion.isEmpty()) {
    getAppsVersionComponents(stringVersion, majorVersion, minorVersion, patchVersion, channel, channelVersion);
  }
}

bool AppsVersion::isStableVersion() const { return channel == Channel::Stable || channel == Channel::ReleaseCandidate; }

QString AppsVersion::toString() const {
  QString channelString = channel == Channel::Stable
                              ? ""
                              : QString(".%0%1").arg(getStringFromChannel(channel), QString::number(channelVersion));
  return QString("v%0.%1.%2%3")
      .arg(QString::number(majorVersion), QString::number(minorVersion), QString::number(patchVersion), channelString);
}

bool AppsVersion::operator==(const AppsVersion& other) const {
  return majorVersion == other.majorVersion && minorVersion == other.minorVersion &&
         patchVersion == other.patchVersion && channel == other.channel && channelVersion == other.channelVersion;
}

bool AppsVersion::operator!=(const AppsVersion& other) const { return !(*this == other); }

bool AppsVersion::operator<(const AppsVersion& other) const {
  if (majorVersion < other.majorVersion) {
    return true;
  }
  if (majorVersion > other.majorVersion) {
    return false;
  }

  if (minorVersion < other.minorVersion) {
    return true;
  }
  if (minorVersion > other.minorVersion) {
    return false;
  }

  if (patchVersion < other.patchVersion) {
    return true;
  }
  if (patchVersion > other.patchVersion) {
    return false;
  }

  if (channel < other.channel) {
    return true;
  }
  if (channel > other.channel) {
    return false;
  }

  if (channelVersion < other.channelVersion) {
    return true;
  }
  if (channelVersion > other.channelVersion) {
    return false;
  }

  return false;
}

bool getAppsInfoComponents(const QString& infoString, QString& appName, AppsVersion& version, int& commitCount,
                           QString& commitHash, QString& branchName, QDateTime& commitDate) {
  // Version template
  // <APP_NAME>-<VERSION>[-<COMMITS_COUNT>-<COMMIT_HASH>]-<GIT_BRANCH>.<COMMIT_DATE>
  // Release versions won't have commits count and commit hash

  QRegularExpression versionInfoExpression(
      "(VahanaVR|Studio)-[0-9.a-zA-Z]+-?\\d*-[a-zA-Z0-9]*-[a-zA-Z0-9-]+\\.\\d{4}-\\d{2}-\\d{2}");
  if (!versionInfoExpression.match(infoString).hasMatch()) {
    LogManager::getInstance()->writeToLogFile("Error: the current version doesn't respect the version template");
    return false;
  }

  QStringList components = infoString.split("-");
  appName = components.takeFirst();
  version.parseVersionString(components.takeFirst().mid(1));

  // parse commit infos
  commitCount = 0;
  commitHash.clear();
  branchName.clear();

  branchName = components.join("-");
  components = branchName.split(".");
  // parse timestamp
  if (components.size() > 1) {
    branchName = components.takeFirst();
    commitDate = QDateTime::fromString(components.takeFirst(), "yyyy-MM-dd");
  }
  bool isDevVersion = !branchName.endsWith(STABLE_STUDIO) && !branchName.endsWith(STABLE_VAHANAVR);
  if (isDevVersion) {
    components = branchName.split("-");
    if (components.size() < 3) {
      LogManager::getInstance()->writeToLogFile("Error: the current version doesn't respect the version template");
      return false;
    }
    commitCount = components.takeFirst().toInt();
    commitHash = components.takeFirst();
    branchName = components.join("-");
  }

  return true;
}

AppsInfo::AppsInfo(const QString& stringAppInfo)
    : appName(QStringLiteral("")),
      version(),
      commitCount(0),
      commitHash((QStringLiteral(""))),
      branchName((QStringLiteral(""))),
      commitTimeStamp() {
  getAppsInfoComponents(stringAppInfo, appName, version, commitCount, commitHash, branchName, commitTimeStamp);
}

bool AppsInfo::isDevVersion() const { return branchName != STABLE_STUDIO && branchName != STABLE_VAHANAVR; }

QString AppsInfo::toString() const {
  return QString("%0-%1").arg(appName, version.toString()) +
         (isDevVersion() ? QString("-%0-%1").arg(commitHash, branchName) : QString());
}

}  // namespace Helper
}  // namespace VideoStitch
