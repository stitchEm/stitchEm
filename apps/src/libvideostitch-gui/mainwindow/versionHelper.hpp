// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QDateTime>

#include "libvideostitch-gui/common.hpp"

namespace VideoStitch {
namespace Helper {

// Channels, in the order of stability
enum class Channel { Unknown, Alpha, Beta, ReleaseCandidate, Stable };

QString getStringFromChannel(Channel channel);
Channel getChannelFromString(const QString& str);

/**
 * @brief The AppsVersion represents a version number of VS
 * example of string = v1.2.3.beta4
 * majorVersion = 1
 * minorVersion = 2
 * patchVersion = 3
 * channel = beta (optional)
 * channelVersion = 4 (optional)
 */
class VS_GUI_EXPORT AppsVersion {
 public:
  explicit AppsVersion(const QString& stringVersion = QString());

  bool parseVersionString(const QString& stringVersion);

  bool isStableVersion() const;
  QString toString() const;
  bool operator==(const AppsVersion& other) const;
  bool operator!=(const AppsVersion& other) const;
  bool operator<(const AppsVersion& other) const;

 private:
  int majorVersion;
  int minorVersion;
  int patchVersion;
  Channel channel;
  int channelVersion;
};

/**
 * @brief The AppsVersion represents a version number of VS
 * example of string = VahanaVR-v1.1.5.beta3-16246-g42c3f6ce8a-vahanavr12x.2011-10-05
 * version = v1.1.5.beta3
 * commitCount = 16246 (optional)
 * commitHash = g42c3f6ce8a (optional)
 * branchName = vahanavr12x
 * commidDate = 05-10-2011
 */
class VS_GUI_EXPORT AppsInfo {
 public:
  explicit AppsInfo(const QString& stringAppInfo = QString());

  const AppsVersion& getVersion() const { return version; }
  QDateTime getTimeStamp() const { return commitTimeStamp; }
  bool isDevVersion() const;
  QString toString() const;

 private:
  QString appName;
  AppsVersion version;
  int commitCount;
  QString commitHash;
  QString branchName;
  QDateTime commitTimeStamp;
};

}  // namespace Helper
}  // namespace VideoStitch
