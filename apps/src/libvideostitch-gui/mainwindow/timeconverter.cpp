// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "timeconverter.hpp"

#include "libvideostitch/logging.hpp"

#include <QStringList>
#include <cmath>

QString TimeConverter::frameToTimeDisplay(const frameid_t currentFrame, VideoStitch::FrameRate frameRate) {
  if ((currentFrame == -1) || (frameRate.num <= 0) || (frameRate.den == 0)) {
    return "-1";
  }

  const qint64 seconds = (currentFrame * frameRate.den) / frameRate.num;
  const QString secondsStr = QString("%0").arg(seconds % 60, 2, 10, QLatin1Char('0'));

  const qint64 minutes = seconds / 60;
  const QString minutesStr = QString("%0").arg(minutes % 60, 2, 10, QLatin1Char('0'));

  const qint64 remainder = currentFrame - ceil((seconds * frameRate.num) / double(frameRate.den));
  const QString framesStr = QString("%0").arg(remainder, 2, 10, QLatin1Char('0'));
  QString ret = minutesStr + ":" + secondsStr + ":" + framesStr;
  const int hours = minutes / 60;
  if (hours > 0) {
    ret = QString::number(hours) + ":" + ret;
  }
  return ret;
}

QString TimeConverter::dateToTimeDisplay(const mtime_t currentDate) {
  if (currentDate == -1) {
    return "-1";
  }

  const qint64 seconds = currentDate / 1000000;
  const QString secondsStr = QString("%0").arg(seconds % 60, 2, 10, QLatin1Char('0'));

  const qint64 minutes = seconds / 60;
  const QString minutesStr = QString("%0").arg(minutes % 60, 2, 10, QLatin1Char('0'));

  const qint64 remainderInMicroSeconds = currentDate - seconds * 1000000;
  const QString framesStr = QString("%0").arg(remainderInMicroSeconds / 10000, 2, 10, QLatin1Char('0'));
  QString ret = minutesStr + ":" + secondsStr + ":" + framesStr;
  const int hours = minutes / 60;
  if (hours > 0) {
    QString hoursStr = QString::number(hours);
    ret = hoursStr + ":" + ret;
  }
  return ret;
}

frameid_t TimeConverter::timeDisplayToFrame(const QString time, VideoStitch::FrameRate frameRate, bool* ok) {
  if (time == "0") {
    return 0;
  }

  if (time == "-1") {
    return -1;
  }
  QStringList elements = time.split(":");
  if (elements.size() < 3) {
    return 0;
  }

  // Partial time, complete with 00
  for (int i = 0; i < elements.size(); ++i) {
    if (elements.at(i).isEmpty()) {
      elements[i] = "00";
    }
  }

  // Parse the frame number
  double frameNumber = elements.last().toDouble(ok);
  if (ok && !*ok) {
    return 0;
  }
  elements.pop_back();

  // Parse the seconds
  double seconds = elements.last().toDouble(ok);
  if (ok && !*ok) {
    return 0;
  }
  elements.pop_back();

  // Parse the minutes
  seconds += elements.last().toDouble(ok) * 60.0;
  if (ok && !*ok) {
    return 0;
  }
  elements.pop_back();

  // Parse the hours, if there are...
  if (elements.size()) {
    seconds += elements.last().toDouble(ok) * 3600.0;
    if (ok && !*ok) {
      return 0;
    }
    elements.pop_back();
  }
  frameNumber += seconds * double(frameRate.num) / double(frameRate.den);
  return ceil(frameNumber);
}

bool TimeConverter::isLongerThanAnHour(const frameid_t curFrame, VideoStitch::FrameRate frameRate) {
  return frameToTimeDisplay(curFrame, frameRate).split(":").size() >= 4;
}

int TimeConverter::numberOfDigitsOfIntegerPart(VideoStitch::FrameRate frameRate) {
  return QString::number(std::floor(frameRate.num / frameRate.den)).size();
}

bool TimeConverter::hasMoreThanThreeIntDigits(VideoStitch::FrameRate frameRate) {
  return numberOfDigitsOfIntegerPart(frameRate) >= 3;
}
