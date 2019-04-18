// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/common.hpp"
#include "libvideostitch/config.hpp"
#include "libvideostitch/frame.hpp"

class TimeConverter {
 public:
  /**
   * @brief Converts the frame value into a time format string.
   * @param currentFrame The frame value.
   * @param frameRate The frame rate to use.
   * @return A string representing the frame in time format.
   */
  static VS_GUI_EXPORT QString frameToTimeDisplay(const frameid_t currentFrame, VideoStitch::FrameRate);

  /**
   * @brief Converts a high precision time value into a time string format.
   * @param currentDate The current date.
   * @return The time string format.
   */
  static VS_GUI_EXPORT QString dateToTimeDisplay(const mtime_t currentDate);

  /**
   * @brief Given the time string it calculates the frame number.
   * @param time The time in string format.
   * @param ok True if the conversion was successful.
   * @return The frame number.
   */
  static VS_GUI_EXPORT frameid_t timeDisplayToFrame(const QString time, VideoStitch::FrameRate, bool* ok = nullptr);

  /**
   * @brief Calculates if the frame number converted into a time format is bigger than 59:59:59:xxx
   * @param curFrame The frame number.
   * @return True if the time is longer than 59:59:59:xxx.
   */
  static VS_GUI_EXPORT bool isLongerThanAnHour(const frameid_t curFrame, VideoStitch::FrameRate);

  /**
   * @brief Checks the number of digits of the integer part of the fps value.
   * @return The number of digits.
   */
  static VS_GUI_EXPORT int numberOfDigitsOfIntegerPart(VideoStitch::FrameRate);

  /**
   * @brief Checks if the fps is over 99.
   * @return True if has three digits.
   */
  static VS_GUI_EXPORT bool hasMoreThanThreeIntDigits(VideoStitch::FrameRate);
};
