// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "h264codec.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"

H264Codec::H264Codec(QWidget *const parent) : MpegLikeCodec(parent) {}

QString H264Codec::getKey() const { return QStringLiteral("h264"); }

bool H264Codec::meetsSizeRequirements(int width, int height) const {
  // check that the width and height are below the limits, and are even numbers
  return (width <= H264_MAX_WIDTH && height <= H264_MAX_WIDTH && width % 2 == 0 && height % 2 == 0);
}
void H264Codec::correctSizeToMeetRequirements(int &width, int &height) {
  bool limits_reached = false;
  bool odd_dimension = false;

  // correct the width and height to be within the codec limits
  if (!(width <= H264_MAX_WIDTH && height <= H264_MAX_WIDTH)) {
    double ratio;
    if (width != 0 && height != 0) {
      ratio = (double)width / (double)height;
    } else {
      // set a default output aspect ratio
      ratio = 2.;
    }
    if (ratio >= 1.0) {
      width = H264_MAX_WIDTH;
      height = H264_MAX_WIDTH / ratio;
    } else {
      width = H264_MAX_WIDTH * ratio;
      height = H264_MAX_WIDTH;
    }
    limits_reached = true;
    MsgBoxHandler::getInstance()->generic(tr("H264 codec is limited to %0 x %1. Setting the resolution to %2 x %3.")
                                              .arg(QString::number(H264_MAX_WIDTH), QString::number(H264_MAX_WIDTH),
                                                   QString::number(width), QString::number(height)),
                                          tr("Warning"), WARNING_ICON);
  }

  // check that the H.264 limits are even numbers
  Q_ASSERT(H264_MAX_WIDTH % 2 == 0);

  // correct the width and height so that they are even numbers
  if (width % 2) {
    /* increment width to avoid enlarging the vertical field of view */
    ++width;
    odd_dimension = true;
  }
  if (height % 2) {
    /* decrement height to avoid enlarging the vertical field of view */
    --height;
    odd_dimension = true;
  }

  // display a message to inform the user about the corrections
  // but do not display the second one if the first one was already displayed
  if (!limits_reached && odd_dimension) {
    MsgBoxHandler::getInstance()->generic(
        tr("H264 codec is limited to even dimensions. Setting the resolution to %0 x %1.")
            .arg(QString::number(width), QString::number(height)),
        tr("Warning"), WARNING_ICON);
  }
}
