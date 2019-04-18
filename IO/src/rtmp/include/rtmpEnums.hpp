// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

namespace VideoStitch {
namespace IO {
enum PacketType {
  PacketType_VideoDisposable,
  PacketType_VideoLow,
  PacketType_VideoHigh,
  PacketType_VideoHighest,
  PacketType_VideoSPS,
  PacketType_Audio
};

enum ColorPrimaries {
  ColorPrimaries_BT709 = 1,
  ColorPrimaries_Unspecified,
  ColorPrimaries_BT470M = 4,
  ColorPrimaries_BT470BG,
  ColorPrimaries_SMPTE170M,
  ColorPrimaries_SMPTE240M,
  ColorPrimaries_Film,
  ColorPrimaries_BT2020
};

enum ColorTransfer {
  ColorTransfer_BT709 = 1,
  ColorTransfer_Unspecified,
  ColorTransfer_BT470M = 4,
  ColorTransfer_BT470BG,
  ColorTransfer_SMPTE170M,
  ColorTransfer_SMPTE240M,
  ColorTransfer_Linear,
  ColorTransfer_Log100,
  ColorTransfer_Log316,
  ColorTransfer_IEC6196624,
  ColorTransfer_BT1361,
  ColorTransfer_IEC6196621,
  ColorTransfer_BT202010,
  ColorTransfer_BT202012
};

enum ColorMatrix {
  ColorMatrix_GBR = 0,
  ColorMatrix_BT709,
  ColorMatrix_Unspecified,
  ColorMatrix_BT470M = 4,
  ColorMatrix_BT470BG,
  ColorMatrix_SMPTE170M,
  ColorMatrix_SMPTE240M,
  ColorMatrix_YCgCo,
  ColorMatrix_BT2020NCL,
  ColorMatrix_BT2020CL
};

enum class RTMPConnectionStatus {
  Disconnected,
  Connecting,
  Connected,
};
}  // namespace IO
}  // namespace VideoStitch
