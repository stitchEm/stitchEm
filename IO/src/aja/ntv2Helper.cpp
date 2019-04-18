// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ntv2Helper.hpp"

FrameRate aja2vsFrameRate(const NTV2FrameRate frameRate) {
  switch (frameRate) {
    case NTV2_FRAMERATE_6000:
      return FrameRate(60, 1);
    case NTV2_FRAMERATE_5994:
      return FrameRate(60000, 1001);
    case NTV2_FRAMERATE_3000:
      return FrameRate(30, 1);
    case NTV2_FRAMERATE_2997:
      return FrameRate(30000, 1001);
    case NTV2_FRAMERATE_2500:
      return FrameRate(25, 1);
    case NTV2_FRAMERATE_2400:
      return FrameRate(24, 1);
    case NTV2_FRAMERATE_2398:
      return FrameRate(24000, 1001);
    case NTV2_FRAMERATE_5000:
      return FrameRate(50, 1);
    case NTV2_FRAMERATE_4800:
      return FrameRate(48, 1);
    case NTV2_FRAMERATE_4795:
      return FrameRate(48000, 1001);
    case NTV2_FRAMERATE_12000:
      return FrameRate(120, 1);
    case NTV2_FRAMERATE_11988:
      return FrameRate(120000, 1001);
    case NTV2_FRAMERATE_1500:
      return FrameRate(15, 1);
    case NTV2_FRAMERATE_1498:
      return FrameRate(15000, 1001);
    default:
      return FrameRate(1, 1);
  }
}

NTV2FrameBufferFormat vs2ajaPixelFormat(const PixelFormat pixelFmt) {
  switch (pixelFmt) {
    case RGBA:
      return NTV2_FBF_ABGR;
    case RGB:
      return NTV2_FBF_24BIT_RGB;
    case BGRU:
      return NTV2_FBF_ARGB;
    case UYVY:
      return NTV2_FBF_8BIT_YCBCR;
    case YUY2:
      return NTV2_FBF_8BIT_YCBCR_YUY2;
    default:
      return NTV2_FBF_INVALID;
  }
}

NTV2VideoFormat vs2ajaDisplayFormat(const DisplayMode displayFmt) {
  for (int format = int(NTV2_FORMAT_FIRST_HIGH_DEF_FORMAT); format < int(NTV2_MAX_NUM_VIDEO_FORMATS); ++format) {
    const FrameRate frameRate = aja2vsFrameRate(GetNTV2FrameRateFromVideoFormat(NTV2VideoFormat(format)));
    if (frameRate.num == displayFmt.framerate.num && frameRate.den == displayFmt.framerate.den &&
        int64_t(GetDisplayWidth(NTV2VideoFormat(format))) == displayFmt.width &&
        int64_t(GetDisplayHeight(NTV2VideoFormat(format))) == displayFmt.height &&
        IsProgressivePicture(NTV2VideoFormat(format)) != displayFmt.interleaved &&
        IsPSF(NTV2VideoFormat(format)) == displayFmt.psf) {
      return NTV2VideoFormat(format);
    }
  }
  return NTV2_FORMAT_UNKNOWN;
}

PixelFormat aja2vsPixelFormat(const NTV2FrameBufferFormat pixelFmt) {
  switch (pixelFmt) {
    case NTV2_FBF_10BIT_YCBCR:
      return YUV422P10;
    case NTV2_FBF_8BIT_YCBCR:
      return UYVY;
    case NTV2_FBF_ARGB:
      return PixelFormat::Unknown;
    case NTV2_FBF_RGBA:
      return RGBA;
    case NTV2_FBF_10BIT_RGB:
      return PixelFormat::Unknown;
    case NTV2_FBF_8BIT_YCBCR_YUY2:
      return YUY2;
    case NTV2_FBF_ABGR:
      return BGRU;

    // NTV2_FBF_LAST_SD_FBF = NTV2_FBF_ABGR,
    case NTV2_FBF_10BIT_DPX:
    case NTV2_FBF_10BIT_YCBCR_DPX:
    case NTV2_FBF_8BIT_DVCPRO:
    case NTV2_FBF_8BIT_QREZ:
    case NTV2_FBF_8BIT_HDV:
      return PixelFormat::Unknown;

    case NTV2_FBF_24BIT_RGB:
      return RGB;
    case NTV2_FBF_24BIT_BGR:
      return BGR;
    case NTV2_FBF_10BIT_YCBCRA:
    case NTV2_FBF_10BIT_DPX_LITTLEENDIAN:
    case NTV2_FBF_48BIT_RGB:
      return PixelFormat::Unknown;

    case NTV2_FBF_PRORES:
    case NTV2_FBF_PRORES_DVCPRO:
    case NTV2_FBF_PRORES_HDV:
      return YUV422P10;

    case NTV2_FBF_10BIT_RGB_PACKED:
    case NTV2_FBF_10BIT_ARGB:
    case NTV2_FBF_16BIT_ARGB:
    case NTV2_FBF_UNUSED_23:
    case NTV2_FBF_10BIT_RAW_RGB:
    case NTV2_FBF_10BIT_RAW_YCBCR:
      return PixelFormat::Unknown;

    default:
      return PixelFormat::Unknown;
  }
}

DisplayMode aja2vsDisplayFormat(const NTV2VideoFormat displayFmt) {
  const int64_t width = GetDisplayWidth(displayFmt);
  const int64_t height = GetDisplayHeight(displayFmt);
  switch (displayFmt) {
      // HD
    case NTV2_FORMAT_1080psf_2398:
      return DisplayMode(width, height, false, {24000, 1001}, true);
    case NTV2_FORMAT_1080p_2398:
      return DisplayMode(width, height, false, {24000, 1001});
    case NTV2_FORMAT_1080psf_2400:
      return DisplayMode(width, height, false, {24, 1}, true);
    case NTV2_FORMAT_1080p_2400:
      return DisplayMode(width, height, false, {24, 1});
    case NTV2_FORMAT_1080p_2500:
      return DisplayMode(width, height, false, {25, 1});
    case NTV2_FORMAT_1080p_2997:
      return DisplayMode(width, height, false, {30000, 1001});
    case NTV2_FORMAT_1080p_3000:
      return DisplayMode(width, height, false, {30, 1});
    // B formats
    case NTV2_FORMAT_1080i_5000:  // same as NTV2_FORMAT_1080p_5000
      return DisplayMode(width, height, true, {25, 1});
    case NTV2_FORMAT_1080i_5994:  // same as case NTV2_FORMAT_1080p_5994
      return DisplayMode(width, height, true, {30000, 1001});
    case NTV2_FORMAT_1080i_6000:  // same as case NTV2_FORMAT_1080p_6000:
      return DisplayMode(width, height, true, {30, 1});

      // 1080 x 2K
    case NTV2_FORMAT_1080p_2K_2398:
      return DisplayMode(width, height, false, {24000, 1001});
    case NTV2_FORMAT_1080p_2K_2400:
      return DisplayMode(width, height, false, {24, 1});
    case NTV2_FORMAT_1080psf_2K_2398:
      return DisplayMode(width, height, false, {24000, 1001}, true);
    case NTV2_FORMAT_1080psf_2K_2400:
      return DisplayMode(width, height, false, {24, 1}, true);
    case NTV2_FORMAT_1080psf_2K_2500:
      return DisplayMode(width, height, false, {25, 1}, true);
    case NTV2_FORMAT_1080p_2K_2500:
      return DisplayMode(width, height, false, {25, 1});

      // 2K
    case NTV2_FORMAT_2K_1498:
      return DisplayMode(width, height, false, {15000, 1001});
    case NTV2_FORMAT_2K_1500:
      return DisplayMode(width, height, false, {15, 1});
    case NTV2_FORMAT_2K_2398:
      return DisplayMode(width, height, false, {24000, 1001});
    case NTV2_FORMAT_2K_2400:
      return DisplayMode(width, height, false, {24, 1});
    case NTV2_FORMAT_2K_2500:
      return DisplayMode(width, height, false, {25, 1});

      // 4k UHD
    case NTV2_FORMAT_4x1920x1080psf_2398:
      return DisplayMode(width, height, false, {24000, 1001}, true);
    case NTV2_FORMAT_4x1920x1080p_2398:
      return DisplayMode(width, height, false, {24000, 1001});
    case NTV2_FORMAT_4x1920x1080psf_2400:
      return DisplayMode(width, height, false, {24, 1}, true);
    case NTV2_FORMAT_4x1920x1080p_2400:
      return DisplayMode(width, height, false, {24, 1});
    case NTV2_FORMAT_4x1920x1080psf_2500:
      return DisplayMode(width, height, false, {25, 1}, true);
    case NTV2_FORMAT_4x1920x1080p_2500:
      return DisplayMode(width, height, false, {25, 1});
    case NTV2_FORMAT_4x1920x1080psf_2997:
      return DisplayMode(width, height, false, {30000, 1001}, true);
    case NTV2_FORMAT_4x1920x1080p_2997:
      return DisplayMode(width, height, false, {30000, 1001});
    case NTV2_FORMAT_4x1920x1080psf_3000:
      return DisplayMode(width, height, false, {30, 1}, true);
    case NTV2_FORMAT_4x1920x1080p_3000:
      return DisplayMode(width, height, false, {30, 1});
    case NTV2_FORMAT_4x2048x1080p_5000:
      return DisplayMode(width, height, false, {50, 1});
    case NTV2_FORMAT_4x1920x1080p_5994:
      return DisplayMode(width, height, false, {60000, 1001});
    case NTV2_FORMAT_4x1920x1080p_6000:
      return DisplayMode(width, height, false, {60, 1});

      // 4K DCI
    case NTV2_FORMAT_4x2048x1080psf_2398:
      return DisplayMode(width, height, false, {24000, 1001}, true);
    case NTV2_FORMAT_4x2048x1080p_2398:
      return DisplayMode(width, height, false, {24000, 1001});
    case NTV2_FORMAT_4x2048x1080psf_2400:
      return DisplayMode(width, height, false, {24, 1}, true);
    case NTV2_FORMAT_4x2048x1080p_2400:
      return DisplayMode(width, height, false, {24, 1});
    case NTV2_FORMAT_4x2048x1080psf_2500:
      return DisplayMode(width, height, false, {25, 1}, true);
    case NTV2_FORMAT_4x2048x1080p_2500:
      return DisplayMode(width, height, false, {25, 1});
    case NTV2_FORMAT_4x2048x1080psf_2997:
      return DisplayMode(width, height, false, {30000, 1001}, true);
    case NTV2_FORMAT_4x2048x1080p_2997:
      return DisplayMode(width, height, false, {30000, 1001});
    case NTV2_FORMAT_4x2048x1080psf_3000:
      return DisplayMode(width, height, false, {30, 1}, true);
    case NTV2_FORMAT_4x2048x1080p_3000:
      return DisplayMode(width, height, false, {30, 1});
    case NTV2_FORMAT_4x2048x1080p_4795:
      return DisplayMode(width, height, false, {48000, 1001});
    case NTV2_FORMAT_4x2048x1080p_4800:
      return DisplayMode(width, height, false, {48, 1});
    case NTV2_FORMAT_4x1920x1080p_5000:
      return DisplayMode(width, height, false, {50, 1});
    case NTV2_FORMAT_4x2048x1080p_5994:
      return DisplayMode(width, height, false, {60000, 1001});
    case NTV2_FORMAT_4x2048x1080p_6000:
      return DisplayMode(width, height, false, {60, 1});
    case NTV2_FORMAT_4x2048x1080p_11988:
      return DisplayMode(width, height, false, {120000, 1001});
    case NTV2_FORMAT_4x2048x1080p_12000:
      return DisplayMode(width, height, false, {120, 1});
    default:
    case NTV2_FORMAT_UNKNOWN:
      return DisplayMode(0, 0, false, {1, 1});
  }
}

TimecodeFormat NTV2FrameRate2TimecodeFormat(const NTV2FrameRate inFrameRate) {
  switch (inFrameRate) {
    case NTV2_FRAMERATE_6000:
      return kTCFormat60fps;
    case NTV2_FRAMERATE_5994:
      return kTCFormat60fpsDF;
    case NTV2_FRAMERATE_4800:
      return kTCFormat48fps;
    case NTV2_FRAMERATE_4795:
      return kTCFormat48fps;
    case NTV2_FRAMERATE_3000:
      return kTCFormat30fps;
    case NTV2_FRAMERATE_2997:
      return kTCFormat30fpsDF;
    case NTV2_FRAMERATE_2500:
      return kTCFormat25fps;
    case NTV2_FRAMERATE_2400:
      return kTCFormat24fps;
    case NTV2_FRAMERATE_5000:
      return kTCFormat50fps;
    default:
      return kTCFormatUnknown;
  }
}

ULWord GetRP188RegisterForInput(const NTV2InputSource inInputSource) {
  switch (inInputSource) {
    case NTV2_INPUTSOURCE_SDI1:
      return kRegRP188InOut1DBB;
      break;  //	reg 29
    case NTV2_INPUTSOURCE_SDI2:
      return kRegRP188InOut2DBB;
      break;  //	reg 64
    case NTV2_INPUTSOURCE_SDI3:
      return kRegRP188InOut3DBB;
      break;  //	reg 268
    case NTV2_INPUTSOURCE_SDI4:
      return kRegRP188InOut4DBB;
      break;  //	reg 273
    case NTV2_INPUTSOURCE_SDI5:
      return kRegRP188InOut5DBB;
      break;  //	reg 342
    case NTV2_INPUTSOURCE_SDI6:
      return kRegRP188InOut6DBB;
      break;  //	reg 418
    case NTV2_INPUTSOURCE_SDI7:
      return kRegRP188InOut7DBB;
      break;  //	reg 427
    case NTV2_INPUTSOURCE_SDI8:
      return kRegRP188InOut8DBB;
      break;  //	reg 436
    default:
      return 0;
  }
}
