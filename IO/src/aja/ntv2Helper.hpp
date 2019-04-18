// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/frame.hpp"
#include "libvideostitch/plugin.hpp"

#include <ajatypes.h>
#include <ajastuff/common/types.h>
#include <ntv2publicinterface.h>
#include <ntv2utils.h>
#include <ntv2card.h>
#include <ntv2rp188.h>

using namespace VideoStitch;
using namespace Plugin;

FrameRate aja2vsFrameRate(const NTV2FrameRate frameRate);

NTV2FrameBufferFormat vs2ajaPixelFormat(const PixelFormat pixelFmt);

NTV2VideoFormat vs2ajaDisplayFormat(const DisplayMode displayFmt);

PixelFormat aja2vsPixelFormat(const NTV2FrameBufferFormat pixelFmt);

DisplayMode aja2vsDisplayFormat(const NTV2VideoFormat displayFmt);

TimecodeFormat NTV2FrameRate2TimecodeFormat(const NTV2FrameRate inFrameRate);

ULWord GetRP188RegisterForInput(const NTV2InputSource inInputSource);
