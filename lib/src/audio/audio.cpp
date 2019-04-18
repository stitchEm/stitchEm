// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audio/resampler.hpp"
#include "util/plugin.hpp"

#include "libvideostitch/audio.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <thread>

using namespace VideoStitch::Plugin;

namespace VideoStitch {
namespace Audio {

Samples::Samples()
    : samples{},
      nbSamples(0),
      timestamp(-1),
      delete_([](data_buffer_t&) {}),
      depth(SamplingDepth::SD_NONE),
      rate(SamplingRate::SR_NONE),
      layout(UNKNOWN) {}

Samples::Samples(const SamplingRate r, const SamplingDepth d, const ChannelLayout l, mtime_t timestamp,
                 data_buffer_t& data, size_t nbSamples, deleter del)
    : Samples(r, d, l, timestamp, data.data(), nbSamples, del) {}

Samples::Samples(const SamplingRate r, const SamplingDepth d, const ChannelLayout l, mtime_t timestamp,
                 data_buffer_t& data, size_t nbSamples)
    : Samples(r, d, l, timestamp, data.data(), nbSamples) {}

Samples::Samples(const SamplingRate r, const SamplingDepth d, const ChannelLayout l, mtime_t timestamp, uint8_t** data,
                 size_t nbSamples, deleter del)
    : nbSamples(nbSamples), timestamp(timestamp), delete_(del), depth(d), rate(r), layout(l) {
  alloc(data);
}

Samples::Samples(const SamplingRate r, const SamplingDepth d, const ChannelLayout l, mtime_t timestamp, uint8_t** data,
                 size_t nbSamples)
    : nbSamples(nbSamples), timestamp(timestamp), depth(d), rate(r), layout(l) {
  delete_ = [](data_buffer_t& samples) {
    for (uint8_t*& ptr : samples) {
      delete[] ptr;
      ptr = nullptr;
    }
  };
  alloc(data);
}

Samples::~Samples() { delete_(samples); }

Samples::Samples(Samples&& o) : depth(o.depth), rate(o.rate), layout(o.layout) {
  // steal that music
  nbSamples = o.nbSamples;
  timestamp = o.timestamp;
  delete_ = o.delete_;
  samples = o.samples;
  o.clear();
}

Samples& Samples::operator=(Samples&& o) {
  rate = o.rate;
  depth = o.depth;
  layout = o.layout;
  // delete current audio signal
  delete_(samples);
  // steal that music
  nbSamples = o.nbSamples;
  timestamp = o.timestamp;
  delete_ = o.delete_;
  samples = o.samples;
  o.clear();
  return *this;
}

Samples Samples::clone() const {
  data_buffer_t newdata{};
  if (isInterleaved(depth)) {
    const size_t dataSize = nbSamples * getSampleSizeFromSamplingDepth(depth) * getNbChannelsFromChannelLayout(layout);
    newdata[0] = new uint8_t[dataSize];
    memcpy(newdata[0], samples[0], dataSize);
  } else {
    ChannelMap mask = SPEAKER_FRONT_LEFT;
    const size_t dataSize = nbSamples * getSampleSizeFromSamplingDepth(depth);
    for (int i = 0; i < MAX_AUDIO_CHANNELS; ++i) {
      if (mask & layout) {
        newdata[i] = new uint8_t[dataSize];
        memcpy(newdata[i], samples[i], dataSize);
      }
      mask = (ChannelMap)(mask << 1);
    }
  }
  return Samples(rate, depth, layout, timestamp, newdata, nbSamples);
}

void Samples::alloc(uint8_t** data) {
  samples = {};
  if (depth == SamplingDepth::UINT8 || depth == SamplingDepth::INT16 || depth == SamplingDepth::INT24 ||
      depth == SamplingDepth::INT32 || depth == SamplingDepth::FLT || depth == SamplingDepth::DBL) {
    // For interleaved data allocate first channel only
    samples[0] = data[0];
  } else {
    ChannelMap mask = SPEAKER_FRONT_LEFT;
    for (int i = 0; i < MAX_AUDIO_CHANNELS; ++i) {
      if (mask & layout) {
        samples[i] = data[i];
      }
      mask = (ChannelMap)(mask << 1);
    }
  }
}

void Samples::clear() {
  samples = {};
  nbSamples = 0;
  timestamp = -1;
  delete_ = [](data_buffer_t& samples) {
    for (uint8_t* ptr : samples) {
      delete[] ptr;
    }
  };
}

Status Samples::drop(size_t n) {
  if (n > nbSamples) {
    std::stringstream errmsg;
    errmsg << "nb of samples to drop " << n << " larger than nbSamples available " << nbSamples;
    return {Origin::AudioPipeline, ErrType::RuntimeError, errmsg.str()};
  }

  auto _drop = [this, n](int chan, int nbChannels) {
    uint8_t* newSamples = new uint8_t[getSampleSizeFromSamplingDepth(depth) * (nbSamples - n) * nbChannels];
    memcpy(newSamples, samples[chan] + getSampleSizeFromSamplingDepth(depth) * n * nbChannels,
           getSampleSizeFromSamplingDepth(depth) * (nbSamples - n) * nbChannels);
    delete[] samples[chan];
    samples[chan] = newSamples;
  };

  mapSamples(_drop);

  nbSamples -= n;

  assert(rate != SamplingRate::SR_NONE);
  timestamp += (mtime_t)std::round(n * 1000000. / getIntFromSamplingRate(rate));
  return Status::OK();
}

Status Samples::append(const Samples& other) {
  if (other.nbSamples == 0) {  // Nothing to append in fact
    return Status::OK();
  }
  if (other.depth != depth || other.layout != layout || other.rate != rate) {
    std::stringstream errmsg;
    errmsg << "Unexpected sample format to append : depth " << getStringFromSamplingDepth(other.depth) << " layout "
           << getStringFromChannelLayout(other.layout) << " rate " << getIntFromSamplingRate(other.rate)
           << " instead of depth " << getStringFromSamplingDepth(depth) << " layout "
           << getStringFromChannelLayout(layout) << " rate " << getIntFromSamplingRate(rate);
    return {Origin::AudioPipeline, ErrType::RuntimeError, errmsg.str()};
  }

  auto _append = [&](int chan, int nbChannels) {
    uint8_t* newSamples =
        new uint8_t[getSampleSizeFromSamplingDepth(depth) * (nbSamples + other.nbSamples) * nbChannels];
    memcpy(newSamples, samples[chan], getSampleSizeFromSamplingDepth(depth) * nbSamples * nbChannels);
    delete[] samples[chan];
    memcpy(newSamples + getSampleSizeFromSamplingDepth(depth) * nbSamples * nbChannels, other.samples[chan],
           getSampleSizeFromSamplingDepth(depth) * other.nbSamples * nbChannels);
    samples[chan] = newSamples;
  };

  mapSamples(_append);

  nbSamples += other.nbSamples;
  return Status::OK();
}

int getChannelIndexFromChannelMap(ChannelMap speaker) {
  switch (speaker) {
    case SPEAKER_FRONT_LEFT:
      return 0;
    case SPEAKER_FRONT_RIGHT:
      return 1;
    case SPEAKER_SIDE_LEFT:
      return 2;
    case SPEAKER_SIDE_RIGHT:
      return 3;
    case SPEAKER_FRONT_CENTER:
      return 4;
    case SPEAKER_BACK_CENTER:
      return 5;
    case SPEAKER_LOW_FREQUENCY:
      return 6;
    case SPEAKER_BACK_LEFT:
      return 7;
    case SPEAKER_BACK_RIGHT:
      return 8;
    case SPEAKER_FRONT_LEFT_OF_CENTER:
      return 9;
    case SPEAKER_FRONT_RIGHT_OF_CENTER:
      return 10;
    case SPEAKER_TOP_CENTER:
      return 11;
    case SPEAKER_TOP_FRONT_LEFT:
      return 12;
    case SPEAKER_TOP_FRONT_CENTER:
      return 13;
    case SPEAKER_TOP_FRONT_RIGHT:
      return 14;
    case SPEAKER_TOP_BACK_LEFT:
      return 15;
    case SPEAKER_TOP_BACK_CENTER:
      return 16;
    case SPEAKER_TOP_BACK_RIGHT:
      return 17;
    // By default we support the ACN-ordering for the ambisonic ie (WYZX)
    case SPEAKER_AMB_W:
      return 18;
    case SPEAKER_AMB_X:
      return 19;
    case SPEAKER_AMB_Y:
      return 20;
    case SPEAKER_AMB_Z:
      return 21;
    case SPEAKER_AMB_V:
      return 22;
    case SPEAKER_AMB_T:
      return 23;
    case SPEAKER_AMB_R:
      return 24;
    case SPEAKER_AMB_S:
      return 25;
    case SPEAKER_AMB_U:
      return 26;
    case SPEAKER_AMB_Q:
      return 27;
    case SPEAKER_AMB_O:
      return 28;
    case SPEAKER_AMB_M:
      return 29;
    case SPEAKER_AMB_K:
      return 30;
    case SPEAKER_AMB_L:
      return 31;
    case SPEAKER_AMB_N:
      return 32;
    case SPEAKER_AMB_P:
      return 33;
    case NO_SPEAKER:
      return 34;
    default:
      return -1;
  }
}

ChannelMap getChannelMapFromChannelIndex(int i) {
  if (getChannelIndexFromChannelMap(SPEAKER_FRONT_LEFT) == i) {
    return SPEAKER_FRONT_LEFT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_FRONT_RIGHT) == i) {
    return SPEAKER_FRONT_RIGHT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_FRONT_CENTER) == i) {
    return SPEAKER_FRONT_CENTER;
  } else if (getChannelIndexFromChannelMap(SPEAKER_LOW_FREQUENCY) == i) {
    return SPEAKER_LOW_FREQUENCY;
  } else if (getChannelIndexFromChannelMap(SPEAKER_BACK_LEFT) == i) {
    return SPEAKER_BACK_LEFT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_BACK_RIGHT) == i) {
    return SPEAKER_BACK_RIGHT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_FRONT_LEFT_OF_CENTER) == i) {
    return SPEAKER_FRONT_LEFT_OF_CENTER;
  } else if (getChannelIndexFromChannelMap(SPEAKER_FRONT_RIGHT_OF_CENTER) == i) {
    return SPEAKER_FRONT_RIGHT_OF_CENTER;
  } else if (getChannelIndexFromChannelMap(SPEAKER_BACK_CENTER) == i) {
    return SPEAKER_BACK_CENTER;
  } else if (getChannelIndexFromChannelMap(SPEAKER_SIDE_LEFT) == i) {
    return SPEAKER_SIDE_LEFT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_SIDE_RIGHT) == i) {
    return SPEAKER_SIDE_RIGHT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_TOP_CENTER) == i) {
    return SPEAKER_TOP_CENTER;
  } else if (getChannelIndexFromChannelMap(SPEAKER_TOP_FRONT_LEFT) == i) {
    return SPEAKER_TOP_FRONT_LEFT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_TOP_FRONT_CENTER) == i) {
    return SPEAKER_TOP_FRONT_CENTER;
  } else if (getChannelIndexFromChannelMap(SPEAKER_TOP_FRONT_RIGHT) == i) {
    return SPEAKER_TOP_FRONT_RIGHT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_TOP_BACK_LEFT) == i) {
    return SPEAKER_TOP_BACK_LEFT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_TOP_BACK_CENTER) == i) {
    return SPEAKER_TOP_BACK_CENTER;
  } else if (getChannelIndexFromChannelMap(SPEAKER_TOP_BACK_RIGHT) == i) {
    return SPEAKER_TOP_BACK_RIGHT;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_W) == i) {
    return SPEAKER_AMB_W;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_X) == i) {
    return SPEAKER_AMB_X;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_Y) == i) {
    return SPEAKER_AMB_Y;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_Z) == i) {
    return SPEAKER_AMB_Z;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_R) == i) {
    return SPEAKER_AMB_R;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_S) == i) {
    return SPEAKER_AMB_S;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_T) == i) {
    return SPEAKER_AMB_T;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_U) == i) {
    return SPEAKER_AMB_U;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_V) == i) {
    return SPEAKER_AMB_V;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_K) == i) {
    return SPEAKER_AMB_K;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_L) == i) {
    return SPEAKER_AMB_L;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_M) == i) {
    return SPEAKER_AMB_M;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_N) == i) {
    return SPEAKER_AMB_N;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_O) == i) {
    return SPEAKER_AMB_O;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_P) == i) {
    return SPEAKER_AMB_P;
  } else if (getChannelIndexFromChannelMap(SPEAKER_AMB_Q) == i) {
    return SPEAKER_AMB_Q;
  } else {
    return NO_SPEAKER;
  }
}

int getIntFromBlockSize(BlockSize bs) {
  switch (bs) {
    case BlockSize::BS_32:
      return 32;
    case BlockSize::BS_64:
      return 64;
    case BlockSize::BS_128:
      return 128;
    case BlockSize::BS_256:
      return 256;
    case BlockSize::BS_512:
      return 512;
    case BlockSize::BS_1024:
      return 1024;
    case BlockSize::BS_2048:
      return 2048;
    case BlockSize::BS_4096:
      return 4096;
    case BlockSize::BS_NONE:
      return 0;
  }
  return 0;
}

double getDblFromBlockSize(BlockSize bs) { return static_cast<double>(getIntFromBlockSize(bs)); }

int getIntFromSamplingRate(SamplingRate samplingRate) {
  switch (samplingRate) {
    case SamplingRate::SR_NONE:
      return 0;
    case SamplingRate::SR_22050:
      return 22050;
    case SamplingRate::SR_32000:
      return 32000;
    case SamplingRate::SR_44100:
      return 44100;
    case SamplingRate::SR_48000:
      return 48000;
    case SamplingRate::SR_88200:
      return 88200;
    case SamplingRate::SR_96000:
      return 96000;
    case SamplingRate::SR_176400:
      return 176400;
    case SamplingRate::SR_192000:
      return 192000;
    default:
      return 0;
  }
}

double getDblFromSamplingRate(SamplingRate samplingRate) {
  return static_cast<double>(getIntFromSamplingRate(samplingRate));
}

BlockSize getBlockSizeFromInt(const int bs) {
  if (getIntFromBlockSize(BlockSize::BS_32) == bs) {
    return BlockSize::BS_32;
  } else if (getIntFromBlockSize(BlockSize::BS_64) == bs) {
    return BlockSize::BS_64;
  } else if (getIntFromBlockSize(BlockSize::BS_128) == bs) {
    return BlockSize::BS_128;
  } else if (getIntFromBlockSize(BlockSize::BS_256) == bs) {
    return BlockSize::BS_256;
  } else if (getIntFromBlockSize(BlockSize::BS_512) == bs) {
    return BlockSize::BS_512;
  } else if (getIntFromBlockSize(BlockSize::BS_1024) == bs) {
    return BlockSize::BS_1024;
  } else if (getIntFromBlockSize(BlockSize::BS_2048) == bs) {
    return BlockSize::BS_2048;
  } else if (getIntFromBlockSize(BlockSize::BS_4096) == bs) {
    return BlockSize::BS_4096;
  } else {
    return BlockSize::BS_NONE;
  }
}

SamplingRate getSamplingRateFromInt(const int samplingRateInt) {
  if (getIntFromSamplingRate(SamplingRate::SR_22050) == samplingRateInt) {
    return SamplingRate::SR_22050;
  } else if (getIntFromSamplingRate(SamplingRate::SR_32000) == samplingRateInt) {
    return SamplingRate::SR_32000;
  } else if (getIntFromSamplingRate(SamplingRate::SR_44100) == samplingRateInt) {
    return SamplingRate::SR_44100;
  } else if (getIntFromSamplingRate(SamplingRate::SR_48000) == samplingRateInt) {
    return SamplingRate::SR_48000;
  } else if (getIntFromSamplingRate(SamplingRate::SR_88200) == samplingRateInt) {
    return SamplingRate::SR_88200;
  } else if (getIntFromSamplingRate(SamplingRate::SR_96000) == samplingRateInt) {
    return SamplingRate::SR_96000;
  } else if (getIntFromSamplingRate(SamplingRate::SR_176400) == samplingRateInt) {
    return SamplingRate::SR_176400;
  } else if (getIntFromSamplingRate(SamplingRate::SR_192000) == samplingRateInt) {
    return SamplingRate::SR_192000;
  } else {
    return SamplingRate::SR_NONE;
  }
}

SamplingFormat getSamplingFormatFromSamplingDepth(SamplingDepth samplingDepth) {
  switch (samplingDepth) {
    case SamplingDepth::UINT8:
    case SamplingDepth::INT16:
    case SamplingDepth::INT24:
    case SamplingDepth::INT32:
    case SamplingDepth::FLT:
    case SamplingDepth::DBL:
      return SamplingFormat::INTERLEAVED;
    case SamplingDepth::UINT8_P:
    case SamplingDepth::INT16_P:
    case SamplingDepth::INT24_P:
    case SamplingDepth::INT32_P:
    case SamplingDepth::FLT_P:
    case SamplingDepth::DBL_P:
      return SamplingFormat::PLANAR;
    default:
      //      Logger::get(Logger::Warning) << "getSamplingFormatFromSamplingDepth: unknown samplingDepth value: " <<
      //      samplingDepth << std::endl;
      return SamplingFormat::FORMAT_UNKNOWN;
  }
}

std::size_t getSampleSizeFromSamplingDepth(SamplingDepth samplingDepth) {
  switch (samplingDepth) {
    case SamplingDepth::UINT8:
    case SamplingDepth::UINT8_P:
      return sizeof(uint8_t);
    case SamplingDepth::INT16:
    case SamplingDepth::INT16_P:
      return sizeof(int16_t);
    case SamplingDepth::INT24:
    case SamplingDepth::INT24_P:
      return 3 * sizeof(uint8_t);
    case SamplingDepth::INT32:
    case SamplingDepth::INT32_P:
      return sizeof(int32_t);
    case SamplingDepth::FLT:
    case SamplingDepth::FLT_P:
      return sizeof(float);
    case SamplingDepth::DBL:
    case SamplingDepth::DBL_P:
      return sizeof(double);
    default:
      //      Logger::get(Logger::Warning) << "getSampleSizeFromSamplingDepth: unknown samplingDepth value: " <<
      //      samplingDepth << std::endl;
      return 0;
  }
}

const char* getStringFromSamplingDepth(SamplingDepth samplingDepth) {
  switch (samplingDepth) {
    case SamplingDepth::SD_NONE:
      return "No sampling depth";
    case SamplingDepth::UINT8:
      return "s8";
    case SamplingDepth::INT16:
      return "s16";
    case SamplingDepth::INT32:
      return "s32";
    case SamplingDepth::FLT:
      return "flt";
    case SamplingDepth::DBL:
      return "dbl";
    case SamplingDepth::UINT8_P:
      return "s8p";
    case SamplingDepth::INT16_P:
      return "s16p";
    case SamplingDepth::INT32_P:
      return "s32p";
    case SamplingDepth::FLT_P:
      return "fltp";
    case SamplingDepth::DBL_P:
      return "dblp";
    default:
      return "";
  }
}

SamplingDepth getSamplingDepthFromString(const char* samplingDepthStr) {
  if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::UINT8), samplingDepthStr) == 0) {
    return SamplingDepth::UINT8;
  } else if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::INT16), samplingDepthStr) == 0) {
    return SamplingDepth::INT16;
  } else if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::INT32), samplingDepthStr) == 0) {
    return SamplingDepth::INT32;
  } else if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::FLT), samplingDepthStr) == 0) {
    return SamplingDepth::FLT;
  } else if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::DBL), samplingDepthStr) == 0) {
    return SamplingDepth::DBL;
  } else if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::UINT8_P), samplingDepthStr) == 0) {
    return SamplingDepth::UINT8_P;
  } else if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::INT16_P), samplingDepthStr) == 0) {
    return SamplingDepth::INT16_P;
  } else if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::INT32_P), samplingDepthStr) == 0) {
    return SamplingDepth::INT32_P;
  } else if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::FLT_P), samplingDepthStr) == 0) {
    return SamplingDepth::FLT_P;
  } else if (std::strcmp(getStringFromSamplingDepth(SamplingDepth::DBL_P), samplingDepthStr) == 0) {
    return SamplingDepth::DBL_P;
  } else {
    return SamplingDepth::SD_NONE;
  }
}

const char* getStringFromChannelType(ChannelMap map) {
  switch (map) {
    case SPEAKER_FRONT_LEFT:
      return "front left";
    case SPEAKER_FRONT_RIGHT:
      return "front right";
    case SPEAKER_FRONT_CENTER:
      return "front center";
    case SPEAKER_LOW_FREQUENCY:
      return "front low frequency";
    case SPEAKER_BACK_LEFT:
      return "back left";
    case SPEAKER_BACK_RIGHT:
      return "back right";
    case SPEAKER_FRONT_LEFT_OF_CENTER:
      return "front left of center";
    case SPEAKER_FRONT_RIGHT_OF_CENTER:
      return "front right of center";
    case SPEAKER_BACK_CENTER:
      return "back center";
    case SPEAKER_SIDE_LEFT:
      return "side left";
    case SPEAKER_SIDE_RIGHT:
      return "side right";
    case SPEAKER_TOP_CENTER:
      return "top center";
    case SPEAKER_TOP_FRONT_LEFT:
      return "top front left";
    case SPEAKER_TOP_FRONT_CENTER:
      return "top front center";
    case SPEAKER_TOP_FRONT_RIGHT:
      return "top front right";
    case SPEAKER_TOP_BACK_LEFT:
      return "top back left";
    case SPEAKER_TOP_BACK_CENTER:
      return "top back center";
    case SPEAKER_TOP_BACK_RIGHT:
      return "top back right";
    case SPEAKER_AMB_W:
      return "amb w";
    case SPEAKER_AMB_X:
      return "amb x";
    case SPEAKER_AMB_Y:
      return "amb y";
    case SPEAKER_AMB_Z:
      return "amb z";
    case SPEAKER_AMB_R:
      return "amb r";
    case SPEAKER_AMB_S:
      return "amb s";
    case SPEAKER_AMB_T:
      return "amb t";
    case SPEAKER_AMB_U:
      return "amb u";
    case SPEAKER_AMB_V:
      return "amb v";
    case SPEAKER_AMB_K:
      return "amb k";
    case SPEAKER_AMB_L:
      return "amb l";
    case SPEAKER_AMB_M:
      return "amb m";
    case SPEAKER_AMB_N:
      return "amb n";
    case SPEAKER_AMB_O:
      return "amb o";
    case SPEAKER_AMB_P:
      return "amb p";
    case SPEAKER_AMB_Q:
      return "amb q";
    case NO_SPEAKER:
      return "no speaker";
  }
  return "no speaker";
}

const char* getStringFromChannelLayout(ChannelLayout channelLayout) {
  switch (channelLayout) {
    case MONO:
      return "mono";
    case STEREO:
      return "stereo";
    case _2POINT1:
      return "2.1";
    case _3DUMMY:
      return "3(dummy)";
    case _2_1:
      return "3.0(back)";
    case SURROUND:
      return "3.0";
    case _3POINT1:
      return "3.1";
    case _4POINT0:
      return "4.0";
    case _4POINT1:
      return "4.1";
    case _2_2:
      return "quad(side)";
    case QUAD:
      return "quad";
    case _5POINT0:
      return "5.0(side)";
    case _5POINT1:
      return "5.1(side)";
    case _5POINT0_BACK:
      return "5.0";
    case _5POINT1_BACK:
      return "5.1";
    case _6POINT0:
      return "6.0";
    case _6POINT0_FRONT:
      return "6.0(front)";
    case HEXAGONAL:
      return "hexagonal";
    case _6POINT1:
      return "6.1";
    case _6POINT1_BACK:
      return "6.1";
    case _6POINT1_FRONT:
      return "6.1(side)";
    case _7POINT0:
      return "7.0";
    case _7POINT0_FRONT:
      return "7.0(front)";
    case _7POINT1:
      return "7.1";
    case _7POINT1_WIDE:
      return "7.1(wide-side)";
    case _7POINT1_WIDE_BACK:
      return "7.1(wide)";
    case OCTAGONAL:
      return "octagonal";
    case _8DUMMY:
      return "8(dummy)";
    case AMBISONICS_WXY:
      return "amb_wxy";
    case AMBISONICS_WXYZ:
      return "amb_wxyz";
    case AMBISONICS_2ND:
      return "amb_2nd";
    case AMBISONICS_3RD:
      return "amb_3rd";
    case UNKNOWN:
      return "unknown";
  }
  return "unknown";
}

int getNbChannelsFromChannelLayout(ChannelLayout channelLayout) {
  switch (channelLayout) {
    case MONO:
      return 1;
    case STEREO:
      return 2;
    case _3DUMMY:
    case _2POINT1:
    case _2_1:
    case SURROUND:
    case AMBISONICS_WXY:
      return 3;
    case _3POINT1:
    case _4POINT0:
    case _2_2:
    case QUAD:
    case AMBISONICS_WXYZ:
      return 4;
    case _4POINT1:
    case _5POINT0:
    case _5POINT0_BACK:
      return 5;
    case _5POINT1:
    case _5POINT1_BACK:
    case _6POINT0:
    case _6POINT0_FRONT:
    case HEXAGONAL:
      return 6;
    case _6POINT1:
    case _6POINT1_BACK:
    case _6POINT1_FRONT:
    case _7POINT0:
    case _7POINT0_FRONT:
      return 7;
    case _7POINT1:
    case _7POINT1_WIDE:
    case _7POINT1_WIDE_BACK:
    case OCTAGONAL:
    case _8DUMMY:
      return 8;
    case AMBISONICS_2ND:
      return 9;
    case AMBISONICS_3RD:
      return 16;
    case UNKNOWN:
      return 0;
  }
  return 0;
}

ChannelLayout getAChannelLayoutFromNbChannels(size_t nbChannels) {
  if (nbChannels == 1) {
    return MONO;
  } else if (nbChannels == 2) {
    return STEREO;
  } else if (nbChannels == 3) {
    return _3DUMMY;
  } else if (nbChannels == 4) {
    return _2_2;
  } else if (nbChannels == 5) {
    return _5POINT0;
  } else if (nbChannels == 6) {
    return _6POINT0;
  } else if (nbChannels == 7) {
    return _6POINT1;
  } else if (nbChannels == 8) {
    return _8DUMMY;
  } else {
    return UNKNOWN;
  }
}

ChannelLayout getChannelLayoutFromString(const char* channelLayout) {
  if (std::strcmp(getStringFromChannelLayout(MONO), channelLayout) == 0) {
    return MONO;
  } else if (std::strcmp(getStringFromChannelLayout(STEREO), channelLayout) == 0) {
    return STEREO;
  } else if (std::strcmp(getStringFromChannelLayout(_2POINT1), channelLayout) == 0) {
    return _2POINT1;
  } else if (std::strcmp(getStringFromChannelLayout(_2_1), channelLayout) == 0) {
    return _2_1;
  } else if (std::strcmp(getStringFromChannelLayout(SURROUND), channelLayout) == 0) {
    return SURROUND;
  } else if (std::strcmp(getStringFromChannelLayout(_3POINT1), channelLayout) == 0) {
    return _3POINT1;
  } else if (std::strcmp(getStringFromChannelLayout(_4POINT0), channelLayout) == 0) {
    return _4POINT0;
  } else if (std::strcmp(getStringFromChannelLayout(_4POINT1), channelLayout) == 0) {
    return _4POINT1;
  } else if (std::strcmp(getStringFromChannelLayout(_2_2), channelLayout) == 0) {
    return _2_2;
  } else if (std::strcmp(getStringFromChannelLayout(QUAD), channelLayout) == 0) {
    return QUAD;
  } else if (std::strcmp(getStringFromChannelLayout(_5POINT0), channelLayout) == 0) {
    return _5POINT0;
  } else if (std::strcmp(getStringFromChannelLayout(_5POINT1), channelLayout) == 0) {
    return _5POINT1;
  } else if (std::strcmp(getStringFromChannelLayout(_5POINT0_BACK), channelLayout) == 0) {
    return _5POINT0_BACK;
  } else if (std::strcmp(getStringFromChannelLayout(_5POINT1_BACK), channelLayout) == 0) {
    return _5POINT1_BACK;
  } else if (std::strcmp(getStringFromChannelLayout(_6POINT0), channelLayout) == 0) {
    return _6POINT0;
  } else if (std::strcmp(getStringFromChannelLayout(_6POINT0_FRONT), channelLayout) == 0) {
    return _6POINT0_FRONT;
  } else if (std::strcmp(getStringFromChannelLayout(HEXAGONAL), channelLayout) == 0) {
    return HEXAGONAL;
  } else if (std::strcmp(getStringFromChannelLayout(_6POINT1), channelLayout) == 0) {
    return _6POINT1;
  } else if (std::strcmp(getStringFromChannelLayout(_6POINT1_BACK), channelLayout) == 0) {
    return _6POINT1_BACK;
  } else if (std::strcmp(getStringFromChannelLayout(_6POINT1_FRONT), channelLayout) == 0) {
    return _6POINT1_FRONT;
  } else if (std::strcmp(getStringFromChannelLayout(_7POINT0), channelLayout) == 0) {
    return _7POINT0;
  } else if (std::strcmp(getStringFromChannelLayout(_7POINT0_FRONT), channelLayout) == 0) {
    return _7POINT0_FRONT;
  } else if (std::strcmp(getStringFromChannelLayout(_7POINT1), channelLayout) == 0) {
    return _7POINT1;
  } else if (std::strcmp(getStringFromChannelLayout(_7POINT1_WIDE), channelLayout) == 0) {
    return _7POINT1_WIDE;
  } else if (std::strcmp(getStringFromChannelLayout(_7POINT1_WIDE_BACK), channelLayout) == 0) {
    return _7POINT1_WIDE_BACK;
  } else if (std::strcmp(getStringFromChannelLayout(OCTAGONAL), channelLayout) == 0) {
    return OCTAGONAL;
  } else if (std::strcmp(getStringFromChannelLayout(AMBISONICS_WXY), channelLayout) == 0) {
    return AMBISONICS_WXY;
  } else if (std::strcmp(getStringFromChannelLayout(AMBISONICS_WXYZ), channelLayout) == 0) {
    return AMBISONICS_WXYZ;
  } else if (std::strcmp(getStringFromChannelLayout(AMBISONICS_2ND), channelLayout) == 0) {
    return AMBISONICS_2ND;
  } else if (std::strcmp(getStringFromChannelLayout(AMBISONICS_3RD), channelLayout) == 0) {
    return AMBISONICS_3RD;
  }
  return UNKNOWN;
}

void convertSamplesToMonoDouble(const Audio::Samples& samples, Audio::AudioTrack& snd, const int nChannels,
                                const Audio::SamplingDepth sampleDepth) {
  switch (sampleDepth) {
    case Audio::SamplingDepth::UINT8_P: {
      uint8_t* s = samples.getSamples()[0];
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        if (*s > 0) {
          snd.push_back(((double)*s++ - 128.0) / 127.0);
        } else {
          snd.push_back(((double)*s++ - 128.0) / 128.0);
        }
      }
      break;
    }
    case Audio::SamplingDepth::INT16_P: {
      int16_t* s = reinterpret_cast<int16_t*>(samples.getSamples()[0]);
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        if (*s > 0) {
          snd.push_back(((double)*s++) / 32767.0);
        } else {
          snd.push_back(((double)*s++) / 32768.0);
        }
      }
      break;
    }
    case Audio::SamplingDepth::INT24_P: {
      uint8_t* s = samples.getSamples()[0];
      uint32_t sh;
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        sh = 0;
        memcpy(&sh, s, 3);
        int32_t shi = static_cast<int32_t>(sh << 8);
        if (shi > 0) {
          snd.push_back(((double)shi) / 2147483392.0);
        } else {
          snd.push_back(((double)shi) / 2147483648.0);
        }
        s += 3;
      }
      break;
    }
    case Audio::SamplingDepth::INT32_P: {
      int32_t* s = reinterpret_cast<int32_t*>(samples.getSamples()[0]);
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        if (*s > 0) {
          snd.push_back(((double)*s++) / 2147483647.0);
        } else {
          snd.push_back(((double)*s++) / 2147483648.0);
        }
      }
      break;
    }
    case Audio::SamplingDepth::FLT_P: {
      float* s = reinterpret_cast<float*>(samples.getSamples()[0]);
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        snd.push_back((double)*s++);
      }
      break;
    }
    case Audio::SamplingDepth::DBL_P: {
      double* s = reinterpret_cast<double*>(samples.getSamples()[0]);
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        snd.push_back(*s++);
      }
      break;
    }
    case Audio::SamplingDepth::UINT8: {
      uint8_t* s = samples.getSamples()[0];
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        if (*s > 128) {
          snd.push_back(((double)*s - 128.0) / 127.0);
        } else {
          snd.push_back(((double)*s - 128.0) / 128.0);
        }
        s += nChannels;
      }
      break;
    }
    case Audio::SamplingDepth::INT16: {
      int16_t* s = reinterpret_cast<int16_t*>(samples.getSamples()[0]);
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        if (*s > 0) {
          snd.push_back(((double)*s) / 32767.0);
        } else {
          snd.push_back(((double)*s) / 32768.0);
        }
        s += nChannels;
      }
      break;
    }
    case Audio::SamplingDepth::INT24: {
      uint8_t* s = samples.getSamples()[0];
      uint32_t sh;
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        sh = 0;
        memcpy(&sh, s, 3);
        int32_t shi = static_cast<int32_t>(sh << 8);
        if (shi > 0) {
          snd.push_back(((double)shi) / 2147483392.0);
        } else {
          snd.push_back(((double)shi) / 2147483648.0);
        }
        s += nChannels * 3;
      }
      break;
    }
    case Audio::SamplingDepth::INT32: {
      int32_t* s = reinterpret_cast<int32_t*>(samples.getSamples()[0]);
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        if (*s > 0) {
          snd.push_back(((double)*s) / 2147483647.0);
        } else {
          snd.push_back(((double)*s) / 2147483648.0);
        }
        s += nChannels;
      }
      break;
    }
    case Audio::SamplingDepth::FLT: {
      float* s = reinterpret_cast<float*>(samples.getSamples()[0]);
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        snd.push_back((double)*s);
        s += nChannels;
      }
      break;
    }
    case Audio::SamplingDepth::DBL: {
      double* s = reinterpret_cast<double*>(samples.getSamples()[0]);
      for (size_t i = 0; i < samples.getNbOfSamples(); i++) {
        snd.push_back(*s);
        s += nChannels;
      }
      break;
    }
    default:
      break;
  }
}

template <typename Functor>
void Samples::mapSamples(const Functor& execFunctor) {
  switch (getSamplingFormatFromSamplingDepth(depth)) {
    case SamplingFormat::INTERLEAVED: {
      execFunctor(0, getNbChannelsFromChannelLayout(layout));
      break;
    }

    case SamplingFormat::PLANAR: {
      int64_t mask = 0x1;
      for (int i = 0; i < MAX_AUDIO_CHANNELS; ++i) {
        if (layout & mask) execFunctor(i, 1);
        mask <<= 1;
      }
      break;
    }

    default:
      assert(false && "Sampling format is unknown");
  }
}

}  // namespace Audio
}  // namespace VideoStitch
