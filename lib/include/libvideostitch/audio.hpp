// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "status.hpp"

#include <array>
#include <vector>

namespace VideoStitch {
namespace Ptv {
class Value;
}

namespace Audio {

/** Internal audio sample representation */
typedef double audioSample_t;

/**
 * Channel mapping mask.
 */
enum ChannelMap : int64_t {
  SPEAKER_FRONT_LEFT = 0x1,
  SPEAKER_FRONT_RIGHT = 0x2,
  SPEAKER_SIDE_LEFT = 0x4,
  SPEAKER_SIDE_RIGHT = 0x8,
  SPEAKER_FRONT_CENTER = 0x10,
  SPEAKER_BACK_CENTER = 0x20,
  SPEAKER_LOW_FREQUENCY = 0x40,
  SPEAKER_BACK_LEFT = 0x80,
  SPEAKER_BACK_RIGHT = 0x100,
  SPEAKER_FRONT_LEFT_OF_CENTER = 0x200,
  SPEAKER_FRONT_RIGHT_OF_CENTER = 0x400,
  SPEAKER_TOP_CENTER = 0x800,
  SPEAKER_TOP_FRONT_LEFT = 0x1000,
  SPEAKER_TOP_FRONT_CENTER = 0x2000,
  SPEAKER_TOP_FRONT_RIGHT = 0x4000,
  SPEAKER_TOP_BACK_LEFT = 0x8000,
  SPEAKER_TOP_BACK_CENTER = 0x10000,
  SPEAKER_TOP_BACK_RIGHT = 0x20000,
  SPEAKER_AMB_W = 0x40000,
  SPEAKER_AMB_X = 0x80000,
  SPEAKER_AMB_Y = 0x100000,
  SPEAKER_AMB_Z = 0x200000,
  SPEAKER_AMB_R = 0x400000,
  SPEAKER_AMB_S = 0x800000,
  SPEAKER_AMB_T = 0x1000000,
  SPEAKER_AMB_U = 0x2000000,
  SPEAKER_AMB_V = 0x4000000,
  SPEAKER_AMB_K = 0x8000000,
  SPEAKER_AMB_L = 0x10000000,
  SPEAKER_AMB_M = 0x20000000,
  SPEAKER_AMB_N = 0x40000000,
  SPEAKER_AMB_O = 0x80000000,
  SPEAKER_AMB_P = 0x100000000,
  SPEAKER_AMB_Q = 0x200000000,
  NO_SPEAKER = 0x400000000
};

/**
 * Sampling rate.
 * @brief Defines the sampling rate of the audio signal
 */
enum class SamplingRate : int {
  SR_NONE = 0,
  SR_22050 = 1,
  SR_32000 = 2,
  SR_44100 = 3,
  SR_48000 = 4,
  SR_88200 = 5,
  SR_96000 = 6,
  SR_176400 = 7,
  SR_192000 = 8
};

/**
 * Sampling depth.
 * @brief Defines the sampling depth of the audio signal
 */
enum class SamplingDepth {
  SD_NONE,
  // interleaved
  UINT8,
  INT16,
  INT24,
  INT32,
  FLT,
  DBL,
  // planar
  UINT8_P,
  INT16_P,
  INT24_P,
  INT32_P,
  FLT_P,
  DBL_P
};

#define MAX_AUDIO_CHANNELS 35

/**
 * @brief Defines if the audio signal is planar or interleaved
 */
enum SamplingFormat { FORMAT_UNKNOWN, INTERLEAVED, PLANAR };

/**
 * Channel layout.
 * @brief Defines the layout of the audio signal
 */
enum ChannelLayout : int64_t {
  UNKNOWN,
  MONO = SPEAKER_FRONT_LEFT,
  STEREO = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT,
  _3DUMMY = STEREO | SPEAKER_SIDE_LEFT,
  _2POINT1 = STEREO | SPEAKER_LOW_FREQUENCY,
  _2_1 = STEREO | SPEAKER_BACK_CENTER,
  SURROUND = STEREO | SPEAKER_FRONT_CENTER,
  _3POINT1 = SURROUND | SPEAKER_LOW_FREQUENCY,
  _4POINT0 = SURROUND | SPEAKER_BACK_CENTER,
  _4POINT1 = _4POINT0 | SPEAKER_LOW_FREQUENCY,
  _2_2 = STEREO | SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT,
  QUAD = STEREO | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT,
  _5POINT0 = SURROUND | SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT,
  _5POINT1 = _5POINT0 | SPEAKER_LOW_FREQUENCY,
  _5POINT0_BACK = SURROUND | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT,
  _5POINT1_BACK = _5POINT0_BACK | SPEAKER_LOW_FREQUENCY,
  _6POINT0 = _5POINT0 | SPEAKER_BACK_CENTER,
  _6POINT0_FRONT = _2_2 | SPEAKER_FRONT_LEFT_OF_CENTER | SPEAKER_FRONT_RIGHT_OF_CENTER,
  HEXAGONAL = _5POINT0_BACK | SPEAKER_BACK_CENTER,
  _6POINT1 = _5POINT1 | SPEAKER_BACK_CENTER,
  _6POINT1_BACK = _5POINT1_BACK | SPEAKER_BACK_CENTER,
  _6POINT1_FRONT = _6POINT0_FRONT | SPEAKER_LOW_FREQUENCY,
  _7POINT0 = _5POINT0 | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT,
  _7POINT0_FRONT = _5POINT0 | SPEAKER_FRONT_LEFT_OF_CENTER | SPEAKER_FRONT_RIGHT_OF_CENTER,
  _8DUMMY = _6POINT1 | SPEAKER_BACK_LEFT,
  _7POINT1 = _5POINT1 | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT,
  _7POINT1_WIDE = _5POINT1 | SPEAKER_FRONT_LEFT_OF_CENTER | SPEAKER_FRONT_RIGHT_OF_CENTER,
  _7POINT1_WIDE_BACK = _5POINT1_BACK | SPEAKER_FRONT_LEFT_OF_CENTER | SPEAKER_FRONT_RIGHT_OF_CENTER,
  OCTAGONAL = _5POINT0 | SPEAKER_BACK_LEFT | SPEAKER_BACK_CENTER | SPEAKER_BACK_RIGHT,
  AMBISONICS_WXY = SPEAKER_AMB_W | SPEAKER_AMB_X | SPEAKER_AMB_Y,
  AMBISONICS_WXYZ = AMBISONICS_WXY | SPEAKER_AMB_Z,
  AMBISONICS_2ND = AMBISONICS_WXYZ | SPEAKER_AMB_R | SPEAKER_AMB_S | SPEAKER_AMB_T | SPEAKER_AMB_U | SPEAKER_AMB_V,
  AMBISONICS_3RD = AMBISONICS_2ND | SPEAKER_AMB_K | SPEAKER_AMB_L | SPEAKER_AMB_M | SPEAKER_AMB_N | SPEAKER_AMB_O |
                   SPEAKER_AMB_P | SPEAKER_AMB_Q
};

/**
 * Block size.
 * @brief Defines the block size in samples for audio processors.
 */
enum class BlockSize { BS_NONE, BS_32, BS_64, BS_128, BS_256, BS_512, BS_1024, BS_2048, BS_4096 };

static inline double getDefaultSamplingRate() { return 44100.0; }
static inline int getDefaultBlockSize() { return 512; }
static inline std::string getAudioGeneratorId() { return "audiogen"; }

/**
 * @brief Determines if the  audio signal is planar or interleaved
 */
VS_EXPORT SamplingFormat getSamplingFormatFromSamplingDepth(SamplingDepth samplingDepth);

/**
 * @brief Return the size (bytes) of one sample
 */
VS_EXPORT std::size_t getSampleSizeFromSamplingDepth(SamplingDepth samplingDepth);

/**
 * @brief Return an index corresponding to the channel map.
 */
VS_EXPORT int getChannelIndexFromChannelMap(ChannelMap speaker);

/**
 * @brief Return a channel map from an integer value
 */
VS_EXPORT ChannelMap getChannelMapFromChannelIndex(int i);

/**
 * @brief Return an int representation of a block size.
 * This can be used when serializing a block size enum.
 */
VS_EXPORT int getIntFromBlockSize(BlockSize bs);

/**
 * @brief Return a double representation of a block size.
 * This can be used when serializing a block size enum.
 */
VS_EXPORT double getDblFromBlockSize(BlockSize bs);

/**
 * @brief Return an int representation of a sampling rate.
 * This can be used when serializing a sampling rate enum.
 */
VS_EXPORT int getIntFromSamplingRate(SamplingRate samplingRate);

/**
 * @brief Return a double representation of a sampling rate.
 * This can be used when serializing a sampling rate enum.
 */
VS_EXPORT double getDblFromSamplingRate(SamplingRate samplingRate);

/**
 * @brief Return a sampling rate from an integer value
 * This can be used when parsing a sampling rate enum.
 */
VS_EXPORT SamplingRate getSamplingRateFromInt(const int samplingRateInt);

/**
 * @brief Return a block size from an integer value
 * This can be used when parsing a block size enum.
 */
VS_EXPORT BlockSize getBlockSizeFromInt(const int bs);

/**
 * @brief Return a string representation of a sampling depth.
 * This can be used when serializing a sampling depth enum.
 */
VS_EXPORT const char* getStringFromSamplingDepth(SamplingDepth samplingDepth);

/**
 * @brief Return a sampling rate from a string.
 * This can be used when parsing a sampling rate enum.
 */
VS_EXPORT SamplingDepth getSamplingDepthFromString(const char* samplingDepthStr);

/**
 * @brief Return a string representation of a channel map.
 * This can be used when serializing a channel map.
 */
VS_EXPORT const char* getStringFromChannelType(ChannelMap map);

/**
 * @brief Return a string representation of a sampling depth.
 * This can be used when serializing a sampling depth enum.
 */
VS_EXPORT const char* getStringFromChannelLayout(ChannelLayout channelLayout);

/**
 * @brief Return the number of channels used by this layout.
 */
VS_EXPORT int getNbChannelsFromChannelLayout(ChannelLayout channelLayout);

/**
 * @brief Return a channel layout with this number of channels.
 */
VS_EXPORT ChannelLayout getAChannelLayoutFromNbChannels(size_t nbChannels);

/**
 * @brief Return a sampling rate from a string.
 * This can be used when parsing a sampling rate enum.
 */
VS_EXPORT ChannelLayout getChannelLayoutFromString(const char* channelLayout);

/**
 * Where the audio samples channel planes will be written.
 * Up to 18 channels passthru is supported.
 * Audio channel samples are either interleaved into a sample
 * frame or planar, either contiguous or padded (see Spec).
 * The memory of each plane is not allocated.
 * The ownership is transfered to the stitcher.
 */
class VS_EXPORT Samples {
 public:
  typedef std::array<uint8_t*, MAX_AUDIO_CHANNELS> data_buffer_t;

  typedef void (*deleter)(data_buffer_t&);

  /**
   * @brief Default constructor. Generate an invalid buffer.
   */
  Samples();

  /**
   * @brief Construct from raw data.
   * @param r Sampling rate
   * @param d Sample depth
   * @param l Channels layout
   * @param timestamp First sample timestamp in microseconds
   * @param data Raw data. Ownership is transferred to newly created Samples object.
   * @param nbSamples Number of samples in the buffer
   * @param delete_ Optional deleter functor
   */
  Samples(const SamplingRate r, const SamplingDepth d, const ChannelLayout l, mtime_t timestamp, data_buffer_t& data,
          size_t nbSamples, deleter delete_);

  /**
   * @brief Construct from raw data.
   * @param r Sampling rate
   * @param d Sample depth
   * @param l Channels layout
   * @param timestamp First sample timestamp in microseconds
   * @param data Raw data. Ownership is transferred to newly created Samples object.
   * @param nbSamples Number of samples in the buffer
   */
  Samples(const SamplingRate r, const SamplingDepth d, const ChannelLayout l, mtime_t timestamp, data_buffer_t& data,
          size_t nbSamples);

  /**
   * @brief Construct from raw data.
   * @param r Sampling rate
   * @param d Sample depth
   * @param l Channels layout
   * @param timestamp First sample timestamp in microseconds
   * @param data Raw data. Ownership is transferred to newly created Samples object.
   * @param nbSamples Number of samples in the buffer
   * @param delete_ Optional deleter functor
   * TODO deprecate
   */
  Samples(const SamplingRate r, const SamplingDepth d, const ChannelLayout l, mtime_t timestamp, uint8_t** data,
          size_t nbSamples, deleter delete_);
  /**
   * @brief Construct from raw data.
   * @param r Sampling rate
   * @param d Sample depth
   * @param l Channels layout
   * @param timestamp First sample timestamp in microseconds
   * @param data Raw data. Ownership is transferred to newly created Samples object.
   * @param nbSamples Number of samples in the buffer
   * TODO deprecate
   */
  Samples(const SamplingRate r, const SamplingDepth d, const ChannelLayout l, mtime_t timestamp, uint8_t** data,
          size_t nbSamples);
  ~Samples();

  /**
   * @brief Copy constructor with move semantics
   * @param s The object to copy
   */
  Samples(Samples&& s);

  /**
   * @brief Copying audio samples is undefined
   */
  Samples(const Samples&) = delete;

  /**
   * @brief Copying audio samples is undefined
   */
  Samples& operator=(const Samples&) = delete;

  /**
   * @brief Assign operator with move semantics
   * @param s The object to assign
   */
  Samples& operator=(Samples&& s) ;

  /**
   * @brief Assign operator with move semantics
   * @param s The object to assign
   */
  Samples clone() const;

  /**
   * @brief Get the number of audio samples
   * @return The audio samples
   */
  size_t getNbOfSamples() const { return nbSamples; }

  /**
   * @brief Get the audio samples (const)
   * @return s The constant reference to the audio samples
   */
  const data_buffer_t& getSamples() const { return samples; }

  /**
   * @brief Get the samples timestamp.
   * @return The samples timestamp : in 1 / fps unit (like the video's frameId)
   */
  mtime_t getTimestamp() const { return timestamp; }

  /**
   * @brief Set the samples timestamp.
   * @param t The samples timestamp : in 1 / fps unit (like the video's frameId)
   */
  void setTimestamp(mtime_t t) { timestamp = t; }

  /**
   * @brief Drop the @param nb first samples of the buffer
   */
  Status drop(size_t nb);

  /**
   * @brief Append the @param other samples at the end of the buffer
   */
  Status append(const Audio::Samples& other);

  /**
   * @brief Get the sampling rate.
   * @return The sampling rate.
   */
  SamplingRate getSamplingRate() const { return rate; }

  /**
   * @brief Get the sampling depth.
   * @return The sampling depth.
   */
  SamplingDepth getSamplingDepth() const { return depth; }

  /**
   * @brief Get the channels layout.
   * @return The channels layout.
   */
  ChannelLayout getChannelLayout() const { return layout; }

  /**
   * @brief Set the samples deleter.
   * @param del The functor
   */
  void setDeleter(deleter del) { delete_ = del; }

 private:
  // TODO use data_buffer_t&
  void alloc(uint8_t** data);
  void clear();

  template <typename Functor>
  void mapSamples(const Functor& execFunctor);

  data_buffer_t samples;
  size_t nbSamples;
  mtime_t timestamp;
  deleter delete_;

  SamplingDepth depth;
  SamplingRate rate;
  ChannelLayout layout;
};

}  // namespace Audio
}  // namespace VideoStitch
