// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// An object to hold and pass around audio samples internally.

#include "libvideostitch/audioBlock.hpp"

#include <cassert>

namespace VideoStitch {
namespace Audio {

AudioTrack::~AudioTrack() {}

AudioTrack::AudioTrack(AudioTrack&& o) : std::vector<audioSample_t>(std::move(o)), channel_(o.channel_) {}

AudioTrack& AudioTrack::operator=(AudioTrack&& o) {
  std::vector<audioSample_t>::operator=(std::move(o));
  channel_ = o.channel_;
  return *this;
}

AudioTrack::AudioTrack(ChannelMap c) : channel_(c) {}

ChannelMap AudioTrack::channel() const { return channel_; }

void AudioTrack::setChannel(ChannelMap c) { channel_ = c; }

/// -----------
#define INIT_DATA_BUFFER                                                                                               \
  {                                                                                                                    \
    {                                                                                                                  \
      AudioTrack(NO_SPEAKER), AudioTrack(SPEAKER_FRONT_LEFT), AudioTrack(SPEAKER_FRONT_RIGHT),                         \
          AudioTrack(SPEAKER_SIDE_LEFT), AudioTrack(SPEAKER_SIDE_RIGHT), AudioTrack(SPEAKER_FRONT_CENTER),             \
          AudioTrack(SPEAKER_BACK_CENTER), AudioTrack(SPEAKER_LOW_FREQUENCY), AudioTrack(SPEAKER_BACK_LEFT),           \
          AudioTrack(SPEAKER_BACK_RIGHT), AudioTrack(SPEAKER_FRONT_LEFT_OF_CENTER),                                    \
          AudioTrack(SPEAKER_FRONT_RIGHT_OF_CENTER), AudioTrack(SPEAKER_TOP_CENTER),                                   \
          AudioTrack(SPEAKER_TOP_FRONT_LEFT), AudioTrack(SPEAKER_TOP_FRONT_CENTER),                                    \
          AudioTrack(SPEAKER_TOP_FRONT_RIGHT), AudioTrack(SPEAKER_TOP_BACK_LEFT), AudioTrack(SPEAKER_TOP_BACK_CENTER), \
          AudioTrack(SPEAKER_TOP_BACK_RIGHT), AudioTrack(SPEAKER_AMB_W), AudioTrack(SPEAKER_AMB_X),                    \
          AudioTrack(SPEAKER_AMB_Y), AudioTrack(SPEAKER_AMB_Z), AudioTrack(SPEAKER_AMB_R), AudioTrack(SPEAKER_AMB_S),  \
          AudioTrack(SPEAKER_AMB_T), AudioTrack(SPEAKER_AMB_U), AudioTrack(SPEAKER_AMB_V), AudioTrack(SPEAKER_AMB_K),  \
          AudioTrack(SPEAKER_AMB_L), AudioTrack(SPEAKER_AMB_M), AudioTrack(SPEAKER_AMB_N), AudioTrack(SPEAKER_AMB_O),  \
          AudioTrack(SPEAKER_AMB_P), AudioTrack(SPEAKER_AMB_Q),                                                        \
    }                                                                                                                  \
  }

namespace {
AudioBlock::data_buffer_t initDataBuffer = INIT_DATA_BUFFER;
}

AudioBlock::AudioBlock(ChannelLayout layout, mtime_t timestamp) : layout_(layout), timestamp_(timestamp) {
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    data_[i] = initDataBuffer[i];
  }
}

AudioBlock::AudioBlock(unsigned char nbTracks, mtime_t timestamp) : timestamp_(timestamp) {
  int layout = 0;
  for (unsigned char i = 0; i < nbTracks; ++i) {
    layout <<= layout;
    layout &= 1;
  }
  layout_ = static_cast<ChannelLayout>(layout);
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    data_[i] = initDataBuffer[i];
  }
}

AudioBlock::AudioBlock(const AudioBlock& o) : layout_(o.layout_), timestamp_(o.timestamp_) {
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    data_[i] = o.data_[i];
  }
}

AudioBlock::AudioBlock(AudioBlock&& o) : layout_(o.layout_), timestamp_(o.timestamp_) {
  // for some reason, MVSC tries to generate a copy ctor
  // for the std::array when invoking its move ctor ¯\_(ツ)_/¯
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    data_[i] = std::move(o.data_[i]);
  }
}

AudioBlock& AudioBlock::operator=(AudioBlock&& o) {
  layout_ = o.layout_;
  timestamp_ = o.timestamp_;
  // for some reason, MVSC tries to generate a copy ctor
  // for the std::array when invoking its move ctor ¯\_(ツ)_/¯
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    data_[i] = std::move(o.data_[i]);
  }
  return *this;
}

AudioBlock::~AudioBlock() {}

void AudioBlock::setChannelLayout(const ChannelLayout layout) { layout_ = layout; }

void AudioBlock::setTimestamp(mtime_t time) { timestamp_ = time; }

mtime_t AudioBlock::getTimestamp() const { return timestamp_; }

ChannelLayout AudioBlock::getLayout() const { return layout_; }

void AudioBlock::clear() {
  for (auto& track : *this) {
    track.resize(0);
  }
}

AudioBlock::iterator AudioBlock::begin() { return (iterator(layout_, data_.begin() + 1)); }
AudioBlock::const_iterator AudioBlock::begin() const {
  return (const_iterator(layout_, const_cast<AudioBlock*>(this)->data_.begin() + 1));
}
AudioBlock::const_iterator AudioBlock::cbegin() const { return this->begin(); }
AudioBlock::iterator AudioBlock::end() { return iterator(); }
AudioBlock::const_iterator AudioBlock::end() const { return const_iterator(); }
AudioBlock::const_iterator AudioBlock::cend() const { return this->end(); }

#define GETAUDIOTRACK(i)                  \
  {                                       \
    switch (i) {                          \
      case SPEAKER_FRONT_LEFT:            \
        return data_[1];                  \
      case SPEAKER_FRONT_RIGHT:           \
        return data_[2];                  \
      case SPEAKER_SIDE_LEFT:             \
        return data_[3];                  \
      case SPEAKER_SIDE_RIGHT:            \
        return data_[4];                  \
      case SPEAKER_FRONT_CENTER:          \
        return data_[5];                  \
      case SPEAKER_BACK_CENTER:           \
        return data_[6];                  \
      case SPEAKER_LOW_FREQUENCY:         \
        return data_[7];                  \
      case SPEAKER_BACK_LEFT:             \
        return data_[8];                  \
      case SPEAKER_BACK_RIGHT:            \
        return data_[9];                  \
      case SPEAKER_FRONT_LEFT_OF_CENTER:  \
        return data_[10];                 \
      case SPEAKER_FRONT_RIGHT_OF_CENTER: \
        return data_[11];                 \
      case SPEAKER_TOP_CENTER:            \
        return data_[12];                 \
      case SPEAKER_TOP_FRONT_LEFT:        \
        return data_[13];                 \
      case SPEAKER_TOP_FRONT_CENTER:      \
        return data_[14];                 \
      case SPEAKER_TOP_FRONT_RIGHT:       \
        return data_[15];                 \
      case SPEAKER_TOP_BACK_LEFT:         \
        return data_[16];                 \
      case SPEAKER_TOP_BACK_CENTER:       \
        return data_[17];                 \
      case SPEAKER_TOP_BACK_RIGHT:        \
        return data_[18];                 \
      case SPEAKER_AMB_W:                 \
        return data_[19];                 \
      case SPEAKER_AMB_X:                 \
        return data_[20];                 \
      case SPEAKER_AMB_Y:                 \
        return data_[21];                 \
      case SPEAKER_AMB_Z:                 \
        return data_[22];                 \
      case SPEAKER_AMB_R:                 \
        return data_[23];                 \
      case SPEAKER_AMB_S:                 \
        return data_[24];                 \
      case SPEAKER_AMB_T:                 \
        return data_[25];                 \
      case SPEAKER_AMB_U:                 \
        return data_[26];                 \
      case SPEAKER_AMB_V:                 \
        return data_[27];                 \
      case SPEAKER_AMB_K:                 \
        return data_[28];                 \
      case SPEAKER_AMB_L:                 \
        return data_[29];                 \
      case SPEAKER_AMB_M:                 \
        return data_[30];                 \
      case SPEAKER_AMB_N:                 \
        return data_[31];                 \
      case SPEAKER_AMB_O:                 \
        return data_[32];                 \
      case SPEAKER_AMB_P:                 \
        return data_[33];                 \
      case SPEAKER_AMB_Q:                 \
        return data_[34];                 \
    }                                     \
    return data_[0];                      \
  }

AudioBlock::reference AudioBlock::operator[](size_type i) { GETAUDIOTRACK(i); }
AudioBlock::const_reference AudioBlock::operator[](size_type i) const { GETAUDIOTRACK(i); }
AudioBlock::reference AudioBlock::at(size_type i) { return (*this)[i]; }
AudioBlock::const_reference AudioBlock::at(size_type i) const { return (*this)[i]; }

AudioBlock& AudioBlock::operator+=(const AudioBlock& rhs) {
  assert(rhs.layout_ == this->layout_);
  assert(rhs.numSamples() == this->numSamples());
  size_t nbSamples = this->numSamples();
  for (AudioTrack& track : *this) {
    for (size_t i = 0; i < nbSamples; ++i) {
      track[i] += rhs[track.channel()][i];
    }
  }
  return *this;
}

void AudioBlock::swap(AudioBlock& o) {
  std::swap(timestamp_, o.timestamp_);
  std::swap(layout_, o.layout_);
  std::swap(data_, o.data_);
}

AudioBlock::size_type AudioBlock::size() {
  // SWAR algorithm
  uint32_t i = (uint32_t)layout_;
  i = i - ((i >> 1) & 0x55555555);
  i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
  return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

AudioBlock::size_type AudioBlock::max_size() { return MAX_AUDIO_CHANNELS; }

bool AudioBlock::empty() const {
  for (const auto& track : data_) {
    if (!track.empty()) {
      return false;
    }
  }
  return true;
}

void AudioBlock::resize(size_t nSamples) {
  if (layout_ == UNKNOWN) {
    return;
  }
  for (auto& track : *this) {
    track.resize(nSamples);
  }
}

void AudioBlock::assign(size_t nSamples, audioSample_t val) {
  if (layout_ == UNKNOWN) {
    return;
  }
  for (auto& track : *this) {
    track.assign(nSamples, val);
  }
}

size_t AudioBlock::numSamples() const { return begin()->size(); }

/// const_iterator

AudioBlock::const_iterator::const_iterator() : mask_(NO_SPEAKER) {}
AudioBlock::const_iterator::const_iterator(ChannelLayout l, data_buffer_t::iterator ptr)
    : mask_(1), layout_(l), ptr_(ptr) {
  if (!(layout_ & mask_)) {
    advance();
  }
}
AudioBlock::const_iterator::const_iterator(const const_iterator& o)
    : mask_(o.mask_), layout_(o.layout_), ptr_(o.ptr_) {}
AudioBlock::const_iterator::const_iterator(const iterator& o) : mask_(o.mask_), layout_(o.layout_), ptr_(o.ptr_) {}
AudioBlock::const_iterator::~const_iterator() {}

channel_t AudioBlock::const_iterator::channel() const { return (channel_t)mask_; }

AudioBlock::const_iterator& AudioBlock::const_iterator::operator=(const const_iterator& o) {
  mask_ = o.mask_;
  layout_ = o.layout_;
  ptr_ = o.ptr_;
  return *this;
}
bool AudioBlock::const_iterator::operator==(const const_iterator& o) const { return mask_ == o.mask_; }
bool AudioBlock::const_iterator::operator!=(const const_iterator& o) const { return mask_ != o.mask_; }

AudioBlock::const_iterator& AudioBlock::const_iterator::operator++() {
  advance();
  return *this;
}

void AudioBlock::const_iterator::advance() {
  do {
    ++ptr_;
    mask_ <<= 1;
  } while (!(layout_ & mask_) && mask_ != NO_SPEAKER);
}

AudioBlock::const_iterator::const_reference AudioBlock::const_iterator::operator*() const { return *ptr_; }
AudioBlock::const_iterator::const_pointer AudioBlock::const_iterator::operator->() const { return ptr_; }

/// iterator

AudioBlock::iterator::iterator() {}
AudioBlock::iterator::iterator(ChannelLayout l, data_buffer_t::iterator ptr) : const_iterator(l, ptr) {}
AudioBlock::iterator::iterator(const iterator& o) : const_iterator(o) {}
AudioBlock::iterator::~iterator() {}

AudioBlock::iterator& AudioBlock::iterator::operator=(const iterator& o) {
  mask_ = o.mask_;
  layout_ = o.layout_;
  ptr_ = o.ptr_;
  return *this;
}
bool AudioBlock::iterator::operator==(const iterator& o) const { return mask_ == o.mask_; }
bool AudioBlock::iterator::operator!=(const iterator& o) const { return mask_ != o.mask_; }

AudioBlock::iterator& AudioBlock::iterator::operator++() {
  advance();
  return *this;
}

AudioBlock::iterator::reference AudioBlock::iterator::operator*() const { return *ptr_; }
AudioBlock::iterator::pointer AudioBlock::iterator::operator->() const { return ptr_; }

}  // namespace Audio
}  // namespace VideoStitch
