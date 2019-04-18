// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// An object to hold and pass around audio samples internally.

#pragma once

#include "audio.hpp"

#include <vector>
#include <array>

namespace VideoStitch {
namespace Audio {

/** \class AudioTrack
 * \brief An object for holding and passing around audio data
 * An audio channel holds the samples from a single track.
 */
class AudioTrack : public std::vector<audioSample_t> {
 public:
  /**
   * @brief Constructs a track with the given channel type
   * @param c Channel type
   */
  explicit AudioTrack(ChannelMap c = NO_SPEAKER);

  ~AudioTrack();

  AudioTrack& operator=(AudioTrack&&);
  AudioTrack(AudioTrack&&);

  /**
   * @brief Returns the channel type
   * @return The channel type of the corresponding track
   */
  ChannelMap channel() const;

  /**
   * @brief Sets the channel type
   * @param Channel type to set
   * @return void
   */
  void setChannel(ChannelMap);

  // XXX TODO FIXME temporary
  AudioTrack clone() const {
    AudioTrack myclone = *this;
    return myclone;
  }

 private:
  friend class AudioBlock;
  AudioTrack& operator=(const AudioTrack&) = default;
  AudioTrack(const AudioTrack&) = default;

  ChannelMap channel_;
};

/// \class AudioBlock
/// \brief An object for holding and passing around audio data
/// An AudioBlock is a sequence of audio channels.
class VS_EXPORT AudioBlock {
 public:
  typedef AudioTrack value_type;
  typedef AudioTrack& reference;
  typedef const AudioTrack& const_reference;
  typedef std::ptrdiff_t difference_type;
  typedef channel_t size_type;

  typedef std::array<AudioTrack, MAX_AUDIO_CHANNELS> data_buffer_t;

  explicit AudioBlock(ChannelLayout layout = UNKNOWN, mtime_t timestamp = 0);
  explicit AudioBlock(unsigned char nbTracks, mtime_t timestamp = 0);

  // AudioBlock(const AudioBlock&) = delete;
  // AudioBlock& operator=(const AudioBlock&) = delete;

  AudioBlock(AudioBlock&& o);
  AudioBlock& operator=(AudioBlock&& o);

  ~AudioBlock();

  /// \fn void setChannelLayout(const ChannelLayout layout)
  /// \param layout The new channel layout
  void setChannelLayout(const ChannelLayout layout);

  /// \fn void setTimestamp(mtime_t time)
  /// \param time The new timestamp, in microseconds
  void setTimestamp(mtime_t time);

  /// \fn void getTimestamp()
  /// \return The object's timestamp, in microseconds
  mtime_t getTimestamp() const;

  /// \fn void getLayout()
  /// \return The channel layout
  ChannelLayout getLayout() const;

  /// \fn void clear()
  /// \brief Clear all audio data
  void clear();

  /// \fn void swap()
  /// \brief swap an audio block with an other
  void swap(AudioBlock&);

  /// \fn size_type size()
  /// \return return the number of tracks in the audio block
  size_type size();

  /// \fn void swap()
  /// \brief swap an audio block with an other
  size_type max_size();

  /// \fn bool empty()
  /// \return true if the container is empty, false otherwise
  bool empty() const;

  /// \fn void resize()
  /// \brief resize all tracks of the audio block
  /// \param size in samples
  void resize(size_t n);

  /// \fn void assign()
  /// \brief assign nSamples at the value val
  /// \param size in samples
  /// \param value to set the samples
  void assign(size_t nSamples, audioSample_t val);

  /// \fn void numSamples()
  /// \return return number of samples in the block
  size_t numSamples() const;

  class iterator;
  /// \class AudioBlock::iterator
  /// \brief Iterator on the different channels of the AudioBlock (immutable version)
  class const_iterator {
   public:
    typedef std::ptrdiff_t difference_type;
    typedef const AudioTrack value_type;
    typedef const AudioTrack& const_reference;
    typedef data_buffer_t::const_iterator const_pointer;
    typedef std::random_access_iterator_tag iterator_category;

    const_iterator();
    const_iterator(ChannelLayout, std::array<AudioTrack, MAX_AUDIO_CHANNELS>::iterator);
    const_iterator(const const_iterator&);
    explicit const_iterator(const iterator&);
    ~const_iterator();

    channel_t channel() const;

    const_iterator& operator=(const const_iterator&);
    bool operator==(const const_iterator&) const;
    bool operator!=(const const_iterator&) const;

    const_iterator& operator++();

    const_reference operator*() const;
    const_pointer operator->() const;
    const_reference operator[](size_type) const;

   protected:
    void advance();

    int64_t mask_ = 1;
    ChannelLayout layout_;
    std::array<AudioTrack, MAX_AUDIO_CHANNELS>::iterator ptr_;
  };
  /// \class AudioBlock::iterator
  /// \brief Iterator on the different channels of the AudioBlock (mutable version)
  class iterator : public const_iterator {
   public:
    typedef std::ptrdiff_t difference_type;
    typedef AudioTrack value_type;
    typedef AudioTrack& reference;
    typedef data_buffer_t::iterator pointer;
    typedef std::random_access_iterator_tag iterator_category;

    iterator();
    iterator(ChannelLayout, data_buffer_t::iterator);
    iterator(const iterator&);
    ~iterator();

    iterator& operator=(const iterator&);
    bool operator==(const iterator&) const;
    bool operator!=(const iterator&) const;

    iterator& operator++();

    reference operator*() const;
    pointer operator->() const;
    reference operator[](size_type) const;
  };

  iterator begin();
  const_iterator begin() const;
  const_iterator cbegin() const;
  iterator end();
  const_iterator end() const;
  const_iterator cend() const;

  reference operator[](size_type);
  const_reference operator[](size_type) const;
  reference at(size_type);
  const_reference at(size_type) const;
  AudioBlock& operator+=(const AudioBlock& rhs);
  friend AudioBlock operator+(AudioBlock& lhs, const AudioBlock& rhs) {
    lhs += rhs;
    return lhs;
  }

  // XXX TODO FIXME temporary
  AudioBlock clone() const {
    AudioBlock myclone = *this;
    return myclone;
  }

 private:
  AudioBlock(const AudioBlock& o);  // XXX TODO FIXME delete me
  AudioBlock& operator=(const AudioBlock&) = default;

  ChannelLayout layout_;
  mtime_t timestamp_;
  data_buffer_t data_;
};

}  // namespace Audio
}  // namespace VideoStitch
