// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// A class to read wave files

#pragma once

#include "audioObject.hpp"
#include "config.hpp"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <locale>

namespace VideoStitch {
namespace Audio {

enum class SampleFormat { S8, S16, S24, S32, F32, F64, UNKNOWN };

class WavReaderBlocking {
 public:
  friend class WavReader;

  enum AudioFormat { PCM = 0x0001, FLOAT = 0x0003, A_LAW = 0x0006, MU_LAW = 0x0007, EXTENDED = 0xFFFE };

  explicit WavReaderBlocking(const char* fname);

  ~WavReaderBlocking();

  void printInfo() const;

  SampleFormat sampleFormat(void) const;
  int sampleRate() const;
  int channels() const;
  int getnSamples() const { return (_dataLen / _blockAlign); }
  unsigned int channelMask() const;

  double currentPosition();

  void read(float** data, const unsigned int nSamples);  // nchannels x nSamples
  void read(float* data, const unsigned int nSamples);   // Interleaved

  void seek(double time);

 private:
  std::string _fname;
  std::uint32_t _fSize;

  std::uint32_t _formatBlockLength;

  std::uint16_t _fmtCode;
  std::uint16_t _nChannels;
  std::uint32_t _sampleRate;
  std::uint32_t _bytesPerSecond;
  std::uint16_t _blockAlign;
  std::uint16_t _bitsPerSample;

  std::uint16_t _extSize;

  std::uint16_t _nValidBits;
  std::uint32_t _channelMask;
  std::uint32_t _guid0;
  std::uint16_t _guid1;
  std::uint16_t _guid2;
  std::uint8_t _guid3[8];

  std::uint32_t _dataStart;
  std::uint32_t _dataLen;
  std::uint32_t _dataPos;

  std::ifstream _f;

  std::string _prettyTime() const;
  void _parseHeader();
};

///
/// \class WavReader
/// \brief A class to read wave files
///
class VS_EXPORT WavReader : public AudioObject {
 public:
  ///
  /// \brief Constructs a wavereader object
  /// \param fname File name of the file to open
  ///
  explicit WavReader(const char* fname);
  explicit WavReader(const std::string& fname) : WavReader(fname.c_str()) {}
  ~WavReader() {}

  ///
  /// \brief Reads nSamples of the input file,
  ///        C-style function to manipulate C-arrays the memory has to be managed by the user
  /// \param nSamples number of samples to read
  ///
  void step(float** data, uint32_t nSamples);

  ///
  /// \brief Reads all the samples of the input file,
  ///         C-style function to manipulate C-arrays the memory has to be managed by the user
  ///
  void step(float** data);

  ///
  /// \brief Reads nSamples of the input file,
  ///        C++ function to manipulate AudioBlock the memory is managed by the class itself
  /// \param nSamples number of samples to read
  ///
  void step(AudioBlock& buf, uint32_t nSamples);

  ///
  /// \brief Reads all the samples of the input file,
  ///         C++ function to manipulate AudioBlock the memory is managed by the class itself
  ///
  void step(AudioBlock& buf);

  using AudioObject::step;

  ///
  /// \brief Reads the number of channels of the input file
  ///
  int getChannels() { return wavreader.channels(); }

  ///
  /// \brief Reads the channel mask of the input file
  ///
  unsigned int getChannelMask() { return wavreader.channelMask(); }

  ///
  /// \brief Reads the total number of samples of the input file
  ///
  int getnSamples() { return wavreader.getnSamples(); }

  ///
  /// \brief Prints information about the input file
  ///
  void printInfo() { wavreader.printInfo(); }

  ChannelLayout getLayout() const { return layout; }

 private:
  WavReaderBlocking wavreader;
  ChannelLayout layout;
};

///
/// \class WavWriter
/// \brief A class to write wave files
///        handle only AudioBlocks
///
class VS_EXPORT WavWriter : public AudioObject {
 public:
  WavWriter() : AudioObject("wavwriter", AudioFunction::SINK), _fileSize(0), _fileName("") {}

  ///
  /// \brief Constructs a wave writer object
  /// \param fname File name of the output file
  /// \param layout format of the audio
  /// \param sampleRate sampling rate of the output file
  ///
  WavWriter(const char* fileName, const ChannelLayout layout, const double sampleRate);
  WavWriter(const std::string& fileName, const ChannelLayout layout, const double sampleRate)
      : WavWriter(fileName.c_str(), layout, sampleRate) {}
  ~WavWriter() {}

  ///
  /// \brief Writes all the samples contained in the input audioBlock to the output file
  /// \param buf Audioblock to write. To manipulate AudioBlock
  ///
  void step(AudioBlock& buf);
  void step(const AudioBlock& buf);
  using AudioObject::step;

  void step(AudioTrack& track);
  void step(const audioSample_t* data, const size_t size);

  ///
  /// \brief Close the output file.
  ///        Calls this function after writing all samples. It writes the size of the file in the header
  ///
  void close();

  WavWriter& operator=(const WavWriter& o) {
    _fileName = std::move(o._fileName);
    _f.open(_fileName, std::ios::out);
    _fileSize = std::move(o._fileSize);
    return *this;
  }

  void append(const audioSample_t* data, const size_t size);

  void updateFileSize();
  void flush() {
    updateFileSize();
    _f.flush();
  }

 private:
  uint32_t _fileSize;
  std::string _fileName;
  std::ofstream _f;
  void _writeHeader();

};  // class WavWriter

}  // namespace Audio
}  // namespace VideoStitch
