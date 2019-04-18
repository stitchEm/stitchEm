// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/audioWav.hpp"

#include <vector>
#include <cmath>

namespace VideoStitch {
namespace Audio {

///////////////////////////////////////////////////////////
//
//               AudioObject::WavWriter
//
///////////////////////////////////////////////////////////
//
//  Wav file writer
//
//

WavWriter::WavWriter(const char *fileName, const ChannelLayout layout, const double sampleRate)
    : AudioObject("wavwriter", AudioFunction::SINK, getNbChannelsFromChannelLayout(layout), 0, sampleRate),
      _fileSize(0),
      _fileName(fileName) {
  _f.open(fileName, std::ios::out);
  assert(_f.is_open());
  _writeHeader();
}

void WavWriter::updateFileSize() {
  // Write file size field
  _f.seekp(4);  // Bytes [ 4 .. 7 ]
  _f.write((char *)&_fileSize, 4);

  // Write data size field
  _f.seekp(40);  // Bytes [ 40 .. 43 ]
  std::uint32_t dataSize = _fileSize - 44;
  _f.write((char *)&dataSize, 4);
  _f.seekp(std::ios_base::end);
}

void WavWriter::close() {
  updateFileSize();
  // Close file
  _f.close();
}

void WavWriter::step(const AudioBlock &buf) { step(const_cast<AudioBlock &>(buf)); }

void WavWriter::step(AudioBlock &buf) {
  size_t nSamples = buf.numSamples();
  if (nSamples > 0) {
    int nChannels = getNbChannelsFromChannelLayout(buf.getLayout());
    float *interleaved = new float[(int)nSamples * nChannels];
    for (size_t s = 0; s < nSamples; ++s) {
      int c = 0;
      for (const auto &track : buf) {
        interleaved[s * nChannels + c] = static_cast<float>(track[s]);
        c++;
      }
    }
    size_t bufferSize = nSamples * nChannels * sizeof(float);
    _fileSize += static_cast<uint32_t>(bufferSize);
    _f.write((const char *)interleaved, bufferSize);
    delete[] interleaved;
  }
}

// To be tested
void WavWriter::step(AudioTrack &track) {
  for (auto sample : track) {
    float value = (float)sample;
    _f.write((char *)&value, 4);
    _fileSize += 4;
  }
  //  std::cout << _fileName << " _fileSize " << _fileSize << std::endl;
}

void WavWriter::step(const audioSample_t *data, const size_t size) {
  for (size_t s = 0; s < size; s++) {
    float value = (float)data[s];
    _f.write((char *)&value, sizeof(float));
    _fileSize += 4;
  }
}

void WavWriter::append(const audioSample_t *data, const size_t size) {
  if (!_f.is_open()) _f.open(_fileName, std::ios::in | std::ios::out);
  _f.seekp(std::ios_base::end);
  step(data, size);
  close();
}

void WavWriter::_writeHeader() {
  uint32_t u32;
  uint16_t u16;

  _f.write("RIFF", 4);
  u32 = 0;
  _f.write((char *)&u32, 4);  // Will have to write the file size here on closing
  _f.write("WAVEfmt ", 8);
  u32 = 16;  // Format block length
  _f.write((char *)&u32, 4);
  u16 = 3;  // Float sample format
  _f.write((char *)&u16, 2);
  u16 = (std::uint16_t)getInputs();  // Channels
  _f.write((char *)&u16, 2);
  u32 = (std::uint32_t)lround(getSampleRate());  // Samples per second (Fs)
  _f.write((char *)&u32, 4);
  //  u32 *= (std::uint32_t) ( getInputs() * 4 );     // Bytes per second [ Fs * nChans * ( bits_per_samp / 8 ) ]
  u32 = 0;  // FIXME on windows this field is not well written
  _f.write((char *)&u32, 4);
  u16 = ((std::uint16_t)getInputs() * 4);  // Block align [ nChannels * sizeof (float) ]
  _f.write((char *)&u16, 2);
  u16 = 32;  // Bits per sample
  _f.write((char *)&u16, 2);
  _f.write("data", 4);
  u32 = 0;  // Will have to write the data size here on closing
  _f.write((char *)&u32, 4);

  _fileSize = 44;
}

}  // namespace Audio
}  // namespace VideoStitch
