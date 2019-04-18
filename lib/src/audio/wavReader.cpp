// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#if !defined(__clang_analyzer__)

#include "libvideostitch/audioWav.hpp"

namespace VideoStitch {
namespace Audio {

WavReader::WavReader(const char *fname) : AudioObject("wavReader", AudioFunction::SINK), wavreader(fname) {
  setnInputs(0);
  setnOutputs(wavreader.channels());
  setSampleRate(wavreader.sampleRate());
  layout = getAChannelLayoutFromNbChannels(wavreader.channels());
}

void WavReader::step(float **data, uint32_t nSamples) { wavreader.read(data, nSamples); }

void WavReader::step(float **data) {
  uint32_t nSamples = wavreader.getnSamples();
  step(data, nSamples);
}

void WavReader::step(AudioBlock &buf, uint32_t nSamples) {
  uint32_t nChannels = wavreader.channels();
  buf.setChannelLayout(layout);
  float **data = (float **)malloc(nChannels * sizeof(float *));
  for (uint32_t c = 0; c < nChannels; c++) {
    data[c] = (float *)malloc(nSamples * sizeof(float));
  }
  wavreader.read(data, nSamples);
  uint32_t i = 0;
  for (auto &track : buf) {
    for (size_t s = 0; s < nSamples; s++) {
      track.push_back(data[i][s]);
    }
    i++;
  }

  for (uint32_t c = 0; c < nChannels; c++) {
    free(data[c]);
  }
  free(data);
}

void WavReader::step(AudioBlock &buf) {
  uint32_t nSamples = wavreader.getnSamples();
  step(buf, nSamples);
}

///////////////////////////////////////////////////////////
//
//                   Wave file reader
//
///////////////////////////////////////////////////////////

WavReaderBlocking::WavReaderBlocking(const char *fileName)
    : _fSize(0),
      _formatBlockLength(0),
      _fmtCode(0),
      _nChannels(0),
      _sampleRate(0),
      _bytesPerSecond(0),
      _blockAlign(0),
      _bitsPerSample(0),
      _extSize(0),
      _nValidBits(0),
      _channelMask(0),
      _guid0(0),
      _guid1(0),
      _guid2(0),
      _dataStart(0),
      _dataLen(0),
      _dataPos(0) {
  memset(_guid3, 0, sizeof(_guid3));
  _fname.assign(fileName);
  _f.open(fileName, std::ios::in);
  assert(_f.is_open());
  _parseHeader();
}

WavReaderBlocking::~WavReaderBlocking() { _f.close(); }

SampleFormat WavReaderBlocking::sampleFormat(void) const {
  switch (_fmtCode) {
    case PCM:
    checkPCM:
      switch (_bitsPerSample) {
        case 8:
          return SampleFormat::S8;
        case 16:
          return SampleFormat::S16;
        case 24:
          return SampleFormat::S24;
        case 32:
          return SampleFormat::S32;
      }
      return SampleFormat::UNKNOWN;

    case FLOAT:
    checkFloat:
      switch (_bitsPerSample) {
        case 32:
          return SampleFormat::F32;
        case 64:
          return SampleFormat::F64;
      }
      return SampleFormat::UNKNOWN;

    case EXTENDED:
      switch (_guid0) {
        case PCM:
          goto checkPCM;
        case FLOAT:
          goto checkFloat;
      }
      return SampleFormat::UNKNOWN;
  }

  return SampleFormat::UNKNOWN;
}

int WavReaderBlocking::sampleRate() const { return (int)_sampleRate; }

int WavReaderBlocking::channels() const { return (int)_nChannels; }

unsigned int WavReaderBlocking::channelMask() const { return (unsigned int)_channelMask; }

void WavReaderBlocking::printInfo() const {
  std::cout << "printInfo in wavreader blocking" << std::endl;

  const int width = 30;

  std::cout << "Information for " << _fname << std::endl << std::endl;

  std::cout << std::setw(width) << "   File size:";
  std::cout << std::dec << _fSize + 8 << " bytes" << std::endl;

  std::cout << std::setw(width) << "   Running Time: ";
  std::cout << _prettyTime() << std::endl << std::endl;

  std::cout << std::setw(width) << "   Sample Format:";
  switch (this->sampleFormat()) {
    case SampleFormat::S8:
      std::cout << "8-bit PCM" << std::endl;
      break;
    case SampleFormat::S16:
      std::cout << "16-bit PCM" << std::endl;
      break;
    case SampleFormat::S24:
      std::cout << "24-bit PCM" << std::endl;
      break;
    case SampleFormat::S32:
      std::cout << "32-bit PCM" << std::endl;
      break;
    case SampleFormat::F32:
      std::cout << "32-bit Float" << std::endl;
      break;
    case SampleFormat::F64:
      std::cout << "64-bit Float" << std::endl;
      break;
    default:
      std::cout << "Unknown" << std::endl;
      break;
  }

  std::cout << std::setw(width) << "   Sample Rate:";
  std::cout << std::dec << _sampleRate << std::endl;

  std::cout << std::setw(width) << "   Channels:";
  std::cout << std::dec << _nChannels << std::endl;

  double br;
  std::cout << std::setw(width) << "   Bit Rate:";
  if ((_bytesPerSecond * 8) >= 1000000) {
    br = (double)(_bytesPerSecond * 8) / 1000000.0;
    std::cout << std::setprecision(3) << br << " Mbps" << std::endl << std::endl;
  } else {
    br = (double)(_bytesPerSecond * 8) / 1000.0;
    std::cout << std::setprecision(3) << br << " kbps" << std::endl << std::endl;
  }

  if (_formatBlockLength < 40) {
    return;
  }

  std::cout << "Extended information" << std::endl << std::endl;

  std::cout << std::setw(width) << "   Valid Audio Bits:";
  std::cout << std::dec << _nValidBits << std::endl;

  std::cout << std::setw(width) << "   Channel Mask:";
  std::cout << std::hex << _channelMask << std::endl;

  std::cout << std::setw(width) << "   GUID:";
  char buf[32];
  sprintf(buf, "%08x", _guid0);
  std::cout << std::hex << buf << "-";
  sprintf(buf, "%04x", _guid1);
  std::cout << std::hex << buf << "-";
  sprintf(buf, "%04x", _guid2);
  std::cout << std::hex << buf << "-";
  sprintf(buf, "%02x%02x-%02x%02x%02x%02x%02x%02x", _guid3[0], _guid3[1], _guid3[2], _guid3[3], _guid3[4], _guid3[5],
          _guid3[6], _guid3[7]);
  std::cout << std::hex << buf;
  std::cout << std::endl << std::endl;
}

double WavReaderBlocking::currentPosition() {
  unsigned int sampleCount = (_dataPos - _dataStart) / _blockAlign;
  return (1.0 / (double)_sampleRate) * (double)sampleCount;
}

static const float FACTOR_32 = (float)2147483647.0;
static const float FACTOR_16 = (float)32767.0;
static const float FACTOR_8 = (float)127.0;

void WavReaderBlocking::read(float **data, const unsigned int nSamples) {
  unsigned int bytesRemaining;
  unsigned int bytesToRead;
  unsigned int samplesToRead;

start:

  bytesRemaining = (_dataStart + _dataLen) - _dataPos;
  bytesToRead = nSamples * _blockAlign;

  if (bytesRemaining > bytesToRead) {
    samplesToRead = nSamples;
  } else if (bytesRemaining == 0) {
    /* Handle end-of-stream */
    //		switch ( wf->eos_action ) {
    //			case WAV_EOS_ACTION_REMOVE:
    //				return WAV_EOS_ACTION_REMOVE;
    //			case WAV_EOS_ACTION_LOOP:
    _dataPos = _dataStart;  // Loop
    goto start;
    //			default:
    //				return -1;
    //		}

  } else {
    /* Check that the number of bytes left corresponds to a correct number of samples */
    assert(bytesRemaining % (_bitsPerSample / 8) == 0);

    /* We won't be filling all the way, so zero the output buffer */
    for (int i = 0; i < _nChannels; i++) {
      memset(data[i], 0, sizeof(float) * nSamples);
    }
    samplesToRead = bytesRemaining / _blockAlign;
  }

  _f.seekg(_dataPos);

  if ((_fmtCode == FLOAT) || ((_fmtCode == EXTENDED) && (_guid0 == FLOAT))) {
    /* Float data */
    switch (_bitsPerSample) {
      case 32: {
        for (unsigned int i = 0; i < samplesToRead; i++) {
          for (int j = 0; j < _nChannels; j++) {
            _f.read((char *)&data[j][i], sizeof(float));
          }
        }
        break;
      }

      case 64:
        std::cout << "64-bit float data not implemented yet" << std::endl;
        assert(false);
        break;
      default:
        break;
    }

  } else {
    /* Integer data */
    switch (_bitsPerSample) {
      case 8: {
        std::int8_t sample;
        for (unsigned int i = 0; i < samplesToRead; i++) {
          for (unsigned int j = 0; j < _nChannels; j++) {
            _f.read((char *)&sample, 1);
            data[j][i] = (float)sample / FACTOR_8;
          }
        }
        break;
      }
      case 16: {
        std::int16_t sample;
        for (unsigned int i = 0; i < samplesToRead; i++) {
          for (unsigned int j = 0; j < _nChannels; j++) {
            _f.read((char *)&sample, 2);
            data[j][i] = (float)sample / FACTOR_16;
          }
        }
        break;
      }
      case 24: {
        std::int32_t sample;
        for (unsigned int i = 0; i < samplesToRead; i++) {
          for (unsigned int j = 0; j < _nChannels; j++) {
            _f.read((char *)&sample, 3);
            sample = sample << 8;
            data[j][i] = (float)sample / FACTOR_32;
          }
        }
        break;
      }
      case 32: {
        int32_t sample;
        for (unsigned int i = 0; i < samplesToRead; i++) {
          for (unsigned int j = 0; j < _nChannels; j++) {
            _f.read((char *)&sample, 4);
            data[j][i] = (float)sample / FACTOR_32;
          }
        }
        break;
      }
    }
  }

  _dataPos = (unsigned int)_f.tellg();
}

// Interleaved output
void WavReaderBlocking::read(float *data, const unsigned int nSamples) {
  unsigned int bytesRemaining;
  unsigned int bytesToRead;
  unsigned int samplesToRead;

start:

  bytesRemaining = (_dataStart + _dataLen) - _dataPos;
  bytesToRead = nSamples * _blockAlign;

  if (bytesRemaining > bytesToRead) {
    samplesToRead = nSamples;
  } else if (bytesRemaining == 0) {
    /* Handle end-of-stream */
    //		switch ( wf->eos_action ) {
    //			case WAV_EOS_ACTION_REMOVE:
    //				return WAV_EOS_ACTION_REMOVE;
    //			case WAV_EOS_ACTION_LOOP:
    _dataPos = _dataStart;  // Loop
    goto start;
    //			default:
    //				return -1;
    //		}

  } else {
    /* Check that the number of bytes left corresponds to a correct number of samples */
    assert(bytesRemaining % _blockAlign == 0);

    /* We won't be filling all the way, so zero the output buffer */
    memset(data, 0, sizeof(float) * nSamples * _nChannels);
    samplesToRead = bytesRemaining / _blockAlign;
  }

  _f.seekg(_dataPos);

  if ((_fmtCode == FLOAT) || ((_fmtCode == EXTENDED) && (_guid0 == FLOAT))) {
    /* Float data */
    switch (_bitsPerSample) {
      case 32: {
        for (unsigned int i = 0; i < samplesToRead * _nChannels; i++) {
          _f.read((char *)&data[i], sizeof(float));
        }
        break;
      }
      case 64:
        std::cout << "64-bit float data not implemented yet" << std::endl;
        assert(false);
        break;
      default:
        break;
    }

  } else {
    /* Integer data */
    switch (_bitsPerSample) {
      case 8: {
        std::int8_t sample;
        for (unsigned int i = 0; i < samplesToRead * _nChannels; i++) {
          _f.read((char *)&sample, 1);
          data[i] = (float)sample / FACTOR_8;
        }
        break;
      }
      case 16: {
        std::int16_t sample;
        for (unsigned int i = 0; i < samplesToRead * _nChannels; i++) {
          _f.read((char *)&sample, 2);
          data[i] = (float)sample / FACTOR_16;
        }
        break;
      }
      case 24: {
        std::int32_t sample;
        for (unsigned int i = 0; i < samplesToRead * _nChannels; i++) {
          _f.read((char *)&sample, 3);
          sample = sample << 8;
          data[i] = (float)sample / FACTOR_32;
        }
        break;
      }
      case 32: {
        int32_t sample;
        for (unsigned int i = 0; i < samplesToRead * _nChannels; i++) {
          _f.read((char *)&sample, 4);
          data[i] = (float)sample / FACTOR_32;
        }
        break;
      }
    }
  }

  _dataPos = (unsigned int)_f.tellg();
}

void WavReaderBlocking::seek(double time) {
  if (time < 0) {
    time = 0;
  }

  unsigned int t = _dataStart + ((std::uint32_t)(time / (1 / (double)_sampleRate)) * _blockAlign);
  _dataPos = t;
}

std::string WavReaderBlocking::_prettyTime() const {
  int h = 0, m = 0, s = 0, ms = 0;
  float totalSeconds = ((float)_dataLen / (float)_blockAlign) / (float)_sampleRate;
  int iSeconds = (int)totalSeconds;

  ms = (int)((totalSeconds - (int)totalSeconds) * 1000);
  s = iSeconds % 60;
  m = ((iSeconds - s) / 60) % 60;
  h = ((iSeconds - s - (m * 60)) / 3600);

  return ((h != 0) ? (std::to_string(h) + ":") : "") + ((m != 0) ? (std::to_string(m) + ":") : "") +
         ((s != 0) ? (std::to_string(s)) : "00") + "." + std::to_string(ms);
}

void WavReaderBlocking::_parseHeader() {
  char buf[256];

  _f.read(buf, 4);
  if (0 != strncmp(buf, "RIFF", 4)) {
    std::cout << "Not a WAV file" << std::endl;
    _f.close();
  }
  _f.read((char *)&_fSize, sizeof(_fSize));
  _f.read(buf, 4);
  if (0 != strncmp(buf, "WAVE", 4)) {
    std::cout << "Not a WAV file" << std::endl;
    _f.close();
    return;
  }

  // Get file format info
  bool found = false;
  std::uint32_t s;
  do {
    _f.read(buf, 4);
    if (0 != strncmp(buf, "fmt ", 4)) {
      _f.read((char *)&s, sizeof(s));
      _f.seekg(s, std::ios::cur);
    } else {
      found = true;
    }
  } while (!found);

  _f.read((char *)&_formatBlockLength, sizeof(_formatBlockLength));
  if ((_formatBlockLength != 16) && (_formatBlockLength != 18) && (_formatBlockLength != 40)) {
    std::cout << "Unexpected format length: " << _formatBlockLength << std::endl;
    _f.close();
    return;
  }

  _f.read((char *)&_fmtCode, sizeof(_fmtCode));
  if ((_fmtCode != PCM) && (_fmtCode != FLOAT) && (_fmtCode != EXTENDED)) {
    std::cout << "Unknown format: " << _fmtCode << std::endl;
    _f.close();
    return;
  }

  _f.read((char *)&_nChannels, sizeof(_nChannels));
  _f.read((char *)&_sampleRate, sizeof(_sampleRate));
  _f.read((char *)&_bytesPerSecond, sizeof(_bytesPerSecond));
  _f.read((char *)&_blockAlign, sizeof(_blockAlign));
  _f.read((char *)&_bitsPerSample, sizeof(_bitsPerSample));

  assert(_blockAlign == ((_bitsPerSample / 8) * _nChannels));
  // TODO: fix this field on windows.
  // assert(_bytesPerSecond == ( _sampleRate * _blockAlign ));

  if (_formatBlockLength > 16) {
    _f.read((char *)&_extSize, sizeof(_extSize));
  }

  if (_formatBlockLength > 18) {
    _f.read((char *)&_nValidBits, sizeof(_nValidBits));
    _f.read((char *)&_channelMask, sizeof(_channelMask));
    _f.read((char *)&_guid0, sizeof(_guid0));
    _f.read((char *)&_guid1, sizeof(_guid1));
    _f.read((char *)&_guid2, sizeof(_guid2));
    _f.read((char *)&_guid3, sizeof(_guid3));
  }

  // Find start of audio data
  _f.seekg(12);
  found = false;
  do {
    _f.read(buf, 4);
    if (0 != strncmp(buf, "data", 4)) {
      _f.read((char *)&s, sizeof(s));
      _f.seekg(s, std::ios::cur);
    } else {
      _f.read((char *)&_dataLen, sizeof(_dataLen));
      _dataStart = (unsigned int)_f.tellg();
      _dataPos = _dataStart;
      found = true;
    }
  } while (!found);
}

}  // namespace Audio
}  // namespace VideoStitch

#endif  // !defined(__clang_analyzer__)
