// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "resampler.hpp"

#include "libvideostitch/logging.hpp"

#include <vector>
#include <sstream>

namespace VideoStitch {
namespace Audio {

static std::fstream audioDebugFile;

void dumpAudio(audioSample_t *data, const int nSamples) {
  // DEBUG Dump output
  if (!audioDebugFile.is_open()) {
    audioDebugFile.open("/tmp/debug.raw", std::ios::out);
  }
  for (int s = 0; s < nSamples; s++) {
    float tmp = (float)data[s];
    audioDebugFile.write((char *)&tmp, sizeof(float));
  }
}

// Helper to convert AudioBlock to Samples
void samples2AudioBlock(AudioBlock &out, const Samples &in) {
  ChannelLayout layout = in.getChannelLayout();
  float **indata = (float **)in.getSamples().data();
  size_t nSamples = in.getNbOfSamples();
  AudioBlock ablock(layout, in.getTimestamp());
  if (in.getSamplingDepth() != SamplingDepth::FLT_P) {
    // Manage only float planar for the moment
    return;
  }
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    AudioTrack track(getChannelMapFromChannelIndex(i));
    if (layout & getChannelMapFromChannelIndex(i)) {
      for (size_t s = 0; s < nSamples; s++) {
        track.push_back((audioSample_t)indata[i][s]);
      }
    }
    ablock[getChannelMapFromChannelIndex(i)] = track.clone();
  }
  out.swap(ablock);
}

void audioBlock2Samples(Samples &out, const AudioBlock &in) {
  std::array<audioSample_t *, MAX_AUDIO_CHANNELS> r;
  for (auto &v : r) {
    v = nullptr;
  }

  size_t nbSamples = in.begin()->size();

  for (const AudioTrack &track : in) {
    int channel = getChannelIndexFromChannelMap(track.channel());
    assert(channel >= 0);
    assert(channel < (int)r.size());
    // no two audio tracks can share the same channel
    assert(r[channel] == nullptr);
    audioSample_t *samples = new audioSample_t[nbSamples];
    for (size_t s = 0; s < nbSamples; s++) {
      samples[s] = track[s];
    }
    r[channel] = samples;
  }
  out = Samples(SamplingRate::SR_48000, SamplingDepth::DBL_P, in.getLayout(), in.getTimestamp(), (uint8_t **)r.data(),
                nbSamples);
}

uint8_t **allocSamplesData(size_t nbSamples, SamplingDepth d, ChannelLayout layout) {
  uint8_t **tmp = new uint8_t *[MAX_AUDIO_CHANNELS];
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    if (getChannelMapFromChannelIndex(i) & layout) {
      tmp[i] = (uint8_t *)new uint8_t[nbSamples * getSampleSizeFromSamplingDepth(d)]();
    } else {
      tmp[i] = nullptr;
    }
  }
  return tmp;
}

void freeSamplesData(uint8_t **tmp) {
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    if (tmp[i]) {
      delete[] tmp[i];
    }
  }
  delete[] tmp;
}

// Constructor
AudioResampler::AudioResampler(const SamplingRate inRate, const SamplingDepth inDepth, const SamplingRate outRate,
                               const SamplingDepth outDepth, const ChannelLayout layout, const int blockSize)
    : _offsettime(0),
      _inRate((double)getIntFromSamplingRate(inRate)),
      _outRate((double)getIntFromSamplingRate(outRate)),
      _inDepth(inDepth),
      _outDepth(outDepth),
      _layout(layout),
      _blockSizeIn(blockSize),
#ifndef R8BLIB_UNSUPPORTED
      _internalBuf(blockSize),
#endif
      _outData(nullptr),
      _keepPtr(nullptr),
      _dump(false),
      _iWriteCount(0) {

  _blockSizeOut = (int)(_blockSizeIn * (_outRate / _inRate) + 0.5);

  if (_outRate > 0. && _outRate != _inRate) {  // In case where no resampling is needed
    for (int i = 0; i < getNbChannelsFromChannelLayout(layout); i++) {
#ifndef R8BLIB_UNSUPPORTED
      _resamps[i] = new r8b::CDSPResampler24(_inRate, _outRate, (int)_blockSizeIn);
#endif
    }
  }

  if (_dump) {
    _outFile.open("/tmp/out.wav", std::ios::out);
    _inFile.open("/tmp/in.wav", std::ios::out);
  }
}

AudioResampler *AudioResampler::create(const SamplingRate inRate, const SamplingDepth inDepth,
                                       const SamplingRate outRate, const SamplingDepth outDepth,
                                       const ChannelLayout layout, const size_t blockSize) {
  return new AudioResampler(inRate, inDepth, outRate, outDepth, layout, (int)blockSize);
}

void AudioResampler::alloc() {
  int nChannels = getNbChannelsFromChannelLayout(_layout);
  _outData = new audioSample_t *[nChannels];
  _keepPtr = new audioSample_t *[nChannels];
  for (int c = 0; c < nChannels; c++) {
    _outData[c] = new audioSample_t[_blockSizeOut];
    _keepPtr[c] = _outData[c];
  }
}

AudioResampler::~AudioResampler() {
  if (_outData != nullptr && _keepPtr != nullptr) {
    int nChannels = getNbChannelsFromChannelLayout(_layout);
    // Free the original output pointers
    // This needs to be done since the r8b resampler can change the output memory
    for (int i = 0; i < nChannels; i++) {
      if (_keepPtr[i] != _outData[i]) {
        delete[] _keepPtr[i];
      } else {
        delete[] _outData[i];
      }
    }
    delete[] _keepPtr;
    delete[] _outData;
  }
}

int AudioResampler::resample(const audioSample_t *in, size_t nbSamplesin, audioSample_t *&out,
                             const uint32_t channelIndex) {
  /// Note from the r8b documentation :
  /// This variable receives the pointer to the resampled data.
  /// On function's return, this pointer may point to the address within the "in" input buffer,
  /// or to *this object's internal buffer. In real-time applications it is suggested to pass
  /// this pointer to the next output audio block and consume any data left from the previous
  /// output audio block first before calling the process() function again.
  /// The buffer pointed to by the "out" on return may be owned by the resampler,
  /// so it should not be freed by the caller.
  if (_outRate > 0. && _outRate != _inRate) {
#ifndef R8BLIB_UNSUPPORTED
    return _resamps[channelIndex]->process(const_cast<double *>(in), static_cast<int>(nbSamplesin), out);
#else
    return 0;
#endif
  }

  memcpy(out, in, nbSamplesin * sizeof(*in));
  return static_cast<int>(nbSamplesin);
}

void AudioResampler::resample(const Audio::Samples &audioSamplesIn, AudioBlock &audioBlockOut) {
  if (_outDepth == SamplingDepth::SD_NONE) {
    return;
  }

  if (_outRate <= 0.) {
    return;
  }

  ChannelLayout layout = audioSamplesIn.getChannelLayout();
  AudioBlock block(layout);
  mtime_t outTime = 0, inTime = audioSamplesIn.getTimestamp();

  // Allocate memory for the first call
  if (_outData == nullptr) {
    alloc();
  }

#ifndef R8BLIB_UNSUPPORTED
  const Samples::data_buffer_t &in = audioSamplesIn.getSamples();
  int iChannel = 0;
  bool interleaved = isInterleaved(_inDepth);
#endif

  if ((int)audioSamplesIn.getNbOfSamples() > _blockSizeIn) {
    Logger::get(Logger::Warning) << "[audio_resampler] too many input samples given " << audioSamplesIn.getNbOfSamples()
                                 << " > " << _blockSizeIn << std::endl;
    assert(false);
    return;
  }

#ifndef R8BLIB_UNSUPPORTED
  for (int i = 0; i < MAX_AUDIO_CHANNELS; i++) {
    if (layout & getChannelMapFromChannelIndex(i)) {
      AudioTrack track(getChannelMapFromChannelIndex(i));
      // Convert input data to dbl_p if needed before resampling it here
      int nResampled = 0;
      if (interleaved) {
        nResampled = convertInterleaveData(in[0], (int)audioSamplesIn.getNbOfSamples(), _inDepth, layout, iChannel,
                                           _internalBuf.getPtr());
      } else {
        nResampled =
            convertToInternalFormat(in[i], (int)audioSamplesIn.getNbOfSamples(), _inDepth, _internalBuf.getPtr());
      }

      if (_dump) {  /// DEBUG purpose
        dumpInput(getChannelMapFromChannelIndex(i));
      }

      nResampled = resample(_internalBuf.getPtr(), nResampled, _outData[iChannel], iChannel);

      if (i == 0) {
        _offsettime -=
            (mtime_t)(((double)audioSamplesIn.getNbOfSamples() / _inRate - (double)nResampled / _outRate) * 1000000.);
        outTime = inTime + _offsettime;
        block.setTimestamp(outTime);
      }

      // Copy data
      for (int s = 0; s < nResampled; s++) {
        track.push_back(_outData[iChannel][s]);
      }

      if (_dump) {  /// DEBUG purpose
        dumpOutput(nResampled, iChannel, getChannelMapFromChannelIndex(i));
      }
      block[getChannelMapFromChannelIndex(i)] = track.clone();
      iChannel++;
    }
  }
#else
  if (_iWriteCount == 0) {
    Logger::get(Logger::Warning) << "[audio_resampler] audio resampling not supported, missing r8B library"
                                 << std::endl;
  }
  _iWriteCount++;
  return;
#endif
  audioBlockOut.swap(block);
  _iWriteCount++;
}

void AudioResampler::resample(const AudioBlock &audioBlockIn, Audio::Samples &audioSamplesOut) {
  if (_outDepth == SamplingDepth::SD_NONE) {
    return;
  }

  if (_outRate <= 0.) {
    return;
  }

  int iChannel = 0;
  uint8_t **out = nullptr;

  // Allocate memory for the first call
  if (_outData == nullptr) {
    alloc();
  }
  size_t nbSamplesIn = audioBlockIn.begin()->size();
  size_t nbSamplesOut = 0;
  if (nbSamplesIn == 0) {
    return;
  }

  for (auto &track : audioBlockIn) {
    if (!(track.channel() & _layout)) {
      continue;
    }
    nbSamplesOut = resample(track.data(), nbSamplesIn, _outData[iChannel], iChannel);

    if (!isInterleaved(_outDepth) && nbSamplesOut > 0) {
      convertToSamplesPlanar(_outData[iChannel], nbSamplesOut, _outDepth);
      // Copy out data
      if (out == nullptr) {
        out = allocSamplesData(nbSamplesOut, _outDepth, _layout);
      }
      assert(out[getChannelIndexFromChannelMap(track.channel())] != nullptr && _outData[iChannel] != nullptr);
      memcpy((void *)out[getChannelIndexFromChannelMap(track.channel())], (void *)_outData[iChannel],
             nbSamplesOut * getSamplingDepthSize(_outDepth));
    }

    iChannel++;
  }

  mtime_t outTime = 0, inTime = audioBlockIn.getTimestamp();
  _offsettime -= (mtime_t)(((double)nbSamplesIn / _inRate - (double)nbSamplesOut / _outRate) * 1000000.);
  outTime = inTime + _offsettime;

  if (!isInterleaved(_outDepth) && nbSamplesOut > 0) {
    uint8_t **tmp = allocSamplesData(nbSamplesOut, _outDepth, _layout);
    convertToLayout((uint8_t **)out, tmp, (int)nbSamplesOut, _outDepth, audioBlockIn.getLayout(), _layout);
    // audioSamplesOut will take the ownership of the memory
    audioSamplesOut =
        Samples(getSamplingRateFromInt(static_cast<int>(_outRate)), _outDepth, _layout, outTime, tmp, nbSamplesOut);
    delete[] tmp;
    freeSamplesData(out);
  }

  // convert to good sample format
  if (isInterleaved(_outDepth) && nbSamplesOut > 0) {
    int nOutChannels = getNbChannelsFromChannelLayout(_layout);
    uint8_t **tmp = allocSamplesData(nbSamplesOut, SamplingDepth::DBL_P, _layout);
    // convertToLayout() needs an array of [MAX_AUDIO_CHANNELS] which is not the case for _outData
    // Use an intermediate pointer to fix this.
    std::vector<uint8_t *> arraySamples(MAX_AUDIO_CHANNELS);
    for (int i = 0, j = 0; i < MAX_AUDIO_CHANNELS && j < nOutChannels; i++) {
      if (getChannelMapFromChannelIndex(i) & _layout) {
        arraySamples[i] = (uint8_t *)_outData[j++];
      }
    }

    convertToLayout((uint8_t **)arraySamples.data(), (uint8_t **)tmp, (int)nbSamplesOut, SamplingDepth::DBL_P,
                    audioBlockIn.getLayout(), _layout);

    // convertToSamplesInterleaved() needs flat indexing, which is
    // not the case for many formats. Use an intermediate pointer
    // to fix this.
    std::vector<uint8_t *> flatSamples(nOutChannels);
    for (int i = 0, j = 0; i < MAX_AUDIO_CHANNELS && j < nOutChannels; i++) {
      if (getChannelMapFromChannelIndex(i) & _layout) {
        flatSamples[j++] = tmp[i];
      }
    }

    out = new uint8_t *[MAX_AUDIO_CHANNELS];
    out[0] = new uint8_t[nOutChannels * nbSamplesOut * getSamplingDepthSize(_outDepth)];
    convertToSamplesInterleaved((audioSample_t **)flatSamples.data(), nOutChannels, nbSamplesOut, out[0], _outDepth);

    // audioSamplesOut will take the ownership of the memory
    audioSamplesOut =
        Samples(getSamplingRateFromInt(static_cast<int>(_outRate)), _outDepth, _layout, outTime, out, nbSamplesOut);
    delete[] out;
    freeSamplesData(tmp);
  }
}

void AudioResampler::dumpInput(const ChannelMap channelType) {
  /// DEBUG Dump input
#ifndef R8BLIB_UNSUPPORTED
  if (channelType == SPEAKER_FRONT_LEFT) {
    for (int s = 0; s < _blockSizeIn; s++) {
      float tmp = (float)_internalBuf[s];
      _inFile.write((char *)&tmp, sizeof(float));
    }
  }
#endif
}

void AudioResampler::dumpOutput(const int nResampled, const int iChannel, const ChannelMap channelType) {
  // DEBUG Dump output
  if (channelType == SPEAKER_FRONT_LEFT) {
    for (int s = 0; s < nResampled; s++) {
      float tmp = (float)_outData[iChannel][s];
      _outFile.write((char *)&tmp, sizeof(float));
    }
  }
}

}  // namespace Audio
}  // namespace VideoStitch
