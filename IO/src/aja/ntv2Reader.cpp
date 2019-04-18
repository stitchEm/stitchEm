// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ntv2Reader.hpp"

#include "ajastuff/system/systemtime.h"
#include "ajastuff/system/process.h"
#include "ntv2utils.h"
#include "ntv2signalrouter.h"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

static const int AUDIOSIZE_MAX(401 * 1024);
// AJA supports only 8 or 16 channels, we support only stereo so let's keep only 8 channels
static const unsigned NB_AJA_CHANNELS(8);

namespace VideoStitch {
namespace Input {

NTV2Reader* NTV2Reader::create(readerid_t id, const Ptv::Value* config, const int64_t width, const int64_t height) {
  std::string name;
  if (Parse::populateString("NTV2 reader", *config, "name", name, true) != VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Missing NTV2 device name (\"name\" field) in the configuration." << std::endl;
    return nullptr;
  }

  int deviceIndex = 0;
  if (Parse::populateInt("NTV2 reader", *config, "device", deviceIndex, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "NTV2 device index (\"device\") couldn't be retrieved. Aborting." << std::endl;
    return nullptr;
  }

  ULWord channelNumber = 1;
  if (Parse::populateInt("NTV2 reader", *config, "channel", channelNumber, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "NTV2 channel index (\"channel\") couldn't be retrieved. Aborting." << std::endl;
    return nullptr;
  }

  bool audio = false;
  Parse::populateBool("Audio Capture Switch", *config, "audio", audio, false);

  FrameRate frameRate;
  if (!config->has("frame_rate")) {
    Logger::get(Logger::Error) << "Frame rate (\"frame_rate\") couldn't be retrieved. Aborting." << std::endl;
    return nullptr;
  } else {
    const Ptv::Value* fpsConf = config->has("frame_rate");
    if ((Parse::populateInt("NTV2 reader", *fpsConf, "num", frameRate.num, false) !=
         VideoStitch::Parse::PopulateResult_Ok) ||
        (Parse::populateInt("NTV2 reader", *fpsConf, "den", frameRate.den, false) !=
         VideoStitch::Parse::PopulateResult_Ok)) {
      Logger::get(Logger::Error) << "Frame rate (\"frame_rate\") couldn't be retrieved. Aborting." << std::endl;
      return nullptr;
    }
  }

  bool interlaced;
  if (Parse::populateBool("NTV2 reader", *config, "interleaved", interlaced, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "NTV2 interlacing (\"interleaved\") couldn't be retrieved. Aborting." << std::endl;
    return nullptr;
  }

  NTV2Reader* reader = new NTV2Reader(id, width, height, (UWord)deviceIndex, audio,
                                      ::GetNTV2ChannelForIndex(channelNumber), frameRate, interlaced);
  const AJAStatus status = reader->init();
  if (AJA_SUCCESS(status)) {
    return reader;
  } else {
    delete reader;
    return nullptr;
  }
}

// ------------ Initialization -------------------
NTV2Reader::NTV2Reader(readerid_t id, const int64_t width, const int64_t height, const UWord deviceIndex,
                       const bool withAudio, const NTV2Channel channel, FrameRate fps, bool interlaced)
    : Reader(id),
      VideoReader(width, height, 2 * width * height, UYVY, Host, fps, 0, NO_LAST_FRAME,
                  false /* not a procedural reader */, nullptr),
      AudioReader(Audio::STEREO, Audio::SamplingRate::SR_48000, Audio::SamplingDepth::INT32),
      producerThread(nullptr),
      deviceIndex(deviceIndex),
      withAudio(withAudio),
      inputChannel(channel),
      frameRate(NTV2_FRAMERATE_UNKNOWN),
      audioSystem(NTV2_AUDIOSYSTEM_1),
      startFrameId(-1),
      timeCodeSource(NTV2_TCSOURCE_DEFAULT),
      interlaced(interlaced),
      AJAStop(false),
      globalQuit(false),
      videoBufferSize(0),
      videoTS(-1),
      audioTS(0),
      nbSamplesRead(0),
      audioBufferSize(0) {
  memset(aVHostBuffer, 0, sizeof(aVHostBuffer));
  device = NTV2Device::getDevice(deviceIndex);
  audioBuff.reserve(AUDIOSIZE_MAX / sizeof(ajasample_t) / NB_AJA_CHANNELS *
                    Audio::getNbChannelsFromChannelLayout(AudioReader::getSpec().layout));
}

NTV2Reader::~NTV2Reader() {
  //	Stop my capture and consumer threads, then destroy them...
  quit();
  delete producerThread;
  producerThread = nullptr;

  if (device) {
    //	Unsubscribe from input vertical event...
    device->device.UnsubscribeInputVerticalEvent(inputChannel);
  }

  //	Free all the buffers in the ring
  for (unsigned bufferNdx = 0; bufferNdx < CIRCULAR_BUFFER_SIZE; ++bufferNdx) {
    if (aVHostBuffer[bufferNdx].videoBuffer) {
      delete aVHostBuffer[bufferNdx].videoBuffer;
    }
    if (aVHostBuffer[bufferNdx].audioBuffer) {
      delete aVHostBuffer[bufferNdx].audioBuffer;
    }
  }
}

void NTV2Reader::quit() {
  globalQuit = true;
  noSignal = true;
  AJAStop = true;
  if (producerThread) {
    while (producerThread->Active()) {
      AJATime::Sleep(10);
    }
  }

  delete frameRateVS;
}

// -------------------- Initialization

AJAStatus NTV2Reader::init() {
  AJAStatus status(AJA_STATUS_SUCCESS);

  if (!device) {
    return AJA_STATUS_DISABLED;
  }
  // Sometimes other applications disable some or all of the frame buffers, so turn on ours now
  device->device.EnableChannel(inputChannel);
  inputSource = ::NTV2ChannelToInputSource(inputChannel);

  if (NTV2_INPUT_SOURCE_IS_SDI(inputSource)) {
    timeCodeSource = ::NTV2ChannelToTimecodeIndex(inputChannel);
  } else if (NTV2_INPUT_SOURCE_IS_ANALOG(inputSource)) {
    timeCodeSource = NTV2_TCSOURCE_LTC1;
  } else {
    timeCodeSource = NTV2_TCSOURCE_DEFAULT;
  }

  // Set up the video, audio and routing
  displayMode = DisplayMode(VideoReader::getSpec().width, VideoReader::getSpec().height, interlaced,
                            VideoReader::getSpec().frameRate);
  videoFormat = vs2ajaDisplayFormat(displayMode);
  Logger::get(Logger::Info) << "Aja used video format: " << NTV2VideoFormatToString(videoFormat) << std::endl;
  if (!Is4KFormat(videoFormat)) {
    setupVideo(inputChannel);
    routeInputSignal(inputChannel);
  } else {
    if (inputChannel == NTV2_CHANNEL1) {
      setupVideo(NTV2_CHANNEL1);
      routeInputSignal(NTV2_CHANNEL1);
      setupVideo(NTV2_CHANNEL2);
      routeInputSignal(NTV2_CHANNEL2);
      setupVideo(NTV2_CHANNEL3);
      routeInputSignal(NTV2_CHANNEL3);
      setupVideo(NTV2_CHANNEL4);
      routeInputSignal(NTV2_CHANNEL4);
    } else if (inputChannel == NTV2_CHANNEL5) {
      setupVideo(NTV2_CHANNEL5);
      routeInputSignal(NTV2_CHANNEL5);
      setupVideo(NTV2_CHANNEL6);
      routeInputSignal(NTV2_CHANNEL6);
      setupVideo(NTV2_CHANNEL7);
      routeInputSignal(NTV2_CHANNEL7);
      setupVideo(NTV2_CHANNEL8);
      routeInputSignal(NTV2_CHANNEL8);
    }
  }
  status = setupAudio();
  if (AJA_FAILURE(status)) {
    return status;
  }

  // setup AutoCirculate
  setupHostBuffers();
  device->device.AutoCirculateStop(inputChannel);
  {
    std::unique_lock<std::mutex> l(audioBuffMutex);
    device->device.AutoCirculateInitForInput(
        inputChannel, 7,                                     //  Number of frames to circulate
        withAudio ? audioSystem : NTV2_AUDIOSYSTEM_INVALID,  //  Which audio system (if any)?
        AUTOCIRCULATE_WITH_RP188);                           //  Include timecode
  }
  mInputTransfer.acFrameBufferFormat = NTV2_FBF_8BIT_YCBCR;

  frameRateVS = new FrameRate(displayMode.framerate.num, displayMode.framerate.den);

  run();

  return AJA_STATUS_SUCCESS;
}

AJAStatus NTV2Reader::setupVideo(NTV2Channel channel) {
  // Set the input video format regarding conf
  displayMode = DisplayMode(VideoReader::getSpec().width, VideoReader::getSpec().height, interlaced,
                            VideoReader::getSpec().frameRate);
  videoFormat = vs2ajaDisplayFormat(displayMode);
  if (videoFormat == NTV2_FORMAT_UNKNOWN) {
    Logger::get(Logger::Error) << "Unknown format set" << endl;
    return AJA_STATUS_NOINPUT;
  }
  device->device.SetVideoFormat(videoFormat, false, false, channel);
  device->device.SetReference(::NTV2InputSourceToReferenceSource(inputSource));
  device->device.SetRP188Mode(channel, NTV2_RP188_INPUT);
  device->device.SetFrameBufferFormat(channel, NTV2_FBF_8BIT_YCBCR);
  device->device.GetFrameRate(&frameRate, channel);

  // Enable and subscribe to the interrupts for the channel to be used...
  device->device.EnableInputInterrupt(channel);
  device->device.SubscribeInputVerticalEvent(channel);

  return AJA_STATUS_SUCCESS;
}

AJAStatus NTV2Reader::setupAudio() {
  // Have the audio system capture audio from the designated device input...
  NTV2EmbeddedAudioInput embeddedAudio;
  if (!device->device.GetEmbeddedAudioInput(embeddedAudio)) {
    return AJA_STATUS_FALSE;
  }
  NTV2AudioSource audioSource = ::NTV2InputSourceToAudioSource(inputSource);
  if (!device->device.SetAudioSystemInputSource(audioSystem, audioSource, embeddedAudio)) {
    return AJA_STATUS_FALSE;
  }
  if (!device->device.SetNumberAudioChannels(NB_AJA_CHANNELS, audioSystem)) {
    return AJA_STATUS_FALSE;
  }
  if (!device->device.SetAudioRate(NTV2_AUDIO_48K, audioSystem)) {
    return AJA_STATUS_FALSE;
  }
  if (!device->device.SetAudioBufferSize(NTV2_AUDIO_BUFFER_BIG, audioSystem)) {
    return AJA_STATUS_FALSE;
  }
  return AJA_STATUS_SUCCESS;
}

void NTV2Reader::setupHostBuffers() {
  // Let my circular buffer know when it's time to quit...
  aVCircularBuffer.SetAbortFlag(&AJAStop);

  bool vancEnabled = false;
  bool wideVanc = false;
  device->device.GetEnableVANCData(&vancEnabled, &wideVanc);
  videoBufferSize = GetVideoWriteSize(videoFormat, NTV2_FBF_8BIT_YCBCR, vancEnabled, wideVanc);
  audioBufferSize = AUDIOSIZE_MAX;

  //	Allocate and add each in-host AVDataBuffer to my circular buffer member variable...
  for (unsigned bufferNdx = 0; bufferNdx < CIRCULAR_BUFFER_SIZE; ++bufferNdx) {
    aVHostBuffer[bufferNdx].videoBuffer = reinterpret_cast<uint32_t*>(new uint8_t[videoBufferSize]);
    aVHostBuffer[bufferNdx].videoBufferSize = videoBufferSize;
    aVHostBuffer[bufferNdx].audioBuffer = reinterpret_cast<uint32_t*>(new uint8_t[audioBufferSize]);
    aVHostBuffer[bufferNdx].audioBufferSize = audioBufferSize;
    aVCircularBuffer.Add(&aVHostBuffer[bufferNdx]);
  }
}

void NTV2Reader::routeInputSignal(NTV2Channel channel) {
  NTV2CrosspointID inputIdentifier;
  NTV2InputCrosspointID fbfInputSelect;

  switch (channel) {
    default:
    case NTV2_CHANNEL1:
      inputIdentifier = NTV2_XptSDIIn1;
      break;
    case NTV2_CHANNEL2:
      inputIdentifier = NTV2_XptSDIIn2;
      break;
    case NTV2_CHANNEL3:
      inputIdentifier = NTV2_XptSDIIn3;
      break;
    case NTV2_CHANNEL4:
      inputIdentifier = NTV2_XptSDIIn4;
      break;
    case NTV2_CHANNEL5:
      inputIdentifier = NTV2_XptSDIIn5;
      break;
    case NTV2_CHANNEL6:
      inputIdentifier = NTV2_XptSDIIn6;
      break;
    case NTV2_CHANNEL7:
      inputIdentifier = NTV2_XptSDIIn7;
      break;
    case NTV2_CHANNEL8:
      inputIdentifier = NTV2_XptSDIIn8;
      break;
  }

  switch (channel) {
    default:
    case NTV2_CHANNEL1:
      fbfInputSelect = NTV2_XptFrameBuffer1Input;
      break;
    case NTV2_CHANNEL2:
      fbfInputSelect = NTV2_XptFrameBuffer2Input;
      break;
    case NTV2_CHANNEL3:
      fbfInputSelect = NTV2_XptFrameBuffer3Input;
      break;
    case NTV2_CHANNEL4:
      fbfInputSelect = NTV2_XptFrameBuffer4Input;
      break;
    case NTV2_CHANNEL5:
      fbfInputSelect = NTV2_XptFrameBuffer5Input;
      break;
    case NTV2_CHANNEL6:
      fbfInputSelect = NTV2_XptFrameBuffer6Input;
      break;
    case NTV2_CHANNEL7:
      fbfInputSelect = NTV2_XptFrameBuffer7Input;
      break;
    case NTV2_CHANNEL8:
      fbfInputSelect = NTV2_XptFrameBuffer8Input;
      break;
  }

  router.AddConnection(fbfInputSelect, inputIdentifier);

  //	Disable SDI output from the SDI input being used,
  //	but only if the device supports bi-directional SDI,
  //	and only if the input being used is an SDI input...
  if (::NTV2DeviceHasBiDirectionalSDI(NTV2Device::getDevice(deviceIndex)->deviceID)) {
    device->device.SetSDITransmitEnable(channel, false);
  }

  // Replace the device's current signal routing with this new one...
  device->device.ApplySignalRoute(router);
}

AJAStatus NTV2Reader::run() {
  // Check that there's a valid input to capture
  if (NTV2DeviceHasBiDirectionalSDI(NTV2Device::getDevice(deviceIndex)->deviceID)) {
    device->device.SetSDITransmitEnable(inputChannel, false);
  }

  if (device->device.GetInputVideoFormat(inputSource) == NTV2_FORMAT_UNKNOWN) {
    Logger::get(Logger::Warning) << "No video signal present on the input connector" << std::endl;
  }

  // Start the capture threads
  startProducerThread();

  return AJA_STATUS_SUCCESS;
}

// -------------------- Stitcher thread

ReadStatus NTV2Reader::readFrame(mtime_t& date, unsigned char* video) {
  // Wait for the next frame to become ready to read out
  if (noSignal) {
    // We assume there is no signal but there's something to read. In this case will be an empty frame.
    // TODO: manage no signal status for IO to avoid blocking the reading process.
    date = 0;
    return ReadStatus::OK();
  }

  AVDataBuffer* frameData = aVCircularBuffer.StartConsumeNextBuffer();
  if (frameData) {
    ULWord frames, seconds, minutes, hours;
    frameData->rp188.GetRP188Secs(seconds);
    frameData->rp188.GetRP188Frms(frames);
    frameData->rp188.GetRP188Mins(minutes);
    frameData->rp188.GetRP188Hrs(hours);
    memcpy(video, frameData->videoBuffer, frameData->videoBufferSize);
    date = (mtime_t)(1000000 * (uint64_t(seconds) + 60 * uint64_t(minutes) + 3600 * uint64_t(hours))) +
           frameRateVS->frameToTimestamp(frames);
    if (date <= videoTS) {
      Logger::get(Logger::Verbose) << "NTV2 timecode not monotonic, inventing." << std::endl;
      date = videoTS + frameRateVS->frameToTimestamp(1);
    }
    videoTS = date;
    // Now release and recycle the buffer
    aVCircularBuffer.EndConsumeNextBuffer();
  }

  return ReadStatus::OK();
}

Status NTV2Reader::seekFrame(frameid_t) { return VideoStitch::Status::OK(); }

Status NTV2Reader::seekFrame(mtime_t) { return VideoStitch::Status::OK(); }

ReadStatus NTV2Reader::readSamples(size_t nbSamples, Audio::Samples& samples) {
  // 24 bits PCM samples, using the 3 most significant bytes of an integer
  // Two modes : 8 channels or 16 channels
  // Samples are interleaved
  uint16_t nbChan = Audio::getNbChannelsFromChannelLayout(AudioReader::getSpec().layout);
  std::unique_lock<std::mutex> lk(audioBuffMutex);
  size_t actuallyRead = std::min(nbSamples * nbChan, audioBuff.size());

  Audio::Samples::data_buffer_t raw;
  raw[0] = new uint8_t[sizeof(ajasample_t) * actuallyRead];
  memcpy(raw[0], audioBuff.data(), actuallyRead * sizeof(ajasample_t));
  audioBuff.erase(audioBuff.begin(), audioBuff.begin() + actuallyRead);

  time_t currentTS =
      audioTS + nbSamplesRead * 1000000 / uint64_t(getIntFromSamplingRate(AudioReader::getSpec().sampleRate));
  samples = Audio::Samples(AudioReader::getSpec().sampleRate, AudioReader::getSpec().sampleDepth,
                           AudioReader::getSpec().layout, currentTS, raw, actuallyRead / nbChan);

  if (audioBuff.size() != 0) {
    nbSamplesRead += uint64_t(actuallyRead / nbChan);
  }

  return ReadStatus::OK();
}

bool NTV2Reader::eos() { return false; }

size_t NTV2Reader::available() { return audioBuff.size() / sizeof(ajasample_t); }

// -------------------- Producer thread

void NTV2Reader::startProducerThread() {
  //	Create and start the capture thread...
  producerThread = new AJAThread();
  producerThread->Attach(producerThreadStatic, this);
  producerThread->SetPriority(AJA_ThreadPriority_High);
  producerThread->SetThreadName("AJAReaderThread");
  producerThread->Start();
}

void NTV2Reader::producerThreadStatic(AJAThread* thread, void* ctx) {
  NTV2Reader* reader = reinterpret_cast<NTV2Reader*>(ctx);
  reader->captureFrames();
}

bool NTV2Reader::InputSignalHasTimecode(void) const {
  const ULWord regNum(GetRP188RegisterForInput(inputSource));
  ULWord regValue = 0;

  // Bit 16 of the RP188 DBB register will be set if there is timecode embedded in the input signal...
  return (regNum && device->device.ReadRegister(regNum, &regValue) && regValue & BIT(16));
}

void NTV2Reader::captureFrames() {
  device->device.AutoCirculateStart(inputChannel);
  while (!globalQuit) {
    AUTOCIRCULATE_STATUS acStatus;
    device->device.AutoCirculateGetStatus(inputChannel, acStatus);
    if (acStatus.IsRunning() && acStatus.HasAvailableInputFrame()) {
      noSignal = false;

      //  At this point, there's at least one fully-formed frame available in the device's
      //  frame buffer to transfer to the host. Reserve an AVDataBuffer to "produce", and
      //  use it in the next transfer from the device->device...
      AVDataBuffer* captureData(aVCircularBuffer.StartProduceNextBuffer());
      if (!captureData) {
        continue;
      }

      mInputTransfer.SetBuffers(captureData->videoBuffer, captureData->videoBufferSize, captureData->audioBuffer,
                                captureData->audioBufferSize, captureData->ancBuffer, captureData->ancBufferSize);

      //  Do the transfer from the device into our host AVDataBuffer...
      device->device.AutoCirculateTransfer(inputChannel, mInputTransfer);
      captureData->audioBufferSize = mInputTransfer.acTransferStatus.acAudioTransferSize;

      NTV2_RP188 defaultTC;
      if (NTV2_IS_VALID_TIMECODE_INDEX(timeCodeSource) && InputSignalHasTimecode()) {
        //	Use the timecode that was captured by AutoCirculate...
        mInputTransfer.GetInputTimeCode(defaultTC, timeCodeSource);
      }
      if (defaultTC.IsValid()) {
        captureData->rp188 = defaultTC;  //	Stuff it in the captureData
      } else {
        // Invent a timecode (based on frame count)...
        const NTV2FrameRate ntv2FrameRate = ::GetNTV2FrameRateFromVideoFormat(videoFormat);
        const TimecodeFormat tcFormat = NTV2FrameRate2TimecodeFormat(ntv2FrameRate);
        const CRP188 inventedTC(mInputTransfer.acTransferStatus.acFramesProcessed, 0, 0, 10, tcFormat);
        captureData->rp188 = inventedTC;
      }

      if (captureData->audioBufferSize <= AUDIOSIZE_MAX) {
        std::unique_lock<std::mutex> lk(audioBuffMutex);
        ULWord frames, seconds, minutes, hours;
        captureData->rp188.GetRP188Secs(seconds);
        captureData->rp188.GetRP188Frms(frames);
        captureData->rp188.GetRP188Mins(minutes);
        captureData->rp188.GetRP188Hrs(hours);

        if (audioBuff.size() == 0) {
          nbSamplesRead = 0;
          audioTS = (mtime_t)(1000000 * (uint64_t(seconds) + 60 * uint64_t(minutes) + 3600 * uint64_t(hours))) +
                    frameRateVS->frameToTimestamp(frames);
        }

        // Convert signal to stereo only as we support only stereo for the moment
        int nbSamplesPerChannel = captureData->audioBufferSize / sizeof(ajasample_t) / NB_AJA_CHANNELS;
        for (int i = 0; i < nbSamplesPerChannel; i++) {
          audioBuff.push_back(captureData->audioBuffer[i * NB_AJA_CHANNELS]);
          audioBuff.push_back(captureData->audioBuffer[i * NB_AJA_CHANNELS + 1]);
        }
      }

      //  Signal that we're done "producing" the frame, making it available for future "consumption"...
      aVCircularBuffer.EndProduceNextBuffer();
      //  if A/C running and frame(s) are available for transfer
    } else {
      if (!device->device.WaitForInputVerticalInterrupt(inputChannel)) {
        noSignal = true;
      }
    }
  }  //  loop til quit signaled

  //  Stop AutoCirculate...
  device->device.AutoCirculateStop(inputChannel);
  aVCircularBuffer.Clear();
}

}  // namespace Input
}  // namespace VideoStitch
