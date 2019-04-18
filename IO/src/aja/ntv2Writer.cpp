// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ntv2Writer.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/frame.hpp"

#include "ajastuff/common/timecode.h"
#include "ajastuff/system/systemtime.h"
#include "ajastuff/system/process.h"
#include "ajastuff/system/thread.h"
#include "ntv2utils.h"

#include <cmath>

#include <future>

/**
  @brief  The maximum number of bytes of 48KHz audio that can be transferred for a single frame.
      Worst case, assuming 16 channels of audio (max), 4 bytes per sample, and 67 msec per frame
      (assuming the lowest possible frame rate of 14.98 fps)...
      48,000 samples per second requires 3,204 samples x 4 bytes/sample x 16 = 205,056 bytes
      201K will suffice, with 768 bytes to spare
**/
static const uint32_t AUDIOBYTES_MAX_48K(201 * 1024);

/**
  @brief  The maximum number of bytes of 96KHz audio that can be transferred for a single frame.
      Worst case, assuming 16 channels of audio (max), 4 bytes per sample, and 67 msec per frame
      (assuming the lowest possible frame rate of 14.98 fps)...
      96,000 samples per second requires 6,408 samples x 4 bytes/sample x 16 = 410,112 bytes
      401K will suffice, with 512 bytes to spare
**/
static const uint32_t AUDIOBYTES_MAX_96K(401 * 1024);

static const std::string kAjaOutTag("AJAoutput");

namespace VideoStitch {
namespace Output {

Output* NTV2Writer::create(const Ptv::Value& config, const std::string& name, const char*, unsigned width,
                           unsigned height, FrameRate framerate) {
  // Parse device index
  int deviceIndex = 0;
  if (Parse::populateInt("NTV2 writer", config, "device", deviceIndex, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::error(kAjaOutTag) << "NTV2 device index (\"device\") couldn't be retrieved. Aborting." << std::endl;
    return nullptr;
  }

  // Parse video channel
  ULWord channelNumber = 1;
  if (Parse::populateInt("NTV2 writer", config, "channel", channelNumber, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::error(kAjaOutTag) << "NTV2 channel index (\"channel\") couldn't be retrieved. Aborting." << std::endl;
    return nullptr;
  }

  // Parse display mode
  unsigned aja_width = width;
  Parse::populateInt("NTV2 writer", config, "width", aja_width, false);
  unsigned aja_height = height;
  Parse::populateInt("NTV2 writer", config, "height", aja_height, false);
  bool aja_interleaved = false;
  Parse::populateBool("NTV2 writer", config, "interleaved", aja_interleaved, false);
  bool aja_psf = false;
  Parse::populateBool("NTV2 writer", config, "psf", aja_psf, false);

  // Parse offset in the display mode
  unsigned offset_x = 0;
  unsigned offset_y = 0;
  Parse::populateInt("NTV2 writer", config, "offset_x", offset_x, false);
  Parse::populateInt("NTV2 writer", config, "offset_y", offset_y, false);

  // Check the panorama fits in the display mode
  if ((width + offset_x) > aja_width || (height + offset_y) > aja_height) {
    Logger::error(kAjaOutTag) << "NTV2 writer width and height should be smaller than display mode."
                                 " Output size: "
                              << aja_width << "x" << aja_height << " Panorama size: " << width << "x" << height
                              << " Panorama offset: " << offset_x << "x" << offset_y << " Aborting." << std::endl;
    return nullptr;
  }

  // Parse frame rate
  FrameRate aja_frameRate = framerate;
  const Ptv::Value* fpsConf = config.has("frame_rate");
  if (fpsConf) {
    if ((Parse::populateInt("NTV2 writer", *fpsConf, "num", aja_frameRate.num, true) !=
         VideoStitch::Parse::PopulateResult_Ok) ||
        (Parse::populateInt("NTV2 writer", *fpsConf, "den", aja_frameRate.den, true) !=
         VideoStitch::Parse::PopulateResult_Ok)) {
      Logger::error(kAjaOutTag) << "NTV2 frame rate (\"frame_rate\") couldn't be retrieved. Aborting." << std::endl;
      return nullptr;
    }
  }

  // Parse audio
  bool audio = false;
  Parse::populateBool("Audio Player Switch", config, "audio", audio, false);

  // Verify the video format is correct
  const NTV2VideoFormat videoFormat =
      vs2ajaDisplayFormat(DisplayMode(aja_width, aja_height, aja_interleaved, aja_frameRate, aja_psf));
  if (videoFormat == NTV2_FORMAT_UNKNOWN) {
    Logger::error(kAjaOutTag) << "NTV2 Video format " << ::NTV2VideoFormatToString(videoFormat)
                              << " not supported. Aborting." << std::endl;
    return nullptr;
  }

  // Verify the video channel is correct
  const NTV2Channel videoChannel = GetNTV2ChannelForIndex(channelNumber);
  if (videoChannel == NTV2_CHANNEL_INVALID) {
    Logger::error(kAjaOutTag) << "NTV2 Video channel" << ::NTV2ChannelToString(videoChannel)
                              << " not supported. Aborting." << std::endl;
    return nullptr;
  }

  if (!NTV2Device::getDevice(deviceIndex)) {
    Logger::get(Logger::Error) << "NTV2 Video device" << deviceIndex << " not found. Aborting." << std::endl;
    return nullptr;
  }

  // Check for channels configuration (4K setups)
  if (!checkChannelConf(aja_width, aja_height, channelNumber)) {
    return nullptr;
  }

  NTV2Writer* writer = new NTV2Writer(name, deviceIndex, audio, videoChannel, videoFormat, width, height, offset_x,
                                      offset_y, aja_frameRate);
  const AJAStatus status = writer->_init();
  if (AJA_SUCCESS(status)) {
    return writer;
  } else {
    delete writer;
    return nullptr;
  }
}

NTV2Writer::NTV2Writer(const std::string& name, const UWord deviceIndex, const bool withAudio,
                       const NTV2Channel channel, const NTV2VideoFormat format, unsigned width, unsigned height,
                       unsigned offset_x, unsigned offset_y, FrameRate framerate)
    : Output(name),
      VideoWriter(width, height, framerate, UYVY),
      AudioWriter(Audio::SamplingRate::SR_48000, Audio::SamplingDepth::INT32, Audio::ChannelLayout::STEREO),
      consumerThread(nullptr),
      producerThread(nullptr),
      deviceIndex(deviceIndex),
      outputNb(1),
      withAudio(withAudio),
      outputChannel(channel),
      videoFormat(format),
      audioSystem(NTV2_AUDIOSYSTEM_1),
      nbAJAChannels(0),
      currentSample(0),
      globalQuit(false),
      AJAStop(false),
      videoBufferSize(0),
      audioBufferSize(0),
      nbSamplesPerFrame(framerate.den * Audio::getIntFromSamplingRate(getSamplingRate()) / framerate.num),
      videoBuffer(CIRCULAR_BUFFER_SIZE * getExpectedFrameSize()),
      audioBuffer(CIRCULAR_BUFFER_SIZE * nbSamplesPerFrame),  // We support stereo only for the moment
      doLevelConversion(false),
      doMultiChannel(true),
      offset_x(offset_x),
      offset_y(offset_y),
      preRollFrames(0),
      producedFrames(0) {
  Logger::info(kAjaOutTag) << "AJA output used video format: " << NTV2VideoFormatToString(videoFormat) << " "
                           << videoFormat << std::endl;
  memset(aVHostBuffer, 0, sizeof(aVHostBuffer));
  initSinTableFor16Channels();
}

NTV2Writer::~NTV2Writer() {
  // Stop my playout and producer threads, then destroy them...
  quit();
  NTV2Device::getDevice(deviceIndex)->device.UnsubscribeOutputVerticalEvent(outputChannel);
  NTV2Device::getDevice(deviceIndex)->device.AutoCirculateFlush(outputChannel);
  NTV2Device::getDevice(deviceIndex)->device.DisableChannel(outputChannel);

  for (unsigned int ndx = 0; ndx < CIRCULAR_BUFFER_SIZE; ++ndx) {
    if (aVHostBuffer[ndx].videoBuffer) {
      delete aVHostBuffer[ndx].videoBuffer;
      aVHostBuffer[ndx].videoBuffer = nullptr;
    }
    if (aVHostBuffer[ndx].audioBuffer) {
      delete aVHostBuffer[ndx].audioBuffer;
      aVHostBuffer[ndx].audioBuffer = nullptr;
    }
  }  //  for each buffer in the ring
}

void NTV2Writer::pushVideo(const Frame& videoFrame) {
  std::unique_lock<std::mutex> lk(frameMutex);

  int64_t nbFramesInVideoBuff = videoBuffer.size() / getExpectedFrameSize();
  Logger::debug(kAjaOutTag) << "Push VIDEO : video buff size " << nbFramesInVideoBuff << " frames at ts "
                            << videoFrame.pts << std::endl;
  videoBuffer.push((uint8_t*)videoFrame.planes[0], getExpectedFrameSize());
}

void NTV2Writer::pushAudio(Audio::Samples& audioSamples) {
  if (withAudio) {
    std::unique_lock<std::mutex> lk(frameMutex);
    int nbVSChannels = Audio::getNbChannelsFromChannelLayout(audioSamples.getChannelLayout());
    const int32_t* audioInData = (const int32_t*)audioSamples.getSamples()[0];
    for (int i = 0; i < audioSamples.getNbOfSamples(); i++) {
      for (int c = 0; c < nbVSChannels; c++) {
        audioBuffer.push(*audioInData);
        audioInData++;
      }
      for (int c = nbVSChannels; c < int(nbAJAChannels); c++) {
        audioBuffer.push(0);
      }
    }
    Logger::debug(kAjaOutTag) << "Push AUDIO : audio buffer size " << audioBuffer.size() / nbAJAChannels / 48. << " ms "
                              << " at ts " << audioSamples.getTimestamp() << std::endl;
  }
}

void NTV2Writer::quit() {
  //  Set the global 'quit' flag, and wait for the threads to go inactive...
  globalQuit = true;
  AJAStop = true;

  if (producerThread) {
    while (producerThread->Active()) {
      AJATime::Sleep(10);
    }
  }
  producerThread->Stop();

  if (consumerThread) {
    while (consumerThread->Active()) {
      AJATime::Sleep(10);
    }
  }
  consumerThread->Stop();
}

AJAStatus NTV2Writer::_init() {
  AJAStatus status(AJA_STATUS_SUCCESS);

  if (::NTV2DeviceCanDoMultiFormat(NTV2Device::getDevice(deviceIndex)->deviceID) && doMultiChannel) {
    NTV2Device::getDevice(deviceIndex)->device.SetMultiFormatMode(true);
  } else if (::NTV2DeviceCanDoMultiFormat(NTV2Device::getDevice(deviceIndex)->deviceID)) {
    NTV2Device::getDevice(deviceIndex)->device.SetMultiFormatMode(false);
  }

  if (outputNb != 1) {
    if (!::NTV2DeviceCanDo4KVideo(NTV2Device::getDevice(deviceIndex)->deviceID)) {
      Logger::info(kAjaOutTag) << "AJA device does not supports 4K" << std::endl;
      return AJA_STATUS_UNSUPPORTED;
    }
  }

  // Set up the video, audio and routing
  if (!Is4KFormat(videoFormat)) {
    setupVideo(outputChannel);
    routeOutputSignal(outputChannel);
  } else {
    if (outputChannel == NTV2_CHANNEL1) {
      setupVideo(NTV2_CHANNEL1);
      routeOutputSignal(NTV2_CHANNEL1);
      setupVideo(NTV2_CHANNEL2);
      routeOutputSignal(NTV2_CHANNEL2);
      setupVideo(NTV2_CHANNEL3);
      routeOutputSignal(NTV2_CHANNEL3);
      setupVideo(NTV2_CHANNEL4);
      routeOutputSignal(NTV2_CHANNEL4);
    } else if (outputChannel == NTV2_CHANNEL5) {
      setupVideo(NTV2_CHANNEL5);
      routeOutputSignal(NTV2_CHANNEL5);
      setupVideo(NTV2_CHANNEL6);
      routeOutputSignal(NTV2_CHANNEL6);
      setupVideo(NTV2_CHANNEL7);
      routeOutputSignal(NTV2_CHANNEL7);
      setupVideo(NTV2_CHANNEL8);
      routeOutputSignal(NTV2_CHANNEL8);
    }
  }
  status = setupAudio();
  if (AJA_FAILURE(status)) {
    return status;
  }

  //  Set up the circular buffers, the device signal routing, and playout AutoCirculate...
  setupHostBuffers();
  setupOutputAutoCirculate();
  run();
  return AJA_STATUS_SUCCESS;
}

AJAStatus NTV2Writer::setupVideo(NTV2Channel channel) {
  if (videoFormat == NTV2_FORMAT_UNKNOWN) {
    NTV2Device::getDevice(deviceIndex)->device.GetVideoFormat(&videoFormat, NTV2_CHANNEL1);
  }

  if (!::NTV2DeviceCanDoVideoFormat(NTV2Device::getDevice(deviceIndex)->deviceID, videoFormat)) {
    Logger::error(kAjaOutTag) << "AJA device cannot handle display format: '" << ::NTV2VideoFormatToString(videoFormat)
                              << "'" << std::endl;
    return AJA_STATUS_UNSUPPORTED;
  }

  //  Configure the device to handle the requested video format...
  NTV2Device::getDevice(deviceIndex)->device.SetVideoFormat(videoFormat, false, false, channel);
  NTV2Device::getDevice(deviceIndex)->device.SetFrameBufferFormat(channel, NTV2_FBF_8BIT_YCBCR);
  NTV2Device::getDevice(deviceIndex)->device.SetReference(NTV2_REFERENCE_FREERUN);
  NTV2Device::getDevice(deviceIndex)->device.EnableChannel(channel);
  NTV2Device::getDevice(deviceIndex)->device.SetEnableVANCData(false);  //  No VANC with RGB pixel formats (for now)
  NTV2Device::getDevice(deviceIndex)->device.SetMode(channel, NTV2_MODE_DISPLAY);

  if (outputNb != 1) {
    const int idx = GetIndexForNTV2Channel(channel);
    for (int i = idx; i < idx + outputNb; ++i) {
      const NTV2Channel chan = GetNTV2ChannelForIndex(i);
      NTV2Device::getDevice(deviceIndex)->device.SetFrameBufferFormat(chan, NTV2_FBF_8BIT_YCBCR);
      NTV2Device::getDevice(deviceIndex)->device.EnableChannel(chan);
      NTV2Device::getDevice(deviceIndex)->device.SetMode(chan, NTV2_MODE_DISPLAY);
      NTV2Device::getDevice(deviceIndex)->device.SetVideoFormat(videoFormat, false, false, chan);
    }
  }

  //  Subscribe the output interrupt -- it's enabled by default...
  NTV2Device::getDevice(deviceIndex)->device.SubscribeOutputVerticalEvent(channel);
  if (outputDestHasRP188BypassEnabled()) {
    disableRP188Bypass();
  }
  return AJA_STATUS_SUCCESS;
}

AJAStatus NTV2Writer::setupAudio() {
  static const NTV2AudioSystem channelToAudioSystem[] = {NTV2_AUDIOSYSTEM_1, NTV2_AUDIOSYSTEM_2, NTV2_AUDIOSYSTEM_3,
                                                         NTV2_AUDIOSYSTEM_4, NTV2_AUDIOSYSTEM_5, NTV2_AUDIOSYSTEM_6,
                                                         NTV2_AUDIOSYSTEM_7, NTV2_AUDIOSYSTEM_8, NTV2_NUM_AUDIOSYSTEMS};

  const uint16_t numberOfAudioChannels(::NTV2DeviceGetMaxAudioChannels(NTV2Device::getDevice(deviceIndex)->deviceID));

  if (::NTV2DeviceGetNumAudioStreams(NTV2Device::getDevice(deviceIndex)->deviceID)) {
    audioSystem = channelToAudioSystem[outputChannel];
  }

  NTV2Device::getDevice(deviceIndex)->device.SetNumberAudioChannels(numberOfAudioChannels, audioSystem);
  nbAJAChannels = uint32_t(numberOfAudioChannels);
  NTV2Device::getDevice(deviceIndex)->device.SetAudioRate(NTV2_AUDIO_48K, audioSystem);
  //  How big should the on-device audio buffer be?   1MB? 2MB? 4MB? 8MB?
  //  For this demo, 4MB will work best across all platforms (Windows, Mac & Linux)...
  NTV2Device::getDevice(deviceIndex)->device.SetAudioBufferSize(NTV2_AUDIO_BUFFER_BIG, audioSystem);

  //  Set up the output audio embedders...
  NTV2Device::getDevice(deviceIndex)->device.SetSDIOutputAudioSystem(outputChannel, audioSystem);
  NTV2Device::getDevice(deviceIndex)->device.SetSDIOutputDS2AudioSystem(outputChannel, audioSystem);

  if (outputNb != 1) {
    const int idx = GetIndexForNTV2Channel(outputChannel);
    for (int i = idx; i < idx + outputNb; ++i) {
      NTV2Channel chan = GetNTV2ChannelForIndex(i);
      NTV2Device::getDevice(deviceIndex)->device.SetSDIOutputAudioSystem(chan, audioSystem);
      NTV2Device::getDevice(deviceIndex)->device.SetSDIOutputDS2AudioSystem(chan, audioSystem);
    }
  }

  //  If the last app using the device left it in end-to-end mode (input passthru),
  //  then loopback must be disabled, or else the output will contain whatever audio
  //  is present in whatever signal is feeding the device's SDI input...
  NTV2Device::getDevice(deviceIndex)->device.SetAudioLoopBack(NTV2_AUDIO_LOOPBACK_OFF, audioSystem);

  return AJA_STATUS_SUCCESS;
}

void NTV2Writer::setupHostBuffers() {
  //  Let my circular buffer know when it's time to quit...
  aVCircularBuffer.SetAbortFlag(&AJAStop);

  //  Calculate the size of the video buffer, which depends on video format, pixel format, and whether VANC is included
  //  or not...
  bool vancEnabled = false;
  bool wideVanc = false;
  // NTV2Device::getDevice(deviceIndex)->device.GetEnableVANCData (&vancEnabled, &wideVanc);
  videoBufferSize = GetVideoWriteSize(videoFormat, NTV2_FBF_8BIT_YCBCR, vancEnabled, wideVanc);

  //  Calculate the size of the audio buffer, which mostly depends on the sample rate...
  NTV2AudioRate audioRate(NTV2_AUDIO_48K);
  NTV2Device::getDevice(deviceIndex)->device.GetAudioRate(audioRate, audioSystem);
  audioBufferSize = (audioRate == NTV2_AUDIO_96K) ? AUDIOBYTES_MAX_96K : AUDIOBYTES_MAX_48K;

  uint32_t numSegments = 1;
  uint32_t hostBytesPerSegment = (uint32_t)getExpectedFrameSize();
  uint32_t deviceBytesPerSegment = videoBufferSize;
  // Configure Optional segmented DMA info, for use with specialized data transfers.
  if ((GetDisplayWidth(videoFormat) != getWidth()) || (GetDisplayHeight(videoFormat) != getHeight())) {
    numSegments = getHeight();
    hostBytesPerSegment = (uint32_t)(getExpectedFrameSize() / getHeight());
    deviceBytesPerSegment =
        GetFormatDescriptor(videoFormat, NTV2_FBF_8BIT_YCBCR, vancEnabled, wideVanc).GetBytesPerRow();
  }
  //  Allocate my buffers...
  for (size_t ndx = 0; ndx < CIRCULAR_BUFFER_SIZE; ++ndx) {
    aVHostBuffer[ndx].videoBuffer = reinterpret_cast<uint32_t*>(new uint8_t[videoBufferSize]);
    aVHostBuffer[ndx].inNumSegments = numSegments;
    aVHostBuffer[ndx].inDeviceBytesPerLine = deviceBytesPerSegment;
    aVHostBuffer[ndx].videoBufferSize = hostBytesPerSegment;
    aVHostBuffer[ndx].audioBuffer = withAudio ? reinterpret_cast<uint32_t*>(new uint8_t[audioBufferSize]) : NULL;
    aVHostBuffer[ndx].audioBufferSize = withAudio ? audioBufferSize : 0;

    if (withAudio) {
      ::memset(aVHostBuffer[ndx].audioBuffer, 0x00, audioBufferSize);
    }

    for (uint32_t i = 0; i < videoBufferSize / 4; ++i) {
      // black UYVY on little endian host (YVYU)
      aVHostBuffer[ndx].videoBuffer[i] = 0x10801080;
    }

    aVCircularBuffer.Add(&aVHostBuffer[ndx]);
  }  //  for each AV buffer in my circular buffer

  // Initialize device buffer with black
  if (numSegments > 1) {
    preRollFrames = CIRCULAR_BUFFER_SIZE;
  }
  for (size_t ndx = 0; ndx < preRollFrames; ++ndx) {
    AVDataBuffer* frameData(aVCircularBuffer.StartProduceNextBuffer());
    //  If no frame is available, wait and try again
    if (!frameData) {
      AJATime::Sleep(10);
      continue;
    }
    aVCircularBuffer.EndProduceNextBuffer();
  }
}

void NTV2Writer::setupOutputAutoCirculate() {
  NTV2Device::getDevice(deviceIndex)->device.AutoCirculateStop(outputChannel);
  NTV2Device::getDevice(deviceIndex)
      ->device.AutoCirculateInitForOutput(outputChannel,
                                          8,  // Request 4 frame buffers
                                          withAudio ? audioSystem : NTV2_AUDIOSYSTEM_INVALID,  //  Which audio system?
                                          AUTOCIRCULATE_WITH_RP188);                           //  Add RP188 timecode!
}

//////////////////////////////////////////////
//	This is where the producer thread starts

void NTV2Writer::startProducerThread() {
  //	Create and start the producer thread...
  producerThread = new AJAThread();
  producerThread->Attach(producerThreadStatic, this);
  producerThread->SetPriority(AJA_ThreadPriority_High);
  producerThread->Start();

}  //	StartProducerThread

//  The playout thread function
void NTV2Writer::producerThreadStatic(AJAThread* pThread, void* pContext) {  //  static
  //  Grab the NTV2Writer instance pointer from the pContext parameter,
  NTV2Writer* writer = reinterpret_cast<NTV2Writer*>(pContext);
  if (writer) {
    writer->produceFrames();
  }
}

void NTV2Writer::produceFrames(void) {
  while (!globalQuit) {
    std::unique_lock<std::mutex> lk(frameMutex);
    //	Copy my pre-made test pattern into my video buffer...
    if (videoBuffer.size() >= (size_t)getExpectedFrameSize() &&
        (withAudio && audioBuffer.size() / nbAJAChannels >= nbSamplesPerFrame || !withAudio)) {
      AVDataBuffer* frameData(aVCircularBuffer.StartProduceNextBuffer());
      //  If no frame is available, wait and try again
      if (!frameData) {
        AJATime::Sleep(10);
        continue;
      }
      videoBuffer.pop((uint8_t*)frameData->videoBuffer, getExpectedFrameSize());
      if (withAudio) {
        uint32_t audioFrameAJASize = nbAJAChannels * nbSamplesPerFrame;
        memset((int32_t*)frameData->audioBuffer, 0, audioFrameAJASize * sizeof(uint32_t));  // maybe not useful to test
        frameData->audioBufferSize = audioFrameAJASize * sizeof(int32_t);
        audioBuffer.pop((int32_t*)frameData->audioBuffer, audioFrameAJASize);
      }
      //	Signal that I'm done producing the buffer -- it's now available for playout...
      producedFrames++;
      aVCircularBuffer.EndProduceNextBuffer();
    }
  }
}

//////////////////////////////////////////////
//  This is where the play thread starts

void NTV2Writer::startConsumerThread() {
  //  Create and start the playout thread...
  consumerThread = new AJAThread();
  consumerThread->Attach(consumerThreadStatic, this);
  consumerThread->SetPriority(AJA_ThreadPriority_High);
  consumerThread->SetThreadName("AJAWriterThread");
  consumerThread->Start();

}  //  StartConsumerThread

//  The playout thread function
void NTV2Writer::consumerThreadStatic(AJAThread* pThread, void* pContext) {  //  static
  //  Grab the NTV2Writer instance pointer from the pContext parameter,
  //  then call its playFrames method...
  NTV2Writer* writer = reinterpret_cast<NTV2Writer*>(pContext);
  if (writer) {
    writer->playFrames();
  }
}  //  ConsumerThreadStatic

void NTV2Writer::initSinTableFor16Channels(void) {
  nbSamplesInWavForm = 48000 / 480 * 16;
  for (uint32_t i = 0; i < 48000 / 480; ++i) {
    for (uint32_t c = 0; c < 16; ++c) {
      int32_t tmp = int32_t(sin(2.0 * M_PI * double(i) * double(c + 1) * 480. / 48000.) * 32768.) << 8;
      sinTable16Channels.push_back(tmp);
    }
  }
}

uint32_t NTV2Writer::addAudioToneVS(int32_t* audioBuffer) {
  uint32_t nbAJAChannels;
  uint32_t nbSamples = nbSamplesPerFrame / 2;
  NTV2Device::getDevice(deviceIndex)->device.GetNumberAudioChannels(nbAJAChannels, audioSystem);

  for (uint32_t i = 0; i < nbSamples; ++i) {
    for (uint32_t c = 0; c < nbAJAChannels; ++c) {
      int index = (currentSample + i * nbAJAChannels + c) % nbSamplesInWavForm;
      audioBuffer[i * nbAJAChannels + c] = sinTable16Channels[index];
    }
  }
  currentSample += nbSamples * nbAJAChannels;
  currentSample %= nbSamplesInWavForm;
  return nbSamples * nbAJAChannels * sizeof(uint32_t);
}

void NTV2Writer::playFrames(void) {
  outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Connecting, "AJAConnecting");
  bool noSignal = true;
  NTV2Device::getDevice(deviceIndex)->device.AutoCirculateStart(outputChannel);  //  Start it running
  AUTOCIRCULATE_TRANSFER mOutputXferInfo;
  while (!globalQuit) {
    AUTOCIRCULATE_STATUS outputStatus;
    NTV2Device::getDevice(deviceIndex)->device.AutoCirculateGetStatus(outputChannel, outputStatus);

    //  Check if there's room for another frame on the card...
    if (outputStatus.CanAcceptMoreOutputFrames()) {
      //  Wait for the next frame to become ready to "consume"...
      AVDataBuffer* playData(aVCircularBuffer.StartConsumeNextBuffer());
      if (playData) {
        if (noSignal) {
          noSignal = false;
          outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Connected,
                                          "AJAConnected");
        }
        if (preRollFrames > 0) {
          //  Transfer the initial black frames to initialize the device for playout...
          mOutputXferInfo.SetVideoBuffer(playData->videoBuffer,
                                         GetVideoWriteSize(videoFormat, NTV2_FBF_8BIT_YCBCR, false, false));
          preRollFrames--;
        } else {
          //  Transfer the frame to the device for playout...
          mOutputXferInfo.SetVideoBuffer(playData->videoBuffer, playData->videoBufferSize);
          if (playData->inNumSegments > 1) {
            mOutputXferInfo.EnableSegmentedDMAs(playData->inNumSegments, playData->videoBufferSize,
                                                playData->videoBufferSize, playData->inDeviceBytesPerLine);
            mOutputXferInfo.acInVideoDMAOffset = playData->inDeviceBytesPerLine * offset_y + offset_x * 2;
          }
        }
        if (withAudio) {
          mOutputXferInfo.SetAudioBuffer(playData->audioBuffer, playData->audioBufferSize);
        } else {
          mOutputXferInfo.SetAudioBuffer(nullptr, 0);
        }
        NTV2Device::getDevice(deviceIndex)->device.AutoCirculateTransfer(outputChannel, mOutputXferInfo);
        aVCircularBuffer.EndConsumeNextBuffer();  //  Signal that the frame has been "consumed"
      }
    } else {
      if (!NTV2Device::getDevice(deviceIndex)->device.WaitForOutputVerticalInterrupt(outputChannel)) {
        noSignal = true;
        if (!AJAStop) {
          Logger::warning(kAjaOutTag) << "No valid signal for this configuration. Aborting." << std::endl;
          outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                          "AJADisConnected");
          AJAStop = true;
        }
      }
    }
  }  //  loop til quit signaled

  //  Stop AutoCirculate...
  NTV2Device::getDevice(deviceIndex)->device.AutoCirculateStop(outputChannel);

}  //  PlayFrames

AJAStatus NTV2Writer::run() {
  //  Start my consumer threads...
  startConsumerThread();
  startProducerThread();
  return AJA_STATUS_SUCCESS;

}  //  Run

bool NTV2Writer::outputDestHasRP188BypassEnabled(void) {
  bool result(false);
  const ULWord regNum(getRP188RegisterForOutput(outputDestination));
  ULWord regValue(0);

  //
  //  Bit 23 of the RP188 DBB register will be set if output timecode will be
  //  grabbed directly from an input (bypass source)...
  //
  if (regNum && NTV2Device::getDevice(deviceIndex)->device.ReadRegister(regNum, &regValue) && regValue & BIT(23)) {
    result = true;
  }

  return result;
}

void NTV2Writer::disableRP188Bypass(void) {
  const ULWord regNum(getRP188RegisterForOutput(outputDestination));
  //  Clear bit 23 of my output destination's RP188 DBB register...
  if (regNum) {
    NTV2Device::getDevice(deviceIndex)->device.WriteRegister(regNum, 0, BIT(23), BIT(23));
  }
}

ULWord NTV2Writer::getRP188RegisterForOutput(const NTV2OutputDestination inOutputDest) {
  switch (inOutputDest) {
    case NTV2_OUTPUTDESTINATION_SDI1:
      return kRegRP188InOut1DBB;  //  reg 29
    case NTV2_OUTPUTDESTINATION_SDI2:
      return kRegRP188InOut2DBB;  //  reg 64
    case NTV2_OUTPUTDESTINATION_SDI3:
      return kRegRP188InOut3DBB;  //  reg 268
    case NTV2_OUTPUTDESTINATION_SDI4:
      return kRegRP188InOut4DBB;  //  reg 273
    case NTV2_OUTPUTDESTINATION_SDI5:
      return kRegRP188InOut5DBB;  //  reg 29
    case NTV2_OUTPUTDESTINATION_SDI6:
      return kRegRP188InOut6DBB;  //  reg 64
    case NTV2_OUTPUTDESTINATION_SDI7:
      return kRegRP188InOut7DBB;  //  reg 268
    case NTV2_OUTPUTDESTINATION_SDI8:
      return kRegRP188InOut8DBB;  //  reg 273
    default:
      return 0;
  }  //  switch on output destination
}

bool NTV2Writer::checkChannelConf(unsigned width, unsigned height, int chan) {
  if (height > 1080 && !(chan == 0 || chan == 4)) {
    Logger::error(kAjaOutTag) << " Using 4K output with multiple SDI output without using channel 1 or 5 as reference"
                              << std::endl;
    Logger::error(kAjaOutTag) << " In this configuration you should use output from 1 to 4 or 5 to 8. Using channel "
                              << chan << std::endl;
    return false;
  }
  return true;
}

void NTV2Writer::routeOutputSignal(NTV2Channel chan) {
  NTV2Device::getDevice(deviceIndex)
      ->device.SetSDIOutputStandard(outputChannel, ::GetNTV2StandardFromVideoFormat(videoFormat));
  if (::NTV2DeviceHasBiDirectionalSDI(NTV2Device::getDevice(deviceIndex)->deviceID)) {
    NTV2Device::getDevice(deviceIndex)->device.SetSDITransmitEnable(chan, true);
  }

  NTV2OutputCrosspointID outputXpt(NTV2_XptBlack);
  NTV2InputCrosspointID sdiInputSelect(NTV2_XptSDIOut1Input);

  switch (chan) {
    case NTV2_CHANNEL1:
      outputXpt = NTV2_XptFrameBuffer1RGB;
      sdiInputSelect = NTV2_XptSDIOut1Input;
      break;
    case NTV2_CHANNEL2:
      outputXpt = NTV2_XptFrameBuffer2RGB;
      sdiInputSelect = NTV2_XptSDIOut2Input;
      break;
    case NTV2_CHANNEL3:
      outputXpt = NTV2_XptFrameBuffer3RGB;
      sdiInputSelect = NTV2_XptSDIOut3Input;
      break;
    case NTV2_CHANNEL4:
      outputXpt = NTV2_XptFrameBuffer4RGB;
      sdiInputSelect = NTV2_XptSDIOut4Input;
      break;
    case NTV2_CHANNEL5:
      outputXpt = NTV2_XptFrameBuffer5RGB;
      sdiInputSelect = NTV2_XptSDIOut5Input;
      break;
    case NTV2_CHANNEL6:
      outputXpt = NTV2_XptFrameBuffer6RGB;
      sdiInputSelect = NTV2_XptSDIOut6Input;
      break;
    case NTV2_CHANNEL7:
      outputXpt = NTV2_XptFrameBuffer7RGB;
      sdiInputSelect = NTV2_XptSDIOut7Input;
      break;
    case NTV2_CHANNEL8:
      outputXpt = NTV2_XptFrameBuffer8RGB;
      sdiInputSelect = NTV2_XptSDIOut8Input;
      break;
  }

  router.AddConnection(sdiInputSelect, outputXpt);

  NTV2Device::getDevice(deviceIndex)->device.ApplySignalRoute(router);
}

}  // namespace Output
}  // namespace VideoStitch
