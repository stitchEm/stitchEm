// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rtmpClient.hpp"
#include "metadataParser.hpp"
#include "amfIncludes.hpp"
#include "amfIMU.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/orah/exposureData.hpp"

#include <thread>
#include <string>

#include "librtmp/log.h"

static std::string RTMPtag("RTMP Client");

static const mtime_t INIT_TIME_STAMP = -1;

namespace VideoStitch {

namespace IO {
// Shared with rtmpPublisher through rtmpStructures.hpp
std::mutex rtmpInitMutex;
}  // namespace IO

namespace Input {

static const AVal av_onMetaData = mAVC("onMetaData");

static const AVal av_videocodecid = mAVC("videocodecid");
static const AVal av_width = mAVC("width");
static const AVal av_height = mAVC("height");
static const AVal av_framerate = mAVC("framerate");

static const AVal av_audiocodecid = mAVC("audiocodecid");
static const AVal av_audiosamplerate = mAVC("audiosamplerate");
static const AVal av_audiosamplesize = mAVC("audiosamplesize");
static const AVal av_stereo = mAVC("stereo");

static const AVal av_onTextData = mAVC("onTextData");
static const AVal av_onTextLine = mAVC("text");

static const auto CONNECTION_RETRY_TIMEOUT = std::chrono::milliseconds(1000);
static const auto DECODE_TIMEOUT = std::chrono::milliseconds(1000);

RTMPClient* RTMPClient::create(readerid_t id, const Ptv::Value* config, const int64_t width, const int64_t height) {
  std::string displayName;
  if (Parse::populateString("RTMP client", *config, "name", displayName, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::error(RTMPtag) << "Missing URL/name (\"name\" field) in the configuration " << std::endl;
    return nullptr;
  }

  //------------------------ Video
  std::string dec;
  Parse::populateString("RTMP client", *config, "decoder", dec, false);

  PotentialValue<VideoDecoder::Type> decoderType = VideoDecoder::parseDecoderType(dec);
  if (!decoderType.ok()) {
    Logger::error(RTMPtag) << "Cannot open stream with unknown video decoder type" << std::endl;
    return nullptr;
  }

  VideoStitch::PixelFormat fmt = NV12;
  FrameRate frameRate;
  frameRate.den = 1;
  frameRate.num = 30;

  if (config->has("frame_rate")) {
    const Ptv::Value* fpsConf = config->has("frame_rate");
    if ((Parse::populateInt("RTMP client", *fpsConf, "num", frameRate.num, false) !=
         VideoStitch::Parse::PopulateResult_Ok) ||
        (Parse::populateInt("RTMP client", *fpsConf, "den", frameRate.den, false) !=
         VideoStitch::Parse::PopulateResult_Ok)) {
      Logger::error(RTMPtag) << "Frame rate (\"frame_rate\") couldn't be retrieved in the configuration" << std::endl;
      return nullptr;
    }
  } else {
    Logger::error(RTMPtag) << "Frame rate (\"frame_rate\") couldn't be retrieved in the configuration" << std::endl;
    return nullptr;
  }

  //------------------------ Audio
  int audioChan = 0;
  int audioSamplerate = 0;
  std::string audioSampleDepth;

  if (Parse::populateInt("RTMP client", *config, "audio_channels", audioChan, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::info(RTMPtag) << "Missing audio channels (\"audio_channels\" field) in the configuration " << std::endl;
  }
  if (Parse::populateInt("RTMP client", *config, "audio_samplerate", audioSamplerate, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::info(RTMPtag) << "Missing audio samplerate (\"audio_samplerate\" field) in the configuration " << std::endl;
  }
  if (Parse::populateString("RTMP client", *config, "audio_sample_depth", audioSampleDepth, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::info(RTMPtag) << "Missing audio sample depth (\"audio_sample_depth\" field) in the configuration "
                          << std::endl;
  }

  Audio::ChannelLayout chanLayout = Audio::ChannelLayout::STEREO;

  switch (audioChan) {
    case 1:
      chanLayout = Audio::ChannelLayout::MONO;
      break;
    case 2:
      chanLayout = Audio::ChannelLayout::STEREO;
      break;
    case 3:
      chanLayout = Audio::ChannelLayout::_2POINT1;
      break;
    case 4:
      chanLayout = Audio::ChannelLayout::_3POINT1;
      break;
    case 5:
      chanLayout = Audio::ChannelLayout::_4POINT1;
      break;
    case 6:
      chanLayout = Audio::ChannelLayout::_5POINT1;
      break;
    case 7:
      chanLayout = Audio::ChannelLayout::_6POINT1;
      break;
    case 8:
      chanLayout = Audio::ChannelLayout::_7POINT1;
      break;
    default:
      chanLayout = Audio::ChannelLayout::UNKNOWN;
      break;
  }

  Audio::SamplingRate srate = Audio::getSamplingRateFromInt(audioSamplerate);

  Audio::SamplingDepth sdepth = Audio::getSamplingDepthFromString(audioSampleDepth.c_str());

  FrameRate frameRateIMU = {1, 1};

  return new RTMPClient(id, config, displayName, width, height, fmt, frameRate, decoderType.value(), chanLayout, srate,
                        sdepth, frameRateIMU);
}

RTMPClient::RTMPClient(readerid_t id, const Ptv::Value* config, const std::string& displayName, int64_t width,
                       int64_t height, VideoStitch::PixelFormat fmt, FrameRate framerate,
                       VideoDecoder::Type decoderType, Audio::ChannelLayout chanLayout, Audio::SamplingRate srate,
                       Audio::SamplingDepth sdepth, FrameRate frameRateIMU)
    : Reader(id),
      VideoReader(width, height, VideoStitch::getFrameDataSize((int32_t)width, (int32_t)height, fmt), fmt,
                  VideoDecoder::decoderAddressSpace(decoderType), framerate, 0, NO_LAST_FRAME, false, /*no procedural*/
                  nullptr),
      AudioReader(chanLayout, srate, sdepth),
      MetadataReader(frameRateIMU),
      SinkReader(),
      videoDecoder(nullptr),
      inputVideoTimestamp(INIT_TIME_STAMP),
      decoderType(decoderType),
      URL(displayName),
      config(config->clone()),
      rtmpConnectionStatus(IO::RTMPConnectionStatus::Disconnected),
      stopping(false),
      videoSem(0),
      readThread(new std::thread(&RTMPClient::readLoop, this)),
      decodeThread(new std::thread(&RTMPClient::decodeLoop, this)),
      audioDecodeThread(nullptr)
#ifdef USE_AVFORMAT
      ,
      avSink()
#endif
{
}

RTMPClient::~RTMPClient() {
  stopping = true;

  if (videoDecoder) {
    videoDecoder->stop();
  }

  readThread->join();
  delete readThread;
  videoSem.notify();
  decodeThread->join();
  delete decodeThread;

  if (audioDecodeThread) {
    {
      std::lock_guard<std::mutex> lock(audioPktQueueMutex);
      stoppingAudioQueue = true;
    }
    audioPktQueueCond.notify_one();
    audioDecodeThread->join();
    delete audioDecodeThread;
  }

  {
    std::lock_guard<std::mutex> lock(frameMu);
    stoppingFrames = true;
  }
  frameCV.notify_one();

  while (frames.size() > 0) {
    Frame frame = frames.front();
    videoDecoder->releaseFrame(frame.second);
    frames.pop();
  }

  removeSink();
  if (config) {
    delete config;
  }

  delete videoDecoder;
}

// --------------------------- Handshake ------------------------

bool RTMPClient::connect() {
  setConnectionStatus(IO::RTMPConnectionStatus::Connecting);

  bool success = false;
  std::string failReason;

#if defined(_WIN32)
  WORD version;
  WSADATA wsaData;
  version = MAKEWORD(1, 1);
  WSAStartup(version, &wsaData);
#endif

  if (rtmp) {
    Logger::info(RTMPtag) << "Attempting to close previous rtmp connection on " << URL << std::endl;
  }

  rtmp = std::unique_ptr<RTMP, std::function<void(RTMP*)>>(RTMP_Alloc(), [](RTMP* data) {
    RTMP_Close(data);
    RTMP_Free(data);
  });

  {
    std::lock_guard<std::mutex> lock(IO::rtmpInitMutex);

    // global SSL init may not be thread safe
    RTMP_Init(rtmp.get());
  }

  RTMP_LogLevel level = Logger::getLevel() >= Logger::LogLevel::Debug ? RTMP_LOGERROR : RTMP_LOGCRIT;
  RTMP_LogSetLevel(level);

  rtmp->Link.swfUrl.av_len = rtmp->Link.tcUrl.av_len;
  rtmp->Link.swfUrl.av_val = rtmp->Link.tcUrl.av_val;
  rtmp->Link.flashVer.av_val = (char*)"FMLE/3.0 (compatible; FMSc/1.0)";
  rtmp->Link.flashVer.av_len = (int)strlen(rtmp->Link.flashVer.av_val);

  if (!RTMP_SetupURL(rtmp.get(), const_cast<char*>(URL.c_str()))) {
    failReason = std::string("Could not parse the URL");
    goto end;
  }

  if (!RTMP_Connect(rtmp.get(), nullptr)) {
    failReason = std::string("Could not connect");
    goto end;
  }

#if defined(_WIN32)
  int sendBufferSize = 131072;
  setsockopt(rtmp->m_sb.sb_socket, SOL_SOCKET, SO_SNDBUF, (char*)&sendBufferSize, sizeof(sendBufferSize));
  int actualSendBufferSize, actualSendBufferSizeLen = sizeof(actualSendBufferSize);
  getsockopt(rtmp->m_sb.sb_socket, SOL_SOCKET, SO_SNDBUF, (char*)&actualSendBufferSize, &actualSendBufferSizeLen);
  Logger::info(RTMPtag) << "Using a " << actualSendBufferSize << " bytes TCP send buffer" << std::endl;
#endif

  if (!RTMP_ConnectStream(rtmp.get(), 0)) {
    failReason = std::string("Cannot connect stream");
    goto end;
  }

  success = true;

end:
  if (!success) {
    rtmp.reset();
    Logger::warning(RTMPtag) << "Connection to " << URL << " failed: " << failReason.c_str() << std::endl;
    setConnectionStatus(IO::RTMPConnectionStatus::Disconnected);
    return false;
  } else {
    flushAudio();
    flushVideo();
    Logger::info(RTMPtag) << "Connected to " << URL << std::endl;
    setConnectionStatus(IO::RTMPConnectionStatus::Connected);
    return true;
  }
}

void RTMPClient::setConnectionStatus(IO::RTMPConnectionStatus newStatus) {
  {
    std::lock_guard<std::mutex> lock(frameMu);
    rtmpConnectionStatus = newStatus;
  }
  frameCV.notify_one();
}

// ---------------------- Packets receiving -------------------

void RTMPClient::metaDataParse(AMFObject* amfObj) {
  int width = 0;
  int height = 0;
  double framerate = 0;

  AMFObjectProperty prop;

  // VIDEO
  if (RTMP_FindFirstMatchingProperty(amfObj, &av_width, &prop)) {
    width = int(AMFProp_GetNumber(&prop));
  }

  if (RTMP_FindFirstMatchingProperty(amfObj, &av_height, &prop)) {
    height = int(AMFProp_GetNumber(&prop));
  }

  if (RTMP_FindFirstMatchingProperty(amfObj, &av_framerate, &prop)) {
    framerate = AMFProp_GetNumber(&prop);
  }

  if (RTMP_FindFirstMatchingProperty(amfObj, &av_videocodecid, &prop)) {
    if (width != getWidth()) {
      Logger::warning(RTMPtag) << "Width of the stream doesn't match the one in project file " << width << " vs "
                               << getWidth() << std::endl;
    }
    if (height != getHeight()) {
      Logger::warning(RTMPtag) << "Height of the stream doesn't match the one in project file " << height << " vs "
                               << getHeight() << std::endl;
    }

    FrameRate frProject = this->VideoReader::getSpec().frameRate;

    if ((frProject.den == 0) || (std::abs(framerate - ((double)(frProject.num) / frProject.den)) > 1e-6)) {
      Logger::warning(RTMPtag) << "Framerate of the stream does not match the one in the project file " << framerate
                               << " vs " << frProject << std::endl;
    }

    Logger::info(RTMPtag) << "Width " << width << std::endl;
    Logger::info(RTMPtag) << "Height " << height << std::endl;
    Logger::info(RTMPtag) << "Framerate " << frProject << std::endl;

    AMFDataType type = AMFProp_GetType(&prop);
    if (type == AMF_NUMBER) {
      int codecId = int(AMFProp_GetNumber(&prop));
      Logger::info(RTMPtag) << "VideocodecID : " << codecId << std::endl;
    } else {
      AVal string;
      AMFProp_GetString(&prop, &string);
      Logger::info(RTMPtag) << "VideocodecID : " << string.av_val << std::endl;
    }

    if (!videoDecoder) {
      auto potDecoder = VideoDecoder::createVideoDecoder(width, height, frProject, decoderType);
      if (potDecoder.ok()) {
        videoDecoder = potDecoder.release();
      }
    }
  }

  // AUDIO
  if (RTMP_FindFirstMatchingProperty(amfObj, &av_audiocodecid, &prop)) {
    int samplerate = 0;
    int samplesize = 0;
    int nbChans = 1;

    // audio metadata
    if (RTMP_FindFirstMatchingProperty(amfObj, &av_audiosamplerate, &prop)) {
      samplerate = int(AMFProp_GetNumber(&prop));
    }
    if (RTMP_FindFirstMatchingProperty(amfObj, &av_audiosamplesize, &prop)) {
      samplesize = int(AMFProp_GetNumber(&prop));
    }
    if (RTMP_FindFirstMatchingProperty(amfObj, &av_stereo, &prop)) {
      if (AMFProp_GetBoolean(&prop)) {
        nbChans = 2;
      }
    }

    bool noAudio = false;
    if (samplerate != getIntFromSamplingRate(AudioReader::getSpec().sampleRate)) {
      Logger::warning(RTMPtag) << "samplerate of the stream doesn't match the one in project file " << samplerate
                               << " vs " << getIntFromSamplingRate(AudioReader::getSpec().sampleRate) << std::endl;
      noAudio = true;
    }
    if (nbChans != getNbChannelsFromChannelLayout(AudioReader::getSpec().layout)) {
      Logger::warning(RTMPtag) << "Channel layout of the stream doesn't match the one in project file " << nbChans
                               << " vs " << getNbChannelsFromChannelLayout(AudioReader::getSpec().layout) << std::endl;
      noAudio = true;
    }
    if (samplesize / 8 != (int)getSampleSizeFromSamplingDepth(AudioReader::getSpec().sampleDepth)) {
      Logger::warning(RTMPtag) << "Sampledepth of the stream doesn't match the one in project file " << samplesize / 8
                               << " vs " << getSampleSizeFromSamplingDepth(AudioReader::getSpec().sampleDepth)
                               << std::endl;
      noAudio = true;
    }

    if (!noAudio) {
      RTMP_FindFirstMatchingProperty(amfObj, &av_audiocodecid, &prop);

      AMFDataType type = AMFProp_GetType(&prop);
      if (!audioDecoder) {
        Logger::info(RTMPtag) << "Creating AudioDecoder" << std::endl;
        audioDecoder = AudioDecoder::createAudioDecoder(type, &prop, &audioStream, samplerate, samplesize, nbChans);
      } else {
        Logger::info(RTMPtag) << "Skipping AudioDecoder creation" << std::endl;
      }

      if (!audioDecodeThread) {
        audioDecodeThread = new std::thread(&RTMPClient::audioDecodeLoop, this);
      }
    }
  }
}

videoreaderid_t RTMPClient::getOrahInputID() const {
  if (URL.size() < 3) {
    return -1;
  }

  std::string orahStreamID = URL.substr(URL.size() - 3, URL.size() - 1);
  if (orahStreamID == "0_0") {
    return 0;
  } else if (orahStreamID == "0_1") {
    return 1;
  } else if (orahStreamID == "1_0") {
    return 2;
  } else if (orahStreamID == "1_1") {
    return 3;
  }
  return -1;
}

#ifdef USE_AVFORMAT
void RTMPClient::metaDataParseOnText(AMFObject* amfObj, const uint32_t timestamp) {
#else
void RTMPClient::metaDataParseOnText(AMFObject* amfObj, const uint32_t) {
#endif
  AMFObjectProperty prop;

  if (RTMP_FindFirstMatchingProperty(amfObj, &av_onTextLine, &prop)) {
    AVal textLine;
    AMFProp_GetString(&prop, &textLine);

#ifdef USE_AVFORMAT
    VideoStitch::IO::Packet avpkt;
    avpkt.data = Span<unsigned char>((unsigned char*)textLine.av_val, textLine.av_len);
    avpkt.pts = avpkt.dts = timestamp * 1000;
    avSink.pushMetadataPacket(avpkt);
#endif

    std::string textLineStr = "";
    if (textLine.av_len > 0 && textLine.av_len < 4096) {
      textLineStr = std::string(textLine.av_val, textLine.av_val + textLine.av_len);
    } else {
      Logger::error(RTMPtag) << "Metadata packet encountered invalid line length: " << textLine.av_len << std::endl;
    }

    std::map<videoreaderid_t, Metadata::Exposure> exposureParsed;
    std::map<videoreaderid_t, Metadata::WhiteBalance> whiteBalanceParsed;
    std::map<videoreaderid_t, Metadata::ToneCurve> toneCurveParsed;
    std::pair<bool, IMU::Measure> imuParsed;

    videoreaderid_t cameraID = getOrahInputID();
    if (!(cameraID == 0 || cameraID == 2)) {
      Logger::error(RTMPtag) << "Metadata packet on unexpected or unknown stream encountered: " << URL << std::endl;
      return;
    }

    MetadataParser::parse(textLineStr, cameraID, imuParsed, exposureParsed, whiteBalanceParsed, toneCurveParsed);

    if (imuParsed.first) {
      const IMU::Measure& imuData = imuParsed.second;
      std::lock_guard<std::mutex> imuLock(imuQueueMutex);
      /// disable gathering of imuData until vectors are replaced with circular buffers in
      /// VideoStitch::Stab::IMUStabilization
      imuQueue.push(imuData);
      while (imuQueue.size() > 25) {
        imuQueue.pop();
      }
    }

    if (exposureParsed.size()) {
      std::lock_guard<std::mutex> exposureLock(exposureQueueMutex);
      exposureQueue.push(exposureParsed);
    }

    if (toneCurveParsed.size()) {
      std::lock_guard<std::mutex> toneCurveLock(toneCurveQueueMutex);
      toneCurveQueue.push(toneCurveParsed);
    }
  }
}

bool RTMPClient::metaDataParseIMU(AMFObject* amfObj, VideoStitch::IMU::Measure& imuData) {
  AMFObjectProperty prop;

  bool returnCode = true;

  // IMU
  if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imuid), &prop)) {
    // Accelerometer data
    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_acc_x), &prop)) {
      imuData.imu_acc_x = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }

    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_acc_y), &prop)) {
      imuData.imu_acc_y = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }

    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_acc_z), &prop)) {
      imuData.imu_acc_z = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }

    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_gyr_x), &prop)) {
      imuData.imu_gyr_x = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }

    // Gyroscope data
    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_gyr_y), &prop)) {
      imuData.imu_gyr_y = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }
    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_gyr_z), &prop)) {
      imuData.imu_gyr_z = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }

    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_mag_x), &prop)) {
      imuData.imu_mag_x = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }

    // Magnetometer data
    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_mag_y), &prop)) {
      imuData.imu_mag_y = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }

    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_mag_z), &prop)) {
      imuData.imu_mag_z = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }

    if (RTMP_FindFirstMatchingProperty(amfObj, &(VideoStitch::IMU::av_imu_temperature), &prop)) {
      imuData.imu_temperature = int(AMFProp_GetNumber(&prop));
    } else {
      returnCode = false;
    }

  } else {
    returnCode = false;
  }

  return returnCode;
}

// ----------------------- Network client thread (receive loop) -----------------------

void RTMPClient::readAudioPacket(const RTMPPacket& packet) {
  if (packet.m_body == nullptr) {
    return;
  }

  VideoStitch::IO::DataPacket packetData((const unsigned char*)packet.m_body, packet.m_nBodySize);
  packetData.timestamp = packet.m_nTimeStamp;

  packetData.type = VideoStitch::IO::PacketType_Audio;
  {
    std::unique_lock<std::mutex> lk(audioPktQueueMutex);
    audioPktQueue.push(packetData);
    Logger::debug(RTMPtag) << "Audio packet: " << URL << " at " << packetData.timestamp << std::endl;
  }
  audioPktQueueCond.notify_one();
}

void RTMPClient::flushAudio() {
  {
    std::unique_lock<std::mutex> lock(audioPktQueueMutex);
    std::queue<VideoStitch::IO::DataPacket> empty;
    std::swap(audioPktQueue, empty);  // that's how you clear std::queue
  }

  {
    std::lock_guard<std::mutex> lock(audioStream.audioBufferMutex);
    audioStream.stream.clear();
    audioStream.cnts = 0;
  }
  Logger::info(RTMPtag) << "Flushing audio " << URL << std::endl;
}

#ifndef USE_AVFORMAT
Status RTMPClient::addSink(const Ptv::Value*, const mtime_t, const mtime_t) {
  return {Origin::Output, ErrType::UnsupportedAction, "[RTMP Client] Sink not implemented"};
}
#else
Status RTMPClient::addSink(const Ptv::Value* sinkConfig, const mtime_t videoTimeStamp, const mtime_t audioTimeStamp) {
  std::string targetDir;
  if (Parse::populateString("RTMP client", *sinkConfig, "target_dir", targetDir, true) != Parse::PopulateResult_Ok) {
    Logger::error(RTMPtag) << "Missing sink target directory (\"target_dir\" field) in the configuration" << std::endl;
    return {Origin::Output, ErrType::UnsupportedAction, "[RTMP Client] Sink needs a target output directory"};
  }
  std::string outName = (URL.size() < 3) ? URL : URL.substr(URL.size() - 3, URL.size() - 1);
  Parse::populateString("RTMP client", *config, "out_name", outName, false);
  outName = targetDir + "/" + outName;

  Ptv::Value* avConfig = Ptv::Value::emptyObject();
  avConfig->push("filename", Ptv::Value::stringObject(outName));

  /* sinkConfig values will overrides default config values */
  std::string outType("mov");
  Parse::populateString("RTMP client", *config, "out_type", outType, false);
  Parse::populateString("RTMP client", *sinkConfig, "out_type", outType, false);
  avConfig->push("type", Ptv::Value::stringObject(outType));

  avConfig->push("video_codec", Ptv::Value::stringObject(videoDecoder->name()));
  if (audioDecoder) {
    avConfig->push("audio_codec", Ptv::Value::stringObject(audioDecoder->name()));
    avConfig->push("sample_format",
                   Ptv::Value::stringObject(getStringFromSamplingDepth(AudioReader::getSpec().sampleDepth)));
    avConfig->push("sampling_rate", Ptv::Value::intObject(getIntFromSamplingRate(AudioReader::getSpec().sampleRate)));
    avConfig->push("channel_layout",
                   Ptv::Value::stringObject(getStringFromChannelLayout(AudioReader::getSpec().layout)));
  }

  avConfig->push("metadata_codec", Ptv::Value::stringObject("mov_text"));

  int64_t maxMuxedSize;
  if (Parse::populateInt("RTMP client", *config, "max_video_file_chunk", maxMuxedSize, false) ==
      Parse::PopulateResult_Ok) {
    Parse::populateInt("RTMP client", *sinkConfig, "max_video_file_chunk", maxMuxedSize, false);
    avConfig->push("max_video_file_chunk", Ptv::Value::intObject(maxMuxedSize));
  }
  return avSink.create(*avConfig, (unsigned)getWidth(), (unsigned)getHeight(), this->VideoReader::getSpec().frameRate,
                       videoHeader, videoTimeStamp, audioTimeStamp);
}
#endif

void RTMPClient::removeSink() {
#ifdef USE_AVFORMAT
  avSink.destroy();
#endif
}

void RTMPClient::readVideoPacket(const RTMPPacket& packet) {
  if (packet.m_body == nullptr) {
    return;
  }

  // Check delta
  if ((inputVideoTimestamp != INIT_TIME_STAMP) && (this->VideoReader::getSpec().frameRate.num != 0)) {
    int64_t millisecondsPerFrame =
        this->VideoReader::getSpec().frameRate.den * 1000 / this->VideoReader::getSpec().frameRate.num +
        1;  // +1 to take rounding into account
    int64_t delta = (int64_t)packet.m_nTimeStamp - (int64_t)inputVideoTimestamp;
    if (delta < 0 || delta > millisecondsPerFrame) {
      Logger::warning(RTMPtag) << "Video input is not monotonous, got " << inputVideoTimestamp << " + " << delta
                               << " for " << URL << std::endl;
    }
  }

  inputVideoTimestamp = packet.m_nTimeStamp;

  Span<const unsigned char> packetData((const unsigned char*)packet.m_body, packet.m_nBodySize);
  mtime_t packetTimestamp = packet.m_nTimeStamp;

  if (!videoDecoder) {
    Logger::warning(RTMPtag) << "Received video packet, but no video decoder has been created yet" << std::endl;
  }

  switch (packetData[1]) {
    case 0: {
      // codec config packet inside the FLV
      if (videoDecoder) {
        videoDecoder->decodeHeader(packetData, packetTimestamp, videoHeader);
        Logger::debug(RTMPtag) << "Video header: " << URL << " at " << packetTimestamp << std::endl;
      }
      break;
    }
    case 1: {  // frame
      if (videoDecoder) {
        VideoStitch::IO::Packet avpkt;
        if (!videoDecoder->demux(packetData, packetTimestamp, avpkt)) {
          break;
        }
#ifdef USE_AVFORMAT
        avSink.pushVideoPacket(avpkt);
#endif
        if (videoDecoder->decodeAsync(avpkt)) {
          videoSem.notify();
        }
        Logger::debug(RTMPtag) << "Video packet: " << URL << " at " << packetTimestamp << std::endl;
      }
      break;
    }
    case 2: {
      Logger::info(RTMPtag) << "End of stream" << std::endl;
    } break;

    default:
      Logger::error(RTMPtag) << "Unknown video packet type " << (int)packetData[1] << std::endl;
      assert(false && "RTMPClient : unknown video packet type");
  }
}

void RTMPClient::readInfoPacket(const RTMPPacket& packet) {
  // GET metadata packet to know codec and metadata to use
  AMFObject amfObj;
  int res = AMF_Decode(&amfObj, packet.m_body, packet.m_nBodySize, FALSE);
  if (res < 0) {
    Logger::error(RTMPtag) << "Error decoding info packet" << std::endl;
    return;
  }

  AMFObjectProperty* amfObjProp = AMF_GetProp(&amfObj, NULL, 0);

  AVal metastring;
  AMFProp_GetString(amfObjProp, &metastring);

  std::string metastringStr = "";
  if (metastring.av_len > 0 && metastring.av_len < 256) {
    metastringStr = std::string(metastring.av_val, metastring.av_val + metastring.av_len);
  }
  Logger::debug(RTMPtag) << "readInfoPacket(): metastring: <" << metastringStr << ">" << std::endl;

  if (AVMATCH(&metastring, &av_onMetaData)) {
    // get metadata packet: let's get parameter, create decoder and launch decoding script
    AMF_Dump(&amfObj);
    metaDataParse(&amfObj);
  }

  if (AVMATCH(&metastring, &av_onTextData)) {
    AMF_Dump(&amfObj);
    metaDataParseOnText(&amfObj, packet.m_nTimeStamp);
  }

  if (AVMATCH(&metastring, &(VideoStitch::IMU::av_imuid))) {
    AMF_Dump(&amfObj);
    VideoStitch::IMU::Measure imuData;
    imuData.timestamp = (mtime_t)packet.m_nTimeStamp * 1000;  //  milliseconds to microseconds conversion
    bool imuRetrievedCorrectly = metaDataParseIMU(&amfObj, imuData);
    if (imuRetrievedCorrectly) {
      std::unique_lock<std::mutex> imuLock(imuQueueMutex);
      //      imuQueue.push(imuData);
      if (imuQueue.size() > 25) {  ///< TODO: set the max size of que imu queue
        imuQueue.pop();
      }
    }
  }

  AMF_Reset(&amfObj);
}

void RTMPClient::readLoop() {
  for (;;) {
    if (stopping) {
      return;
    }

    if (!rtmp || !RTMP_IsConnected(rtmp.get()) || RTMP_IsTimedout(rtmp.get())) {
      setConnectionStatus(IO::RTMPConnectionStatus::Disconnected);
      Logger::info(RTMPtag) << "Reconnecting to " << URL << std::endl;
      if (!connect()) {
        std::this_thread::sleep_for(CONNECTION_RETRY_TIMEOUT);
        continue;
      }
    }

    RTMPPacket packet = {};
    while (RTMP_IsConnected(rtmp.get()) && RTMP_ReadPacket(rtmp.get(), &packet)) {
      const bool isReady = RTMPPacket_IsReady(&packet);
      if (isReady) {
        readPacket(packet);
        RTMPPacket_Free(&packet);
        break;
      }
    }
  }
}

void RTMPClient::readPacket(const RTMPPacket& packet) {
  switch (packet.m_packetType) {
    case RTMP_PACKET_TYPE_AUDIO:
      // deactivate till audio decoding work
      readAudioPacket(packet);
      break;

    case RTMP_PACKET_TYPE_VIDEO:
      readVideoPacket(packet);
      break;

    case RTMP_PACKET_TYPE_INFO:
      readInfoPacket(packet);
      break;

    default:
      // to parse
      break;
  }
}

//------------------------- Audio Decode Loop -------------------------------

void RTMPClient::audioDecodeLoop() {
  assert(getNbChannelsFromChannelLayout(AudioReader::getSpec().layout));
  assert(getSampleSizeFromSamplingDepth(AudioReader::getSpec().sampleDepth));
  assert(getIntFromSamplingRate(AudioReader::getSpec().sampleRate));

  VideoStitch::IO::DataPacket packetData;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(audioPktQueueMutex);
      audioPktQueueCond.wait(lock, [this] { return audioPktQueue.size() > 0 || stoppingAudioQueue; });
      if (stoppingAudioQueue) {
        return;
      }

      // pop a rtmp packet
      packetData = audioPktQueue.front();
      audioPktQueue.pop();
    }

    if (!audioDecoder) {
      Logger::warning(RTMPtag) << "Won't decode, no audioDecoder yet" << std::endl;
      continue;
    }

    VideoStitch::IO::Packet avpkt;
    audioDecoder->demux(packetData, avpkt);
#ifdef USE_AVFORMAT
    avSink.pushAudioPacket(avpkt);
#endif
    if (audioDecoder->decode(&packetData)) {
      std::lock_guard<std::mutex> lock(audioStream.audioBufferMutex);
      if (audioStream.cnts == 0) {
        audioStream.cnts = MTOCNTIME(packetData.timestamp * 1000);
      }

      mtime_t endTS =
          CNTOMTIME(audioStream.cnts) +
          ((audioStream.stream.size() / (getNbChannelsFromChannelLayout(AudioReader::getSpec().layout) *
                                         getSampleSizeFromSamplingDepth(AudioReader::getSpec().sampleDepth))) *
           (mtime_t)1000000) /
              getIntFromSamplingRate(AudioReader::getSpec().sampleRate);
      if (((mtime_t)packetData.timestamp * 1000 - endTS) / 1000 > 1) {
        Logger::error(RTMPtag) << "Audio packet lost, sync might be lost : "
                               << ((mtime_t)packetData.timestamp * 1000 - endTS) / 1000 << " ms " << std::endl;
      }
    }
  }
}

// ----------------------- Stitcher Controller decoding thread -----------------------

void RTMPClient::decodeFrame() {
  Frame frame;
  {
    std::unique_lock<std::mutex> lk(frameMu);

    if (frames.size() >= CIRCULAR_BUFFER_LEN) {
      Logger::warning(RTMPtag) << "Stitcher late, dropped one frame on " << URL << std::endl;
      frame = frames.front();
      frames.pop();
      videoDecoder->releaseFrame(frame.second);
    }
  }
  if (videoDecoder->synchronize(frame.first, frame.second)) {
    {
      std::unique_lock<std::mutex> lk(frameMu);
      frames.push(frame);
    }
    frameCV.notify_one();
  }
}

void RTMPClient::flushVideo() {
  if (rtmpConnectionStatus != IO::RTMPConnectionStatus::Connected) {
    if (videoDecoder) {
      auto framesNum = videoDecoder->flush();
      while (framesNum) {
        decodeFrame();
        framesNum--;
      }
      if (frames.size()) {
        Logger::info(RTMPtag) << "Flushing video " << URL << std::endl;
      }
      while (frames.size() > 0) {
        videoDecoder->releaseFrame(frames.front().second);
        frames.pop();
      }

      frameCV.notify_one();
    }
  }
}

void RTMPClient::decodeLoop() {
  for (;;) {
    if (stopping) {
      return;
    }

    if (!videoSem.wait_for(unsigned(DECODE_TIMEOUT.count()))) {
      Logger::info(RTMPtag) << "Decoder timeout, no input packets received for " << URL << std::endl;
      frameCV.notify_one();
      continue;
    }

    if (!videoDecoder) {
      continue;
    }

    decodeFrame();
  }
}

ReadStatus RTMPClient::readFrame(mtime_t& date, unsigned char* video) {
  Frame frame;
  {
    std::unique_lock<std::mutex> lock(frameMu);
    frameCV.wait(lock, [this]() {
      return frames.size() > 0 || stoppingFrames || rtmpConnectionStatus == IO::RTMPConnectionStatus::Disconnected;
    });

    if (stoppingFrames || frames.empty() || rtmpConnectionStatus != IO::RTMPConnectionStatus::Connected) {
      return ReadStatus::fromCode<ReadStatusCode::EndOfFile>();
    }

    frame = frames.front();
    frames.pop();
  }
  date = frame.first;

  videoDecoder->copyFrame(video, date, frame.second);
  videoDecoder->releaseFrame(frame.second);

  // Check latency
  if (date) {
    mtime_t currentLatency = inputVideoTimestamp - date / 1000; /* from us to ms */
    ;

    if (this->updateLatency(currentLatency)) {
      Logger::verbose(RTMPtag) << "Video latency increased to " << currentLatency << " ms for " << URL << std::endl;
      Logger::verbose(RTMPtag) << frames.size() << " video frames are waiting for reader synchronization" << std::endl;
    }
  }

  return ReadStatus::OK();
}

size_t RTMPClient::available() {
  assert(getNbChannelsFromChannelLayout(AudioReader::getSpec().layout));
  assert(getSampleSizeFromSamplingDepth(AudioReader::getSpec().sampleDepth));

  return audioStream.stream.size() / (getSampleSizeFromSamplingDepth(AudioReader::getSpec().sampleDepth) *
                                      getNbChannelsFromChannelLayout(AudioReader::getSpec().layout));
}

bool RTMPClient::eos() {
  return false;  // XXX TODO FIXME vlad :)
}

Status RTMPClient::readIMUSamples(std::vector<VideoStitch::IMU::Measure>& imuData) {
  std::unique_lock<std::mutex> lockQueueIMU(imuQueueMutex);
  imuData.reserve(imuQueue.size());
  while (!imuQueue.empty()) {
    imuData.push_back(imuQueue.front());
    imuQueue.pop();
  }
  return Status::OK();
}

Input::MetadataReader::MetadataReadStatus RTMPClient::readExposure(
    std::map<videoreaderid_t, Metadata::Exposure>& exposure) {
  std::lock_guard<std::mutex> lock(exposureQueueMutex);
  if (!exposureQueue.empty()) {
    std::map<videoreaderid_t, Metadata::Exposure> expmap = exposureQueue.front();
    for (const auto& kv : expmap) {
      exposure[kv.first] = kv.second;
    }
    exposureQueue.pop();
  }

  using MetaStatus = Input::MetadataReader::MetadataReadStatus;
  if (!exposureQueue.empty()) {
    return MetaStatus::fromCode<MetaStatus::StatusCode::MoreDataAvailable>();
  }
  return MetaStatus::OK();
}

Input::MetadataReader::MetadataReadStatus RTMPClient::readWhiteBalance(
    std::map<videoreaderid_t, Metadata::WhiteBalance>&) {
  return Input::MetadataReader::MetadataReadStatus::OK();
}

Input::MetadataReader::MetadataReadStatus RTMPClient::readToneCurve(
    std::map<videoreaderid_t, Metadata::ToneCurve>& toneCurve) {
  std::lock_guard<std::mutex> lock(toneCurveQueueMutex);
  if (!toneCurveQueue.empty()) {
    std::map<videoreaderid_t, Metadata::ToneCurve> tcmap = toneCurveQueue.front();
    for (const auto& kv : tcmap) {
      toneCurve[kv.first] = kv.second;
    }
    toneCurveQueue.pop();
  }

  using MetaStatus = Input::MetadataReader::MetadataReadStatus;
  if (!toneCurveQueue.empty()) {
    return MetaStatus::fromCode<MetaStatus::StatusCode::MoreDataAvailable>();
  }
  return MetaStatus::OK();
}

ReadStatus RTMPClient::readSamples(size_t nbSamples, Audio::Samples& samples) {
  assert(getIntFromSamplingRate(AudioReader::getSpec().sampleRate));

  size_t availSmpl = available();

  if (availSmpl < nbSamples) {
    return ReadStatus::fromCode<ReadStatusCode::EndOfFile>();
  }

  uint64_t readBytes = nbSamples * getNbChannelsFromChannelLayout(AudioReader::getSpec().layout) *
                       getSampleSizeFromSamplingDepth(AudioReader::getSpec().sampleDepth);
  Audio::Samples::data_buffer_t raw;
  raw[0] = new uint8_t[readBytes];
  std::lock_guard<std::mutex> lk(audioStream.audioBufferMutex);
  audioStream.stream.pop(raw[0], readBytes);
  samples = Audio::Samples(AudioReader::getSpec().sampleRate, AudioReader::getSpec().sampleDepth,
                           AudioReader::getSpec().layout, CNTOMTIME(audioStream.cnts), raw, nbSamples);
  audioStream.cnts +=
      cntime_t(std::round(nbSamples * 100000000000.0 / getIntFromSamplingRate(AudioReader::getSpec().sampleRate)));

  return ReadStatus::OK();
}

}  // namespace Input
}  // namespace VideoStitch
