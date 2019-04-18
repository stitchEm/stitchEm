// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rtmpPublisher.hpp"
#include "videoEncoder.hpp"
#include "audioEncoder.hpp"
#include "amfIncludes.hpp"
#include "amfIMU.hpp"

#include "aacEncoder.hpp"

#include "libvideostitch/logging.hpp"

#include <thread>
#include <unordered_map>

#define SEND_MOCK_IMU_METADATA 0

static std::string RTMPtag("RTMP Publisher");

static const int64_t INIT_TIME_STAMP = -(1 << 30);

namespace VideoStitch {
namespace Output {

static const AVal av_setDataFrame = mAVC("@setDataFrame");
static const AVal av_onMetaData = mAVC("onMetaData");
static const AVal av_width = mAVC("width");
static const AVal av_height = mAVC("height");
static const AVal av_encoder = mAVC("encoder");

const int RTMPPublisher::rtmpTimeout = 5;
// x264Encoder with "veryslow" preset adds latency of 75 video frames corresponding to ~130 AAC audio packet
const size_t RTMPPublisher::AUDIO_BUFFER_LIMIT = 150;
const size_t RTMPPublisher::VIDEO_BUFFER_LIMIT = 50;
const size_t RTMPPublisher::PACKET_QUEUE_LIMIT = 200;

const std::string ConnectingMessage = "RTMPConnecting";
const std::string ConnectedMessage = "Streaming";
const std::string CongestionMessage = "Network Congestion";
const std::string RTMPConnectionRefusedMessage = "RTMPConnectionRefused";
const std::string BadUrlMessage = "BadUrl";
const std::string NetworkErrorMessage = "NetworkError";

Potential<RTMPPublisher> RTMPPublisher::create(const Ptv::Value* config,
                                               const VideoStitch::Plugin::VSWriterPlugin::Config& runtime) {
  std::string encoder = "x264";
  bool deprecated_encoder_mode = false;
  if (Parse::populateString("RTMP", *config, "encoder", encoder, false) == VideoStitch::Parse::PopulateResult_Ok) {
    Logger::info(RTMPtag) << "'encoder' parameter deprecated, use 'video_codec' instead" << std::endl;
    deprecated_encoder_mode = true;
  }
  if (Parse::populateString("RTMP", *config, "video_codec", encoder, false) != VideoStitch::Parse::PopulateResult_Ok) {
    if (deprecated_encoder_mode == false) {
      Logger::info(RTMPtag) << "No encoder defined, using x264 on CPU" << std::endl;
    }
  }
  std::string audio_encoder = "";
  if (Parse::populateString("RTMP", *config, "audio_codec", audio_encoder, false) == Parse::PopulateResult_WrongType) {
    Logger::info(RTMPtag) << "No audio encoder defined, disabling audio" << std::endl;
  }
  std::string pubUser, pubPasswd;
  Parse::populateString("RTMP", *config, "pub_user", pubUser, false);
  Parse::populateString("RTMP", *config, "pub_passwd", pubPasswd, false);

  std::string userAgent;
  std::string codecVer;
  if (Parse::populateString("RTMP", *config, "user_agent", userAgent, false) != VideoStitch::Parse::PopulateResult_Ok) {
    userAgent = "FMSc/1.0";
    codecVer = "Vahana v2.0";
  } else {
    codecVer = userAgent;
  }
  const std::string flashVer = "FMLE/3.0 (compatible; " + userAgent + ")";

  PixelFormat format = Unknown;
  const std::unordered_map<std::string, PixelFormat> pixelFormatMap = {
      {"x264", YV12},  {"h264", YV12},       {"mock", YV12},      {"qsv", NV12}, {"h264_qsv", NV12}, {"hevc_qsv", NV12},
#ifdef SUP_NVENC_M
      {"nvenc", YV12}, {"h264_nvenc", YV12}, {"hevc_nvenc", YV12}
#else
      {"nvenc", NV12}, {"h264_nvenc", NV12}, {"hevc_nvenc", NV12}
#endif
  };
  if (pixelFormatMap.find(encoder) != pixelFormatMap.end()) {
    format = pixelFormatMap.at(encoder);
  }

  if (format == Unknown) {
    return Potential<RTMPPublisher>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, RTMPtag,
                                    "Could not find pixel format for Video Encoder : " + encoder);
  }
  AddressSpace outputType = (encoder.find("nvenc") != std::string::npos) ? Device : Host;

  auto videoEncoder =
      VideoEncoder::createVideoEncoder(*config, runtime.width, runtime.height, runtime.framerate, encoder);

  if (!videoEncoder.ok()) {
    return Potential<RTMPPublisher>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, RTMPtag,
                                    "Could not create Video Encoder", videoEncoder.status());
  }

  auto audioEncoder =
      AudioEncoder::createAudioEncoder(*config, runtime.rate, runtime.depth, runtime.layout, audio_encoder);

  int32_t minBitrate = -1;
  Parse::populateInt("RTMP", *config, "bitrate_min", minBitrate, false);

  return new RTMPPublisher(runtime, std::move(videoEncoder), std::move(audioEncoder), pubUser, pubPasswd, flashVer,
                           codecVer, format, outputType, minBitrate);
}

RTMPPublisher::RTMPPublisher(const VideoStitch::Plugin::VSWriterPlugin::Config& runtime,
                             Potential<VideoEncoder> newVideoEncoder, std::unique_ptr<AudioEncoder> newAudioEncoder,
                             const std::string& pubUser, const std::string& pubPasswd, const std::string& flashVer,
                             const std::string& codecVer, VideoStitch::PixelFormat format, AddressSpace type,
                             int32_t minBitrate)
    : Output(runtime.name),
      VideoWriter(runtime.width, runtime.height, runtime.framerate, format, type),
      AudioWriter(runtime.rate, runtime.depth, runtime.layout),
      audioEncoder(std::move(newAudioEncoder)),
      videoEncoder(std::move(newVideoEncoder)),
      inputVideoTimestamp(0),
      URL(runtime.name),
      rtmpConnectionStatus(IO::RTMPConnectionStatus::Disconnected),
      stopping(false),
      pubUser(pubUser),
      pubPasswd(pubPasswd),
      flashVer(flashVer),
      queuedMax(PACKET_QUEUE_LIMIT / 4),
      minBitrate(minBitrate),
      bitRateTimeOut(0),
      dropped(false) {
  // https://helpx.adobe.com/adobe-media-server/dev/adding-metadata-live-stream.html
  metaDataPacketBuffer.resize(2048);

  char* enc = metaDataPacketBuffer.data();
  char* pend = metaDataPacketBuffer.data() + metaDataPacketBuffer.size();
  enc = AMF_EncodeString(enc, pend, &av_setDataFrame);
  enc = AMF_EncodeString(enc, pend, &av_onMetaData);
  *enc++ = AMF_OBJECT;
  enc = AMF_EncodeNamedNumber(enc, pend, &av_width, double(getWidth()));
  enc = AMF_EncodeNamedNumber(enc, pend, &av_height, double(getHeight()));
  enc = videoEncoder->metadata(enc, pend);
  if (audioEncoder) {
    enc = audioEncoder->metadata(enc, pend);
  }

  const AVal av_VideoStitchVersion = mAVC(codecVer.c_str());

  enc = AMF_EncodeNamedString(enc, pend, &av_encoder, &av_VideoStitchVersion);

  *enc++ = 0;
  *enc++ = 0;
  *enc++ = AMF_OBJECT_END;

  metaDataPacketBuffer.resize(enc - metaDataPacketBuffer.data());

  sendThread = new std::thread(&RTMPPublisher::sendLoop, this);
}

RTMPPublisher::~RTMPPublisher() {
  {
    std::lock_guard<std::mutex> lk(dataMutex);
    stopping = true;
  }
  // wake up the send thread so it won't wait for more data
  sendCond.notify_all();
  sendThread->join();
  delete sendThread;
}

void RTMPPublisher::updateConfig(const Ptv::Value& config) {
  if (videoEncoder.ok() && videoEncoder->dynamicBitrateSupported()) {
    std::lock_guard<std::mutex> lk(updateMutex);

    uint32_t targetBitRate = videoEncoder->getBitRate();
    Parse::populateInt("RTMP", config, "bitrate", targetBitRate, false);
    uint32_t bufferSize = uint32_t(-1);
    Parse::populateInt("RTMP", config, "buffer_size", bufferSize, false);
    int32_t newMinBitrate = -1;
    Parse::populateInt("RTMP", config, "bitrate_min", newMinBitrate, false);
    minBitrate = newMinBitrate;
    videoEncoder->setMaxBitRate(targetBitRate);
    videoEncoder->setBitRate(targetBitRate, bufferSize);
    if (rtmpConnectionStatus == IO::RTMPConnectionStatus::Connected) {
      outputEventManager.publishEvent(OutputEventManager::EventType::Connected, ConnectedMessage);
    }
  }
}

// ----------------------- Encoding thread -----------------------

void RTMPPublisher::pushMetadataIMU(const VideoStitch::IMU::Measure& imuData) {
  std::vector<char> mdpBuffer(2048);
  char* enc = mdpBuffer.data();
  char* pend = mdpBuffer.data() + mdpBuffer.size();
  enc = AMF_EncodeString(enc, pend, &av_setDataFrame);
  enc = AMF_EncodeString(enc, pend, &(VideoStitch::IMU::av_imuid));
  *enc++ = AMF_OBJECT;
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imuid), 1);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_acc_x), imuData.imu_acc_x);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_acc_y), imuData.imu_acc_y);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_acc_z), imuData.imu_acc_z);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_gyr_x), imuData.imu_gyr_x);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_gyr_y), imuData.imu_gyr_y);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_gyr_z), imuData.imu_gyr_z);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_mag_x), imuData.imu_mag_x);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_mag_y), imuData.imu_mag_y);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_mag_z), imuData.imu_mag_z);
  enc = AMF_EncodeNamedNumber(enc, pend, &(VideoStitch::IMU::av_imu_temperature), imuData.imu_temperature);

  *enc++ = 0;
  *enc++ = 0;
  *enc++ = AMF_OBJECT_END;

  mdpBuffer.resize(enc - mdpBuffer.data());

  RTMPPacket packet;
  RTMPPacket_Alloc(&packet, (int)mdpBuffer.size());
  packet.m_nChannel = 0x03;  // control channel (invoke)
  packet.m_headerType = RTMP_PACKET_SIZE_LARGE;
  packet.m_packetType = RTMP_PACKET_TYPE_INFO;
  packet.m_nTimeStamp = uint32_t(imuData.timestamp / 1000);  //  microseconds to milliseconds conversion
  packet.m_hasAbsTimestamp = TRUE;
  memcpy(packet.m_body, mdpBuffer.data(), mdpBuffer.size());
  packet.m_nBodySize = (uint32_t)mdpBuffer.size();

  packet.m_nInfoField2 = 0;
  packet.m_nBytesRead = 0;
  packet.m_chunk = nullptr;

  if (!RTMP_SendPacket(rtmp.get(), &packet, FALSE)) {
    Logger::error(RTMPtag) << "RTMP_SendPacket failed" << std::endl;
  }
  RTMPPacket_Free(&packet);
}

void RTMPPublisher::pushVideo(const Frame& videoFrame) {
  // encode the video frame
  std::vector<VideoStitch::IO::DataPacket> videoPackets;
  inputVideoTimestamp = videoFrame.pts / 1000; /* from us to ms */
  if (videoEncoder->encode(videoFrame, videoPackets)) {
    std::lock_guard<std::mutex> lk(dataBufferMutex);
    bufferedVideo.insert(bufferedVideo.end(), videoPackets.begin(), videoPackets.end());
  }

#if SEND_MOCK_IMU_METADATA
  VideoStitch::IMU::Measure imuData;
  imuData.timestamp = timestamp + 1000;  // arbitrarily offset by 1 ms the timestamp of the closest video frame
  imuData.imu_acc_x = 10;
  imuData.imu_acc_y = 11;
  imuData.imu_acc_z = 12;
  imuData.imu_gyr_x = 20;
  imuData.imu_gyr_y = 21;
  imuData.imu_gyr_z = 22;
  imuData.imu_mag_x = 30;
  imuData.imu_mag_y = 31;
  imuData.imu_mag_z = 32;
  imuData.imu_temperature = 100;

  static unsigned int metadataFrame = 0;
  metadataFrame++;

  if ((metadataFrame % 10) == 0) {
    pushMetadataIMU(imuData);
  }
#endif

  // Muxing
  mux();
}

void RTMPPublisher::pushAudio(Audio::Samples& audioSamples) {
  // encode the audio frame
  if (audioEncoder && audioSamples.getNbOfSamples() > 0) {
    std::vector<VideoStitch::IO::DataPacket> audioPackets;
    mtime_t audioTimestamp = audioSamples.getTimestamp();

    if (audioEncoder->encode(audioTimestamp, audioSamples.getSamples().data(),
                             (unsigned int)audioSamples.getNbOfSamples(), audioPackets)) {
      std::lock_guard<std::mutex> lk(dataBufferMutex);
      bufferedAudio.insert(bufferedAudio.end(), audioPackets.begin(), audioPackets.end());
    }
  }

  // Muxing
  mux();
}

void RTMPPublisher::mux() {
  std::lock_guard<std::mutex> lk(dataBufferMutex);
  // Muxing
  // publish the video frame, and audio data (and keep rtmp packets in order)
  if (audioEncoder) {
    while (bufferedAudio.size() > 0 && bufferedVideo.size() > 0) {
      if ((int64_t)bufferedAudio.front().timestamp < (int64_t)bufferedVideo.front().timestamp) {
        sendPacketFromQueue(bufferedAudio);
      } else {
        sendPacketFromQueue(bufferedVideo);
      }
    }

    // if after muxing we still have a lot (more than specified limit) of packets in one of the buffers - we send them,
    // as the most likely reason for that is that we've lost one of the sources
    sendPacketsUntilLimit(bufferedAudio, AUDIO_BUFFER_LIMIT);
    sendPacketsUntilLimit(bufferedVideo, VIDEO_BUFFER_LIMIT);
  } else {
    // no audio, no muxing
    sendPacketsUntilLimit(bufferedVideo, 0);
  }
}

// --------------------------- Handshake ------------------------

bool RTMPPublisher::publishHeaders() {
  bool success = true;

  Logger::info(RTMPtag) << "Attempting to send header packets" << std::endl;

  RTMPPacket packet;
  RTMPPacket_Alloc(&packet, (int)metaDataPacketBuffer.size());
  packet.m_nChannel = 0x03;  // control channel (invoke)
  packet.m_headerType = RTMP_PACKET_SIZE_LARGE;
  packet.m_packetType = RTMP_PACKET_TYPE_INFO;
  packet.m_nTimeStamp = 0;
  packet.m_nInfoField2 = rtmp->m_stream_id;
  packet.m_hasAbsTimestamp = TRUE;
  memcpy(packet.m_body, metaDataPacketBuffer.data(), metaDataPacketBuffer.size());
  packet.m_nBodySize = (uint32_t)metaDataPacketBuffer.size();
  if (!RTMP_SendPacket(rtmp.get(), &packet, FALSE)) {
    Logger::error(RTMPtag) << "Cannot publish headers, RTMP_SendPacket failed!" << std::endl;
    rtmpConnectionStatus = IO::RTMPConnectionStatus::Disconnected;
    outputEventManager.publishEvent(OutputEventManager::EventType::Disconnected, RTMPConnectionRefusedMessage);
    success = false;
    goto end;
  }

  if (videoEncoder.ok()) {
    std::vector<VideoStitch::IO::DataPacket> videoPackets;

    videoEncoder->header(videoPackets);
    VideoStitch::IO::DataPacket* pkt = videoPackets.data();
    if (pkt) {
      RTMPPacket sequence_packet;
      RTMPPacket_Alloc(&sequence_packet, (int)pkt->size());
      sequence_packet.m_nChannel = 0x04;  // video channel
      sequence_packet.m_headerType = RTMP_PACKET_SIZE_LARGE;
      sequence_packet.m_packetType = RTMP_PACKET_TYPE_VIDEO;
      sequence_packet.m_nTimeStamp = 0;
      sequence_packet.m_nInfoField2 = rtmp->m_stream_id;
      sequence_packet.m_hasAbsTimestamp = TRUE;
      memcpy(sequence_packet.m_body, pkt->data(), pkt->size());
      sequence_packet.m_nBodySize = (uint32_t)pkt->size();

      if (!RTMP_SendPacket(rtmp.get(), &sequence_packet, FALSE)) {
        RTMPPacket_Free(&sequence_packet);
        Logger::error(RTMPtag) << "RTMP_SendPacket failed" << std::endl;
        rtmpConnectionStatus = IO::RTMPConnectionStatus::Disconnected;
        outputEventManager.publishEvent(OutputEventManager::EventType::Disconnected, RTMPConnectionRefusedMessage);
        goto end;
      }
      RTMPPacket_Free(&sequence_packet);
      firstVideoPacket = false;
    }
  }

  // the AAC decoder needs to receive its info through a sequence header
  if (audioEncoder) {
    VideoStitch::IO::DataPacket* pkt = audioEncoder->header();
    if (pkt) {
      RTMPPacket sequence_packet;
      RTMPPacket_Alloc(&sequence_packet, (int)pkt->size());
      sequence_packet.m_nChannel = 0x05;  // audio channel
      sequence_packet.m_headerType = RTMP_PACKET_SIZE_LARGE;
      sequence_packet.m_packetType = RTMP_PACKET_TYPE_AUDIO;
      sequence_packet.m_nTimeStamp = 0;
      sequence_packet.m_nInfoField2 = rtmp->m_stream_id;
      sequence_packet.m_hasAbsTimestamp = TRUE;

      memcpy(sequence_packet.m_body, pkt->data(), pkt->size());
      sequence_packet.m_nBodySize = (uint32_t)pkt->size();

      if (!RTMP_SendPacket(rtmp.get(), &sequence_packet, FALSE)) {
        RTMPPacket_Free(&sequence_packet);
        Logger::error(RTMPtag) << "Cannot publish audio headers, RTMP_SendPacket failed!" << std::endl;
        rtmpConnectionStatus = IO::RTMPConnectionStatus::Disconnected;
        outputEventManager.publishEvent(OutputEventManager::EventType::Disconnected, RTMPConnectionRefusedMessage);
        success = false;
        goto end;
      }

      RTMPPacket_Free(&sequence_packet);
    }
  }

  Logger::info(RTMPtag) << "Sent headers to " << URL << ", will start sending data." << std::endl;

  rtmpConnectionStatus = IO::RTMPConnectionStatus::Connected;
  outputEventManager.publishEvent(OutputEventManager::EventType::Connected, ConnectedMessage);

end:
  RTMPPacket_Free(&packet);

  return success;
}

bool RTMPPublisher::connect() {
  if (rtmpConnectionStatus == IO::RTMPConnectionStatus::Connecting) {
    return publishHeaders();
  }

  rtmpConnectionStatus = IO::RTMPConnectionStatus::Connecting;
  outputEventManager.publishEvent(OutputEventManager::EventType::Connecting, ConnectingMessage);

  bool success = false;
  std::string failReason;
  std::string failCode;

#if defined(_WIN32)
  WORD version;
  WSADATA wsaData;
  version = MAKEWORD(1, 1);
  WSAStartup(version, &wsaData);
#endif
  {
    std::lock_guard<std::mutex> lk(rtmpMutex);
    rtmp = std::unique_ptr<RTMP, std::function<void(RTMP*)>>(RTMP_Alloc(), [](RTMP* data) {
      RTMP_Close(data);
      RTMP_Free(data);
    });
    {
      std::lock_guard<std::mutex> lock(IO::rtmpInitMutex);

      // global SSL init may not be thread safe
      RTMP_Init(rtmp.get());
    }
  }

  if (pubUser.size() > 0) {
    rtmp->Link.pubUser.av_val = (char*)pubUser.c_str();
    rtmp->Link.pubUser.av_len = (int)pubUser.size();
  }
  if (pubPasswd.size() > 0) {
    rtmp->Link.pubPasswd.av_val = (char*)pubPasswd.c_str();
    rtmp->Link.pubPasswd.av_len = (int)pubPasswd.size();
  }

  rtmp->Link.swfUrl.av_len = rtmp->Link.tcUrl.av_len;
  rtmp->Link.swfUrl.av_val = rtmp->Link.tcUrl.av_val;
  rtmp->Link.flashVer.av_val = (char*)flashVer.c_str();
  rtmp->Link.flashVer.av_len = (int)flashVer.size();
  rtmp->Link.timeout = rtmpTimeout;

  if (!RTMP_SetupURL(rtmp.get(), const_cast<char*>(URL.c_str()))) {
    failReason = std::string("Could not parse the URL");
    failCode = BadUrlMessage;
    goto end;
  }

  RTMP_EnableWrite(rtmp.get());

  if (!RTMP_Connect(rtmp.get(), nullptr)) {
    failReason = std::string("Could not connect");
    failCode = NetworkErrorMessage;
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
    failReason = std::string("Invalid stream");
    failCode = RTMPConnectionRefusedMessage;
    goto end;
  }

  success = true;

end:
  if (!success) {
    {
      std::lock_guard<std::mutex> lk(rtmpMutex);
      rtmp.reset();
    }
    Logger::error(RTMPtag) << "Connection to " << URL << " failed: " << failReason.c_str() << std::endl;
    rtmpConnectionStatus = IO::RTMPConnectionStatus::Disconnected;
    outputEventManager.publishEvent(OutputEventManager::EventType::Disconnected, failCode);
    return false;
  } else {
    Logger::info(RTMPtag) << "Connected to " << URL << std::endl;
    return publishHeaders();
  }
}

// ---------------------- Packets sending -------------------

/**
 * Send loop : empty the packet buffer
 */
void RTMPPublisher::sendLoop() {
  int64_t initialTimestamp = INIT_TIME_STAMP;
  int64_t lastAudioTimestamp = INIT_TIME_STAMP;
  int64_t lastVideoTimestamp = INIT_TIME_STAMP;

  for (;;) {
    if (stopping) {
      return;
    }

    // reconnection in the background thread
    if (rtmpConnectionStatus != IO::RTMPConnectionStatus::Connected) {
      firstAudioPacket = true;
      firstVideoPacket = true;
      dropped = false;

      Logger::info(RTMPtag) << "Reconnecting to " << URL << std::endl;
      if (!connect()) {
        if (!stopping) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        continue;
      }

      // flush the queue before resuming streaming
      // look for the last keyframe, drop everything before it
      std::lock_guard<std::mutex> lk(dataMutex);
      while (dropGop()) {
      }
    }

    VideoStitch::IO::DataPacket::Storage packetData;
    size_t packetSize;
    VideoStitch::IO::PacketType packetType;
    int64_t packetTimestamp;
    {
      std::unique_lock<std::mutex> lk(dataMutex);
      sendCond.wait(lk, [this] { return queuedPackets.size() > 0 || stopping; });
      if (stopping) {
        return;
      }
      packetType = queuedPackets[0].type;
      packetTimestamp = queuedPackets[0].timestamp;
      packetData = queuedPackets[0].storage();  // Keeps a reference to the storage
      packetSize = queuedPackets[0].size();
      queuedPackets.pop_front();
    }

    if (initialTimestamp == INIT_TIME_STAMP) {
      initialTimestamp = packetTimestamp;
    }

    Logger::debug(RTMPtag) << "Sending " << ((packetType == VideoStitch::IO::PacketType_Audio) ? "audio" : "video")
                           << " packet at timestamp " << packetTimestamp - initialTimestamp
                           << " with a payload of size " << packetSize << std::endl;

    RTMPPacket packet;
    RTMPPacket_Alloc(&packet, (uint32_t)packetSize);
    if (packetType == VideoStitch::IO::PacketType_Audio) {
      // Check delta
      lastAudioTimestamp = lastAudioTimestamp == INIT_TIME_STAMP ? packetTimestamp : lastAudioTimestamp;
      int64_t delta = packetTimestamp - lastAudioTimestamp;
      if (delta < 0) {
        Logger::warning(RTMPtag) << "Audio output is not monotonous, got " << lastAudioTimestamp << " + " << delta
                                 << std::endl;
      }

      lastAudioTimestamp = packetTimestamp;

      packet.m_nChannel = 0x5;
      packet.m_packetType = RTMP_PACKET_TYPE_AUDIO;
      if (firstAudioPacket) {
        packet.m_headerType = RTMP_PACKET_SIZE_LARGE;
        firstAudioPacket = false;
      } else {
        packet.m_headerType = RTMP_PACKET_SIZE_MEDIUM;
      }
    } else {
      // Check latency

      mtime_t currentLatency = inputVideoTimestamp - packetTimestamp;

      if (this->latency < currentLatency) {
        this->latency = currentLatency;

        Logger::verbose(RTMPtag) << "Video latency increased to " << this->getLatency() << " ms for " << URL
                                 << std::endl;
        size_t framesWaiting;
        {
          std::lock_guard<std::mutex> lock(dataBufferMutex);
          framesWaiting = bufferedVideo.size();
        }
        Logger::verbose(RTMPtag) << "  " << framesWaiting << " video frames are waiting for audio synchronization"
                                 << std::endl;
      }
      // Check delta
      lastVideoTimestamp = lastVideoTimestamp == INIT_TIME_STAMP ? packetTimestamp : lastVideoTimestamp;
      if (getFrameRate().num != 0) {
        int64_t millisecondsPerFrame =
            getFrameRate().den * 1000 / getFrameRate().num + 1;  // +1 to take rounding into account
        int64_t delta = packetTimestamp - lastVideoTimestamp;
        if (delta < 0 || delta > millisecondsPerFrame) {
          Logger::warning(RTMPtag) << "Video output is not monotonous, got " << lastVideoTimestamp << " + " << delta
                                   << std::endl;
        }
      }
      lastVideoTimestamp = packetTimestamp;

      packet.m_nChannel = 0x4;
      packet.m_packetType = RTMP_PACKET_TYPE_VIDEO;
      if (firstVideoPacket) {
        packet.m_headerType = RTMP_PACKET_SIZE_LARGE;
        firstVideoPacket = false;
      } else {
        packet.m_headerType = RTMP_PACKET_SIZE_MEDIUM;
      }
    }
    packet.m_nInfoField2 = rtmp->m_stream_id;
    packet.m_hasAbsTimestamp = TRUE;
    packet.m_nTimeStamp = uint32_t(packetTimestamp - initialTimestamp);
    if (packet.m_nTimeStamp != (packetTimestamp - initialTimestamp)) {
      Logger::warning(RTMPtag) << "Packet timestamp overflow : " << packetTimestamp - initialTimestamp << std::endl;
    }

    memcpy(packet.m_body, packetData.get(), packetSize);
    packet.m_nBodySize = (uint32_t)packetSize;

    if (!RTMP_SendPacket(rtmp.get(), &packet, FALSE)) {
      rtmpConnectionStatus = IO::RTMPConnectionStatus::Disconnected;
      if (packet.m_nTimeStamp) {
        Logger::error(RTMPtag) << "RTMP_SendPacket failed" << std::endl;
        outputEventManager.publishEvent(OutputEventManager::EventType::Disconnected, NetworkErrorMessage);
      } else {
        Logger::error(RTMPtag) << "RTMP_SendPacket failed on first packet, check publishing rights." << std::endl;
        outputEventManager.publishEvent(OutputEventManager::EventType::Disconnected, RTMPConnectionRefusedMessage);
      }
    }
    RTMPPacket_Free(&packet);
  }
}

void RTMPPublisher::sendPacket(const VideoStitch::IO::DataPacket& pkt) {
  Logger::debug(RTMPtag) << "Queueing a packet with a payload of size " << pkt.size() << std::endl;
  {
    std::lock_guard<std::mutex> lk(dataMutex);
    if (rtmpConnectionStatus == IO::RTMPConnectionStatus::Connected) {
      std::lock_guard<std::mutex> lk(updateMutex);
      if ((minBitrate > -1) && videoEncoder->dynamicBitrateSupported()) {
        if (bitRateTimeOut < 0) {
          int32_t targetBitRate;
          // test if bitrate should be decreased
          if ((queuedPackets.size() > queuedMax) && ((int32_t)videoEncoder->getBitRate() > minBitrate)) {
            queuedMax = (uint32_t)queuedPackets.size();
            targetBitRate = int32_t(0.9 * videoEncoder->getBitRate());
            if (targetBitRate < minBitrate) {
              targetBitRate = minBitrate;
            }
            videoEncoder->setBitRate(targetBitRate, uint32_t(-1));
            outputEventManager.publishEvent(OutputEventManager::EventType::Connected,
                                            ConnectedMessage + "@" + std::to_string(targetBitRate) + "kbits/s");
            bitRateTimeOut = (int)PACKET_QUEUE_LIMIT / 2;
          } else if ((queuedPackets.size() < PACKET_QUEUE_LIMIT / 8) &&
                     ((int32_t)videoEncoder->getBitRate() < videoEncoder->getMaxBitRate())) {
            // test if bitrate should be increased
            queuedMax = PACKET_QUEUE_LIMIT / 4;
            targetBitRate = int32_t(1.02 * videoEncoder->getBitRate());
            if (targetBitRate > videoEncoder->getMaxBitRate()) {
              targetBitRate = videoEncoder->getMaxBitRate();
              outputEventManager.publishEvent(OutputEventManager::EventType::Connected, ConnectedMessage);
            } else {
              outputEventManager.publishEvent(OutputEventManager::EventType::Connected,
                                              ConnectedMessage + "@" + std::to_string(targetBitRate) + "kbits/s");
            }
            videoEncoder->setBitRate(targetBitRate, uint32_t(-1));
            bitRateTimeOut = (int)PACKET_QUEUE_LIMIT;
          }
        } else {
          bitRateTimeOut--;
        }
      } else if (dropped && (queuedPackets.size() == 0)) {
        /* Network Congestion is gone */
        dropped = false;
        outputEventManager.publishEvent(OutputEventManager::EventType::Connected, ConnectedMessage);
      }
    }
    queuedPackets.push_back(pkt);
    if (queuedPackets.size() > PACKET_QUEUE_LIMIT) {
      Logger::warning(RTMPtag) << "Dropping a GOP, the upload bandwidth is too small on " << URL << std::endl;
      dropGop();
      if (rtmpConnectionStatus == IO::RTMPConnectionStatus::Connected) {
        dropped = true;
        outputEventManager.publishEvent(OutputEventManager::EventType::Connected, CongestionMessage);
        queuedMax = PACKET_QUEUE_LIMIT / 4;
      }
    }
  }

  // this will wake up the RTMP send thread
  sendCond.notify_all();
}

bool RTMPPublisher::dropGop() {
  bool drop = false;
  size_t i;
  // we start at 2, so that successive drops skip the first sps & keyframe and dump
  // the whole GOP if the next keyframe is here
  for (i = 2; i < queuedPackets.size(); ++i) {
    VideoStitch::IO::DataPacket pkt = queuedPackets[i];
    if (pkt.type == VideoStitch::IO::PacketType_VideoSPS || pkt.type == VideoStitch::IO::PacketType_VideoHighest) {
      drop = true;
      break;
    }
  }
  if (drop) {
    queuedPackets.erase(queuedPackets.begin(), queuedPackets.begin() + i);
  }
  return drop;
}

void RTMPPublisher::sendPacketFromQueue(std::deque<IO::DataPacket>& wQueue) {
  if (wQueue.empty()) {
    return;
  }
  sendPacket(wQueue.front());
  wQueue.pop_front();
}

void RTMPPublisher::sendPacketsUntilLimit(std::deque<IO::DataPacket>& wQueue, const size_t limit) {
  if (limit && (wQueue.size() > limit)) {
    Logger::warning(RTMPtag) << "Queue will be flushed " << URL << " " << wQueue.size() << " packets left" << std::endl;
  }
  while (wQueue.size() > limit) {
    sendPacketFromQueue(wQueue);
  }
}
}  // namespace Output
}  // namespace VideoStitch
