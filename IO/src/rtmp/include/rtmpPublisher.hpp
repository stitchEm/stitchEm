// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "rtmpEnums.hpp"
#include "rtmpStructures.hpp"

#include "audioEncoder.hpp"

#include "libvideostitch/imuData.hpp"
#include "libvideostitch/plugin.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include "librtmpIncludes.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <memory>
#include <mutex>
#include <thread>

namespace VideoStitch {
namespace Output {

class VideoEncoder;
class AudioEncoder;

class RTMPPublisher : public VideoWriter, public AudioWriter {
 public:
  virtual ~RTMPPublisher();

  static Potential<RTMPPublisher> create(const Ptv::Value* config,
                                         const VideoStitch::Plugin::VSWriterPlugin::Config& runtime);

  static bool handles(const Ptv::Value* config);

  void updateConfig(const Ptv::Value&) override;
  void pushVideo(const Frame& videoFrame) override;
  void pushMetadataIMU(const VideoStitch::IMU::Measure& imuData);
  void pushAudio(Audio::Samples& audioSamples) override;

 private:
  static const size_t AUDIO_BUFFER_LIMIT;
  static const size_t VIDEO_BUFFER_LIMIT;
  static const size_t PACKET_QUEUE_LIMIT;
  static const int rtmpTimeout;  // in seconds

  RTMPPublisher(const VideoStitch::Plugin::VSWriterPlugin::Config& runtime, Potential<VideoEncoder> videoEncoder,
                std::unique_ptr<AudioEncoder> audioEncoder, const std::string& pubUser, const std::string& pubPasswd,
                const std::string& flashVer, const std::string& codecVer, VideoStitch::PixelFormat fmt,
                AddressSpace type, int32_t minBitrate);

  void mux();
  bool connect();
  bool publishHeaders();

  void sendPacket(const VideoStitch::IO::DataPacket& pkt);
  void sendPacketFromQueue(std::deque<VideoStitch::IO::DataPacket>& wQueue);
  void sendPacketsUntilLimit(std::deque<VideoStitch::IO::DataPacket>& wQueue, const size_t limit = 0);
  bool dropGop();  // drop the current GOP, leaving at least the next keyframe, return false if nothing to drop
                   // you must acquire the dataMutex before calling it

  void sendLoop();

  std::unique_ptr<AudioEncoder> audioEncoder;
  Potential<VideoEncoder> videoEncoder;

  // muxing data structures
  std::deque<VideoStitch::IO::DataPacket> bufferedAudio;
  std::deque<VideoStitch::IO::DataPacket> bufferedVideo;

  bool firstAudioPacket = true;
  bool firstVideoPacket = true;
  std::atomic<mtime_t> inputVideoTimestamp; /* in ms */

  //-----------------------------------------------
  // stream startup stuff
  const std::string URL;

  std::atomic<IO::RTMPConnectionStatus> rtmpConnectionStatus;
  std::atomic<bool> stopping;

  std::vector<char> metaDataPacketBuffer;

  std::thread* sendThread;
  std::condition_variable sendCond;
  std::mutex dataMutex;
  std::deque<VideoStitch::IO::DataPacket> queuedPackets;

  //-----------------------------------------------

  std::unique_ptr<RTMP, std::function<void(RTMP*)>> rtmp;

  const std::string pubUser;
  const std::string pubPasswd;

  const std::string flashVer;

  std::mutex dataBufferMutex;  ///<  protects bufferedAudio and bufferedVideo
  std::mutex rtmpMutex;

  std::mutex updateMutex;
  uint32_t queuedMax;
  int32_t minBitrate;
  int bitRateTimeOut;
  bool dropped;
};

}  // namespace Output
}  // namespace VideoStitch
