// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <algorithm>
#include "NvV4l2Encoder.hpp"
#include "ptvMacro.hpp"

#include <sstream>
#include "libvideostitch/logging.hpp"

#define TEST_ERROR(cond, str)                                                                               \
  if (cond) {                                                                                               \
    return Status(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__, str); \
  }

const std::string __NVENC__("NVENC");

namespace VideoStitch {
namespace Output {

static inline void resetCudaError() { cudaGetLastError(); }

NvV4l2Encoder::NvV4l2Encoder(EncodeConfig& config) : ctx(config), index(0), first_dts(0) {
  queue = new std::deque<std::pair<struct v4l2_buffer, NvBuffer*>>;

  cudaError_t r = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (r != cudaSuccess) {
    Logger::error(__NVENC__) << "cudaStreamCreateWithFlags has returned CUDA error " << r << std::endl;
  }

  enc = NvVideoEncoder::createVideoEncoder("NvV4l2Encoder", 0);
}

NvV4l2Encoder::~NvV4l2Encoder() {
  cudaError_t r = cudaStreamSynchronize(stream);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::warning(__NVENC__) << "cudaStreamSynchronize has returned CUDA error " << r << std::endl;
  }

  r = cudaStreamDestroy(stream);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::warning(__NVENC__) << "cudaStreamDestroy failed with code " << r << std::endl;
  }

  if (ctx.fOutput != NULL) {
    fclose(ctx.fOutput);
  }
  delete queue;
}

void NvV4l2Encoder::supportedEncoders(std::vector<std::string>& codecs) {
  codecs.push_back("h264_nvenc");
  codecs.push_back("hevc_nvenc");
}

Potential<VideoEncoder> NvV4l2Encoder::createNvV4l2Encoder(const Ptv::Value& config, int width, int height,
                                                           FrameRate framerate) {
  std::string bitrate_mode = "";
  std::string filename = "";
  int rcMode = (int)V4L2_MPEG_VIDEO_BITRATE_MODE_CBR;
  int target_usage = 1;  // trade-off between quality and speed, from 1 (quality) to 7 (fastest)
  EncodeConfig encodeConfig = {};
  encodeConfig.bitrate = 5000;  // in Kbps
  encodeConfig.encoder_pixfmt = V4L2_PIX_FMT_H264;
  encodeConfig.fps = framerate;
  encodeConfig.gopLength = 250;
  encodeConfig.encoderProfile = "baseline";
  encodeConfig.encoderLevel = "5.1";
  encodeConfig.width = width;
  encodeConfig.height = height;
  encodeConfig.fOutput = NULL;

  if (Parse::populateInt("RTMP", config, "bitrate", encodeConfig.bitrate, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid bitrate type (int) in Kbps.");
  }
  if (Parse::populateInt("RTMP", config, "target_usage", target_usage, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid target usage (int).");
  }
  encodeConfig.preset = (7 - target_usage) / 2;  // trade-off between quality and speed, from 3 (quality) to 0 (fastest)

  if (Parse::populateInt("RTMP", config, "buffer_size", encodeConfig.vbvSize, false) ==
      Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid vbvSize (int) in Kb.");
  }
  if (Parse::populateString("RTMP", config, "bitrate_mode", bitrate_mode, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid rate control mode type (string).");
  }
  if (bitrate_mode != "") {
    if (stricmp(bitrate_mode.c_str(), "cbr") == 0) {
      rcMode = (int)V4L2_MPEG_VIDEO_BITRATE_MODE_CBR;
    } else if (stricmp(bitrate_mode.c_str(), "vbr") == 0) {
      rcMode = (int)V4L2_MPEG_VIDEO_BITRATE_MODE_VBR;
    }
  }
  if (Parse::populateInt("RTMP", config, "rcmode", rcMode, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid rcmode type (int).");
  }
  encodeConfig.rcMode = (enum v4l2_mpeg_video_bitrate_mode)rcMode;

  if (Parse::populateInt("RTMP", config, "gop", encodeConfig.gopLength, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid goplength type (int).");
  }
  if (Parse::populateInt("RTMP", config, "b_frames", encodeConfig.numB, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid numB type (int).");
  }

  if (Parse::populateString("RTMP", config, "profile", encodeConfig.encoderProfile, false) ==
      Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid profile type (string).");
  }

  if (Parse::populateString("RTMP", config, "level", encodeConfig.encoderLevel, false) ==
      Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid level type (string).");
  }

  std::string codec;
  if (Parse::populateString("RTMP", config, "codec", codec, false) == VideoStitch::Parse::PopulateResult_Ok) {
    Logger::info(__NVENC__) << "'codec' parameter deprecated, use 'video_codec' instead" << std::endl;
  }
  if (Parse::populateString("RTMP", config, "video_codec", codec, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid video_codec type (string). Supported codecs: h264_nvenc, hevc_nvenc");
  }

  if (codec.find("hevc") != std::string::npos) {
    encodeConfig.encoder_pixfmt = V4L2_PIX_FMT_H265;
  }

  if (Parse::populateString("RTMP", config, "bitstream_name", filename, false) == Parse::PopulateResult_WrongType) {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "invalid file name type (string).");
  }
  if (filename != "") {
    encodeConfig.fOutput = fopen(filename.c_str(), "wb");
  }

  auto encoder = new NvV4l2Encoder(encodeConfig);
  Status initEnc = encoder->Initialize();
  if (initEnc.ok()) {
    return Potential<VideoEncoder>(encoder);
  } else {
    return Potential<VideoEncoder>(VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration, __NVENC__,
                                   "could not instantiate the NVENC encoder", initEnc);
  }
}

bool NvV4l2Encoder::encode(const Frame& frame, std::vector<VideoStitch::IO::DataPacket>& packets) {
  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];
  NvBuffer* buffer;

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

  v4l2_buf.m.planes = planes;

  /* Process pending packets if any */
  if (!ProcessOutput(packets)) {
    Logger::error(__NVENC__) << "Failed to ProcessOutput" << std::endl;
    return false;
  }

  if (index < (int)enc->output_plane.getNumBuffers()) {
    buffer = enc->output_plane.getNthBuffer(index);
    v4l2_buf.index = index;
    index++;
  } else {
    int ret = enc->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
    if (ret < 0) {
      Logger::error(__NVENC__) << "Error DQing buffer at output plane" << std::endl;
      return false;
    }
  }

  v4l2_buf.timestamp.tv_sec = frame.pts / 1000000;
  v4l2_buf.timestamp.tv_usec = frame.pts - (v4l2_buf.timestamp.tv_sec * (mtime_t)1000000);

  cudaError_t r;
  for (int i = 0; i < (int)buffer->n_planes; i++) {
    r = cudaMemcpy2DAsync(buffer->planes[i].data, buffer->planes[i].fmt.stride, frame.planes[i], frame.pitches[i],
                          buffer->planes[i].fmt.width * buffer->planes[i].fmt.bytesperpixel,
                          buffer->planes[i].fmt.height, cudaMemcpyDeviceToHost, stream);
    buffer->planes[i].bytesused = buffer->planes[i].fmt.stride * buffer->planes[i].fmt.height;
    if (r != cudaSuccess) {
      resetCudaError();
      Logger::error(__NVENC__) << "cudaMemcpy2DAsync has returned CUDA error " << r << std::endl;
      return false;
    }
  }

  r = cudaStreamSynchronize(stream);
  if (r != cudaSuccess) {
    resetCudaError();
    Logger::error(__NVENC__) << "cudaStreamSynchronize has returned CUDA error " << r << std::endl;
    return false;
  }

  timestamps.push(frame.pts);

  int ret = enc->output_plane.qBuffer(v4l2_buf, NULL);
  if (ret < 0) {
    Logger::error(__NVENC__) << "Error Qing buffer at output plane" << std::endl;
    return false;
  }
  if (v4l2_buf.m.planes[0].bytesused == 0) {
    Logger::warning(__NVENC__) << "Input file read complete" << std::endl;
  }

  return true;
}

const AVal NvV4l2Encoder::av_videocodecid = mAVC("videocodecid");
const AVal NvV4l2Encoder::av_videodatarate = mAVC("videodatarate");
const AVal NvV4l2Encoder::av_framerate = mAVC("framerate");

char* NvV4l2Encoder::metadata(char* enc, char* pend) {
  enc = AMF_EncodeNamedNumber(enc, pend, &av_videocodecid, 7.);
  enc = AMF_EncodeNamedNumber(enc, pend, &av_videodatarate, double(ctx.bitrate));
  enc = AMF_EncodeNamedNumber(enc, pend, &av_framerate, double(ctx.fps.num) / (double)ctx.fps.den);
  return enc;
}

// ----------- private

Status NvV4l2Encoder::Initialize() {
  ctx.profile = getProfile(ctx.encoderProfile.c_str(), ctx.encoder_pixfmt);
  if (ctx.profile == V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE) {
    /* There is no B-frames in H264 Baseline profile */
    Logger::info(__NVENC__) << "B-frames not supported in Baseline profile, discarding b_frames parameters"
                            << std::endl;
    ctx.numB = 0;
  }

  Logger::info(__NVENC__) << "Encoding codec           : "
                          << (ctx.encoder_pixfmt == V4L2_PIX_FMT_H265 ? "HEVC" : "H264") << std::endl;
  Logger::info(__NVENC__) << "         size            : " << ctx.width << "x" << ctx.height << std::endl;
  Logger::info(__NVENC__) << "         bitrate         : " << ctx.bitrate << " kbits/sec" << std::endl;
  Logger::info(__NVENC__) << "         vbvSize         : " << ctx.vbvSize << " kbits" << std::endl;
  Logger::info(__NVENC__) << "         fps             : " << ctx.fps << " frames/sec" << std::endl;
  Logger::info(__NVENC__) << "         rcMode          : "
                          << (ctx.rcMode == V4L2_MPEG_VIDEO_BITRATE_MODE_VBR
                                  ? "VBR"
                                  : ctx.rcMode == V4L2_MPEG_VIDEO_BITRATE_MODE_CBR ? "CBR" : "UNKNOWN")
                          << std::endl;
  if (ctx.gopLength == (int(-1))) {
    Logger::info(__NVENC__) << "         goplength       : INFINITE GOP" << std::endl;
  } else {
    Logger::info(__NVENC__) << "         goplength       : " << ctx.gopLength << std::endl;
  }
  Logger::info(__NVENC__) << "         B frames        : " << ctx.numB << std::endl;
  Logger::info(__NVENC__) << "         profile         : "
                          << ((ctx.profile == V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE)
                                  ? "BASELINE"
                                  : (ctx.profile == V4L2_MPEG_VIDEO_H264_PROFILE_MAIN)
                                        ? "MAIN"
                                        : (ctx.profile == V4L2_MPEG_VIDEO_H264_PROFILE_HIGH)
                                              ? "HIGH"
                                              : (ctx.profile == V4L2_MPEG_VIDEO_H264_PROFILE_HIGH_444_PREDICTIVE)
                                                    ? "HIGH444"
                                                    : (ctx.profile == V4L2_MPEG_VIDEO_H264_PROFILE_STEREO_HIGH)
                                                          ? "STEREO"
                                                          : "UNKNOWN")
                          << std::endl;

  TEST_ERROR(!enc, "Could not create encoder");

  // It is necessary that Capture Plane format be set before Output Plane format.
  // Set encoder capture plane format. It is necessary to set width and height on the capture plane as well
  int ret = enc->setCapturePlaneFormat(ctx.encoder_pixfmt, ctx.width, ctx.height, BITSTREAM_BUFFER_SIZE);
  TEST_ERROR(ret < 0, "Could not set output plane format");

  // Set encoder output plane format
  ret = enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, ctx.width, ctx.height);
  TEST_ERROR(ret < 0, "Could not set output plane format");

  ret = enc->setBitrate(1000 * ctx.bitrate);
  TEST_ERROR(ret < 0, "Could not set encoder bitrate");

  ret = enc->setProfile(ctx.profile);
  TEST_ERROR(ret < 0, "Could not set encoder profile");

  ret = setPreset(ctx.preset);
  TEST_ERROR(ret < 0, "Could not set hw preset");

  if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H264) {
    ret = enc->setLevel(getLevel(ctx.encoderLevel.c_str(), ctx.encoder_pixfmt));
    TEST_ERROR(ret < 0, "Could not set encoder level");
  }

  ret = enc->setRateControlMode(ctx.rcMode);
  TEST_ERROR(ret < 0, "Could not set encoder rate control mode");

  ret = enc->setIDRInterval(ctx.gopLength);
  TEST_ERROR(ret < 0, "Could not set encoder IDR interval");

  ret = enc->setIFrameInterval(ctx.gopLength);
  TEST_ERROR(ret < 0, "Could not set encoder I-Frame interval");

  ret = enc->setFrameRate(ctx.fps.num, ctx.fps.den);
  TEST_ERROR(ret < 0, "Could not set framerate");

  if (ctx.vbvSize) {
    ret = enc->setVirtualBufferSize(1000 * ctx.vbvSize);
  }
  TEST_ERROR(ret < 0, "Could not set virtual buffer size");

  ret = enc->setInsertSpsPpsAtIdrEnabled(true);
  TEST_ERROR(ret < 0, "Could not set insertSPSPPSAtIDR");

  if (ctx.numB) {
    ret = enc->setNumBFrames(ctx.numB);
    TEST_ERROR(ret < 0, "Could not set number of B Frames");
  }

  // Query, Export and Map the output plane buffers so that we can read
  // raw data into the buffers
  ret = enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
  TEST_ERROR(ret < 0, "Could not setup output plane");

  // Query, Export and Map the output plane buffers so that we can write
  // encoded data from the buffers
  ret = enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
  TEST_ERROR(ret < 0, "Could not setup capture plane");

  // output plane STREAMON
  ret = enc->output_plane.setStreamStatus(true);
  TEST_ERROR(ret < 0, "Error in output plane streamon");

  // capture plane STREAMON
  ret = enc->capture_plane.setStreamStatus(true);
  TEST_ERROR(ret < 0, "Error in capture plane streamon");

  enc->capture_plane.setDQThreadCallback(encoder_capture_plane_dq_callback);

  // startDQThread starts a thread internally which calls the
  // encoder_capture_plane_dq_callback whenever a buffer is dequeued
  // on the plane
  enc->capture_plane.startDQThread(this);

  // Enqueue all the empty capture plane buffers
  for (uint32_t i = 0; i < enc->capture_plane.getNumBuffers(); i++) {
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

    v4l2_buf.index = i;
    v4l2_buf.m.planes = planes;

    ret = enc->capture_plane.qBuffer(v4l2_buf, NULL);
    TEST_ERROR(ret < 0, "Error while queueing buffer at capture plane");
  }

  return Status::OK();
}

int NvV4l2Encoder::getProfile(const char* encoderProfile, int /*codec*/) {
  int profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;

  if (encoderProfile && (stricmp(encoderProfile, "baseline") == 0)) {
    profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
  } else if (encoderProfile && (stricmp(encoderProfile, "main") == 0)) {
    profile = V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
  } else if (encoderProfile && (stricmp(encoderProfile, "high") == 0)) {
    profile = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH;
  } else if (encoderProfile && (stricmp(encoderProfile, "constrained_baseline") == 0)) {
    profile = V4L2_MPEG_VIDEO_H264_PROFILE_CONSTRAINED_BASELINE;
  } else if (encoderProfile && (stricmp(encoderProfile, "high444") == 0)) {
    profile = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH_444_PREDICTIVE;
  } else if (encoderProfile && (stricmp(encoderProfile, "stereo") == 0)) {
    profile = V4L2_MPEG_VIDEO_H264_PROFILE_STEREO_HIGH;
  } else {
    if (encoderProfile && (stricmp(encoderProfile, "") != 0)) {
      Logger::error(__NVENC__) << "Unsupported profile " << encoderProfile << std::endl;
    }
    profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
  }

  return profile;
}

enum v4l2_mpeg_video_h264_level NvV4l2Encoder::getLevel(const char* encoderLevel, int /*codec*/) {
  enum v4l2_mpeg_video_h264_level level = V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
  if (encoderLevel) {
    if (stricmp(encoderLevel, "5.1") == 0) {
      level = V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
    } else if (stricmp(encoderLevel, "5.0") == 0) {
      level = V4L2_MPEG_VIDEO_H264_LEVEL_5_0;
    } else if (stricmp(encoderLevel, "4.2") == 0) {
      level = V4L2_MPEG_VIDEO_H264_LEVEL_4_2;
    } else if (stricmp(encoderLevel, "4.1") == 0) {
      level = V4L2_MPEG_VIDEO_H264_LEVEL_4_1;
    } else if (stricmp(encoderLevel, "4.0") == 0) {
      level = V4L2_MPEG_VIDEO_H264_LEVEL_4_0;
    } else if (stricmp(encoderLevel, "3.2") == 0) {
      level = V4L2_MPEG_VIDEO_H264_LEVEL_3_2;
    } else if (stricmp(encoderLevel, "3.1") == 0) {
      level = V4L2_MPEG_VIDEO_H264_LEVEL_3_1;
    } else if (stricmp(encoderLevel, "3.0") == 0) {
      level = V4L2_MPEG_VIDEO_H264_LEVEL_3_0;
    }
  }

  return level;
}

bool NvV4l2Encoder::encoder_capture_plane_dq_callback(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer,
                                                      NvBuffer* shared_buffer, void* arg) {
  NvV4l2Encoder* ctx = reinterpret_cast<NvV4l2Encoder*>(arg);

  if (v4l2_buf == NULL) {
    Logger::error(__NVENC__) << "Error while dequeing buffer from output plane" << std::endl;
    return false;
  }

  // GOT EOS from encoder. Stop dqthread.
  if (buffer->planes[0].bytesused == 0) {
    return false;
  }

  std::lock_guard<std::mutex> lk(ctx->queue_lock);
  ctx->queue->push_back(std::make_pair(*v4l2_buf, buffer));
  return true;
}

bool NvV4l2Encoder::ProcessOutput(std::vector<VideoStitch::IO::DataPacket>& packets) {
  std::unique_lock<std::mutex> lock(queue_lock);
  if (queue->empty()) {
    return true;
  }

  auto buffer = queue->front();
  queue->pop_front();
  queue_lock.unlock();

  if (ctx.fOutput != NULL) {
    fwrite((char*)buffer.second->planes[0].data, 1, buffer.second->planes[0].bytesused, ctx.fOutput);
  }

  mtime_t dts = timestamps.front();
  mtime_t outputTimeStamp = buffer.first.timestamp.tv_sec * (mtime_t)100000 + buffer.first.timestamp.tv_usec / 10;

  /* when there're b frame(s), set dts offset */
  if ((ctx.numB >= 0) && (first_dts == 0)) {
    dts = (mtime_t)(dts - ((1000000.0 * ctx.fps.den) / ctx.fps.num));
    first_dts = 1;
  } else {
    timestamps.pop();
    // TODO : improve timestamp management as it seems that buffer.first.timestamp is not the expected PTS
    outputTimeStamp = dts;
  }

  if (dts > outputTimeStamp) {
    Logger::error(__NVENC__) << "dts " << dts << " value has to be lower than pts " << outputTimeStamp << std::endl;
    dts = outputTimeStamp;
  }

  v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
  if (enc->getMetadata(buffer.first.index, enc_metadata) == 0) {
    // get those packets!
    std::vector<x264_nal_t> nalOut;
    x264_nal_t nal;
    nal.i_type = enc_metadata.KeyFrame ? NAL_SLICE_IDR : NAL_SLICE;
    nal.i_ref_idc = enc_metadata.KeyFrame ? NAL_PRIORITY_HIGHEST : NAL_PRIORITY_HIGH;
    nal.p_payload = (uint8_t*)buffer.second->planes[0].data;
    nal.i_payload = int(buffer.second->planes[0].bytesused);
    nalOut.push_back(nal);

    VideoEncoder::createDataPacket(nalOut, packets, mtime_t(std::round(outputTimeStamp / 1000.0)),
                                   mtime_t(std::round(dts / 1000.0)));
  }
  if (enc->capture_plane.qBuffer(buffer.first, NULL) < 0) {
    Logger::error(__NVENC__) << "Error while Qing buffer at capture plane" << std::endl;
    return false;
  }

  return true;
}

// This function is not yet available through NvVideoEncoder
int NvV4l2Encoder::setPreset(uint32_t preset) {
  v4l2_enc_hw_preset_type_param hwpreset;
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  hwpreset.hw_preset_type = (enum v4l2_enc_hw_preset_type)preset;
  hwpreset.set_max_enc_clock = true;

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEOENC_HW_PRESET_TYPE_PARAM;
  control.string = (char*)&hwpreset;

  int ret = enc->setExtControls(ctrls);
  if (ret < 0) {
    Logger::error(__NVENC__) << "Setting encoder HW preset" << std::endl;
    return false;
  }
  return true;
}
}  // namespace Output
}  // namespace VideoStitch
