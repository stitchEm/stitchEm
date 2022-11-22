// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "filemuxer.hpp"
#include "livemuxer.hpp"
#include "util.hpp"

#include "avMuxer.hpp"
#include "avWriter.hpp"

#include "libvideostitch/config.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/frame.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/audio.hpp"
#include "libvideostitch/gpu_device.hpp"

#ifdef SUP_NVENC
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <sstream>
#include <deque>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#undef PixelFormat
}

static std::string AVtag("libavoutput");

namespace VideoStitch {
namespace Output {

const std::string ConnectingMessage = "AVConnecting";
const std::string ConnectedMessage = "AVConnected";
const std::string DisconnectedMessage = "AVCreationFailed";
static const std::string typeToString(MuxerThreadStatus type) {
  switch (type) {
    case MuxerThreadStatus::CreateError:
      return DisconnectedMessage;
    case MuxerThreadStatus::WriteError:
      return "AVWriteError";
    case MuxerThreadStatus::EncodeError:
      return "AVEncodeError";
    case MuxerThreadStatus::TimeOutError:
      return "TimeOutError";
    case MuxerThreadStatus::NetworkError:
      return "NetworkError";
    case MuxerThreadStatus::OK:
      return "None";
  }
  assert(false);
  return "Invalid error type";
}

#ifdef SUP_NVENC
static inline void resetCudaError() { cudaGetLastError(); }
#endif

// ------------------------------ Configuration ----------------------------

// Check if the codec can support config sample format
static bool check_sample_fmt(const AVCodec* codec, enum AVSampleFormat sample_fmt) {
  const enum AVSampleFormat* p = codec->sample_fmts;
  while (*p != AV_SAMPLE_FMT_NONE) {
    if (*p == sample_fmt) {
      return true;
    }
    p++;
  }
  return false;
}

// parse the configuration to get the pixel format (necessary for the OutputWriter constructor)
VideoStitch::PixelFormat getVideoStitchPixelFormat(const Ptv::Value& config) {
  VideoStitch::PixelFormat fmt = VideoStitch::Unknown;

  // register all codecs, demux and protocols
  Util::Libav::checkInitialization();

  // CODEC
  std::string avCodec(LIBAV_WRITER_DEFAULT_CODEC);
  if (Parse::populateString("LibavOutputWriter", config, "video_codec", avCodec, false) ==
      Parse::PopulateResult_WrongType) {
    return fmt;
  }

  // PIXEL FORMAT
  if (avCodec == "mjpeg") {
    fmt = VideoStitch::YV12;
  } else if (avCodec == "prores") {
    fmt = VideoStitch::YUV422P10;
  } else if ((avCodec == "h264_nvenc") || (avCodec == "hevc_nvenc")) {
    fmt = VideoStitch::NV12;
  } else if ((avCodec == "depth")) {
    fmt = VideoStitch::DEPTH;
  } else {
    // all other codecs
    fmt = VideoStitch::YV12;
  }

  return fmt;
}

// parse the configuration, perform sanity checks
Output* LibavWriter::create(const Ptv::Value& config, const std::string& name, const char*, unsigned width,
                            unsigned height, FrameRate framerate, const Audio::SamplingRate samplingRate,
                            const Audio::SamplingDepth samplingDepth, const Audio::ChannelLayout channelLayout) {
  std::string avCodec(LIBAV_WRITER_DEFAULT_CODEC);
  Parse::populateString("LibavOutputWriter", config, "video_codec", avCodec, false);
  Logger::info(AVtag) << "Creating encoder using " << avCodec << std::endl;
  AddressSpace outputType = ((avCodec == "h264_nvenc") || (avCodec == "hevc_nvenc")) ? Device : Host;

  // force cuda initialization in case it was not already initialized in calling thread
  if ((outputType == Device) && (!GPU::useDefaultBackendDevice().ok())) {
    return nullptr;
  }

  return new LibavWriter(config, name, getVideoStitchPixelFormat(config), outputType, width, height, framerate,
                         samplingRate, samplingDepth, channelLayout);
}

bool LibavWriter::createAudioCodec() {
  std::string audioCodecName;

  // AUDIO CODEC
  if (Parse::populateString("LibavOutputWriter", *m_config, "audio_codec", audioCodecName, false) ==
      Parse::PopulateResult_Ok) {
    // USE LibMP3Lale for Mp3 content
    if (audioCodecName == "mp3") {
      audioCodecName = "libmp3lame";
    }

    const AVCodec* audioCodec = avcodec_find_encoder_by_name(audioCodecName.c_str());
    if (!audioCodec) {
      Logger::error(AVtag) << "Audio codec: " << audioCodecName.c_str() << " not found, disable output." << std::endl;
      return false;
    }

    audioCodecContext = avcodec_alloc_context3(audioCodec);
    if (audioCodecContext == nullptr) {
      Logger::error(AVtag) << "Audio codec: " << audioCodecName.c_str() << " : unable to alloc context" << std::endl;
      return false;
    }
    audioCodecContext->codec_id = audioCodec->id;

    // BITRATE
    int audioBitrate = LIBAV_DEFAULT_AUDIO_BITRATE;
    if (Parse::populateInt("LibavOutputWriter", *m_config, "audio_bitrate", audioBitrate, true) !=
        Parse::PopulateResult_Ok) {
      Logger::error(AVtag) << "Audio codec: " << audioBitrate << " : invalid bitrate" << std::endl;
      return false;
    }

    audioCodecContext->bit_rate = audioBitrate * 1000;
    audioCodecContext->bit_rate_tolerance = 1000;

    // SAMPLE FORMAT
    std::string sample_format;
    if (Parse::populateString("LibavOutputWriter", *m_config, "sample_format", sample_format, true) !=
            Parse::PopulateResult_Ok ||
        av_get_sample_fmt(sample_format.c_str()) == 0) {
      Logger::error(AVtag) << "Audio codec: " << sample_format.c_str() << " : invalid sample format" << std::endl;
      return false;
    }

    if (!check_sample_fmt(audioCodec, av_get_sample_fmt(sample_format.c_str()))) {
      Logger::error(AVtag) << "Audio codec: " << audioCodecName.c_str() << " does not support sample format "
                           << sample_format.c_str() << std::endl;
      return false;
    }
    audioCodecContext->sample_fmt = av_get_sample_fmt(sample_format.c_str());

    // SAMPLING RATE
    if (Parse::populateInt("LibavOutputWriter", *m_config, "sampling_rate", audioCodecContext->sample_rate, true) !=
            Parse::PopulateResult_Ok ||
        (audioCodecContext->sample_rate != 44100 && audioCodecContext->sample_rate != 48000 &&
         audioCodecContext->sample_rate != 32000)) {
      Logger::error(AVtag) << "Audio codec: " << audioCodecContext->sample_rate
                           << " : invalid sample rate (32000, 44100 or 48000)" << std::endl;
      return false;
    }

    // CHANNEL LAYOUT
    std::string channel_layout;
    if (Parse::populateString("LibavOutputWriter", *m_config, "channel_layout", channel_layout, true) !=
        Parse::PopulateResult_Ok) {
      Logger::error(AVtag) << "Audio codec: " << channel_layout.c_str() << " : invalid channel layout" << std::endl;
      return false;
    }
    // Ambisonic format needs to be defined as "amb_wxyz" but also needs to be translated for avcodec
    if (channel_layout == "amb_wxyz") {
      channel_layout = "4.0";
    }
    // CHANNEL MAP
    std::vector<int64_t> channel_map({});
    if (Parse::populateIntList("LibavOutputWriter", *m_config, "channel_map", channel_map, false) ==
        Parse::PopulateResult_WrongType) {
      Logger::error(AVtag) << "Audio channel map: invalid" << std::endl;
      return false;
    }
    int nbChannels = Audio::getNbChannelsFromChannelLayout(Audio::getChannelLayoutFromString(channel_layout.c_str()));
    if (!channel_map.empty() && int(channel_map.size()) != nbChannels) {
      Logger::error(AVtag) << "Audio channel map: invalid number of elements " << channel_map.size()
                           << ", does not match channel layout " << channel_layout << std::endl;
      return false;
    }
    if (!channel_map.empty()) {
      for (auto& map : channel_map) {
        if (!(0 <= map && map < nbChannels)) {
          Logger::error(AVtag) << "Audio channel map: has invalid value " << map << std::endl;
          return false;
        }
      }
      m_channelMap = channel_map;
    }

    if (av_get_channel_layout(channel_layout.c_str()) == 0) {
      Logger::error(AVtag) << "Audio codec: " << channel_layout.c_str() << " : invalid channel layout" << std::endl;
      return false;
    }
    audioCodecContext->channel_layout = av_get_channel_layout(channel_layout.c_str());
    audioCodecContext->channels = av_get_channel_layout_nb_channels(audioCodecContext->channel_layout);

    std::string format(LIBAV_WRITER_DEFAULT_CONTAINER);
    if (Parse::populateString("LibavOutputWriter", *m_config, "type", format, true) == Parse::PopulateResult_Ok) {
      const AVOutputFormat* of = av_guess_format(format.c_str(), nullptr, nullptr);
      if (of && (of->flags & AVFMT_GLOBALHEADER)) {
        audioCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
      }
    }

    // OPEN CONFIGURED AUDIO CODEC
    AVDictionary* opts = nullptr;
    if (audioCodecName == "aac") {
      av_dict_set(&opts, "strict", "experimental", 0);
    }
    int r = avcodec_open2(audioCodecContext, audioCodec, &opts);
    av_dict_free(&opts);
    if (r < 0) {
      Logger::error(AVtag) << "Could not open audio codec: " << audioCodec->name << " avcodec_open2: " << r << " : "
                           << Util::errorString(r) << std::endl;
      return false;
    }

    audioFrame = av_frame_alloc();
    audioFrame->nb_samples = audioCodecContext->frame_size;
    audioFrame->format = audioCodecContext->sample_fmt;
    audioFrame->channel_layout = audioCodecContext->channel_layout;
    audioFrame->channels = audioCodecContext->channels;
    // the codec gives us the frame size, in samples,
    // we calculate the size of the samples buffer in bytes
    const int audioBufferSize =
        av_samples_get_buffer_size(nullptr, audioCodecContext->channels, audioCodecContext->frame_size,
                                   audioCodecContext->sample_fmt, LIBAV_BUFFER_ALIGNMENT);
    if (audioBufferSize < 0) {
      Logger::error(AVtag) << "Could not get buffer size. Error : " << audioBufferSize << " : "
                           << Util::errorString(audioBufferSize) << std::endl;
    }

    avSamples = (const uint8_t*)av_malloc(audioBufferSize);
    if (avSamples == nullptr) {
      Logger::error(AVtag) << "Could not allocate audio samples buffer." << std::endl;
    }

    // setup the data pointers in the AVFrame
    r = avcodec_fill_audio_frame(audioFrame, audioCodecContext->channels, audioCodecContext->sample_fmt, avSamples,
                                 audioBufferSize, 0);
    if (r < 0) {
      Logger::error(AVtag) << "Could not fill audio frame. Error : " << r << " : " << Util::errorString(r) << std::endl;
    }

    Audio::SamplingDepth samplingDepth = Util::sampleFormat(audioCodecContext->sample_fmt);
    if (samplingDepth == Audio::SamplingDepth::SD_NONE) {
      Logger::error(AVtag) << "Audio sampling depth is SamplingDepth::SD_NONE" << std::endl;
    }
    m_sampleFormat = Audio::getSamplingFormatFromSamplingDepth(samplingDepth);
    if (m_sampleFormat == Audio::SamplingFormat::FORMAT_UNKNOWN) {
      Logger::error(AVtag) << "Audio sampling format is SamplingFormat::FORMAT_UNKNOW" << std::endl;
    }
    m_audioFrameSizeInBytes = Audio::getSampleSizeFromSamplingDepth(samplingDepth);
    if (m_audioFrameSizeInBytes == 0) {
      Logger::error(AVtag) << "Audio frame size is 0" << std::endl;
    }
  }
  return true;
}

/* AAC encoder stopped encoding after flushing so it has to be recorded */
/* x264 will stop the lookahead thread during the flush and needs to be restarted */
bool LibavWriter::resetCodec(AVCodecContext* codecContext, MuxerThreadStatus& status) {
  if (codecContext == nullptr) {
    return false;
  } else if (codecContext == audioCodecContext) {
    status = flushAudio();

    int r = avcodec_close(audioCodecContext);
    if (r < 0) {
      Logger::error(AVtag) << "flushing avcodec_close return error " << Util::errorString(r) << std::endl;
      return false;
    }

    const AVCodec* audioCodec = avcodec_find_encoder(audioCodecContext->codec_id);
    if (!audioCodec) {
      Logger::error(AVtag) << "Audio codec: " << audioCodecContext->codec_id << " not found, disable output."
                           << std::endl;
      return false;
    }

    AVDictionary* opts = nullptr;
    if (std::string("aac") == audioCodec->name) {
      av_dict_set(&opts, "strict", "experimental", 0);
    }
    r = avcodec_open2(audioCodecContext, audioCodec, &opts);
    av_dict_free(&opts);
    if (r < 0) {
      Logger::error(AVtag) << "Could not open audio codec: " << audioCodec->name << " avcodec_open2: " << r << " : "
                           << Util::errorString(r) << std::endl;
      return false;
    }
    return true;
  } else if (codecContext == videoCodecContext) {
    status = flushVideo();

    const AVCodec* videoCodec = videoCodecContext->codec;
    AVBufferRef* hw_frames_ctx = nullptr;

    if (videoCodecContext->hw_frames_ctx) {
      hw_frames_ctx = videoCodecContext->hw_frames_ctx;
      videoCodecContext->hw_frames_ctx = av_buffer_ref(videoCodecContext->hw_frames_ctx);
    }

    int r = avcodec_close(videoCodecContext);
    if (r < 0) {
      Logger::error(AVtag) << "flushing avcodec_close return error " << Util::errorString(r) << std::endl;
      return false;
    }

    if (hw_frames_ctx) {
      av_buffer_unref(&videoCodecContext->hw_frames_ctx);
      videoCodecContext->hw_frames_ctx = hw_frames_ctx;
    }

    r = avcodec_open2(videoCodecContext, videoCodec, &codecConfig);
    if (r < 0) {
      Logger::error(AVtag) << "Could not open video codec: " << videoCodec->name << " avcodec_open2: " << r << " : "
                           << Util::errorString(r) << std::endl;
      return false;
    }
    return true;
  }
  return false;
}

bool LibavWriter::createVideoCodec(AddressSpace type, unsigned width, unsigned height, FrameRate framerate) {
  AVBufferRef* hwdevice = nullptr;
  unsigned uEncodeBufferCount = 1;

  // CODEC
  std::string avCodec(LIBAV_WRITER_DEFAULT_CODEC), videoCodecName;
  if (Parse::populateString("LibavOutputWriter", *m_config, "video_codec", avCodec, false) ==
      Parse::PopulateResult_WrongType) {
    return false;
  }
  // MPEG2 - mpeg2Video
  if (avCodec == "mpeg2") {
    videoCodecName = "mpeg2video";
    if (!(width % 4096) || !(height % 4096)) {
      Logger::get(Logger::Warning) << "[libavoutput] The MPEG2 Video codec \"precludes values of [size] that are "
                                      "multiples of 4096.\". Your output may not be readable by other players."
                                   << std::endl;
    }
  } else if (avCodec == "mpeg4") {
    if ((width > 8192) || (height > 8192)) {
      Logger::get(Logger::Error) << "[libavoutput] The MPEG4 Video encoder doesn't support resolutions higher than "
                                    "8192. Your settings are: width="
                                 << width << ", height=" << height << ". Aborting." << std::endl;
      return false;
    } else if ((width % 8) || (height % 8)) {
      Logger::get(Logger::Error) << "[libavoutput] The MPEG4 Video encoder requires resolutions which are multiples of "
                                    "8. Your settings are: width="
                                 << width << ", height=" << height << ". Aborting." << std::endl;
      return false;
    }
    videoCodecName = avCodec;
    // ProRes
  } else if (avCodec == "prores") {
    videoCodecName = "prores_ks";
    // H264 - Libx264
  } else if (avCodec == "h264" || avCodec == "depth") {
    if (width > 4096) {
      Logger::get(Logger::Error) << "[libavoutput] The H264 Video encoder doesn't support resolutions higher than "
                                    "4096. Your settings are: width="
                                 << width << ", height=" << height << ". Aborting." << std::endl;
    } else {
      videoCodecName = "libx264";
    }
  } else if ((avCodec == "h264_nvenc") || (avCodec == "hevc_nvenc")) {
    videoCodecName = avCodec;
  } else if (avCodec == "mjpeg") {
    videoCodecName = avCodec;
  } else {
    Logger::get(Logger::Error) << "[libavoutput] Invalid AV codec '" << avCodec << "'." << std::endl;
    return false;
  }
  const AVCodec* videoCodec = avcodec_find_encoder_by_name(videoCodecName.c_str());
  if (!videoCodec) {
    Logger::get(Logger::Error) << "[libavoutput] Video codec: " << videoCodecName.c_str()
                               << " not found, disable output." << std::endl;
    return false;
  }

  // FFMPEG CONFIGURATION
  std::ostringstream clFFMPEG;  // Command line options for ffmpeg
  std::ostringstream clCodec;   // Command line options for specific codec
  AVDictionary* ffmpegConfig = nullptr;

  // BITRATE
  int avBR = LIBAV_WRITER_DEFAULT_BITRATE;
  if (avCodec != "prores" && avCodec != "mjpeg" && avCodec != "depth") {
    if (Parse::populateInt("LibavOutputWriter", *m_config, "bitrate", avBR, false) == Parse::PopulateResult_WrongType) {
      Logger::get(Logger::Error) << "[libavoutput] Parameter 'bitrate' not found in the configuration. Aborting"
                                 << std::endl;
      return false;
    } else {
      if (avBR < LIBAV_WRITER_MIN_MP4_BITRATE || avBR > LIBAV_WRITER_MAX_MP4_BITRATE) {
        Logger::get(Logger::Error) << "[libavoutput] AV bitrate must me between " << LIBAV_WRITER_MIN_MP4_BITRATE
                                   << " and " << LIBAV_WRITER_MAX_MP4_BITRATE << ". Your setting: " << avBR << "."
                                   << std::endl;
        return false;
      } else {
        clCodec << "-b " << avBR << " ";
      }
    }
  }

  // PASS
  int avNumPass = LIBAV_WRITER_DEFAULT_NUM_PASS;
  if (Parse::populateInt("LibavOutputWriter", *m_config, "pass", avNumPass, false) == Parse::PopulateResult_WrongType) {
  } else if (avNumPass <= 0 || avNumPass > 2) {
    Logger::get(Logger::Error) << "[libavoutput] AV number of pass must me between 1 and 2." << std::endl;
    return false;
  } else {
    clFFMPEG << "-pass " << avNumPass << " ";
  }

  // Q SCALE
  if (avCodec == "mjpeg") {
    int avQScale = LIBAV_WRITER_MIN_QSCALE;
    if (Parse::populateInt("LibavOutputWriter", *m_config, "scale", avQScale, false) ==
        Parse::PopulateResult_WrongType) {
    } else if ((avQScale < LIBAV_WRITER_MIN_QSCALE) || (avQScale > LIBAV_WRITER_MAX_QSCALE)) {
      Logger::get(Logger::Error) << "[libavoutput] AV Quality scale value must be between 1 and "
                                 << LIBAV_WRITER_MAX_QSCALE << " (found " << avQScale << ")" << std::endl;
      return false;
    } else {
      clCodec << "-qmin " << avQScale << " ";
      clCodec << "-qmax " << avQScale << " ";
    }
  }

  // FPS
  double avFR = (double)framerate.num / framerate.den;
  if (avFR <= 1.0 || avFR > 1000.0) {
    Logger::get(Logger::Error) << "[libavoutput] AV frame rate (fps) must me between 1.0 and 1000.0." << std::endl;
    return false;
  } else {
    clFFMPEG << "-r " << avFR << " ";
  }

  // CODEC CONFIGURATION
  // GOP
  int avGOP = LIBAV_WRITER_DEFAULT_GOP_SIZE;
  if (Parse::populateInt("LibavOutputWriter", *m_config, "gop", avGOP, false) == Parse::PopulateResult_WrongType) {
    return false;
  } else if (avGOP < -1 || avGOP > LIBAV_WRITER_MAX_GOP_SIZE) {
    Logger::get(Logger::Error) << "[libavoutput] AV GOP size must me between 1 frame and " << LIBAV_WRITER_MAX_GOP_SIZE
                               << " (" << LIBAV_WRITER_MAX_GOP_SIZE / LIBAV_WRITER_DEFAULT_FRAMERATE << "s)."
                               << std::endl;
    return false;
  } else if (avGOP >= 0) {
    // To set contant gop size
    // see http://forum.doom9.org/showthread.php?t=121116
    // and
    // https://sonnati.wordpress.com/2011/08/19/ffmpeg-%E2%80%93-the-swiss-army-knife-of-internet-streaming-%E2%80%93-part-iii/
    clCodec << "-g " << avGOP << " ";
    if ((avCodec == "h264") || (avCodec == "h264_nvenc")) {
      clCodec << "-keyint_min " << avGOP << " ";
      clCodec << "-sc_threshold " << 0 << " ";
      clCodec << "-mbtree "
              << "0"
              << " ";
    }
  }
  // PRORES or H264 PROFILE
  std::string profile;
  if (avCodec != "depth" &&
      Parse::populateString("LibavOutputWriter", *m_config, "profile", profile, false) == Parse::PopulateResult_Ok) {
    clCodec << "-profile " << profile << " ";
  }
  // H264 LEVEL
  std::string level;
  if (Parse::populateString("LibavOutputWriter", *m_config, "level", level, false) == Parse::PopulateResult_Ok) {
    clCodec << "-level " << level << " ";
  }
  // Configuration preset
  std::string preset;
  if (avCodec != "depth" &&
      Parse::populateString("LibavOutputWriter", *m_config, "preset", preset, false) == Parse::PopulateResult_Ok) {
    clCodec << "-preset " << preset << " ";
  } else if (avCodec == "depth") {
    clCodec << "-preset veryslow ";
  }
  // B FRAMES
  if (avCodec != "mjpeg" && avCodec != "depth") {
    int avNumB = LIBAV_WRITER_DEFAULT_B_FRAMES;
    if (Parse::populateInt("LibavOutputWriter", *m_config, "b_frames", avNumB, false) ==
        Parse::PopulateResult_WrongType) {
    } else if ((avNumB < 0) || (avNumB > LIBAV_WRITER_MAX_B_FRAMES)) {
      Logger::get(Logger::Error) << "[libavoutput] AV B-frames rate must me between 0 and " << LIBAV_WRITER_MAX_B_FRAMES
                                 << " (found " << avNumB << ")" << std::endl;
      return false;
    } else {
      if (((profile == "baseline") && ((avCodec == "h264") || (avCodec == "h264_nvenc"))) ||
          (avCodec == "hevc_nvenc")) { /* There is no B-frames in H264 Baseline profile and no B-frames in hevc_nvenc */
        avNumB = 0;
      }
      clCodec << "-bf " << avNumB << " ";
    }
    if (((avCodec == "h264_nvenc") || (avCodec == "hevc_nvenc")) && (type == Device)) {
      uEncodeBufferCount = avNumB + 4;  // buffers is numb + 1 + 3 pipelining (as in RTMP plugin)
      clCodec << "-delay " << uEncodeBufferCount - 1
              << " ";  // force frame output before there is no more available frames
    }
  }

  // BITRATE MODE
  std::string avRCMode(LIBAV_WRITER_DEFAULT_BITRATE_MODE);
  if (avCodec == "depth") {
    // Depth encoding implies CRF = 0
    clCodec << "-crf 1 ";
  } else {
    if (Parse::populateString("LibavOutputWriter", *m_config, "bitrate_mode", avRCMode, false) ==
        Parse::PopulateResult_WrongType) {
      return false;
    } else if (avRCMode == "CBR") {
      if (avCodec == "mpeg2" || avCodec == "mpeg4") {
        int64_t bufsize = (int64_t)((double)(int64_t(avGOP) * avBR) / (2.0 * avFR));
        clCodec << "-bufsize " << bufsize << " -rc_init_occupancy " << bufsize / 2 << " -minrate " << avBR
                << " -maxrate " << avBR << " ";
      } else {
        int64_t bufsize = (int64_t)((double)(int64_t(avGOP) * avBR) / avFR);
        clCodec << "-bufsize " << bufsize << " -rc_init_occupancy " << bufsize / 2 << " -minrate "
                << int64_t(0.25 * avBR) << " -maxrate " << 2 * avBR << " ";
      }
    } else if (avRCMode == "VBR") {
      // TODO: anything to configure?
    } else {
      Logger::get(Logger::Error) << "[libavoutput] AV bitrate mode must me either \"CBR\" or \"VBR\"" << std::endl;
      return false;
    }
  }

  Util::build_dict(&codecConfig, clCodec.str().c_str(), "codec");

  // stats log file
  std::string statsLogFile;
  if (Parse::populateString("LibavOutputWriter", *m_config, "stats_log_file", statsLogFile, false) ==
      Parse::PopulateResult_Ok) {
    if (!statsLogFile.empty()) {
      av_dict_set(&codecConfig, "stats", statsLogFile.c_str(), 0);
    }
  }

  av_dict_set(&codecConfig, "threads", "auto", 0);
  Util::build_dict(&ffmpegConfig, clFFMPEG.str().c_str(), "other");

  // Command line mode, only for advanced users, this will discard the next configurations
  std::string commandLineOptions;
  if (Parse::populateString("LibavOutputWriter", *m_config, "codec_command_line", commandLineOptions, false) ==
      Parse::PopulateResult_Ok) {
    if (commandLineOptions.empty()) {
      Logger::get(Logger::Error) << "[libavoutput] Command line options are empty" << std::endl;
      return false;
    } else {
      Util::build_dict(&codecConfig, commandLineOptions.c_str(), "codec");
      Logger::get(Logger::Info) << "[libavoutput] Command line: " << commandLineOptions << std::endl;
    }
  }

  videoCodecContext = avcodec_alloc_context3(videoCodec);
  if (videoCodecContext == nullptr) {
    Logger::get(Logger::Error) << "[libavoutput] Video codec: " << videoCodecName.c_str()
                               << " : unable to alloc context" << std::endl;
    return false;
  }
  videoCodecContext->width = width;
  videoCodecContext->height = height;

  // PIXEL FORMAT, COLOR RANGE AND FLAGS
  // NOTE: making sure that the getVideoStitchPixelFormat() function is aligned by using assert()
  if (videoCodecName == "mjpeg") {
    videoCodecContext->pix_fmt = AVPixelFormat::AV_PIX_FMT_YUVJ420P;
    videoCodecContext->color_range = AVCOL_RANGE_JPEG;
    assert(this->getPixelFormat() == VideoStitch::YV12);
  } else if (avCodec == "depth") {
    videoCodecContext->pix_fmt = AVPixelFormat::AV_PIX_FMT_YUVJ420P;
    videoCodecContext->color_range = AVCOL_RANGE_JPEG;  // full-scale
    assert(this->getPixelFormat() == VideoStitch::DEPTH);
  } else if (videoCodecName == "prores_ks") {
    videoCodecContext->pix_fmt = AVPixelFormat::AV_PIX_FMT_YUV422P10;
    videoCodecContext->color_range = AVCOL_RANGE_MPEG;
    assert(this->getPixelFormat() == VideoStitch::YUV422P10);
  } else if ((videoCodecName == "h264_nvenc") || (videoCodecName == "hevc_nvenc")) {
    videoCodecContext->color_range = AVCOL_RANGE_MPEG;
    if (type == Host) {
      videoCodecContext->pix_fmt = AVPixelFormat::AV_PIX_FMT_NV12;
    } else {
      int r;
#ifdef SUP_NVENC
      hwdevice = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
      if (!hwdevice) {
        Logger::get(Logger::Error) << "[libavoutput] Video codec: " << videoCodecName.c_str()
                                   << " : unable to alloc CUDA device" << std::endl;
        return false;
      }
      AVHWDeviceContext* device_ctx = (AVHWDeviceContext*)hwdevice->data;
      if (!GPU::getDefaultBackendDeviceContext(device_ctx->hwctx).ok()) {
        /* We do not include "libavutil/hwcontext_cuda.h" to avoid CUDA dependencies in plugin
           If we could include "libavutil/hwcontext_cuda.h" we would use
           AVCUDADeviceContext *device_hwctx = (AVCUDADeviceContext *)device_ctx->hwctx;
           GPU::getThreadLocalDeviceContext(&device_hwctx->cuda_ctx)
           check that cuda_ctx is the first element of (AVCUDADeviceContext *)device_ctx->hwctx */
        return false;
      }

      r = av_hwdevice_ctx_init(hwdevice);
#else
      r = av_hwdevice_ctx_create(&hwdevice, AV_HWDEVICE_TYPE_CUDA, videoCodecName.c_str(), nullptr, 0);
#endif
      if (r < 0) {
        Logger::get(Logger::Error) << "[libavoutput] Video codec: " << videoCodecName.c_str()
                                   << " : unable to init CUDA device. "
                                   << " Error : " << r << " : " << Util::errorString(r) << std::endl;
        return false;
      }

      videoCodecContext->hw_frames_ctx = av_hwframe_ctx_alloc(hwdevice);

      AVHWFramesContext* hwframe_ctx = (AVHWFramesContext*)videoCodecContext->hw_frames_ctx->data;
      videoCodecContext->pix_fmt = AVPixelFormat::AV_PIX_FMT_CUDA;
      hwframe_ctx->format = videoCodecContext->pix_fmt;
      hwframe_ctx->sw_format = AVPixelFormat::AV_PIX_FMT_NV12;
      hwframe_ctx->width = width;
      hwframe_ctx->height = height;

      r = av_hwframe_ctx_init(videoCodecContext->hw_frames_ctx);
      if (r < 0) {
        Logger::get(Logger::Error) << "[libavoutput] Video codec: " << videoCodecName.c_str()
                                   << " : unable to init CUDA frame context."
                                   << " Error : " << r << " : " << Util::errorString(r) << std::endl;
        av_buffer_unref(&hwdevice);
        return false;
      }
    }
    assert(this->getPixelFormat() == VideoStitch::NV12);
  } else {
    videoCodecContext->pix_fmt = AVPixelFormat::AV_PIX_FMT_YUV420P;
    videoCodecContext->color_range = AVCOL_RANGE_MPEG;
    assert(this->getPixelFormat() == VideoStitch::YV12);
  }
  videoCodecContext->flags |= AV_CODEC_FLAG_PASS1;
  if (atoi(av_dict_get(ffmpegConfig, "pass", nullptr, 0)->value) == 2) {
    videoCodecContext->flags |= AV_CODEC_FLAG_PASS2;
  }

  videoCodecContext->time_base = {framerate.den, framerate.num};

  std::string extraParams;
  if (Parse::populateString("LibavOutputWriter", *m_config, "extra_params", extraParams, false) ==
      Parse::PopulateResult_Ok) {
    Logger::get(Logger::Debug) << "[libavoutput] extra_params : " << extraParams.c_str() << std::endl;
    std::vector<std::string> paramList;
    Util::split(extraParams.c_str(), ',', &paramList);
    auto param = paramList.begin();
    for (; param != paramList.end(); ++param) {
      std::vector<std::string> paramValue;
      Util::split(param->c_str(), '=', &paramValue);

      if (paramValue.size() != 2) {
        Logger::get(Logger::Verbose) << "[libavoutput] extra_params : wrong param (" << paramValue.size()
                                     << " value detected, 2 expected) in " << param->c_str() << std::endl;
      } else {
        Logger::get(Logger::Debug) << "[libavoutput] extra_params : detected param " << paramValue[0].c_str()
                                   << " with value " << paramValue[1].c_str() << " [" << param->c_str() << "]"
                                   << std::endl;
        av_dict_set(&ffmpegConfig, paramValue[0].c_str(), paramValue[1].c_str(), 0);
      }
    }
  }

  std::string format(LIBAV_WRITER_DEFAULT_CONTAINER);
  if (Parse::populateString("LibavOutputWriter", *m_config, "type", format, true) == Parse::PopulateResult_Ok) {
    const AVOutputFormat* of = av_guess_format(format.c_str(), nullptr, nullptr);
    if (of && (of->flags & AVFMT_GLOBALHEADER)) {
      videoCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
  }

  // OPEN CONFIGURED VIDEO CODEC
  const int r = avcodec_open2(videoCodecContext, videoCodecContext->codec, &codecConfig);
  av_dict_free(&ffmpegConfig);

  if (r < 0) {
    Logger::get(Logger::Error) << "[libavoutput] Could not open video codec: " << videoCodec->name
                               << " avcodec_open2: " << Util::errorString(r) << std::endl;
    avcodec_close(videoCodecContext);
    avcodec_free_context(&videoCodecContext);
    videoCodecContext = nullptr;
    if (hwdevice) av_buffer_unref(&hwdevice);
    return false;
  }

  for (int i = 0; i < (int)uEncodeBufferCount; i++) {
    AVFrame* videoFrame;
    // prepare the frames
    videoFrame = av_frame_alloc();
    const int codecWidth = videoCodecContext->width;
    switch (this->getPixelFormat()) {
      case VideoStitch::YV12:
      case VideoStitch::DEPTH:
        videoFrame->linesize[0] = codecWidth;
        videoFrame->linesize[1] = (codecWidth + 1) / 2;
        videoFrame->linesize[2] = (codecWidth + 1) / 2;
        break;
      case VideoStitch::NV12:
        videoFrame->linesize[0] = codecWidth;
        videoFrame->linesize[1] = codecWidth;
        if (this->getExpectedOutputBufferType() == Device) {
          int r = av_hwframe_get_buffer(videoCodecContext->hw_frames_ctx, videoFrame, 0);
          if (r < 0) {
            Logger::get(Logger::Error) << "[libavoutput] Cannot get new Device buffer : " << Util::errorString(r)
                                       << std::endl;
          }
        }
        break;
      case VideoStitch::YUV422P10:
        videoFrame->linesize[0] = codecWidth * 2;
        videoFrame->linesize[1] = codecWidth;
        videoFrame->linesize[2] = codecWidth;
        break;
      default:
        assert(false);
    }
    videoFrame->format = videoCodecContext->pix_fmt;
    videoFrame->width = codecWidth;
    videoFrame->height = videoCodecContext->height;

    videoFrames.push_back(videoFrame);
  }
  return true;
}

LibavWriter::LibavWriter(const Ptv::Value& config, const std::string& name, const VideoStitch::PixelFormat fmt,
                         AddressSpace type, unsigned width, unsigned height, FrameRate framerate,
                         Audio::SamplingRate samplingRate, Audio::SamplingDepth samplingDepth,
                         Audio::ChannelLayout channelLayout)
    : Output(name),
      VideoWriter(width, height, framerate, fmt, type),
      AudioWriter(samplingRate, samplingDepth, channelLayout),
      m_config(config.clone()),
      videoFrames(0),
      codecConfig(nullptr),
      videoCodecContext(nullptr),
      firstVideoPTS(-1),
      audioFrame(nullptr),
      m_sampleFormat(Audio::SamplingFormat::FORMAT_UNKNOWN),
      m_audioFrameSizeInBytes(0),
      audioCodecContext(nullptr),
      m_currentImplNumber(0),
      m_pimplVideo(nullptr),
      m_pimplAudio(nullptr) {
  outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Connecting, ConnectingMessage);

  // register all codecs, demux and protocols
  Util::Libav::checkInitialization();

  std::lock_guard<std::mutex> lk(pimplMu);
  if (createAudioCodec() && createVideoCodec(type, width, height, framerate)) {
    m_pimplVideo.reset(
        AvMuxer_pimpl::create(*m_config, videoCodecContext, audioCodecContext, framerate, m_currentImplNumber++));
    if (hasAudio()) {
      m_pimplAudio = std::dynamic_pointer_cast<AvMuxer_pimpl>(m_pimplVideo);
    }
  }
  if (!m_pimplVideo) {
    Logger::get(Logger::Warning) << "[libavoutput] LibavWriter failed to instanciate implementation during creation."
                                 << std::endl;
    outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                    DisconnectedMessage);
  }

  // Allocate audio data buffer with empty buffers to be consistent with the Audio::Samples::append() API
  for (int i = 0; i < MAX_AUDIO_CHANNELS; ++i) {
    if (Audio::getChannelMapFromChannelIndex(i) & channelLayout) {
      audioData[i] = new uint8_t[0];
    }
  }
  audioBuffer = Audio::Samples(samplingRate, samplingDepth, channelLayout, 0, audioData, 0);
}

LibavWriter::~LibavWriter() {
  MuxerThreadStatus status = close();
  if (status != MuxerThreadStatus::OK) {
    outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                    typeToString(status));
  }
  if (m_config) {
    delete m_config;
  }
  if (codecConfig) {
    av_dict_free(&codecConfig);
  }
}

bool LibavWriter::needsRespawn(std::shared_ptr<AvMuxer_pimpl>& m_pimpl, mtime_t timestamp) {
  // estimate the last video timestamp to avoid approximation due to video timescale
  int64_t frameIdOffset = (int64_t)(round((timestamp - firstVideoPTS) * videoCodecContext->time_base.den /
                                          (1000000.0 * videoCodecContext->time_base.num)));
  mtime_t lastTs = firstVideoPTS + frameIdOffset * (1000000 * int64_t(videoCodecContext->time_base.num)) /
                                       videoCodecContext->time_base.den;
  // update last seen PTS
  if (m_pimpl->m_lastPTS < lastTs) {
    m_pimpl->m_lastPTS = lastTs;
  }
  if (m_pimpl->needsRespawn()) {
    return (((m_pimpl->m_lastPTS - timestamp) * videoCodecContext->time_base.den) <
            (int64_t(500000) * videoCodecContext->time_base.num));
  }
  return false;
}

bool LibavWriter::implReady(std::shared_ptr<AvMuxer_pimpl>& m_pimpl, AVCodecContext* codecContext, mtime_t timestamp) {
  if (m_pimpl) {
    MuxerThreadStatus status = m_pimpl->getStatus();

    mtime_t lastTs = timestamp;
    /* check if we need to reset the implementation */
    if (needsRespawn(m_pimpl, timestamp)) {
      if (!(resetCodec(codecContext, status))) {
        m_pimpl.reset();  // release ownership
        outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                        DisconnectedMessage);
        Logger::error(AVtag) << "LibavWriter : no implementation is instanciated, respawn failed" << std::endl;
        return false;
      }
      if ((m_pimplVideo == m_pimplAudio) || (!hasAudio())) {
        lastTs = m_pimpl->m_lastPTS;
        m_pimpl.reset(AvMuxer_pimpl::create(*m_config, videoCodecContext, audioCodecContext, this->getFrameRate(),
                                            m_currentImplNumber++));
      } else if (m_pimplVideo == m_pimpl) {
        m_pimpl = std::dynamic_pointer_cast<AvMuxer_pimpl>(m_pimplAudio);
      } else {
        m_pimpl = std::dynamic_pointer_cast<AvMuxer_pimpl>(m_pimplVideo);
      }
      if (!m_pimpl) {
        outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                        DisconnectedMessage);
        Logger::error(AVtag) << "LibavWriter : no implementation is instantiated, respawn failed" << std::endl;
        return false;
      }
    }
    if (status != MuxerThreadStatus::OK) {
      std::string errorMessage = typeToString(status);
      Logger::warning(AVtag) << " LibavWriter encountered error " << errorMessage << std::endl;
      outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected, errorMessage);
    }
    if (m_pimpl->firstFrame()) {
      // initialize first seen PTS
      m_pimpl->m_firstPTS = lastTs;
      Logger::debug(AVtag) << "m_firstPTS: " << m_pimpl->m_firstPTS << std::endl;
      outputEventManager.publishEvent(OutputEventManager::EventType::Connected, ConnectedMessage);
    }
    return true;
  } else {
    MuxerThreadStatus status = MuxerThreadStatus::OK;

    Logger::error(AVtag) << "LibavWriter : no implementation is instanciated" << std::endl;
    if (resetCodec(codecContext, status)) {
      m_pimpl.reset(AvMuxer_pimpl::create(*m_config, videoCodecContext, audioCodecContext, this->getFrameRate(),
                                          m_currentImplNumber++));
    }
    if (!m_pimpl) {
      Logger::error(AVtag) << "failed to re-instanciate LibavWriter implementation." << std::endl;
      outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                      DisconnectedMessage);
    }
    return false;
  }
}

void LibavWriter::pushVideo(const Frame& vidFrame) {
  std::lock_guard<std::mutex> lk(pimplMu);
  if (implReady(m_pimplVideo, videoCodecContext, vidFrame.pts)) {
    VideoStitch::Util::SimpleProfiler prof("FFmpeg frame encoding", false, Logger::get(Logger::Verbose));
    if (firstVideoPTS == -1) {
      firstVideoPTS = vidFrame.pts;
      Logger::verbose(AVtag) << "firstVideoPTS: " << firstVideoPTS << std::endl;
    }

    if (videoFrames.size() == 0 || videoCodecContext == nullptr || vidFrame.planes[0] == nullptr) {
      return;
    }

    AVFrame* videoFrame = videoFrames.front();
    videoFrames.pop_front();
    videoFrames.push_back(videoFrame);

    // video
    {
      switch (this->getPixelFormat()) {
        case VideoStitch::YV12:
        case VideoStitch::DEPTH:
          videoFrame->data[0] = (uint8_t*)vidFrame.planes[0];
          videoFrame->data[1] = (uint8_t*)vidFrame.planes[1];
          videoFrame->data[2] = (uint8_t*)vidFrame.planes[2];
          break;
        case VideoStitch::NV12:
          if (this->getExpectedOutputBufferType() == Host) {
            videoFrame->data[0] = (uint8_t*)vidFrame.planes[0];
            videoFrame->data[1] = (uint8_t*)vidFrame.planes[1];
          } else {
#ifdef SUP_NVENC
            cudaError_t err =
                cudaMemcpy2D(videoFrame->data[0], videoFrame->linesize[0], (const void*)vidFrame.planes[0],
                             vidFrame.pitches[0], vidFrame.width, vidFrame.height, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
              resetCudaError();
              Logger::error(AVtag) << "Cannot copy buffer" << std::endl;
              outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                              "Cannot copy buffer");
              return;
            }
            err = cudaMemcpy2D(videoFrame->data[1], videoFrame->linesize[1], (const void*)vidFrame.planes[1],
                               vidFrame.pitches[1], vidFrame.width, vidFrame.height / 2, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
              resetCudaError();
              Logger::error(AVtag) << "Cannot copy buffer" << std::endl;
              outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                              "Cannot copy buffer");
              return;
            }
#else
            AVFrame* videoFrameTmp = av_frame_clone(videoFrame);
            videoFrameTmp->data[0] = (uint8_t*)vidFrame.planes[0];
            videoFrameTmp->data[1] = (uint8_t*)vidFrame.planes[1];
            videoFrameTmp->linesize[0] = (int)vidFrame.pitches[0];
            videoFrameTmp->linesize[1] = (int)vidFrame.pitches[1];
            int r = av_hwframe_transfer_data(videoFrame, videoFrameTmp, 0);
            if (r < 0) {
              Logger::error(AVtag) << "Cannot copy buffer" << std::endl;
              outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected,
                                              "Cannot copy buffer");
              return;
            }
            av_frame_free(&videoFrameTmp);
#endif
          }
          break;
        case VideoStitch::YUV422P10:
          videoFrame->data[0] = (uint8_t*)vidFrame.planes[0];
          videoFrame->data[1] = (uint8_t*)vidFrame.planes[1];
          videoFrame->data[2] = (uint8_t*)vidFrame.planes[2];
          break;
        default:
          assert(false);
      }

      int64_t frameIdOffset =
          (int64_t)(round((firstVideoPTS - m_pimplVideo->m_firstPTS) * videoCodecContext->time_base.den /
                          (1000000.0 * videoCodecContext->time_base.num)));
      videoFrame->pts = (int)(round((vidFrame.pts - firstVideoPTS) * videoCodecContext->time_base.den /
                                    (1000000.0 * videoCodecContext->time_base.num)));
      int64_t frameId = videoFrame->pts + frameIdOffset;
      Logger::verbose(AVtag) << "(VIDEO): date: " << vidFrame.pts << "   frameId: " << frameId
                             << "  videoFrame->pts: " << videoFrame->pts << std::endl;

      if ((videoFrame->buf[0]) && (av_buffer_get_ref_count(videoFrame->buf[0]) > 1) && (m_currentImplNumber < 2)) {
        std::stringstream msg;
        msg << "Video buffer used is already referenced " << av_buffer_get_ref_count(videoFrame->buf[0]);
        Logger::error(AVtag) << msg.str() << std::endl;
        outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected, msg.str());
      }

      Util::AvErrorCode ret = encodeVideoFrame(videoFrame, frameIdOffset);
      if (ret != Util::AvErrorCode::Ok) {
        std::stringstream msg;
        msg << "Error encountered while encoding video frame at " << vidFrame.pts << ". "
            << Util::errorStringFromAvErrorCode(ret);
        Logger::error(AVtag) << msg.str() << std::endl;
        outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected, msg.str());
        assert(false);
      }
    }
  }
}

void LibavWriter::pushAudio(Audio::Samples& audioSamples) {
  std::lock_guard<std::mutex> lk(pimplMu);
  if (hasAudio() && implReady(m_pimplAudio, audioCodecContext,
                              audioSamples.getTimestamp() -
                                  (int64_t(audioBuffer.getNbOfSamples()) * 1000000) / audioCodecContext->sample_rate)) {
    if (audioFrame == nullptr) {
      return;
    }
    mtime_t lastTs = audioSamples.getTimestamp() +
                     (int64_t(audioSamples.getNbOfSamples()) * 1000000) / audioCodecContext->sample_rate;
    if (audioBuffer.getNbOfSamples() == 0) {
      audioBuffer.setTimestamp(audioSamples.getTimestamp());
    }

    Status ret = audioBuffer.append(audioSamples);
    if (!ret.ok()) {
      return;
    }

    int samplesLeft = static_cast<int>(audioBuffer.getNbOfSamples());
    int samplesDone = 0;

    Logger::verbose(AVtag) << "(AUDIO buffer): nbSamples: " << audioBuffer.getNbOfSamples()
                           << "   audioCodec->frame_size: " << audioCodecContext->frame_size << std::endl;

    while (samplesLeft >= audioCodecContext->frame_size) {
      int samplesToEncode = audioCodecContext->frame_size;
      switch (m_sampleFormat) {
        case (Audio::SamplingFormat::PLANAR): {
          int iBuffer = 0;
          for (int i = 0; i < MAX_AUDIO_CHANNELS; ++i) {
            if (Audio::getChannelMapFromChannelIndex(i) & audioBuffer.getChannelLayout()) {
              if (iBuffer < Audio::getNbChannelsFromChannelLayout(audioBuffer.getChannelLayout()) &&
                  (m_channelMap.empty() ? iBuffer : m_channelMap[iBuffer]) < AV_NUM_DATA_POINTERS) {
                audioFrame->data[m_channelMap.empty() ? iBuffer : m_channelMap[iBuffer]] =
                    audioBuffer.getSamples()[i] + samplesDone * m_audioFrameSizeInBytes;
                iBuffer++;
              } else {
                Logger::warning(AVtag) << "Audio channel "
                                       << Audio::getStringFromChannelType(Audio::getChannelMapFromChannelIndex(i))
                                       << " not written" << std::endl;
              }
            }
          }
        } break;
        case (Audio::SamplingFormat::INTERLEAVED):
        case (Audio::SamplingFormat::FORMAT_UNKNOWN):
          std::stringstream msg;
          msg << "Only PLANAR format are supported";
          Logger::error(AVtag) << msg.str() << std::endl;
          outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected, msg.str());
          samplesLeft = 0;
      }

      audioFrame->nb_samples = samplesToEncode;
      audioFrame->pts = (int64_t)round((audioBuffer.getTimestamp() - m_pimplAudio->m_firstPTS +
                                        (samplesDone * 1000000.) / audioCodecContext->sample_rate) *
                                       (double)av_q2d(av_inv_q(audioCodecContext->time_base)) / 1000000.);
      Logger::verbose(AVtag) << "(AUDIO): date: " << audioBuffer.getTimestamp() << "  pts: " << audioFrame->pts
                             << "   samplesToEncode: " << samplesToEncode << "   samplesDone: " << samplesDone
                             << std::endl;

      Util::AvErrorCode ret = encodeAudioFrame(audioFrame);
      if (ret != Util::AvErrorCode::Ok) {
        std::stringstream msg;
        msg << "Error encountered while encoding audio. " << Util::errorStringFromAvErrorCode(ret);
        Logger::error(AVtag) << msg.str() << std::endl;
        outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected, msg.str());
      }
      samplesLeft -= samplesToEncode;
      samplesDone += samplesToEncode;
      audioBuffer.drop(samplesToEncode);
      audioBuffer.setTimestamp(lastTs -
                               (int64_t(audioBuffer.getNbOfSamples()) * 1000000) / audioCodecContext->sample_rate);
    }
  }
}

Util::AvErrorCode LibavWriter::encodeVideoFrame(AVFrame* frame, int64_t frameOffset) {
  Util::AvErrorCode ret = Util::getAvErrorCode(avcodec_send_frame(videoCodecContext, frame));
  if (ret != Util::AvErrorCode::Ok) {
    return ret;
  }

  // since AVCodec 3.1 API, each frame can produce multiple
  // output, we better loop over the output until no more
  // packets are available
  while (ret == Util::AvErrorCode::Ok) {
    auto pkt = newPacket();
    ret = Util::getAvErrorCode(avcodec_receive_packet(videoCodecContext, pkt.get()));
    if (ret == Util::AvErrorCode::Ok) {
      pkt->pts += frameOffset;
      pkt->dts += frameOffset;
      if ((frame == nullptr) && (pkt->duration == 0)) {
        pkt->duration = 1;
      }
      m_pimplVideo->pushVideoPacket(pkt);
    }
  }
  if (ret != Util::AvErrorCode::TryAgain) {
    return ret;
  }

  return Util::AvErrorCode::Ok;
}

Util::AvErrorCode LibavWriter::encodeAudioFrame(AVFrame* frame) {
  Util::AvErrorCode ret = Util::getAvErrorCode(avcodec_send_frame(audioCodecContext, frame));
  if (ret != Util::AvErrorCode::Ok) {
    return ret;
  }
  // since AVCodec 3.1 API, each frame can produce multiple
  // output, we better loop over the output until no more
  // packets are available
  while (ret == Util::AvErrorCode::Ok) {
    auto pkt = newPacket();
    ret = Util::getAvErrorCode(avcodec_receive_packet(audioCodecContext, pkt.get()));
    if (ret == Util::AvErrorCode::Ok) {
      pkt->pts += audioCodecContext->delay;
      pkt->dts += audioCodecContext->delay;
      m_pimplAudio->pushAudioPacket(pkt);
    }
  }
  if (ret != Util::AvErrorCode::TryAgain) {
    return ret;
  }
  return Util::AvErrorCode::Ok;
}

MuxerThreadStatus LibavWriter::flushVideo() {
  MuxerThreadStatus status = MuxerThreadStatus::OK;
  if (!m_pimplVideo) {
    return status;
  }

  int64_t frameIdOffset =
      (int64_t)(round((firstVideoPTS - m_pimplVideo->m_firstPTS) * videoCodecContext->time_base.den /
                      (1000000.0 * videoCodecContext->time_base.num)));
  Util::AvErrorCode ret = encodeVideoFrame(nullptr, frameIdOffset);
  if (ret != Util::AvErrorCode::Ok && ret != Util::AvErrorCode::EndOfFile) {
    std::stringstream msg;
    msg << "Error encountered while flushing video encoder. " << Util::errorStringFromAvErrorCode(ret);
    Logger::error(AVtag) << msg.str() << std::endl;
    outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected, msg.str());
  }

  if (m_pimplVideo.use_count() == 1) {
    status = m_pimplVideo->close();
  }

  return status;
}

MuxerThreadStatus LibavWriter::flushAudio() {
  MuxerThreadStatus status = MuxerThreadStatus::OK;
  if (!m_pimplAudio) {
    return status;
  }
  Util::AvErrorCode ret = encodeAudioFrame(nullptr);
  if (ret != Util::AvErrorCode::Ok && ret != Util::AvErrorCode::EndOfFile) {
    std::stringstream msg;
    msg << "Error encountered while flushing audio encoder. " << Util::errorStringFromAvErrorCode(ret);
    Logger::error(AVtag) << msg.str() << std::endl;
    outputEventManager.publishEvent(VideoStitch::Output::OutputEventManager::EventType::Disconnected, msg.str());
  }

  if (m_pimplAudio.use_count() == 1) {
    status = m_pimplAudio->close();
  }
  return status;
}

MuxerThreadStatus LibavWriter::close() {
  MuxerThreadStatus status = MuxerThreadStatus::OK;
  {
    std::lock_guard<std::mutex> lk(pimplMu);
    // Write the delayed frames
    if (videoCodecContext != nullptr) {
      status = flushVideo();
      m_pimplVideo.reset();
      AVFrame* videoFrame;
      while (videoFrames.size() > 0) {
        videoFrame = videoFrames.front();
        videoFrames.pop_front();
        av_frame_free(&videoFrame);
      }
    }

    if (audioCodecContext != nullptr) {
      status = flushAudio();
      m_pimplAudio.reset();
      av_frame_free(&audioFrame);
    }
  }

  // close codecs only after muxers are done
  if (audioCodecContext) {
    avcodec_close(audioCodecContext);
    avcodec_free_context(&audioCodecContext);
    audioCodecContext = nullptr;
    av_free((void*)avSamples);
  }
  if (videoCodecContext) {
    avcodec_close(videoCodecContext);
    avcodec_free_context(&videoCodecContext);
    videoCodecContext = nullptr;
  }
  return status;
}

}  // namespace Output
}  // namespace VideoStitch
