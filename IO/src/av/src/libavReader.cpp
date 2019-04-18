// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#define NOMINMAX

#include "libavReader.hpp"
#include "util.hpp"
#include "netStreamReader.hpp"

#include "videoReader.hpp"

#include "d3dAllocator.hpp"
#include "d3dDevice.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/frame.hpp"

extern "C" {
#ifdef QUICKSYNC
#include <libavcodec/qsv.h>
#endif
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
}

#ifdef SUP_NVDEC
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <sstream>

static const int64_t NO_FRAME_READ_YET{std::numeric_limits<int64_t>::min()};

// ------------------------- Frame allocator ---------------------------

namespace VideoStitch {
namespace Input {

namespace {

#undef PixelFormat

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

static std::string AVtag("avinput");

VideoStitch::PixelFormat ffmpeg2vs(const AVPixelFormat fmt) {
  switch (fmt) {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
      return VideoStitch::YV12;
    case AV_PIX_FMT_YUV422P10LE:
      return VideoStitch::YUV422P10;
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_CUDA:
      return VideoStitch::NV12;
    default:
      assert(false);
      return Unknown;
  }
}

uint64_t ffmpeg2size(const AVPixelFormat fmt, const uint64_t width, const uint64_t height) {
  switch (fmt) {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_CUDA:
      return (width * height * 3) / 2;
    case AV_PIX_FMT_YUV422P10LE:
      return width * height * 4;
    default:
      assert(false);
      return Unknown;
  }
}

int is_hwaccel_pix_fmt(enum AVPixelFormat pix_fmt) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(pix_fmt);
  return desc->flags & AV_PIX_FMT_FLAG_HWACCEL;
}
}  // namespace

// -------------- Quick Sync specific -----------------------

#ifdef QUICKSYNC

static mfxStatus frame_alloc(mfxHDL pthis, mfxFrameAllocRequest* req, mfxFrameAllocResponse* resp) {
#if defined(_WIN32) || defined(_WIN64)
  QSVContext* decode = (QSVContext*)pthis;

  if (decode->surface_ids) {
    fprintf(stderr, "Multiple allocation requests.\n");
    return MFX_ERR_MEMORY_ALLOC;
  }
  if (!(req->Type & MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET)) {
    fprintf(stderr, "Unsupported surface type: %d\n", req->Type);
    return MFX_ERR_UNSUPPORTED;
  }
  if (req->Info.BitDepthLuma != 8 || req->Info.BitDepthChroma != 8 || req->Info.Shift ||
      req->Info.FourCC != MFX_FOURCC_NV12 || req->Info.ChromaFormat != MFX_CHROMAFORMAT_YUV420) {
    fprintf(stderr, "Unsupported surface properties.\n");
    return MFX_ERR_UNSUPPORTED;
  }

  decode->surface_ids = (mfxMemId*)av_malloc_array(req->NumFrameSuggested, sizeof(*decode->surface_ids));
  decode->surface_used = (int*)av_mallocz_array(req->NumFrameSuggested, sizeof(*decode->surface_used));
  if (!decode->surface_ids || !decode->surface_used) goto fail;

  mfxStatus sts = decode->allocator->AllocFrames(req, resp);

  if (sts != MFX_ERR_NONE) {
    fprintf(stderr, "Error allocating surfaces\n");
    goto fail;
  }

  decode->surface_ids = resp->mids;
  decode->nb_surfaces = resp->NumFrameActual;
  decode->frame_info = req->Info;

  return MFX_ERR_NONE;
fail:
  av_freep(&decode->surface_ids);
  av_freep(&decode->surface_used);
#endif
  return MFX_ERR_MEMORY_ALLOC;
}

static mfxStatus frame_free(mfxHDL pthis, mfxFrameAllocResponse* resp) {
#if defined(_WIN32) || defined(_WIN64)
  QSVContext* decode = (QSVContext*)pthis;
  return decode->allocator->FreeFrames(resp);
#endif
  return MFX_ERR_MEMORY_ALLOC;
}

static mfxStatus frame_lock(mfxHDL pthis, mfxMemId mid, mfxFrameData* ptr) {
#if defined(_WIN32) || defined(_WIN64)
  QSVContext* decode = (QSVContext*)pthis;
  return decode->allocator->LockFrame(mid, ptr);
#endif
  return MFX_ERR_MEMORY_ALLOC;
}

static mfxStatus frame_unlock(mfxHDL pthis, mfxMemId mid, mfxFrameData* ptr) {
#if defined(_WIN32) || defined(_WIN64)
  QSVContext* decode = (QSVContext*)pthis;
  return decode->allocator->UnlockFrame(mid, ptr);
#endif
  return MFX_ERR_MEMORY_ALLOC;
}

static mfxStatus frame_get_hdl(mfxHDL pthis, mfxMemId mid, mfxHDL* hdl) {
#if defined(_WIN32) || defined(_WIN64)
  QSVContext* decode = (QSVContext*)pthis;
  return decode->allocator->GetFrameHDL(mid, hdl);
#endif
  return MFX_ERR_MEMORY_ALLOC;
}

bool QSVContext::initMFX() {
#if defined(_WIN32) || defined(_WIN64)
  // initialize a hardware session
  mfxIMPL impl = MFX_IMPL_HARDWARE_ANY;
  mfxVersion min_version = {{1, 1}};

  mfxStatus sts = MFXInit(impl, &min_version, &session);
  if (sts != MFX_ERR_NONE) {
    Logger::get(Logger::Error) << "Error initializing an MFX session" << std::endl;
    return false;
  }

  // create a device
  mfxU32 adapterNum = 0;
  MFXQueryIMPL(session, &impl);

  // extract the base implementation type
  mfxIMPL baseImpl = MFX_IMPL_BASETYPE(impl);
  const struct {
    // actual implementation
    mfxIMPL impl;
    // adapter's number
    mfxU32 adapterID;
  } implTypes[] = {{MFX_IMPL_HARDWARE, 0},
                   {MFX_IMPL_SOFTWARE, 0},
                   {MFX_IMPL_HARDWARE2, 1},
                   {MFX_IMPL_HARDWARE3, 2},
                   {MFX_IMPL_HARDWARE4, 3}};

  // get corresponding adapter number
  for (mfxU8 i = 0; i < sizeof(implTypes) / sizeof(*implTypes); i++) {
    if (implTypes[i].impl == baseImpl) {
      adapterNum = implTypes[i].adapterID;
      break;
    }
  }

  hwdev = new CD3D9Device();
  if (hwdev == nullptr) {
    return false;
  }
  sts = hwdev->Init(
      nullptr, 0,
      adapterNum);  // XXX TODO FIXME https://software.intel.com/en-us/forums/intel-media-sdk/topic/599935#node-599935

  // handle is needed for HW library only
  mfxHDL hdl = nullptr;
  mfxHandleType hdl_t = MFX_HANDLE_D3D9_DEVICE_MANAGER;
  hwdev->GetHandle(hdl_t, &hdl);
  MFXVideoCORE_SetHandle(session, hdl_t, hdl);

  MFXQueryIMPL(session, &impl);
  if (impl == MFX_IMPL_SOFTWARE) {
    Logger::get(Logger::Error) << "Quick Sync hardware decoding not available." << std::endl;
    return false;
  }

  // wrap the allocator in order to trace the surfaces in use
  allocator = new D3DFrameAllocator;
  D3DAllocatorParams* pd3dAllocParams = new D3DAllocatorParams;
  pd3dAllocParams->pManager = reinterpret_cast<IDirect3DDeviceManager9*>(hdl);
  allocator->Init(pd3dAllocParams);

  mfxFrameAllocator* frame_allocator = new mfxFrameAllocator;
  frame_allocator->pthis = this;
  frame_allocator->Alloc = frame_alloc;
  frame_allocator->Lock = frame_lock;
  frame_allocator->Unlock = frame_unlock;
  frame_allocator->GetHDL = frame_get_hdl;
  frame_allocator->Free = frame_free;

  MFXVideoCORE_SetFrameAllocator(session, frame_allocator);

  surface_ids = nullptr;
#endif
  return true;
}

void QSVContext::freeQSVBuffer(void* opaque, uint8_t* data) {
  int* used = (int*)opaque;
  *used = 0;
  av_freep(&data);
}

int QSVContext::getQSVBuffer(AVCodecContext* avctx, AVFrame* frame, int flags) {
  QSVContext* that = (QSVContext*)avctx->opaque;

  mfxFrameSurface1* surf;
  AVBufferRef* surf_buf;
  int idx;

  for (idx = 0; idx < that->nb_surfaces; idx++) {
    if (!that->surface_used[idx]) break;
  }
  if (idx == that->nb_surfaces) {
    fprintf(stderr, "No free surfaces\n");
    return AVERROR(ENOMEM);
  }

  surf = (mfxFrameSurface1*)av_mallocz(sizeof(*surf));
  if (!surf) return AVERROR(ENOMEM);
  surf_buf = av_buffer_create((uint8_t*)surf, sizeof(*surf), &QSVContext::freeQSVBuffer, &that->surface_used[idx],
                              AV_BUFFER_FLAG_READONLY);
  if (!surf_buf) {
    av_freep(&surf);
    return AVERROR(ENOMEM);
  }

  surf->Info = that->frame_info;
  surf->Data.MemId = &that->surface_ids[idx];

  frame->buf[0] = surf_buf;
  frame->data[3] = (uint8_t*)surf;

  that->surface_used[idx] = 1;

  return 0;
}

#endif

// ---------------------------------------------------------------------

enum AVPixelFormat LibavReader::selectFormat(struct AVCodecContext* avctx, const enum AVPixelFormat* fmt) {
  // select the first hardware-accelerated pixel format if available, else the first one
  const enum AVPixelFormat* backup = fmt;
  while (*fmt != AV_PIX_FMT_NONE && !is_hwaccel_pix_fmt(*fmt)) {
    ++fmt;
  }
  // Quick Sync
#ifdef QUICKSYNC
  if (*fmt == AV_PIX_FMT_QSV) {
    if (!avctx->hwaccel_context) {
      QSVContext* decode = (QSVContext*)avctx->opaque;
      AVQSVContext* qsv = av_qsv_alloc_context();
      if (!qsv) return AV_PIX_FMT_NONE;

      qsv->session = (mfxSession)decode->session;
      qsv->iopattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;

      avctx->hwaccel_context = qsv;
    }

    return AV_PIX_FMT_QSV;
  }
#else
  (void)avctx;
#endif

  if (fmt[0] == AV_PIX_FMT_NONE) {
    return backup[0];
  } else {
    return fmt[0];
  }
}

void LibavReader::findAvStreams(struct AVFormatContext* formatCtx, int& videoIdx, int& audioIdx) {
  if (formatCtx->nb_streams == 0) {
    return;
  }
  // Check for Video Stream
  videoIdx = av_find_best_stream(formatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (videoIdx == AVERROR_STREAM_NOT_FOUND) {
    videoIdx = INVALID_STREAM_ID;
  }
  // Check for Audio stream
  audioIdx = av_find_best_stream(formatCtx, AVMEDIA_TYPE_AUDIO, -1, videoIdx, nullptr, 0);
  if (audioIdx == AVERROR_STREAM_NOT_FOUND) {
    audioIdx = INVALID_STREAM_ID;
  }
  // Ignore any other stream
  for (unsigned streamIndex = 0; streamIndex < formatCtx->nb_streams; ++streamIndex) {
    if ((int)streamIndex != videoIdx && (int)streamIndex != audioIdx) {
      formatCtx->streams[streamIndex]->discard = AVDISCARD_ALL;
    }
  }
}

FFmpegReader::PotentialLibavReader LibavReader::create(const std::string& filename,
                                                       VideoStitch::Plugin::VSReaderPlugin::Config runtime) {
  // register all codecs, demux and protocols
  Util::Libav::checkInitialization();

  AVFormatContext* formatCtx = avformat_alloc_context();
#ifdef QUICKSYNC
  QSVContext* qsvCtx = nullptr;
#endif

  if (!formatCtx) {
    return {Origin::Input, ErrType::RuntimeError, "[" + AVtag + "] Can't allocate format context"};
  }

  AVDictionary* format_opt = nullptr;
  Util::TimeoutHandler* interruptCallback = nullptr;
  if (Util::isStream(filename)) {
    interruptCallback =
        new Util::TimeoutHandler(std::chrono::milliseconds(15000));  // timeout to allow network connection to proceed
    formatCtx->interrupt_callback.callback = &Util::TimeoutHandler::checkInterrupt;
    formatCtx->interrupt_callback.opaque = interruptCallback;
    formatCtx->flags |= AVFMT_FLAG_NONBLOCK;
    // default UDP buffer size define by ffmpeg is too small 64k
    const int LIBAV_UDP_MAX_PKT_SIZE = 256 * 1024;
    av_dict_set_int(&format_opt, "buffer_size", LIBAV_UDP_MAX_PKT_SIZE, 0);
  }

  int r = avformat_open_input(&formatCtx, filename.c_str(), nullptr, &format_opt);
  if (r < 0) {
    std::stringstream msg;
    msg << "[" << AVtag << "] Error when initializing the format context with error number: " << r << " : "
        << Util::errorString(r);
    return {Origin::Input, ErrType::RuntimeError, msg.str()};
  }
  av_dict_free(&format_opt);

  // if you don't call you may miss the first frames
  {
    Util::Libav::Lock sl;
    r = avformat_find_stream_info(formatCtx, nullptr);
    if (r < 0) {
      std::stringstream msg;
      msg << "[" << AVtag << "] Couldn't get additional video stream info" << r << " : " << Util::errorString(r);
      avformat_close_input(&formatCtx);
      avformat_free_context(formatCtx);
      return {Origin::Input, ErrType::RuntimeError, msg.str()};
    }
  }

  // find the first video stream
  int videoIdx = INVALID_STREAM_ID;
  int audioIdx = INVALID_STREAM_ID;
  findAvStreams(formatCtx, videoIdx, audioIdx);
  if (videoIdx == INVALID_STREAM_ID) {
    avformat_close_input(&formatCtx);
    avformat_free_context(formatCtx);
    return {Origin::Input, ErrType::InvalidConfiguration, "[" + AVtag + "] Couldn't find any video stream"};
  }

  // find appropriate decoders
  AVDictionary* opt = nullptr;
  AddressSpace addrSpace = Host;
  AVCodecContext* videoDecoderCtx = nullptr;
  AVCodec* videoCodec = nullptr;
  //  bool success = false;
  if (formatCtx->streams[videoIdx]->codecpar->codec_id == AV_CODEC_ID_H264) {
    // try with a discrete GPU decoder first
    /*videoCodec = avcodec_find_decoder_by_name("h264_cuvid");
    if (videoCodec) {
      Util::Libav::Lock sl;
      r = avcodec_open2(formatCtx->streams[videoIdx]->codec, videoCodec, &opt);
      if (r == 0) {
        Logger::get(Logger::Info) << "Using NVIDIA Pure Video hardware decoder" << std::endl;
        addrSpace = Device;
        // install the callback selecting a hardware-accelerated pixel format
        formatCtx->streams[videoIdx]->codec->get_format = &LibavReader::selectFormat;
        goto VideoOpen;
      }
    }*/

    // then try with an integrated GPU decoder
    /*
    videoCodec = avcodec_find_decoder_by_name("h264_qsv");
    if (videoCodec) {
      Util::Libav::Lock sl;
      qsvCtx = new QSVContext;
      bool r = qsvCtx->initMFX();
      if (r) {
        formatCtx->streams[videoIdx]->codec->opaque = qsvCtx;
        formatCtx->streams[videoIdx]->codec->get_buffer2 = &QSVContext::getQSVBuffer;

        r = avcodec_open2(formatCtx->streams[videoIdx]->codec, videoCodec, &opt);
        if (r == 0) {
          Logger::get(Logger::Info) << "Using Intel QuickSync hardware decoder" << std::endl;
          addrSpace = Host;
          goto VideoOpen;
        } else {
          delete qsvCtx;
        }
      }
    }*/

    // X264 : force single thread as h264 probing seems to miss SPS/PPS and seek fails silently
    av_dict_set(&opt, "threads", "1", 0);
  }

  // finally use a software decoder
  videoCodec = avcodec_find_decoder(formatCtx->streams[videoIdx]->codecpar->codec_id);
  if (videoCodec) {
    Util::Libav::Lock sl;
    videoDecoderCtx = avcodec_alloc_context3(videoCodec);
    if (!videoDecoderCtx) {
      avformat_close_input(&formatCtx);
      avformat_free_context(formatCtx);
      return {Origin::Input, ErrType::RuntimeError, "[" + AVtag + "] Could not allocate context for video decoder"};
    }

    if (avcodec_parameters_to_context(videoDecoderCtx, formatCtx->streams[videoIdx]->codecpar) < 0) {
      avcodec_close(videoDecoderCtx);
      avformat_close_input(&formatCtx);
      avformat_free_context(formatCtx);
      return {Origin::Input, ErrType::RuntimeError,
              "[" + AVtag + "] Could not copy video codec parameters to decoder contex"};
    }

    r = avcodec_open2(videoDecoderCtx, videoCodec, &opt);
    if (r == 0) {
      Logger::info(AVtag) << "Using a software decoder" << std::endl;
      addrSpace = Host;
      goto VideoOpen;
    }

    avcodec_free_context(&videoDecoderCtx);
    avformat_close_input(&formatCtx);
    avformat_free_context(formatCtx);
    return {Origin::Input, ErrType::RuntimeError, "[" + AVtag + "] Could not open video decoder"};
  }

  // can't decode than video :(
  avformat_close_input(&formatCtx);
  avformat_free_context(formatCtx);
  return {Origin::Input, ErrType::UnsupportedAction, "[" + AVtag + "] Could not find decoder for video stream"};

VideoOpen:

  Status status;
  AVCodecContext* audioDecoderCtx = nullptr;
  AVCodec* audioCodec = nullptr;
  struct AVFrame* videoFrame = nullptr;
  struct AVFrame* audioFrame = nullptr;
  int width, height;
  Audio::SamplingRate samplingRate = Audio::SamplingRate::SR_NONE;
  Audio::SamplingDepth samplingDepth = Audio::SamplingDepth::SD_NONE;
  Audio::ChannelLayout layout = Audio::UNKNOWN;
  AVPixelFormat fmt = AV_PIX_FMT_YUV420P;
  LibavReader* reader = nullptr;

  if (audioIdx != INVALID_STREAM_ID) {
    audioCodec = avcodec_find_decoder(formatCtx->streams[audioIdx]->codecpar->codec_id);
    if (!audioCodec) {
      status =
          Status(Origin::Input, ErrType::UnsupportedAction, "[" + AVtag + "] Could not find decoder for audio stream");
      goto CreateFailed;
    }
  }

  // open the audio codec
  {
    Util::Libav::Lock sl;
    if (audioIdx != INVALID_STREAM_ID) {
      audioDecoderCtx = avcodec_alloc_context3(audioCodec);
      if (!audioDecoderCtx) {
        std::stringstream msg;
        msg << "[" << AVtag << "] Unable to allocate the audio decoder context " << r << ": " << Util::errorString(r);
        status = Status(Origin::Input, ErrType::RuntimeError, msg.str());
        goto CreateFailed;
      }

      if (avcodec_parameters_to_context(audioDecoderCtx, formatCtx->streams[audioIdx]->codecpar) < 0) {
        std::stringstream msg;
        msg << "[" << AVtag << "] Unable to copy audio codec parameters to decoder " << r << ": "
            << Util::errorString(r);
        status = Status(Origin::Input, ErrType::RuntimeError, msg.str());
        avcodec_free_context(&audioDecoderCtx);
        audioDecoderCtx = nullptr;
        goto CreateFailed;
      }
      r = avcodec_open2(audioDecoderCtx, audioCodec, &opt);
      if (r < 0) {
        std::stringstream msg;
        msg << "[" << AVtag << "] Unable to open audio decoder " << r << ": " << Util::errorString(r);
        status = Status(Origin::Input, ErrType::RuntimeError, msg.str());
        avcodec_free_context(&audioDecoderCtx);
        audioDecoderCtx = nullptr;
        goto CreateFailed;
      }
    }
  }

  videoFrame = av_frame_alloc();
  if (!videoFrame) {
    status = Status(Origin::Input, ErrType::OutOfResources, "[avinput] Can't allocate video frame");
    goto CreateFailed;
  }
  audioFrame = av_frame_alloc();
  if (!audioFrame) {
    status = Status(Origin::Input, ErrType::OutOfResources, "[avinput] Can't allocate audio frame");
    goto CreateFailed;
  }
  av_dict_free(&opt);

  if (videoCodec->pix_fmts != nullptr) {
    fmt = avcodec_default_get_format(videoDecoderCtx, videoCodec->pix_fmts);
  } else {
    fmt = videoDecoderCtx->pix_fmt;
  }

  // check colorspace
  if (fmt != AV_PIX_FMT_YUV420P && fmt != AV_PIX_FMT_YUVJ420P && fmt != AV_PIX_FMT_NV12 &&
      fmt != AV_PIX_FMT_YUV422P10LE && fmt != AV_PIX_FMT_CUDA) {
    const char* codecName = videoDecoderCtx->codec->name ? videoDecoderCtx->codec->name : "[unknown]";
    const char* containerName = formatCtx->iformat->long_name
                                    ? formatCtx->iformat->long_name
                                    : (formatCtx->iformat->name ? formatCtx->iformat->name : "[unknown]");
    std::stringstream msg;
    msg << "[" << AVtag << "] Unsupported colorspace";
    {
      const char* pixFmtName = av_get_pix_fmt_name(fmt);
      if (pixFmtName) {
        msg << " '" << pixFmtName << "'";
      }
    }
    msg << " for codec '" << codecName << "' in container '" << containerName
        << "'. Only planar YUV 4:2:0 or 4:2:2 are supported.";
    status = Status(Origin::Input, ErrType::InvalidConfiguration, msg.str());
    goto CreateFailed;
  }

  // check size
  width = formatCtx->streams[videoIdx]->codecpar->width;
  height = formatCtx->streams[videoIdx]->codecpar->height;
  if (width != runtime.width || height != runtime.height) {
    std::stringstream msg;
    msg << "[" << AVtag << "] Input size (" << width << "x" << height << ") is different from expected size ("
        << runtime.width << "x" << runtime.height << ")";
    status = Status(Origin::Input, ErrType::InvalidConfiguration, msg.str());
    goto CreateFailed;
  }

  if (audioIdx != INVALID_STREAM_ID) {
    layout = VideoStitch::Util::channelLayout(formatCtx->streams[audioIdx]->codecpar->channel_layout);

    // channels can be valid and channel_layout null (eg with Atomos and Hyperdeck Prores inputs)
    if (layout == Audio::UNKNOWN) {
      layout = Audio::getAChannelLayoutFromNbChannels(formatCtx->streams[audioIdx]->codecpar->channels);
    }
    if (layout == Audio::UNKNOWN) {
      Logger::warning(AVtag) << "Unknown audio channel layout '"
                             << formatCtx->streams[audioIdx]->codecpar->channel_layout << "' ... disabling audio input"
                             << std::endl;
      audioIdx = INVALID_STREAM_ID;
      audioCodec = nullptr;
      av_frame_free(&audioFrame);
    } else {
      samplingDepth = Util::sampleFormat(audioDecoderCtx->sample_fmt);
      const auto audioSampleRate = formatCtx->streams[audioIdx]->codecpar->sample_rate;
      if (audioSampleRate == 22050) {
        samplingRate = Audio::SamplingRate::SR_22050;
      } else if (audioSampleRate == 32000) {
        samplingRate = Audio::SamplingRate::SR_32000;
      } else if (audioSampleRate == 44100) {
        samplingRate = Audio::SamplingRate::SR_44100;
      } else if (audioSampleRate == 48000) {
        samplingRate = Audio::SamplingRate::SR_48000;
      } else {
        Logger::warning(AVtag) << "Unknown audio sample rate " << audioSampleRate << " ... disabling audio input"
                               << std::endl;
        audioIdx = INVALID_STREAM_ID;
        audioCodec = nullptr;
        av_frame_free(&audioFrame);
      }
    }
  }

  if (VideoStitch::Input::netStreamReader::handles(filename)) {
    reader =
        new netStreamReader(runtime.id, filename, width, height, runtime.targetFirstFrame, fmt, addrSpace, formatCtx,
#ifdef QUICKSYNC
                            qsvCtx,
#endif
                            videoDecoderCtx, audioDecoderCtx, videoCodec, audioCodec, videoFrame, audioFrame,
                            interruptCallback, videoIdx, audioIdx, layout, samplingRate, samplingDepth);

  } else {
    reader = new FFmpegReader(runtime.id, filename, width, height, runtime.targetFirstFrame, fmt, addrSpace, formatCtx,
#ifdef QUICKSYNC
                              qsvCtx,
#endif
                              videoDecoderCtx, audioDecoderCtx, videoCodec, audioCodec, videoFrame, audioFrame,
                              interruptCallback, videoIdx, audioIdx, layout, samplingRate, samplingDepth);
  }

  if (interruptCallback) {
    interruptCallback->reset(std::chrono::milliseconds(2000));  // timeout for blocking operations
  }

  if (!reader) {
    std::stringstream msg;
    msg << "[" << AVtag << "] Couldn't create the reader for file '" << filename << "'";
    return {Origin::Input, ErrType::OutOfResources, msg.str()};
  }
  return reader;

CreateFailed:

  Util::Libav::Lock sl;
  if (audioFrame) av_frame_free(&audioFrame);
  if (videoFrame) av_frame_free(&videoFrame);
  if (audioDecoderCtx) {
    avcodec_close(audioDecoderCtx);
    avcodec_free_context(&audioDecoderCtx);
  }
  avcodec_close(videoDecoderCtx);
  avcodec_free_context(&videoDecoderCtx);
  avformat_close_input(&formatCtx);
  avformat_free_context(formatCtx);
  delete interruptCallback;
  return status;
}

LibavReader::LibavReader(const std::string& /* displayName */, const int64_t width, const int64_t height,
                         const int firstFrame, const AVPixelFormat fmt, AddressSpace addrSpace,
                         struct AVFormatContext* formatCtx,
#ifdef QUICKSYNC
                         class QSVContext* qsvCtx,
#endif
                         struct AVCodecContext* videoDecoderCtx, struct AVCodecContext* audioDecoderCtx,
                         struct AVCodec* videoCodec, struct AVCodec* audioCodec, struct AVFrame* videoFrame,
                         struct AVFrame* audioFrame, Util::TimeoutHandler* interruptCallback, const int videoIdx,
                         const int audioIdx, const Audio::ChannelLayout layout, const Audio::SamplingRate samplingRate,
                         const Audio::SamplingDepth samplingDepth)
    : Reader(-1),
      VideoReader(width, height, ffmpeg2size(fmt, width, height), ffmpeg2vs(fmt), addrSpace,
                  {formatCtx->streams[videoIdx]->r_frame_rate.num, formatCtx->streams[videoIdx]->r_frame_rate.den},
                  firstFrame, (int)formatCtx->streams[videoIdx]->nb_frames - 1, false /* not a procedural reader */,
                  nullptr),
      AudioReader(layout, samplingRate, samplingDepth),
      formatCtx(formatCtx),
#ifdef QUICKSYNC
      qsvCtx(qsvCtx),
#endif
      videoDecoderCtx(videoDecoderCtx),
      audioDecoderCtx(audioDecoderCtx),
      videoCodec(videoCodec),
      audioCodec(audioCodec),
      videoFrame(videoFrame),
      audioFrame(audioFrame),
      interruptCallback(interruptCallback),
      videoIdx(videoIdx),
      audioIdx(audioIdx),
      currentVideoPts(NO_FRAME_READ_YET),
      audioBuffer(Audio::getNbChannelsFromChannelLayout(layout)),
      nbSamplesInAudioBuffer(0),
      videoTimeStamp(-1),
      audioTimeStamp(-1),
      expectingIncreasingVideoPts(false) {

  firstVideoFramePts = formatCtx->streams[videoIdx]->start_time;
  if (firstVideoFramePts == AV_NOPTS_VALUE) {
    Logger::warning(AVtag) << "Video stream start_time is undefined (AV_NOPTS_VALUE)" << std::endl;
    firstVideoFramePts = 0;
  }
}

LibavReader::~LibavReader() {
  if (videoFrame) {
    av_frame_free(&videoFrame);
  }
  if (audioFrame) {
    av_frame_free(&audioFrame);
  }

  if (formatCtx) {
    // Close codecs
    {
      Util::Libav::Lock sl;
      if ((unsigned int)videoIdx < formatCtx->nb_streams && videoDecoderCtx) {
        avcodec_free_context(&videoDecoderCtx);
        videoDecoderCtx = nullptr;
      }

      if ((unsigned int)audioIdx < formatCtx->nb_streams && audioDecoderCtx) {
        avcodec_free_context(&audioDecoderCtx);
        audioDecoderCtx = nullptr;
      }
    }
    avformat_close_input(&formatCtx);
  }
  delete interruptCallback;
#ifdef QUICKSYNC
  delete qsvCtx;
#endif
}

// -------------------------- Reading --------------------------

LibavReader::LibavReadStatus LibavReader::readPacket(AVPacket* pkt) {
  // read packet from media
  if (interruptCallback) {
    interruptCallback->reset(std::chrono::milliseconds(1000));
  }
  const int status = av_read_frame(formatCtx, pkt);

  if (status == (int)AVERROR_EOF || eos()) {
    return LibavReader::LibavReadStatus::EndOfPackets;
  } else if (formatCtx->pb && formatCtx->pb->error) {
    Logger::warning(AVtag) << "Stream contains an irrecoverable error - leaving : " << formatCtx->pb->error
                           << std::endl;
    return LibavReader::LibavReadStatus::Error;
  } else if (status < 0) {
    Logger::error(AVtag) << "Unable to read packet from  source with error " << status << " : "
                         << Util::errorString(status) << std::endl;
    return LibavReader::LibavReadStatus::Error;
  }
  // let's try to continue even on status > 0 with unknown error, maybe it's recoverable
  return LibavReader::LibavReadStatus::Ok;
}

// Since FFMPEG 3.1, if we got a frame, we must call this again with
// pkt==NULL to get more output
Util::AvErrorCode LibavReader::avDecodePacket(AVCodecContext* s, AVPacket* pkt, AVFrame* frame, bool* got_frame,
                                              bool flush) {
  *got_frame = false;
  {
    VideoStitch::Util::SimpleProfiler prof("FFmpeg frame decoding ", true, Logger::get(Logger::Debug));
    Util::AvErrorCode ret;

    auto isExpectedEOF = [flush, pkt](Util::AvErrorCode ret) -> bool {
      return ret == Util::AvErrorCode::EndOfFile && (flush || !pkt);
    };

    if (pkt || flush) {
      ret = Util::getAvErrorCode(avcodec_send_packet(s, pkt));

      if (ret != Util::AvErrorCode::Ok) {
        if (!isExpectedEOF(ret)) {
          Logger::get(Logger::Warning) << "[avinput] Error encoutered while sending packet to decoder: "
                                       << Util::errorStringFromAvErrorCode(ret) << std::endl;
        }
        return ret;
      }
    }

    ret = Util::getAvErrorCode(avcodec_receive_frame(s, frame));

    if (ret != Util::AvErrorCode::Ok && ret != Util::AvErrorCode::TryAgain) {
      if (!isExpectedEOF(ret)) {
        Logger::get(Logger::Warning) << "[avinput] Error encoutered while receiving frame from decoder: "
                                     << Util::errorStringFromAvErrorCode(ret) << std::endl;
      }
      return ret;
    }

    if (ret == Util::AvErrorCode::Ok) {
      *got_frame = true;
    }
  }
  return Util::AvErrorCode::Ok;
}

void LibavReader::flushVideoDecoder(bool* got_picture, unsigned char* frame) {
  decodeVideoPacket(got_picture, nullptr, frame, true);
}

void LibavReader::decodeVideoPacket(bool* got_picture, AVPacket* pkt, unsigned char* frame, bool flush) {
  avDecodePacket(videoDecoderCtx, pkt, videoFrame, got_picture, flush);
  if (flush) {
    // flushing
    while (*got_picture && av_frame_get_best_effort_timestamp(videoFrame) <= currentVideoPts) {
      // VSA-5887: if the decoder gives non-increasing PTS when flushing with
      // null packets at the end of the video, just ignore these frames and report
      // EOF immediately, as we don't have code to reorder frames
      Logger::get(Logger::Warning) << "[avinput] Non-increasing timestamps received in EOF flushing, discarding frame."
                                   << std::endl;
      avDecodePacket(videoDecoderCtx, nullptr, videoFrame, got_picture);
    }
  }

  if (*got_picture) {
    int64_t lastVideoPts = currentVideoPts;
    currentVideoPts = av_frame_get_best_effort_timestamp(videoFrame);

    if (expectingIncreasingVideoPts && currentVideoPts <= lastVideoPts) {
      Logger::get(Logger::Warning) << "[avinput] Non-increasing timestamps received: " << currentVideoPts << " after "
                                   << lastVideoPts << std::endl;
    }
    expectingIncreasingVideoPts = true;

    int64_t videoPtsFromVideoStart;

    if (currentVideoPts == AV_NOPTS_VALUE) {
      Logger::get(Logger::Warning) << "[avinput] Invalid timestamps received: " << currentVideoPts << std::endl;
      // try to continue, don't change firstVideoFramePts
      videoPtsFromVideoStart = 0;
    } else {
      // the first presentable video frame should have videoTimeStamp = 0 in the caller's reference
      // if the container has offsets between the starting points of its streams, it needs to be handled internally here
      videoPtsFromVideoStart = currentVideoPts - firstVideoFramePts;
      // Assets videoformat01/source/GOPR0008A.MP4 has a start_time (in pts) of 6006
      // while the best effort timestamp of the first frame is 0
      if (videoPtsFromVideoStart < 0) {
        Logger::get(Logger::Error) << "[avinput] Video frame pts " << currentVideoPts
                                   << " smaller than assumed video start time " << firstVideoFramePts << std::endl;

        // try to continue...
        videoPtsFromVideoStart = 0;
        firstVideoFramePts = currentVideoPts;
      }
    }

    // in AVRational AVStream::time_base units, from the start of the container (not the video stream, see start_time
    // semantics) eg. {1, 90000} => 1/90000th of a second in microseconds, VideoStitch's time unit of choice
    const auto streamTimeBase = formatCtx->streams[videoIdx]->time_base;
    videoTimeStamp = av_rescale(videoPtsFromVideoStart, (1000000 * streamTimeBase.num), streamTimeBase.den);

    // in modern codecs, the resolution can change on the fly
    if (formatCtx->streams[videoIdx]->codecpar->width != videoFrame->width ||
        formatCtx->streams[videoIdx]->codecpar->height != videoFrame->height) {
      Logger::get(Logger::Warning) << "[libavinput] Stream resolution changed (unhandled)" << std::endl;
    }

    if (frame) {
      // first case : CUDA addressing space
      if (VideoReader::getSpec().addressSpace == Device) {
#ifdef SUP_NVDEC
        // NV12 directly on the device
        cudaMemcpy2D((void*)frame, VideoReader::getSpec().width, videoFrame->data[0], videoFrame->linesize[0],
                     VideoReader::getSpec().width, VideoReader::getSpec().height, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D((void*)(frame + VideoReader::getSpec().width * VideoReader::getSpec().height),
                     VideoReader::getSpec().width, videoFrame->data[1], videoFrame->linesize[1],
                     VideoReader::getSpec().width, VideoReader::getSpec().height / 2, cudaMemcpyDeviceToDevice);
        return;
#else
        assert(false);
#endif
      }

      // Host addressing space
      int chromaLineSize = 0, lumaLineSize = 0, chromaLineCount = 0;

      if (VideoReader::getSpec().format == YV12) {
        chromaLineSize = videoFrame->width / 2;
        chromaLineCount = videoFrame->height / 2;
        lumaLineSize = videoFrame->width;
      } else if (VideoReader::getSpec().format == NV12) {
        chromaLineSize = videoFrame->width;
        chromaLineCount = videoFrame->height / 2;
        lumaLineSize = videoFrame->width;
      } else if (VideoReader::getSpec().format == YUV422P10) {
        chromaLineSize = videoFrame->width;
        chromaLineCount = videoFrame->height;
        lumaLineSize = videoFrame->width * 2;
      }
      // y
      unsigned char* ptr = frame;
      for (int line = 0; line < videoFrame->height; line++) {
        memcpy(ptr, videoFrame->data[0] + line * videoFrame->linesize[0], lumaLineSize);
        ptr += lumaLineSize;
      }
      if (VideoReader::getSpec().format == YV12 || VideoReader::getSpec().format == YUV422P10) {
        // planar
        // u
        for (int line = 0; line < chromaLineCount; line++) {
          memcpy(ptr, videoFrame->data[1] + line * videoFrame->linesize[1], chromaLineSize);
          ptr += chromaLineSize;
        }
        // v
        for (int line = 0; line < chromaLineCount; line++) {
          memcpy(ptr, videoFrame->data[2] + line * videoFrame->linesize[2], chromaLineSize);
          ptr += chromaLineSize;
        }
      } else if (VideoReader::getSpec().format == NV12) {
        // interleaved UV
        for (int line = 0; line < chromaLineCount; line++) {
          memcpy(ptr, videoFrame->data[1] + line * videoFrame->linesize[1], chromaLineSize);
          ptr += chromaLineSize;
        }
      }
    }
  }
}

int numSamplesToDropToSyncToStartOfVideo(mtime_t audioTimeStamp, int sampleRate) {
  Logger::info(AVtag) << "Dropping partial audio frame at time " << audioTimeStamp
                      << " to synchronize with the start of the video stream" << std::endl;
  assert(audioTimeStamp <= 0);
  mtime_t sequenceOfSamplesToIgnore = -audioTimeStamp;
  auto numSamplesToDrop = (int)(sequenceOfSamplesToIgnore * sampleRate / 1000000);
  return numSamplesToDrop;
}

void LibavReader::decodeAudioPacket(AVPacket* pkt, bool flush) {
  bool got_sound = true;
  Util::AvErrorCode ret;

  ret = avDecodePacket(audioDecoderCtx, pkt, audioFrame, &got_sound, flush);
  if (pkt && ret != Util::AvErrorCode::Ok && ret != Util::AvErrorCode::EndOfFile) {
    Logger::warning(AVtag) << "Cannot decode audio data for " << formatCtx->filename << ". Skipping packet."
                           << std::endl;
    return;
  }
  // update packet for next iteration
  if (pkt) {
    pkt->data += pkt->size;
    pkt->size = 0;
  }

  while (got_sound) {
    // AVRational AVStream::time_base is the fundamental unit of time (in seconds) in terms of which frame timestamps
    // are represented. VideoStitch uses microseconds.
    if (audioTimeStamp == -1 && pkt) {
      // initialize our timestamp at the one of the first sample in the container
      const auto audioTimeFromVideoStart = pkt->pts - firstVideoFramePts;
      const auto audioTimeBase = formatCtx->streams[audioIdx]->time_base;
      audioTimeStamp = av_rescale(audioTimeFromVideoStart, (1000000 * audioTimeBase.num), audioTimeBase.den);
    }

    // by default we read all audio data from the frame into the samples deque
    auto samplesToIgnoreFromFrame = 0;

    if (audioTimeStamp < 0) {
      Logger::info(AVtag) << "Encountering audio data before the video stream has started" << std::endl;
      auto timestampAfterCurrentFrame =
          audioTimeStamp +
          (audioFrame->nb_samples * 1000000) / Audio::getIntFromSamplingRate(AudioReader::getSpec().sampleRate);
      if (timestampAfterCurrentFrame < 0) {
        // drop this one completely
        Logger::info(AVtag) << "Dropping audio packet at time " << audioTimeStamp
                            << " as video stream has not yet started" << std::endl;
        audioTimeStamp = timestampAfterCurrentFrame;
        return;
      } else {
        // drop samples until video begins
        samplesToIgnoreFromFrame = numSamplesToDropToSyncToStartOfVideo(
            audioTimeStamp, Audio::getIntFromSamplingRate(AudioReader::getSpec().sampleRate));
        audioTimeStamp = 0;
      }
    }

    assert(samplesToIgnoreFromFrame >= 0);
    assert(samplesToIgnoreFromFrame < audioFrame->nb_samples);

    if (av_sample_fmt_is_planar(audioDecoderCtx->sample_fmt)) {
      auto channelBufferSize =
          av_samples_get_buffer_size(nullptr, 1, audioFrame->nb_samples, audioDecoderCtx->sample_fmt, 1);
      auto channelOffset = 0;
      if (samplesToIgnoreFromFrame > 0) {
        channelOffset =
            av_samples_get_buffer_size(nullptr, 1, samplesToIgnoreFromFrame, audioDecoderCtx->sample_fmt, 1);
      }
      assert(channelOffset >= 0);
      assert(channelOffset < channelBufferSize);

      for (int channel = 0; channel < audioDecoderCtx->channels; ++channel) {
        std::deque<uint8_t>& channelSamples = audioBuffer[channel];
        channelSamples.insert(channelSamples.end(), audioFrame->data[channel] + channelOffset,
                              audioFrame->data[channel] + channelBufferSize);
      }
    } else {
      auto audioBufferSize = av_samples_get_buffer_size(nullptr, audioDecoderCtx->channels, audioFrame->nb_samples,
                                                        audioDecoderCtx->sample_fmt, 1);
      auto audioBufferOffset = 0;
      if (samplesToIgnoreFromFrame > 0) {
        audioBufferOffset = av_samples_get_buffer_size(nullptr, audioDecoderCtx->channels, samplesToIgnoreFromFrame,
                                                       audioDecoderCtx->sample_fmt, 1);
      }
      assert(audioBufferOffset >= 0);
      assert(audioBufferOffset < audioBufferSize);

      std::deque<uint8_t>& interleavedSamples = audioBuffer[0];
      interleavedSamples.insert(interleavedSamples.end(), audioFrame->data[0] + audioBufferOffset,
                                audioFrame->data[0] + audioBufferSize);
    }

    nbSamplesInAudioBuffer += audioFrame->nb_samples - samplesToIgnoreFromFrame;
    avDecodePacket(audioDecoderCtx, nullptr, audioFrame, &got_sound);
  }

  return;
}

size_t LibavReader::available() { return nbSamplesInAudioBuffer; }

bool LibavReader::eos() {
  bool ret = formatCtx->pb && formatCtx->pb->eof_reached != 0;
  // VSA-6777: eof_reached immediately true upon opening file with MPEG4 files
  return ret && formatCtx->streams[videoIdx]->codecpar->codec_id != AV_CODEC_ID_MPEG4;
}

#ifndef __clang_analyzer__  // VSA-6998

ReadStatus LibavReader::readSamples(size_t nbSamples, Audio::Samples& audioSamples) {
  // fetch a maximum nbSamples from the buffer, or as much as possible
  Audio::Samples::data_buffer_t raw;
  size_t actuallyRead = std::min(nbSamples, nbSamplesInAudioBuffer);
  if (actuallyRead == 0) {
    return ReadStatus::OK();
  }

  if (av_sample_fmt_is_planar(audioDecoderCtx->sample_fmt)) {
    int channelBufferSize = av_samples_get_buffer_size(nullptr, 1, (int)actuallyRead, audioDecoderCtx->sample_fmt, 1);
    for (int channel = 0; channel < audioDecoderCtx->channels; ++channel) {
      std::deque<uint8_t>& channelBuffer = audioBuffer[channel];
      int vsChannelIndex =
          Audio::getChannelIndexFromChannelMap(Util::getChannelMap(channel, AudioReader::getSpec().layout));
      raw[vsChannelIndex] = new uint8_t[channelBufferSize];
      for (int i = 0; i < channelBufferSize; ++i) {
        raw[vsChannelIndex][i] = channelBuffer.front();
        channelBuffer.pop_front();
      }
    }
  } else {
    int channelBufferSize = av_samples_get_buffer_size(nullptr, audioDecoderCtx->channels, (int)actuallyRead,
                                                       audioDecoderCtx->sample_fmt, 1);
    std::deque<uint8_t>& channelBuffer = audioBuffer[0];
    raw[0] = new uint8_t[channelBufferSize];
    for (int i = 0; i < channelBufferSize; ++i) {
      raw[0][i] = channelBuffer.front();
      channelBuffer.pop_front();
    }
  }

  audioSamples = Audio::Samples(AudioReader::getSpec().sampleRate, AudioReader::getSpec().sampleDepth,
                                AudioReader::getSpec().layout, audioTimeStamp, raw, actuallyRead);

  // set the buffer's timestamp and size accordingly
  audioTimeStamp += (actuallyRead * 1000000) / Audio::getIntFromSamplingRate(AudioReader::getSpec().sampleRate);
  nbSamplesInAudioBuffer -= actuallyRead;

  return ReadStatus::OK();
}

#endif  // __clang_analyzer__

// -------------------------- Seeking ---------------------------------

Status LibavReader::seekFrame(mtime_t) {
  // this is Input::AudioReader::seekFrame
  // seeking with ffmpeg is based on the video stream, an audio seek is a no-op

  // should you decide to provide an implementation here, consider that
  // VideoReader::seekFrame and AudioReader::seekFrame can be called concurrently
  return Status::OK();
}

Status LibavReader::seekFrame(frameid_t targetFrame) {
  const auto videoStream = formatCtx->streams[videoIdx];
  const auto frameRate = av_stream_get_r_frame_rate(videoStream);     // e.g. 24000 / 1001
  const auto videoTimeBase = videoStream->time_base;                  // e.g. 1 / 24000
  const auto ptsMult = av_inv_q(av_mul_q(videoTimeBase, frameRate));  // --> 1001 / 1

  if (!frameRate.num || !frameRate.den) {
    return {Origin::Input, ErrType::InvalidConfiguration, "Video framerate unknown. Aborting seek."};
  }

  if (!videoTimeBase.num || !videoTimeBase.den) {
    return {Origin::Input, ErrType::InvalidConfiguration, "Video codec time base unknown. Aborting seek."};
  }

  // we read and discard at least one frame after seeking to see where we are
  auto seekTargetFrame = targetFrame - 1;
  if (targetFrame <= 0) {
    // seeking to the first frame, see the seekToFirstFrame in seekAndDecode
    seekTargetFrame = 0;
  }

  // If no frame has been read yet we don't know the actual start time of the video container.
  // start_time is not to be trusted, the actual pts of the first video frame might be different.
  // If we were trying to seek with a bogus start_time, we might overshoot, and not be able
  // to seek to a frame near the end of the stream.
  if (currentVideoPts == NO_FRAME_READ_YET) {
    mtime_t curDate;
    const ReadStatus readStatus = readFrame(curDate, nullptr);
    if (!readStatus.ok()) {
      return {Origin::Input, ErrType::RuntimeError, "Cannot read first frame of video stream. Aborting seek."};
    }
  }

  auto targetVideoPts = av_rescale(seekTargetFrame, ptsMult.num, ptsMult.den) + firstVideoFramePts;

  auto frameDuration = av_rescale(1, ptsMult.num, ptsMult.den);

  if (targetVideoPts > firstVideoFramePts + formatCtx->streams[videoIdx]->duration) {
    std::stringstream msg;
    msg << "Trying to seek out of range (frame " << targetFrame << ").";
    return {Origin::Input, ErrType::InvalidConfiguration, msg.str()};
  }

  enum class SeekStatus : unsigned int { AvSeekError, Overshot, Success };
  auto seekAndDecode = [&](int64_t targetPts, bool seekToFirstFrame) {
    // seek to a keyframe ahead of targetPts, never seek beyond targetPts
    // although named BACKWARD, it has nothing to do with the seeking direction
    // (look at the ffmpeg source)
    int seekflags = AVSEEK_FLAG_BACKWARD;

    // convert pts to dts.
    // we look at the initial offset between dts and pts on the first keyframe of the container.
    // have a look at VSA-4138 for details.
    int64_t targetDts = targetPts - (videoStream->start_time - videoStream->first_dts);

    int r = av_seek_frame(formatCtx, videoIdx, targetDts, seekflags);
    if (r < 0) {
      Logger::error(AVtag) << "Seeking in the video stream failed." << r << " : " << Util::errorString(r) << std::endl;
      return SeekStatus::AvSeekError;
    }

    avcodec_flush_buffers(videoDecoderCtx);

    if (audioIdx != INVALID_STREAM_ID) {
      avcodec_flush_buffers(audioDecoderCtx);
    }

    expectingIncreasingVideoPts = false;

    if (seekToFirstFrame) {
      // we don't know for sure whether seeking was successful
      // we can't read to check, so let's hope for the best
      currentVideoPts = targetPts;
      return SeekStatus::Success;
    }

    // check where we are
    mtime_t curDate;
    const ReadStatus readStatus = readFrame(curDate, nullptr);
    if (!readStatus.ok()) {
      return SeekStatus::AvSeekError;
    }

    if (currentVideoPts > targetVideoPts + frameDuration / 2) {
      // We should NEVER end up behind the seeked-frame. We explicitely disallowed this in the av_seek_frame_call
      return SeekStatus::Overshot;
    }

    // decode til we reach the frame before our target
    while (currentVideoPts < targetVideoPts - frameDuration / 2) {
      const ReadStatus readStatus = readFrame(curDate, nullptr);
      if (!readStatus.ok()) {
        return SeekStatus::AvSeekError;
      }
    }

    return SeekStatus::Success;
  };

  bool shouldSeekToFirstFrame = (targetFrame <= 0);
  SeekStatus stat = seekAndDecode(targetVideoPts, shouldSeekToFirstFrame);

  switch (stat) {
    case SeekStatus::Overshot:
      Logger::error(AVtag) << "Could not find correct position in the video stream." << std::endl;
    case SeekStatus::AvSeekError:
      // TODOLATERSTATUS propagate readFrame error
      return {Origin::Input, ErrType::RuntimeError, "Could not seek to target position in video stream"};
    case SeekStatus::Success:
      return Status::OK();
    default:
      return Status::OK();
  }
}

// ------------------------- Probing ----------------------------------

ProbeResult LibavReader::probe(const std::string& filename) {
  // register all codecs, demux and protocols
  Util::Libav::checkInitialization();
  struct AVFormatContext* formatCtx = avformat_alloc_context();
  if (!formatCtx) {
    Logger::error(AVtag) << "Can't allocate format context" << std::endl;
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }

  std::unique_ptr<Util::TimeoutHandler> interruptCallback;
  if (Util::isStream(filename)) {
    interruptCallback.reset(
        new Util::TimeoutHandler(std::chrono::milliseconds(15000)));  // timeout to allow network connection to proceed)
    formatCtx->interrupt_callback.callback = &Util::TimeoutHandler::checkInterrupt;
    formatCtx->interrupt_callback.opaque = interruptCallback.get();
  }

  int r = avformat_open_input(&formatCtx, filename.c_str(), nullptr, nullptr);
  if (r < 0) {
    avformat_free_context(formatCtx);
    Logger::error(AVtag) << "Error when probing the format context: " << Util::errorString(r) << std::endl;
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }

  // if you don't call you may miss the first frames (and the size ...)
  {
    Util::Libav::Lock sl;
    r = avformat_find_stream_info(formatCtx, nullptr);
    if (r < 0) {
      avformat_close_input(&formatCtx);
      avformat_free_context(formatCtx);
      Logger::error(AVtag) << "Couldn't get additional video stream info" << r << " : " << Util::errorString(r)
                           << std::endl;
      return ProbeResult({false, false, -1, -1, -1, -1, false, false});
    }
  }

  int videoIdx = INVALID_STREAM_ID;
  int audioIdx = INVALID_STREAM_ID;
  findAvStreams(formatCtx, videoIdx, audioIdx);
  const bool hasVideo = videoIdx != INVALID_STREAM_ID;
  bool hasAudio = audioIdx != INVALID_STREAM_ID;

  // Check if the audio layout is valid.
  if (hasAudio &&
      Audio::UNKNOWN == VideoStitch::Util::channelLayout(formatCtx->streams[audioIdx]->codecpar->channel_layout) &&
      Audio::UNKNOWN == Audio::getAChannelLayoutFromNbChannels(formatCtx->streams[audioIdx]->codecpar->channels)) {
    hasAudio = false;
    Logger::error(AVtag) << "Invalid audio layout" << std::endl;
  }

  if (!hasVideo && !hasAudio) {
    avformat_close_input(&formatCtx);
    avformat_free_context(formatCtx);
    Logger::error(AVtag) << "Couldn't find any video or audio stream" << std::endl;
    return ProbeResult({true, false, -1, -1, -1, -1, hasAudio, hasVideo});
  }

  if (!hasVideo && hasAudio) {
    avformat_close_input(&formatCtx);
    avformat_free_context(formatCtx);
    Logger::get(Logger::Info) << "Found one audio stream only" << std::endl;
    return ProbeResult({true, false, -1, -1, -1, -1, hasAudio, hasVideo});
  }

  // If it's a video, check for it's last frame
  if (VideoStitch::Input::FFmpegReader::handles(filename)) {
    if (formatCtx->streams[videoIdx]->nb_frames <= 0) {
      // TODO: calculate last frame using duration and time_base
      avformat_close_input(&formatCtx);
      avformat_free_context(formatCtx);
      Logger::error(AVtag) << "Cannot find last frame for: '" << filename << "'. Aborting" << std::endl;
      return ProbeResult({false, false, -1, -1, -1, -1, false, false});
    }
  }

  const int64_t width = formatCtx->streams[videoIdx]->codecpar->width;
  const int64_t height = formatCtx->streams[videoIdx]->codecpar->height;

  /*Clean demuxer and close socket/file before leaving*/
  avformat_close_input(&formatCtx);
  avformat_free_context(formatCtx);

  if (interruptCallback) {
    interruptCallback->reset(std::chrono::milliseconds(2000));  // timeout for blocking operations
  }

  return ProbeResult({true, false, 0, -1, width, height, hasAudio, hasVideo});
}

}  // namespace Input
}  // namespace VideoStitch
