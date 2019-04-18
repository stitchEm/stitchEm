// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videoDecoder.hpp"

#include "frameQueue.hpp"

#include "libvideostitch/logging.hpp"

#include <nvcuvid.h>
#include <cuda_runtime.h>

static const std::string CUVIDtag("NVDEC Decoder");

#define MAX_FRAME_COUNT 20

namespace VideoStitch {
namespace Input {

namespace {
static bool check_cu(CUresult err, const char* func) {
  const char* err_name;
  const char* err_string;

  Logger::get(Logger::Debug) << "Calling " << func << std::endl;

  if (err == CUDA_SUCCESS) return true;

  cuGetErrorName(err, &err_name);
  cuGetErrorString(err, &err_string);

  Logger::get(Logger::Error) << func << " failed" << std::endl;
  if (err_name && err_string) {
    Logger::get(Logger::Error) << " -> " << err_name << ": " << err_string << std::endl;
  }
  Logger::get(Logger::Error) << std::endl;

  return false;
}
}  // namespace

#define CHECK_CU(x) check_cu((x), #x)

class CuvidDecoder : public VideoDecoder {
 public:
  CuvidDecoder()
      : cudecoder(nullptr), width(0), coded_width(0), height(0), coded_height(0), inputTimestamp(0), latency(0) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }

  ~CuvidDecoder() {
    if (cuparser) {
      cuvidDestroyVideoParser(cuparser);
    }
    if (cudecoder) {
      cuvidDestroyDecoder(cudecoder);
    }
    cudaStreamDestroy(stream);
    //    cuCtxDestroy(cuctx);
  }

  bool init(int width, int height, FrameRate framerate) {
    this->width = width;
    this->height = height;
    this->framerate = framerate;

    // context initialization

    int ret = CHECK_CU(cuInit(0));
    if (ret < 0) goto error;

    CUdevice device;
    ret = CHECK_CU(cuDeviceGet(&device, 0));
    if (ret < 0) goto error;

    cudaFree(0);
    ret = CHECK_CU(cuCtxGetCurrent(&cuctx));
    if (ret < 0) goto error;
    if (cuctx == NULL) {
      Logger::get(Logger::Debug) << "Calling cuCtxCreate" << std::endl;
      ret = CHECK_CU(cuCtxCreate(&cuctx, CU_CTX_SCHED_BLOCKING_SYNC, device));
      if (ret < 0) goto error;
    }

    ret = CHECK_CU(cuCtxPopCurrent(nullptr));
    if (ret < 0) goto error;

    CUVIDPARSERPARAMS cuparseinfo;
    memset(&cuparseinfo, 0, sizeof(cuparseinfo));

    cuparseinfo.CodecType = cudaVideoCodec_H264;  // also available : HVEC, VP8, VP9, VC1
    cuparseinfo.ulMaxNumDecodeSurfaces = MAX_FRAME_COUNT;
    cuparseinfo.ulMaxDisplayDelay = 1;
    cuparseinfo.pUserData = this;
    cuparseinfo.pfnSequenceCallback = handleVideoSequence;
    cuparseinfo.pfnDecodePicture = handlePictureDecode;
    cuparseinfo.pfnDisplayPicture = handlePictureDisplay;
    cuparseinfo.ulErrorThreshold = 0;

    ret = CHECK_CU(cuCtxPushCurrent(cuctx));
    if (ret < 0) goto error;

    ret = testDummyDecoder(&cuparseinfo);
    if (ret < 0) goto error;

    ret = CHECK_CU(cuvidCreateVideoParser(&cuparser, &cuparseinfo));
    if (ret < 0) goto error;

    ret = CHECK_CU(cuCtxPopCurrent(nullptr));
    if (ret < 0) goto error;

    return 0;

  error:
    return ret == 0;
  }

  int testDummyDecoder(CUVIDPARSERPARAMS* cuparseinfo) {
    CUVIDDECODECREATEINFO cuinfo;
    memset(&cuinfo, 0, sizeof(cuinfo));

    cuinfo.CodecType = cuparseinfo->CodecType;
    cuinfo.ChromaFormat = cudaVideoChromaFormat_420;
    cuinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;

    cuinfo.ulWidth = 1280;
    cuinfo.ulHeight = 720;
    cuinfo.ulTargetWidth = cuinfo.ulWidth;
    cuinfo.ulTargetHeight = cuinfo.ulHeight;

    cuinfo.target_rect.left = 0;
    cuinfo.target_rect.top = 0;
    cuinfo.target_rect.right = (short)cuinfo.ulWidth;
    cuinfo.target_rect.bottom = (short)cuinfo.ulHeight;

    cuinfo.ulNumDecodeSurfaces = MAX_FRAME_COUNT;
    cuinfo.ulNumOutputSurfaces = 1;
    cuinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;

    cuinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;

    CUvideodecoder cudec = nullptr;
    int ret = CHECK_CU(cuvidCreateDecoder(&cudec, &cuinfo));
    if (ret < 0) return ret;

    ret = CHECK_CU(cuvidDestroyDecoder(cudec));
    if (ret < 0) return ret;

    return 0;
  }

  void decodeHeader(Span<const unsigned char> pkt, mtime_t timestamp, Span<unsigned char>& header) override {
    VideoStitch::IO::Packet avpkt;
    VideoDecoder::demuxHeader(pkt, timestamp, avpkt, bitS);
    header = avpkt.data;
    decodeAsync(avpkt);
  }

  bool demux(Span<const unsigned char> pkt, mtime_t timestamp, VideoStitch::IO::Packet& avpkt) {
    return VideoDecoder::demuxPacket(pkt, timestamp, avpkt, bitS);
  }

  bool decodeAsync(VideoStitch::IO::Packet& avpkt) override {
    int ret = CHECK_CU(cuCtxPushCurrent(cuctx));
    if (ret < 0) {
      return ret == 0;
    }

    CUVIDSOURCEDATAPACKET cupkt;
    memset(&cupkt, 0, sizeof(cupkt));
    cupkt.payload_size = (unsigned long)avpkt.data.size();
    cupkt.payload = (unsigned char*)avpkt.data.begin();
    cupkt.flags = CUVID_PKT_TIMESTAMP;
    cupkt.timestamp = avpkt.dts * 10;  // CUVID timescale is 10 MHz, Packet is 1 MHz
    inputTimestamp = cupkt.timestamp;

    if (!CHECK_CU(cuvidParseVideoData(cuparser, &cupkt))) {
      cuCtxPopCurrent(nullptr);
      return false;
    }
    cuCtxPopCurrent(nullptr);
    return true;
  }

  size_t flush() {
    // XXX TODO FIXME ???
    //    int ret = CHECK_CU(cuCtxPushCurrent(cuctx));
    //    if (ret < 0) {
    //      return 0;
    //    }

    CUVIDSOURCEDATAPACKET cupkt;
    memset(&cupkt, 0, sizeof(cupkt));
    cupkt.flags = CUVID_PKT_ENDOFSTREAM;
    if (!CHECK_CU(cuvidParseVideoData(cuparser, &cupkt))) {
      cuCtxPopCurrent(nullptr);
      return false;
    }
    cuCtxPopCurrent(nullptr);
    return 0;  // XXX TODO FIXME ???
  }

  bool synchronize(mtime_t& date, VideoPtr& videoFrame) override {
    // we are synchronous
    return true;
  }

  void stop() { frameQueue.endDecode(); }

  void copyFrame(unsigned char* videoFrame, mtime_t& date, VideoPtr) override {
    CUVIDPARSERDISPINFO dispinfo;

    if (frameQueue.dequeue(&dispinfo)) {
      // CCtxAutoLock lck(g_CtxLock);
      // Push the current CUDA context (only if we are using CUDA decoding path)
      cuCtxPushCurrent(cuctx);

      CUVIDPROCPARAMS params;
      memset(&params, 0, sizeof(CUVIDPROCPARAMS));

      params.progressive_frame = dispinfo.progressive_frame;
      params.second_field = 0;
      params.top_field_first = dispinfo.top_field_first;

      // map decoded video frame to CUDA surface
      unsigned int decodedPitch = 0;
      CUdeviceptr decodedFrame = 0;
      cuvidMapVideoFrame(cudecoder, dispinfo.picture_index, &decodedFrame, &decodedPitch, &params);

      Logger::get(Logger::Debug) << (dispinfo.progressive_frame ? "Frame" : "Field") << " = "
                                 << /*frameCount <<*/ ", PicIndex = " << dispinfo.picture_index
                                 << ", OutputPTS = " << dispinfo.timestamp << std::endl;

      int offset = 0;
      for (int i = 0; i < 2; i++) {
        CUDA_MEMCPY2D cpy;
        memset(&cpy, 0, sizeof(CUDA_MEMCPY2D));
        cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.srcDevice = decodedFrame;
        cpy.dstDevice = (CUdeviceptr)videoFrame;
        cpy.srcPitch = decodedPitch;
        cpy.dstPitch = width;
        cpy.srcY = offset;
        cpy.WidthInBytes = width;
        cpy.Height = coded_height >> (i ? 1 : 0);

        CHECK_CU(cuMemcpy2DAsync(&cpy, (CUstream)stream));

        offset += coded_height;
        videoFrame += width * height;
      }

      CHECK_CU(cuStreamSynchronize((CUstream)stream));
      date = dispinfo.timestamp / 10;  // NVIDIA CuVid is clocked at 10 MHz, VideoStitch at 1 MHz

      // unmap video frame
      // unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
      cuvidUnmapVideoFrame(cudecoder, decodedFrame);
      // release the frame, so it can be re-used in decoder
      frameQueue.releaseFrame(&dispinfo);

      // Detach from the Current thread
      CHECK_CU(cuCtxPopCurrent(nullptr));
    }
  }

  void releaseFrame(VideoPtr videoSurface) override {}

 private:
  // -------------------- Parser callbacks -----------------------

  static int CUDAAPI handleVideoSequence(void* opaque, CUVIDEOFORMAT* format) {
    CuvidDecoder* that = (CuvidDecoder*)opaque;

    Logger::get(Logger::Debug) << "pfnSequenceCallback" << std::endl;

    assert(that->width == (unsigned int)format->display_area.right);
    assert(that->height == (unsigned int)format->display_area.bottom);

    /* if (format->frame_rate.numerator && format->frame_rate.denominator) {
       assert(that->framerate.num == format->frame_rate.numerator
         && that->framerate.den == format->frame_rate.denominator);
     }*/

    if (that->cudecoder && that->coded_width == format->coded_width && that->coded_height == format->coded_height &&
        that->chroma_format == format->chroma_format && that->codec_type == format->codec) {
      // repeated sequence parameters, continue
      return 1;
    }

    if (that->cudecoder) {
      Logger::get(Logger::Error) << "re-initializing decoder is not supported" << std::endl;
      return 0;
    }

    that->coded_width = format->coded_width;
    that->coded_height = format->coded_height;

    that->chroma_format = format->chroma_format;

    CUVIDDECODECREATEINFO cuinfo;
    memset(&cuinfo, 0, sizeof(cuinfo));

    cuinfo.CodecType = that->codec_type = format->codec;
    cuinfo.ChromaFormat = format->chroma_format;
    cuinfo.OutputFormat = cudaVideoSurfaceFormat_NV12;

    cuinfo.ulWidth = that->coded_width;
    cuinfo.ulHeight = that->coded_height;
    cuinfo.ulTargetWidth = cuinfo.ulWidth;
    cuinfo.ulTargetHeight = cuinfo.ulHeight;

    cuinfo.target_rect.left = 0;
    cuinfo.target_rect.top = 0;
    cuinfo.target_rect.right = (short)cuinfo.ulWidth;
    cuinfo.target_rect.bottom = (short)cuinfo.ulHeight;

    cuinfo.ulNumDecodeSurfaces = MAX_FRAME_COUNT;
    cuinfo.ulNumOutputSurfaces = 1;
    cuinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;

    cuinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;

    // All information gathered, create the decoder
    if (!CHECK_CU(cuvidCreateDecoder(&that->cudecoder, &cuinfo))) {
      return false;
    }
    return true;
  }

  // Called by the video parser to decode a single picture. Since the parser will
  // deliver data as fast as it can, we need to make sure that the picture index
  // we're attempting to use for decode is no longer used for display.
  static int CUDAAPI handlePictureDecode(void* opaque, CUVIDPICPARAMS* picparams) {
    CuvidDecoder* that = reinterpret_cast<CuvidDecoder*>(opaque);
    Logger::get(Logger::Debug) << "pfnDecodePicture idx " << picparams->CurrPicIdx << std::endl;
    bool frameAvailable = that->frameQueue.waitUntilFrameAvailable(picparams->CurrPicIdx);
    if (!frameAvailable) return false;

    // Handle CUDA picture decode (this actually calls the hardware VP/CUDA to decode video frames)
    if (!CHECK_CU(cuvidDecodePicture(that->cudecoder, picparams))) return false;
    return true;
  }

  // Called by the video parser to display a video frame (in the case of field
  // pictures, there may be two decode calls per one display call, since two
  // fields make up one frame).
  static int CUDAAPI handlePictureDisplay(void* opaque, CUVIDPARSERDISPINFO* dispinfo) {
    CuvidDecoder* that = reinterpret_cast<CuvidDecoder*>(opaque);
    Logger::get(Logger::Debug) << "pfnDisplayPicture " << dispinfo->picture_index << std::endl;

    // Check latency
    mtime_t currentLatency = that->inputTimestamp - dispinfo->timestamp;
    if (that->latency < currentLatency) {
      that->latency = currentLatency;
      Logger::verbose(CUVIDtag) << "Video latency increased to " << that->latency / 10000 << " ms" << std::endl;
    }

    that->frameQueue.enqueue(dispinfo);
    return true;
  }

  // --------------------------------

  CUcontext cuctx;
  CUvideodecoder cudecoder;
  CUvideoparser cuparser;

  FrameQueue frameQueue;

  unsigned int width, coded_width;
  unsigned int height, coded_height;
  FrameRate framerate;

  cudaVideoCodec codec_type;
  cudaVideoChromaFormat chroma_format;

  cudaStream_t stream;

  mtime_t inputTimestamp;
  mtime_t latency;
  std::vector<unsigned char> bitS;
};

VideoDecoder* createCuvidDecoder(int width, int height, FrameRate framerate) {
  CuvidDecoder* decoder = new CuvidDecoder();
  if (!decoder->init(width, height, framerate)) {
    return decoder;
  } else {
    Logger::get(Logger::Error) << "RTMP : could not instantiate the NVIDIA decoder" << std::endl;
    return nullptr;
  }
}
}  // namespace Input
}  // namespace VideoStitch
