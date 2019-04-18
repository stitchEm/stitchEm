// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videoDecoder.hpp"

#include "libvideostitch/logging.hpp"

#include "NvVideoDecoder.h"
#include "NvVideoConverter.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <queue>
#include <deque>
#include <utility>

static const std::string NVDECtag("NVDEC Decoder");

#define CHUNK_SIZE 4000000

#define TEST_ERROR(cond, str)                    \
  if (cond) {                                    \
    Logger::error(NVDECtag) << str << std::endl; \
    return false;                                \
  }

namespace VideoStitch {
namespace Input {

static inline void resetCudaError() { cudaGetLastError(); }

class NvV4l2Decoder : public VideoDecoder {
 public:
  NvV4l2Decoder()
      : stopping(false),
        index(0),
        init_capture(false),
        width(0),
        coded_width(0),
        height(0),
        coded_height(0),
        inputTimestamp(0),
        latency(0) {
    conv_output_plane_buf_queue = new std::queue<NvBuffer *>;
    conv_capture_plane_buf_queue = new std::deque<std::pair<struct v4l2_buffer, NvBuffer *>>;

    cudaError_t r = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (r != cudaSuccess) {
      resetCudaError();
      Logger::error(NVDECtag) << "cudaStreamCreateWithFlags has returned CUDA error " << r << std::endl;
    }

    dec = NvVideoDecoder::createVideoDecoder("NvV4l2Decoder", 0);
    // Create converter to convert from BL to PL
    conv = NvVideoConverter::createVideoConverter("conv0");
  }

  ~NvV4l2Decoder() {
    stopping = true;
    // The decoder destructor does all the cleanup i.e set streamoff on output and capture planes,
    // unmap buffers, tell decoder to deallocate buffer (reqbufs ioctl with counnt = 0),
    // and finally call v4l2_close on the fd.
    delete dec;
    delete conv;

    cudaError_t r = cudaStreamDestroy(stream);
    if (r != cudaSuccess) {
      resetCudaError();
      Logger::warning(NVDECtag) << "cudaStreamDestroy failed with code " << r << std::endl;
    }

    delete conv_output_plane_buf_queue;
    delete conv_capture_plane_buf_queue;
  }

  bool init(int width, int height, FrameRate framerate) {
    int ret = 0;

    this->width = width;
    this->height = height;
    this->framerate = framerate;

    TEST_ERROR(!dec, "Could not create decoder");
    TEST_ERROR(!conv, "Could not create converter");

    // Subscribe to Resolution change event
    ret = dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
    TEST_ERROR(ret < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE");

    // Set format on the output plane
    ret = dec->setOutputPlaneFormat(V4L2_PIX_FMT_H264, CHUNK_SIZE);
    TEST_ERROR(ret < 0, "Could not set output plane format");

    ret = dec->enableMetadataReporting();
    TEST_ERROR(ret < 0, "Error while enabling metadata reporting");

    // Query, Export and Map the output plane buffers so that we can read
    // encoded data into the buffers
    ret = dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    TEST_ERROR(ret < 0, "Error while setting up output plane");

    ret = dec->output_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in output plane stream on");

    Logger::info(NVDECtag) << "init successful" << std::endl;

    conv->output_plane.setDQThreadCallback(conv0_output_dqbuf_thread_callback);
    conv->capture_plane.setDQThreadCallback(conv0_capture_dqbuf_thread_callback);

    return true;
  }

  void decodeHeader(Span<const unsigned char> pkt, mtime_t timestamp, Span<unsigned char> &header) override {
    VideoStitch::IO::Packet avpkt;
    VideoDecoder::demuxHeader(pkt, timestamp, avpkt, bitS);
    header = avpkt.data;
    decodeAsync(avpkt);
  }

  bool demux(Span<const unsigned char> pkt, mtime_t timestamp, VideoStitch::IO::Packet &avpkt) {
    return VideoDecoder::demuxPacket(pkt, timestamp, avpkt, bitS);
  }

  bool decodeAsync(VideoStitch::IO::Packet &avpkt) override {
    int ret = 0;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    NvBuffer *buffer;

    if (dec->isInError()) {
      Logger::error(NVDECtag) << "Internal Decoder error" << std::endl;
      return false;
    }

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, sizeof(planes));

    v4l2_buf.m.planes = planes;
    if (index < (int)dec->output_plane.getNumBuffers()) {
      buffer = dec->output_plane.getNthBuffer(index);
      v4l2_buf.index = index;
      index++;
    } else {
      ret = dec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
      if (ret < 0) {
        Logger::error(NVDECtag) << "Error DQing buffer at output plane" << std::endl;
        return false;
      }
    }

    inputTimestamp = avpkt.pts;

    v4l2_buf.timestamp.tv_sec = avpkt.pts / 1000000;
    v4l2_buf.timestamp.tv_usec = avpkt.pts - (v4l2_buf.timestamp.tv_sec * (mtime_t)1000000);

    if (avpkt.data.size() > CHUNK_SIZE) {
      Logger::error(NVDECtag) << "packet size exceed the bitstream buffer size" << std::endl;
      return false;
    }
    memcpy(buffer->planes[0].data, avpkt.data.begin(), avpkt.data.size());
    buffer->planes[0].bytesused = avpkt.data.size();

    v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
    // It is necessary to queue an empty buffer to signal EOS to the decoder
    // i.e. set v4l2_buf.m.planes[0].bytesused = 0 and queue the buffer
    ret = dec->output_plane.qBuffer(v4l2_buf, NULL);
    if (ret < 0) {
      Logger::error(NVDECtag) << "Error Qing buffer at output plane" << std::endl;
      return false;
    }
    if (v4l2_buf.m.planes[0].bytesused == 0) {
      Logger::warning(NVDECtag) << "Input file read complete" << std::endl;
    }

    return true;
  }

  size_t flush() {
    // all the buffers from output plane should be dequeued.
    // and after that capture plane loop should be signalled to stop.
    while (dec->output_plane.getNumQueuedBuffers() > 0 && !dec->isInError()) {
      struct v4l2_buffer v4l2_buf;
      struct v4l2_plane planes[MAX_PLANES];

      memset(&v4l2_buf, 0, sizeof(v4l2_buf));
      memset(planes, 0, sizeof(planes));

      v4l2_buf.m.planes = planes;
      int ret = dec->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
      if (ret < 0) {
        Logger::error(NVDECtag) << "Error DQing buffer at output plane" << std::endl;
        break;
      }
    }
    if (conv) {
      conv->capture_plane.waitForDQThread(-1);
    }
    return 0;  // XXX TODO FIXME ???
  }

  bool synchronize(mtime_t &date, VideoPtr &videoFrame) override {
    struct v4l2_format format;
    struct v4l2_crop crop;
    struct v4l2_event ev;

    if (!init_capture) {
      int32_t min_dec_capture_buffers;
      int ret = 0;
      // Need to wait for the first Resolution change event, so that
      // the decoder knows the stream resolution and can allocate appropriate
      // buffers when we call REQBUFS
      do {
        ret = dec->dqEvent(ev, 1000);
        if (ret < 0) {
          if (errno == EAGAIN) {
            Logger::info(NVDECtag) << "Timed out waiting for first V4L2_EVENT_RESOLUTION_CHANGE" << std::endl;
          } else {
            Logger::info(NVDECtag) << "Error in dequeueing decoder event" << std::endl;
          }
          break;
        }
      } while (ev.type != V4L2_EVENT_RESOLUTION_CHANGE);

      // Get capture plane format from the decoder. This may change after
      // an resolution change event
      ret = dec->capture_plane.getFormat(format);
      TEST_ERROR(ret < 0, "Could not get format from decoder capture plane");

      // Get the display resolution from the decoder
      ret = dec->capture_plane.getCrop(crop);
      TEST_ERROR(ret < 0, "Could not get crop from decoder capture plane");

      Logger::info(NVDECtag) << "Video Resolution: " << crop.c.width << "x" << crop.c.height << std::endl;

      // Not necessary to call VIDIOC_S_FMT on decoder capture plane.
      // But decoder setCapturePlaneFormat function updates the class variables
      ret =
          dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat, format.fmt.pix_mp.width, format.fmt.pix_mp.height);
      TEST_ERROR(ret < 0, "Error in setting decoder capture plane format");

      // Get the minimum buffers which have to be requested on the capture plane
      ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
      TEST_ERROR(ret < 0, "Error while getting value of minimum capture plane buffers");

      // Request (min + 5) buffers, export and map buffers
      ret = dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP, min_dec_capture_buffers + 5, false, false);
      TEST_ERROR(ret < 0, "Error in decoder capture plane setup");

      if (conv) {
        ret = conv->setOutputPlaneFormat(format.fmt.pix_mp.pixelformat, format.fmt.pix_mp.width,
                                         format.fmt.pix_mp.height, V4L2_NV_BUFFER_LAYOUT_BLOCKLINEAR);
        TEST_ERROR(ret < 0, "Error in converter output plane set format");

        ret = conv->setCapturePlaneFormat(V4L2_PIX_FMT_NV12M, crop.c.width, crop.c.height, V4L2_NV_BUFFER_LAYOUT_PITCH);
        TEST_ERROR(ret < 0, "Error in converter capture plane set format");

        ret = conv->setCropRect(0, 0, crop.c.width, crop.c.height);
        TEST_ERROR(ret < 0, "Error while setting crop rect");

        ret = conv->output_plane.setupPlane(V4L2_MEMORY_DMABUF, dec->capture_plane.getNumBuffers(), false, false);
        TEST_ERROR(ret < 0, "Error in converter output plane setup");

        ret = conv->capture_plane.setupPlane(V4L2_MEMORY_MMAP, dec->capture_plane.getNumBuffers(), true, false);
        TEST_ERROR(ret < 0, "Error in converter capture plane setup");

        ret = conv->output_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in converter output plane streamon");

        ret = conv->capture_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in converter output plane streamoff");

        // Add all empty conv output plane buffers to conv_output_plane_buf_queue
        for (uint32_t i = 0; i < conv->output_plane.getNumBuffers(); i++) {
          conv_output_plane_buf_queue->push(conv->output_plane.getNthBuffer(i));
        }

        for (uint32_t i = 0; i < conv->capture_plane.getNumBuffers(); i++) {
          struct v4l2_buffer v4l2_buf;
          struct v4l2_plane planes[MAX_PLANES];

          memset(&v4l2_buf, 0, sizeof(v4l2_buf));
          memset(planes, 0, sizeof(planes));

          v4l2_buf.index = i;
          v4l2_buf.m.planes = planes;
          ret = conv->capture_plane.qBuffer(v4l2_buf, NULL);
          TEST_ERROR(ret < 0, "Error Qing buffer at converter capture plane");
        }

        conv->output_plane.startDQThread(this);
        conv->capture_plane.startDQThread(this);
      }

      // Capture plane STREAMON
      ret = dec->capture_plane.setStreamStatus(true);
      TEST_ERROR(ret < 0, "Error in decoder capture plane streamon");

      // Enqueue all the empty capture plane buffers
      for (uint32_t i = 0; i < dec->capture_plane.getNumBuffers(); i++) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        ret = dec->capture_plane.qBuffer(v4l2_buf, NULL);
        TEST_ERROR(ret < 0, "Error Qing buffer at output plane");
      }
      init_capture = true;
      Logger::verbose(NVDECtag) << "Query and set capture successful" << std::endl;
    }

    NvBuffer *dec_buffer;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, sizeof(planes));
    v4l2_buf.m.planes = planes;

    // Dequeue a filled buffer
    if (dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, -1)) {
      TEST_ERROR(errno != EAGAIN, "Error in decoder capture plane streamon");
    }

    // give the buffer to video converter output plane
    // instead of returning the buffer back to decoder capture plane
    NvBuffer *conv_buffer;
    struct v4l2_buffer conv_output_buffer;
    struct v4l2_plane conv_planes[MAX_PLANES];

    memset(&conv_output_buffer, 0, sizeof(conv_output_buffer));
    memset(conv_planes, 0, sizeof(conv_planes));
    conv_output_buffer.m.planes = conv_planes;

    // Get an empty conv output plane buffer from conv_output_plane_buf_queue
    std::unique_lock<std::mutex> lock(queue_lock);
    queue_cond.wait(lock, [this] { return !conv_output_plane_buf_queue->empty() || stopping; });
    if (stopping) {
      return false;
    }
    conv_buffer = conv_output_plane_buf_queue->front();
    conv_output_plane_buf_queue->pop();

    conv_output_buffer.index = conv_buffer->index;
    conv_output_buffer.timestamp.tv_usec = v4l2_buf.timestamp.tv_usec;
    conv_output_buffer.timestamp.tv_sec = v4l2_buf.timestamp.tv_sec;

    if (conv->output_plane.qBuffer(conv_output_buffer, dec_buffer) < 0) {
      Logger::error(NVDECtag) << "Error while queueing buffer at converter output plane" << std::endl;
    }

    return true;
  }

  void stop() {
    stopping = true;
    dec->abort();
    if (conv) {
      conv->abort();
      queue_cond.notify_all();
    }
    return;
  }

  void copyFrame(unsigned char *videoFrame, mtime_t &date, VideoPtr) override {
    std::unique_lock<std::mutex> lock(outqueue_lock);
    if (conv_capture_plane_buf_queue->empty()) {
      return;
    }
    auto conv_buffer = conv_capture_plane_buf_queue->front();
    conv_capture_plane_buf_queue->pop_front();
    date = conv_buffer.first.timestamp.tv_sec * (mtime_t)1000000 + conv_buffer.first.timestamp.tv_usec;
    outqueue_lock.unlock();

    // copy planes
    uint8_t *out_ptr = videoFrame;
    cudaError_t r;
    for (auto j = 0; j < (int)conv_buffer.second->n_planes; j++) {
      r = cudaMemcpy2DAsync(out_ptr,
                            conv_buffer.second->planes[j].fmt.width * conv_buffer.second->planes[j].fmt.bytesperpixel,
                            conv_buffer.second->planes[j].data, conv_buffer.second->planes[j].fmt.stride,
                            conv_buffer.second->planes[j].fmt.width * conv_buffer.second->planes[j].fmt.bytesperpixel,
                            conv_buffer.second->planes[j].fmt.height, cudaMemcpyHostToDevice, stream);
      out_ptr += conv_buffer.second->planes[j].fmt.width * conv_buffer.second->planes[j].fmt.bytesperpixel *
                 conv_buffer.second->planes[j].fmt.height;
      if (r != cudaSuccess) {
        resetCudaError();
        Logger::error(NVDECtag) << "cudaMemcpy2DAsync has returned CUDA error " << r << std::endl;
        return;
      }
    }
    r = cudaStreamSynchronize(stream);
    if (r != cudaSuccess) {
      resetCudaError();
      Logger::error(NVDECtag) << "cudaStreamSynchronize has returned CUDA error " << r << std::endl;
      return;
    }

    if (conv->capture_plane.qBuffer(conv_buffer.first, NULL) < 0) {
      Logger::error(NVDECtag) << "Error while queueing buffer at converter capture plane" << std::endl;
      return;
    }
  }

  void releaseFrame(VideoPtr videoSurface) override {}

 private:
  // -------------------- Converter callbacks -----------------------
  static bool conv0_output_dqbuf_thread_callback(struct v4l2_buffer *v4l2_buf, NvBuffer *buffer,
                                                 NvBuffer *shared_buffer, void *arg) {
    NvV4l2Decoder *ctx = reinterpret_cast<NvV4l2Decoder *>(arg);
    struct v4l2_buffer dec_capture_ret_buffer;
    struct v4l2_plane planes[MAX_PLANES];

    if (!v4l2_buf) {
      Logger::error(NVDECtag) << "Error while dequeueing conv output plane buffer" << std::endl;
      return false;
    }

    if (v4l2_buf->m.planes[0].bytesused == 0) {
      return false;
    }

    memset(&dec_capture_ret_buffer, 0, sizeof(dec_capture_ret_buffer));
    memset(planes, 0, sizeof(planes));

    dec_capture_ret_buffer.index = shared_buffer->index;
    dec_capture_ret_buffer.m.planes = planes;

    std::lock_guard<std::mutex> lk(ctx->queue_lock);
    ctx->conv_output_plane_buf_queue->push(buffer);

    // Return the buffer dequeued from converter output plane
    // back to decoder capture plane
    if (ctx->dec->capture_plane.qBuffer(dec_capture_ret_buffer, NULL) < 0) {
      return false;
    }

    ctx->queue_cond.notify_all();

    return true;
  }

  static bool conv0_capture_dqbuf_thread_callback(struct v4l2_buffer *v4l2_buf, NvBuffer *buffer,
                                                  NvBuffer *shared_buffer, void *arg) {
    NvV4l2Decoder *ctx = reinterpret_cast<NvV4l2Decoder *>(arg);

    if (!v4l2_buf) {
      Logger::error(NVDECtag) << "Error while dequeueing conv capture plane buffer" << std::endl;
      return false;
    }

    if (v4l2_buf->m.planes[0].bytesused == 0) {
      return false;
    }

    // Check latency
    mtime_t currentLatency =
        ctx->inputTimestamp - v4l2_buf->timestamp.tv_usec - v4l2_buf->timestamp.tv_sec * mtime_t(1000000);
    if (ctx->latency < currentLatency) {
      ctx->latency = currentLatency;
      Logger::verbose(NVDECtag) << "Video latency increased to " << ctx->latency / 1000 << " ms" << std::endl;
    }

    std::lock_guard<std::mutex> lk(ctx->outqueue_lock);
    ctx->conv_capture_plane_buf_queue->push_back(std::make_pair(*v4l2_buf, buffer));

    return true;
  }

  // --------------------------------

  NvVideoDecoder *dec;
  NvVideoConverter *conv;

  std::atomic<bool> stopping;
  std::queue<NvBuffer *> *conv_output_plane_buf_queue;
  std::mutex queue_lock;
  std::condition_variable queue_cond;
  std::deque<std::pair<struct v4l2_buffer, NvBuffer *>> *conv_capture_plane_buf_queue;
  std::mutex outqueue_lock;

  int index;
  bool init_capture;

  unsigned int width, coded_width;
  unsigned int height, coded_height;
  FrameRate framerate;

  mtime_t inputTimestamp;
  mtime_t latency;
  std::vector<unsigned char> bitS;

  cudaStream_t stream;
};

VideoDecoder *createNvV4l2Decoder(int width, int height, FrameRate framerate) {
  NvV4l2Decoder *decoder = new NvV4l2Decoder();
  if (decoder->init(width, height, framerate)) {
    return decoder;
  } else {
    Logger::error(NVDECtag) << "could not instantiate the NVIDIA decoder" << std::endl;
    return nullptr;
  }
}
}  // namespace Input
}  // namespace VideoStitch
