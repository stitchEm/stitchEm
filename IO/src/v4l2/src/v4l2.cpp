// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "v4l2.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <fcntl.h>
#ifndef __ANDROID__
#include <glob.h>
#endif
#include <poll.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

namespace {
// Compares two V4L2 fractions.
static int64_t fcmp(const v4l2_fract* a, const v4l2_fract* b) {
  return (uint64_t)a->numerator * b->denominator - (uint64_t)b->numerator * a->denominator;
}

static const v4l2_fract infinity = {1, 0};
static const v4l2_fract zero = {0, 1};
}  // namespace

namespace VideoStitch {
namespace Input {

bool V4L2Reader::handles(const Ptv::Value* config) {
  return config && config->has("type") && config->has("type")->asString() == "v4l2";
}

V4L2Reader* V4L2Reader::create(const Ptv::Value* config, const Plugin::VSReaderPlugin::Config& runtime) {
  std::string dev = "unknown";
  if (Parse::populateString("v4l2", *config, "name", dev, true) != VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "v4l2: device name (\"name\") couldn't be retrieved. Aborting." << std::endl;
    return nullptr;
  }

  Logger::get(Logger::Debug) << "v4l2: opening device " << dev << std::endl;

  int fd = open(dev.c_str(), O_RDWR, 0);
  if (fd != -1) {
    fcntl(fd, F_SETFD, FD_CLOEXEC);
  } else {
    Logger::get(Logger::Error) << "v4l2: cannot open device " << dev << std::endl;
    return nullptr;
  }

  // get device capabilities
  uint32_t caps;
  v4l2_capability cap;
  if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) {
    Logger::get(Logger::Error) << "v4l2: cannot get device capabilities: " << strerror(errno) << std::endl;
    close(fd);
    return nullptr;
  }

  Logger::get(Logger::Info) << "v4l2: device " << cap.card << " using driver " << cap.driver << " (version "
                            << ((cap.version >> 16) & 0xFF) << "." << ((cap.version >> 8) & 0xFF) << "."
                            << (cap.version & 0xFF) << ") on " << cap.bus_info;
  if (cap.capabilities & V4L2_CAP_DEVICE_CAPS) {
    Logger::get(Logger::Info) << " with capabilities 0x" << cap.device_caps << " (overall 0x" << cap.capabilities << ")"
                              << std::endl;
    caps = cap.device_caps;
  } else {
    Logger::get(Logger::Info) << " with unknown capabilities (overall 0x" << cap.capabilities << ")" << std::endl;
    caps = cap.capabilities;
  }

  if (!(caps & V4L2_CAP_VIDEO_CAPTURE)) {
    Logger::get(Logger::Error) << "v4l2: not a video capture device" << std::endl;
  }

  // setup input
  v4l2_std_id std = V4L2_STD_UNKNOWN;
  v4l2_input input;
  input.index = 0;  // XXX TODO FIXME
  if (ioctl(fd, VIDIOC_ENUMINPUT, &input) < 0) {
    Logger::get(Logger::Error) << "v4l2: invalid video input " << input.index << ": " << strerror(errno) << std::endl;
    close(fd);
    return nullptr;
  }

  if (input.type != V4L2_INPUT_TYPE_CAMERA) {
    Logger::get(Logger::Error) << "v4l2: video input " << input.name << " (" << input.index
                               << ") is not a camera, tuners are not supported" << std::endl;
    close(fd);
    return nullptr;
  }

  // select input
  if (ioctl(fd, VIDIOC_S_INPUT, &input.index) < 0) {
    Logger::get(Logger::Error) << "v4l2: cannot select input " << input.index << ": " << strerror(errno) << std::endl;
    close(fd);
    return nullptr;
  }
  Logger::get(Logger::Info) << "v4l2: selected input " << input.index << std::endl;

  // setup standard
  if (!(input.capabilities & V4L2_IN_CAP_STD)) {
    Logger::get(Logger::Info) << "v4l2: no video standard selection" << std::endl;
  } else {
    // XXX TODO FIXME read std from config?
    if (std == V4L2_STD_UNKNOWN) {
      Logger::get(Logger::Warning) << "v4l2: video standard not set" << std::endl;
      // grab the currently selected standard
      if (ioctl(fd, VIDIOC_G_STD, &std) < 0) {
        Logger::get(Logger::Error) << "v4l2: cannot get video standard" << std::endl;
      }
    }
    if (ioctl(fd, VIDIOC_S_STD, &std) < 0) {
      Logger::get(Logger::Error) << "v4l2: cannot set video standard 0x" << std << ": " << strerror(errno) << std::endl;
      close(fd);
      return nullptr;
    }
    Logger::get(Logger::Info) << "v4l2: video standard set to 0x" << std << std::endl;
  }

  v4l2_format fmt = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE};

  // use the current format if none specified in the config
  uint32_t pixfmt = 0;
  PixelFormat pixfmt_vs = Unknown;
  std::string pixfmt_str = "";
  if (Parse::populateString("v4l2", *config, "pixel_format", pixfmt_str, false) != Parse::PopulateResult_Ok) {
    Logger::get(Logger::Info) << "v4l2: pixel format (\"pixel_format\") couldn't be retrieved from the configuration."
                              << std::endl;
  }
  if (pixfmt_str == "RGBA") {
    pixfmt = V4L2_PIX_FMT_RGB32;
    pixfmt_vs = RGBA;
  } else if (pixfmt_str == "RGB") {
    pixfmt = V4L2_PIX_FMT_RGB24;
    pixfmt_vs = RGB;
  } else if (pixfmt_str == "BGR") {
    pixfmt = V4L2_PIX_FMT_BGR24;
    pixfmt_vs = BGR;
  } else if (pixfmt_str == "BGRU") {
    pixfmt = V4L2_PIX_FMT_BGR32;
    pixfmt_vs = BGRU;
  } else if (pixfmt_str == "UYVY") {
    pixfmt = V4L2_PIX_FMT_UYVY;
    pixfmt_vs = UYVY;
  } else if (pixfmt_str == "YUY2") {
    pixfmt = V4L2_PIX_FMT_YUYV;
    pixfmt_vs = YUY2;
  } else if (pixfmt_str == "YV12") {
    pixfmt = V4L2_PIX_FMT_YVU420;
    pixfmt_vs = YV12;
  } else if (pixfmt_str == "NV12") {
    pixfmt = V4L2_PIX_FMT_NV12;
    pixfmt_vs = NV12;
  } else {
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
      Logger::get(Logger::Error) << "v4l2: cannot get default format: " << strerror(errno) << std::endl;
      close(fd);
      return nullptr;
    }
    switch (fmt.fmt.pix.pixelformat) {
      case V4L2_PIX_FMT_RGB32:
        pixfmt_vs = RGBA;
        break;
      case V4L2_PIX_FMT_RGB24:
        pixfmt_vs = RGB;
        break;
      case V4L2_PIX_FMT_BGR24:
        pixfmt_vs = BGR;
        break;
      case V4L2_PIX_FMT_BGR32:
        pixfmt_vs = BGRU;
        break;
      case V4L2_PIX_FMT_UYVY:
        pixfmt_vs = UYVY;
        break;
      case V4L2_PIX_FMT_YUYV:
        pixfmt_vs = YUY2;
        break;
      case V4L2_PIX_FMT_YVU420:
        pixfmt_vs = YV12;
        break;
      case V4L2_PIX_FMT_NV12:
        pixfmt_vs = NV12;
        break;
    }
    pixfmt = fmt.fmt.pix.pixelformat;
  }

  // find the best possible frame rate and resolution
  v4l2_streamparm parm;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  memset(&parm, 0, sizeof(parm));
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if (ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
    Logger::get(Logger::Error) << "v4l2: cannot get default format: " << strerror(errno) << std::endl;
    close(fd);
    return nullptr;
  }
  fmt.fmt.pix.pixelformat = pixfmt;

  struct v4l2_frmsizeenum fse;
  memset(&fse, 0, sizeof(fse));
  fse.pixel_format = pixfmt;
  struct v4l2_fract best_it = infinity, min_it = zero;
  uint64_t best_area = 0;

  if (runtime.width > 0 && runtime.height > 0) {
    fmt.fmt.pix.width = runtime.width;
    fmt.fmt.pix.height = runtime.height;
    findMaxRate(fd, fmt, &min_it, &best_it);
  } else if (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fse) < 0) {
    // fallback to current format, try to maximize frame rate
    findMaxRate(fd, fmt, &min_it, &best_it);
  } else {
    switch (fse.type) {
      case V4L2_FRMSIZE_TYPE_DISCRETE:
        do {
          struct v4l2_fract cur_it;

          Logger::get(Logger::Info) << "v4l2: frame size " << fse.discrete.width << "x" << fse.discrete.height
                                    << std::endl;
          findMaxRate(fd, fmt, &min_it, &cur_it);

          int64_t c = fcmp(&cur_it, &best_it);
          uint64_t area = fse.discrete.width * fse.discrete.height;
          if (c < 0 || (c == 0 && area > best_area)) {
            best_it = cur_it;
            best_area = area;
            fmt.fmt.pix.width = fse.discrete.width;
            fmt.fmt.pix.height = fse.discrete.height;
          }

          fse.index++;
        } while (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fse) >= 0);

        Logger::get(Logger::Info) << "v4l2: best discrete frame size: " << fmt.fmt.pix.width << "x"
                                  << fmt.fmt.pix.height << std::endl;
        break;
      case V4L2_FRMSIZE_TYPE_STEPWISE:
      case V4L2_FRMSIZE_TYPE_CONTINUOUS:
        Logger::get(Logger::Info) << "v4l2: frame sizes from " << fse.stepwise.min_width << "x"
                                  << fse.stepwise.min_height << " to " << fse.stepwise.max_width << "x"
                                  << fse.stepwise.max_height << " supported";
        if (fse.type == V4L2_FRMSIZE_TYPE_STEPWISE) {
          Logger::get(Logger::Info) << "  with " << fse.stepwise.step_width << "x" << fse.stepwise.step_height
                                    << std::endl;
        } else {
          Logger::get(Logger::Info) << std::endl;
        }

        for (uint32_t width = fse.stepwise.min_width; width <= fse.stepwise.max_width;
             width += fse.stepwise.step_width) {
          for (uint32_t height = fse.stepwise.min_height; height <= fse.stepwise.max_height;
               height += fse.stepwise.step_height) {
            struct v4l2_fract cur_it;

            findMaxRate(fd, fmt, &min_it, &cur_it);

            int64_t c = fcmp(&cur_it, &best_it);
            uint64_t area = width * height;

            if (c < 0 || (c == 0 && area > best_area)) {
              best_it = cur_it;
              best_area = area;
              fmt.fmt.pix.width = width;
              fmt.fmt.pix.height = height;
            }
          }
        }

        Logger::get(Logger::Info) << "v4l2: best frame size: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height
                                  << std::endl;
        break;
    }
  }

  // set the final format
  if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
    Logger::get(Logger::Error) << "v4l2: cannot set format: " << strerror(errno) << std::endl;
    close(fd);
    return nullptr;
  }

  // Get the real format in case the desired is not supported
  if (runtime.width > 0 && runtime.height > 0) {
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
      Logger::error("v4l2") << "Failed to get camera output format: " << strerror(errno) << std::endl;
      close(fd);
      return nullptr;
    }
    if (fmt.fmt.pix.width != runtime.width || fmt.fmt.pix.height != runtime.height ||
        fmt.fmt.pix.pixelformat != pixfmt) {
      Logger::error("v4l2") << "The desired format is not supported: try " << fmt.fmt.pix.width << "x"
                            << fmt.fmt.pix.height << std::endl;
      close(fd);
      return nullptr;
    }
  }

  // now that the final format is set, fetch and override parameters
  if (ioctl(fd, VIDIOC_G_PARM, &parm) < 0) {
    Logger::get(Logger::Error) << "cannot get streaming parameters: " << strerror(errno) << std::endl;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  }
  parm.parm.capture.capturemode = 0;  // normal video mode
  parm.parm.capture.extendedmode = 0;
  if (best_it.denominator != 0) {
    parm.parm.capture.timeperframe = best_it;
  }
  if (ioctl(fd, VIDIOC_S_PARM, &parm) < 0) {
    Logger::get(Logger::Warning) << "v4l2: cannot set streaming parameters: " << strerror(errno) << std::endl;
  }

  // crop depends on frame size
  struct v4l2_cropcap cropcap = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE};
  if (ioctl(fd, VIDIOC_CROPCAP, &cropcap) < 0) {
    Logger::get(Logger::Warning) << "v4l2: cannot get cropping properties: " << strerror(errno) << std::endl;
  } else {
    // reset to the default cropping rectangle
    struct v4l2_crop crop = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        .c = cropcap.defrect,
    };

    if (ioctl(fd, VIDIOC_S_CROP, &crop) < 0) {
      Logger::get(Logger::Warning) << "v4l2: cannot reset cropping limits: " << strerror(errno) << std::endl;
    }
  }

  Logger::get(Logger::Info) << "v4l2: " << fmt.fmt.pix.sizeimage << " bytes for complete image" << std::endl;

  // init I/O method
  uint32_t bufc = 0;
  buffer_t* bufv;
  if (caps & V4L2_CAP_STREAMING) {
    bufc = 4;
    bufv = startMmap(fd, &bufc);
    if (bufv == nullptr) {
      close(fd);
      return nullptr;
    }
  } else if (caps & V4L2_CAP_READWRITE) {
    bufv = nullptr;
  } else {
    Logger::get(Logger::Error) << "v4l2: no supported capture method" << std::endl;
    close(fd);
    return nullptr;
  }

  FrameRate fps{(int)parm.parm.capture.timeperframe.denominator, (int)parm.parm.capture.timeperframe.numerator};
  return new V4L2Reader(runtime.id, fd, bufv, bufc, fmt.fmt.pix.width, fmt.fmt.pix.height, fmt.fmt.pix.sizeimage,
                        pixfmt_vs, fps);
}

buffer_t* V4L2Reader::startMmap(int fd, uint32_t* n) {
  struct v4l2_requestbuffers req;
  memset(&req, 0, sizeof(req));
  req.count = *n;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
    Logger::get(Logger::Error) << "v4l2: cannot allocate buffers: " << strerror(errno) << std::endl;
    return nullptr;
  }

  if (req.count < 2) {
    Logger::get(Logger::Error) << "v4l2: cannot allocate enough buffers" << std::endl;
    return nullptr;
  }

  buffer_t* bufv = (buffer_t*)malloc(req.count * sizeof(*bufv));
  if (bufv == nullptr) {
    return nullptr;
  }

  uint32_t bufc = 0;
  while (bufc < req.count) {
    v4l2_buffer buf;
    memset(&buf, 0, sizeof(v4l2_buffer));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = bufc;

    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
      Logger::get(Logger::Error) << "v4l2: cannot query buffer " << bufc << ": " << strerror(errno) << std::endl;
      stopMmap(fd, bufv, bufc);
      return nullptr;
    }

    bufv[bufc].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (bufv[bufc].start == MAP_FAILED) {
      Logger::get(Logger::Error) << "v4l2: cannot map buffer " << bufc << ": " << strerror(errno) << std::endl;
      stopMmap(fd, bufv, bufc);
      return nullptr;
    }
    bufv[bufc].length = buf.length;
    bufc++;

    if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
      Logger::get(Logger::Error) << "v4l2: cannot queue buffer " << bufc << ": " << strerror(errno) << std::endl;
      stopMmap(fd, bufv, bufc);
      return nullptr;
    }
  }

  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
    Logger::get(Logger::Error) << "v4l2: cannot start streaming: " << strerror(errno) << std::endl;
    stopMmap(fd, bufv, bufc);
    return nullptr;
  }
  *n = bufc;
  return bufv;
}

void V4L2Reader::stopMmap(int fd, buffer_t* bufv, uint32_t bufc) {
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  ioctl(fd, VIDIOC_STREAMOFF, &type);
  for (uint32_t i = 0; i < bufc; i++) {
    munmap(bufv[i].start, bufv[i].length);
  }
  free(bufv);
}

int V4L2Reader::findMaxRate(int fd, struct v4l2_format fmt, const struct v4l2_fract* min_it, struct v4l2_fract* it) {
  struct v4l2_frmivalenum fie;
  memset(&fie, 0, sizeof(fie));
  fie.pixel_format = fmt.fmt.pix.pixelformat;
  fie.width = fmt.fmt.pix.width;
  fie.height = fmt.fmt.pix.height;

  if (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fie) < 0) {
    Logger::get(Logger::Info) << "v4l2: unknown frame intervals: " << strerror(errno) << std::endl;
    // Frame intervals cannot be enumerated. Set the format and then
    // get the streaming parameters to figure out the default frame
    // interval. This is not necessarily the maximum though.
    struct v4l2_format dummy_fmt = fmt;
    struct v4l2_streamparm parm = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE};

    if (ioctl(fd, VIDIOC_S_FMT, &dummy_fmt) < 0 || ioctl(fd, VIDIOC_G_PARM, &parm) < 0) {
      *it = infinity;
      return -1;
    }

    *it = parm.parm.capture.timeperframe;
  } else {
    switch (fie.type) {
      case V4L2_FRMIVAL_TYPE_DISCRETE:
        *it = infinity;
        do {
          if ((fcmp(&fie.discrete, min_it) >= 0) & (fcmp(&fie.discrete, it) < 0)) {
            *it = fie.discrete;
          }
          fie.index++;
        } while (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fie) >= 0);
        Logger::get(Logger::Info) << "v4l2: discrete frame interval: " << it->numerator << "/" << it->denominator
                                  << std::endl;
        break;
      case V4L2_FRMIVAL_TYPE_STEPWISE:
      case V4L2_FRMIVAL_TYPE_CONTINUOUS:
        if (fcmp(&fie.stepwise.max, min_it) < 0) {
          *it = infinity;
          return -1;
        }
        if (fcmp(&fie.stepwise.min, min_it) >= 0) {
          *it = fie.stepwise.min;
          break;
        }
        if (fie.type == V4L2_FRMIVAL_TYPE_CONTINUOUS) {
          *it = *min_it;
          break;
        }

        it->numerator *= fie.stepwise.step.denominator;
        it->denominator *= fie.stepwise.step.denominator;
        while (fcmp(it, min_it) < 0) {
          it->numerator += fie.stepwise.step.numerator;
        }
        break;
    }
  }
  return 0;
}

V4L2Reader::V4L2Reader(int id, int fd, buffer_t* bufv, uint32_t bufc, int64_t width, int64_t height,
                       int64_t frameDataSize, VideoStitch::PixelFormat format, FrameRate frameRate)
    : Reader(id),
      VideoReader(width, height, frameDataSize, format, Host, frameRate, 0, NO_LAST_FRAME, false, nullptr),
      fd(fd),
      bufv(bufv),
      bufc(bufc) {}

V4L2Reader::~V4L2Reader() {
  if (bufv != nullptr) {
    stopMmap(fd, bufv, bufc);
  }
  close(fd);
}

// -----------------------------------

Status V4L2Reader::seekFrame(frameid_t) { return Status::OK(); }

ReadStatus V4L2Reader::readFrame(mtime_t& date, unsigned char* videoFrame) {
  // streaming or reading
  if (bufv == nullptr) {
// we don't care about the return of read
#pragma GCC diagnostic ignored "-Wunused-result"
    read(fd, videoFrame, getSpec().frameDataSize);
  } else {
    v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    // wait for next frame
    if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
      switch (errno) {
        case EAGAIN:
          return ReadStatus::OK();  // XXX TODO FIXME retry!
        case EIO:
        default:
          return ReadStatus::OK();  // XXX TODO FIXME
      }
    }

    // copy frame
    date = getFrameTimestamp(&buf);
    memcpy(videoFrame, bufv[buf.index].start, buf.bytesused);

    // unlock
    if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
      return ReadStatus::OK();  // XXX TODO FIXME
    }
  }

  return ReadStatus::OK();
}

mtime_t V4L2Reader::getFrameTimestamp(const v4l2_buffer* frame) {
  return (frame->timestamp.tv_sec * 1000000) + frame->timestamp.tv_usec;
}

}  // namespace Input

// ------------------ Discovery

namespace Plugin {

V4L2Discovery* V4L2Discovery::create() { return new V4L2Discovery(); }

std::string V4L2Discovery::name() const { return "v4l2"; }

std::string V4L2Discovery::readableName() const { return "Video for Linux Two API"; }

std::vector<Plugin::DiscoveryDevice> V4L2Discovery::inputDevices() {
  std::vector<Plugin::DiscoveryDevice> devices;
#ifndef __ANDROID__
  glob_t globbuf;
  memset(&globbuf, 0, sizeof(globbuf));
  glob("/dev/video*", GLOB_ERR, nullptr, &globbuf);
  for (size_t i = 0; i < globbuf.gl_pathc; ++i) {
    Plugin::DiscoveryDevice device;
    device.name = globbuf.gl_pathv[i];
    device.displayName = device.name;
    device.type = Plugin::DiscoveryDevice::CAPTURE;
    devices.push_back(device);
  }
#endif
  return devices;
}

std::vector<std::string> V4L2Discovery::cards() const { return std::vector<std::string>(); }

}  // namespace Plugin
}  // namespace VideoStitch
