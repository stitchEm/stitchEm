// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif

#include "./pngutil.hpp"

#include "backend/common/imageOps.hpp"

#include "gpu/allocator.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/stream.hpp"
#include "gpu/2dBuffer.hpp"
#include "image/unpack.hpp"

#include "libvideostitch/imageProcessingUtils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace VideoStitch {
namespace Debug {
inline int clamp0255(int32_t v) { return v > 255 ? 255 : (v < 0 ? 0 : v); }

/**
 * Base template class for getters.
 */
template <typename T>
struct BaseGetter {
  typedef T value_type;
};

struct RGB210Getter : public BaseGetter<uint32_t> {
  static int32_t getR(uint32_t v) { return Image::RGB210::r(v); }
  static int32_t getG(uint32_t v) { return Image::RGB210::g(v); }
  static int32_t getB(uint32_t v) { return Image::RGB210::b(v); }
  static int32_t getA(uint32_t v) { return Image::RGB210::a(v); }
};

struct RGBA8888Getter : public BaseGetter<uint32_t> {
  static int32_t getR(uint32_t v) { return Image::RGBA::r(v); }
  static int32_t getG(uint32_t v) { return Image::RGBA::g(v); }
  static int32_t getB(uint32_t v) { return Image::RGBA::b(v); }
  static int32_t getA(uint32_t v) { return Image::RGBA::a(v); }
};

struct RGBSolidGetter : public BaseGetter<uint32_t> {
  static int32_t getR(uint32_t v) { return Image::RGBA::r(v); }
  static int32_t getG(uint32_t v) { return Image::RGBA::g(v); }
  static int32_t getB(uint32_t v) { return Image::RGBA::b(v); }
  static int32_t getA(uint32_t) { return 255; }
};

struct Uint32ValueGetter : public BaseGetter<uint32_t> {
  static int32_t getR(uint32_t v) { return v > 0xffffff ? 0xff : (int32_t)(v & 0xff); }
  static int32_t getG(uint32_t v) { return v > 0xffffff ? 0xff : (int32_t)(v & 0xff00); }
  static int32_t getB(uint32_t v) { return v > 0xffffff ? 0xff : (int32_t)(v & 0xff0000); }
  static int32_t getA(uint32_t) { return 255; }
};

template <int32_t bit>
struct Uint32ValueBitGetter : public BaseGetter<uint32_t> {
  static int32_t getR(uint32_t /*v*/) { return 0xff; }
  static int32_t getG(uint32_t /*v*/) { return 0; }
  static int32_t getB(uint32_t /*v*/) { return 0; }
  static int32_t getA(uint32_t v) { return (v & (1 << bit)) ? 0xff : 0; }
};

template <int64_t minVal, int64_t maxVal>
struct FloatValueGetter : public BaseGetter<float> {
  static int32_t getR(float v) { return clamp0255((int32_t)(getVal(v) + 0.5f)); }
  static int32_t getG(float v) { return getVal(v) > 255.0f ? 255 : 0; }
  static int32_t getB(float v) { return getVal(v) > 255.0f ? 255 : 0; }
  static int32_t getA(float) { return 255; }

 private:
  static float getVal(float v) { return (v - (float)minVal) * 255.0f / (float)(maxVal - minVal); }
};

template <int64_t minVal, int64_t maxVal>
struct FlowValueGetter : public BaseGetter<float2> {
  static int32_t getR(float2 v) { return clamp0255((int32_t)(getVal(v.x))); }
  static int32_t getG(float2 v) { return clamp0255((int32_t)(getVal(v.y))); }
  static int32_t getB(float2 v) { return 0; }
  static int32_t getA(float2 v) { return 255; }

 private:
  static float getVal(float v) { return (v - (float)minVal) * 255.0f / (float)(maxVal - minVal); }
};

template <int64_t minVal, int64_t maxVal>
struct Float4ValueGetter : public BaseGetter<float4> {
  static int32_t getR(float4 v) { return clamp0255((int32_t)(getVal(v.x))); }
  static int32_t getG(float4 v) { return clamp0255((int32_t)(getVal(v.y))); }
  static int32_t getB(float4 v) { return clamp0255((int32_t)(getVal(v.z))); }
  static int32_t getA(float4 v) { return clamp0255((int32_t)(getVal(v.w))); }

 private:
  static float getVal(float v) { return (v - (float)minVal) * 255.0f / (float)(maxVal - minVal); }
};
/**
 * Dumps an RGBA device buffer to disk as PNG.
 * @param filename File to write to.
 * @param devBuffer Device buffer to dump
 * @param width width
 * @param height height
 */
inline void dumpRGB210DeviceBuffer(const char* filename, GPU::Buffer<const uint32_t> devBuffer, int64_t width,
                                   int64_t height) {
  std::vector<uint32_t> hostRGBA(width * height);
  auto surf = Core::OffscreenAllocator::createSourceSurface(width, height, "dumpRGB210DeviceBuffer");
  assert(surf.ok());
  if (surf.ok()) {
    Image::convertRGB210ToRGBA(*surf->pimpl->surface, devBuffer, width, height, GPU::Stream::getDefault());
    GPU::memcpyBlocking(hostRGBA.data(), *surf->pimpl->surface);
    Util::PngReader writer;
    writer.writeRGBAToFile(filename, width, height, hostRGBA.data());
  }
}

/**
 * Dumps an RGBA210 device buffer to disk as PNG.
 * getter one of th getters.
 * transferFp is usually clamp0255
 * @param filename File to write to.
 * @param devBuffer Device buffer to dump
 * @param width width
 * @param height height
 */
template <typename Getter, int (*transferFp)(int32_t)>
inline void dumpRGBADeviceBufferWithTransferFn(const char* filename,
                                               GPU::Buffer<const typename Getter::value_type> devBuffer, int64_t width,
                                               int64_t height) {
  std::vector<typename Getter::value_type> hostRGBA(width * height);
  std::vector<unsigned char> data(width * height * 4);
  GPU::memcpyBlocking(hostRGBA.data(), devBuffer, width * height * sizeof(typename Getter::value_type));
  for (int64_t j = 0; j < width * height; ++j) {
    // if (Image::RGB210::g(hostRGBA[j]) != 0) { std::cout << "g" << Image::RGB210::g(hostRGBA[j]) << " " <<
    // (int)transferFp(Image::RGB210::g(hostRGBA[j])) << std::endl;}
    data[4 * j + 0] = (unsigned char)transferFp(Getter::getR(hostRGBA[j]));
    data[4 * j + 1] = (unsigned char)transferFp(Getter::getG(hostRGBA[j]));
    data[4 * j + 2] = (unsigned char)transferFp(Getter::getB(hostRGBA[j]));
    data[4 * j + 3] = (unsigned char)transferFp(Getter::getA(hostRGBA[j]));
  }
  Util::PngReader writer;
  writer.writeRGBAToFile(filename, width, height, &data.front());
}

inline Status dumpRGBADeviceBuffer(const char* filename, GPU::Buffer<const uint32_t> devBuffer, int64_t width,
                                   int64_t height) {
  std::vector<uint32_t> hostRGBA(width * height);
  FAIL_RETURN(GPU::memcpyBlocking(hostRGBA.data(), devBuffer, width * height * 4));
  Util::PngReader writer;
  writer.writeRGBAToFile(filename, width, height, hostRGBA.data());
  return Status::OK();
}

inline void dumpRGBATexture(const char* filename, GPU::Surface& surface, int64_t width, int64_t height) {
  std::vector<uint32_t> hostRGBA(width * height);
  GPU::memcpyBlocking(hostRGBA.data(), surface);
  Util::PngReader writer;
  writer.writeRGBAToFile(filename, width, height, hostRGBA.data());
}

inline void dumpDepthSurface(const char* filename, GPU::Surface& surface, int64_t width, int64_t height) {
  std::vector<float> hostDepth(width * height);
  GPU::memcpyBlocking(hostDepth.data(), surface);

  std::vector<uint16_t> hostDepthU16;
  hostDepthU16.reserve(width * height);
  for (float val : hostDepth) {
    const float inMilliMeters = val * 1000.f;
    const uint16_t u16 = (uint16_t)std::min((float)std::numeric_limits<uint16_t>::max(), std::round(inMilliMeters));
    hostDepthU16.push_back(u16);
  }

  Util::PngReader writer;
  writer.writeMonochrome16ToFile(filename, width, height, hostDepthU16.data());
}

inline unsigned char binary(unsigned char v) { return v > 0 ? 255 : 0; }
inline unsigned char linear(unsigned char v) { return v; }
inline unsigned char linearFloat(float v) { return (unsigned char)((v < 1.0f ? v : 1.0f) * 255.0); }
/**
 * Dumps a monochrome device buffer to disk as PNG.
 * @param filename File to write to.
 * @param devBuffer Device buffer to dump
 * @param width width
 * @param height height
 */
template <unsigned char (*transferFp)(unsigned char)>
inline void dumpMonochromeDeviceBuffer(std::string filename, GPU::Buffer<const unsigned char> devBuffer, int64_t width,
                                       int64_t height) {
  std::vector<unsigned char> hostMono(width * height);
  std::vector<unsigned char> data(width * height * 4);
  std::cout << "Dumping buffer of size " << devBuffer.byteSize() << std::endl;
  GPU::memcpyBlocking(hostMono.data(), devBuffer, width * height);
  for (int64_t j = 0; j < width * height; ++j) {
    data[4 * j + 0] = transferFp(hostMono[j]);
    data[4 * j + 1] = transferFp(hostMono[j]);
    data[4 * j + 2] = transferFp(hostMono[j]);
    data[4 * j + 3] = 255;
  }
  Util::PngReader writer;
  writer.writeRGBAToFile(filename.c_str(), width, height, &data.front());
}

/**
 * Dumps a monochrome device buffer to disk as PNG.
 * @param filename File to write to.
 * @param devBuffer Device buffer to dump
 * @param width width
 * @param height height
 */
template <unsigned char (*transferFp)(unsigned char)>
inline void dumpMonochromeDeviceBuffer(std::string filename, std::vector<unsigned char> hostMono, int64_t width,
                                       int64_t height) {
  std::vector<unsigned char> data(width * height * 4);
  for (int64_t j = 0; j < width * height; ++j) {
    data[4 * j + 0] = transferFp(hostMono[j]);
    data[4 * j + 1] = transferFp(hostMono[j]);
    data[4 * j + 2] = transferFp(hostMono[j]);
    data[4 * j + 3] = 255;
  }
  Util::PngReader writer;
  writer.writeRGBAToFile(filename.c_str(), width, height, &data.front());
}

/**
 * Dumps a monochrome device buffer to disk as PNG.
 * @param filename File to write to.
 * @param devBuffer Device buffer to dump
 * @param width width
 * @param height height
 */
template <unsigned char (*transferFp)(float)>
inline void dumpMonochromeDeviceBuffer(std::string filename, GPU::Buffer<const float> devBuffer, int64_t width,
                                       int64_t height) {
  std::vector<float> hostMono(width * height);
  std::vector<unsigned char> data(width * height * 4);
  std::cout << "Dumping buffer of size " << devBuffer.byteSize() << std::endl;
  GPU::memcpyBlocking(hostMono.data(), devBuffer, width * height * sizeof(float));
  for (int64_t j = 0; j < width * height; ++j) {
    data[4 * j + 0] = transferFp(hostMono[j]);
    data[4 * j + 1] = transferFp(hostMono[j]);
    data[4 * j + 2] = transferFp(hostMono[j]);
    data[4 * j + 3] = 255;
  }
  Util::PngReader writer;
  writer.writeRGBAToFile(filename.c_str(), width, height, &data.front());
}

template <class T>
inline void dumpRGBAIndexDeviceBuffer(const char* filename, const std::vector<T>& hostRGBA, int64_t width,
                                      int64_t height, const int displayBit = -1) {
  std::vector<unsigned char> data;
  Util::ImageProcessing::convertIndexToRGBA(hostRGBA, data, displayBit);
  Util::PngReader writer;
  writer.writeRGBAToFile(filename, width, height, &data.front());
}

/**
 * Visualize an index buffer to disk as colored PNG.
 * @param filename File to write to.
 * @param devBuffer Device buffer to dump. (Format described in InputsMap)
 * @param width width
 * @param height height
 */
template <class T>
inline Status dumpRGBAIndexDeviceBuffer(const char* filename, GPU::Buffer<const T> devBuffer, int64_t width,
                                        int64_t height, const int displayBit = -1) {
  std::vector<T> hostRGBA(width * height);
  FAIL_RETURN(GPU::memcpyBlocking(hostRGBA.data(), devBuffer, width * height * sizeof(T)));
  dumpRGBAIndexDeviceBuffer(filename, hostRGBA, width, height, displayBit);
  return Status::OK();
}

inline Status dumpRGBACoordinateDeviceBuffer(const std::string& filename, GPU::Buffer<const float2> devBuffer,
                                             int64_t width, int64_t height) {
  std::vector<float2> hostMono(width * height);
  std::vector<unsigned char> data(width * height * 4);
  FAIL_RETURN(GPU::memcpyBlocking(hostMono.data(), devBuffer, width * height * sizeof(float2)));
  float2 minCoord = make_float2(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  float2 maxCoord = make_float2(std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
  for (int i = 0; i < width * height; i++) {
    if (hostMono[i].x < minCoord.x) minCoord.x = hostMono[i].x;
    if (hostMono[i].y < minCoord.y) minCoord.y = hostMono[i].y;
    if (hostMono[i].x > maxCoord.x) maxCoord.x = hostMono[i].x;
    if (hostMono[i].y > maxCoord.y) maxCoord.y = hostMono[i].y;
  }

  for (int64_t j = 0; j < width * height; ++j) {
    data[4 * j + 0] =
        (unsigned char)(std::min(1.0f, (hostMono[j].x - minCoord.x) / (maxCoord.x - minCoord.x)) * 255.0f);
    data[4 * j + 1] =
        (unsigned char)(std::min(1.0f, (hostMono[j].y - minCoord.y) / (maxCoord.y - minCoord.y)) * 255.0f);
    data[4 * j + 2] = 0;
    data[4 * j + 3] = 255;
  }
  Util::PngReader writer;
  writer.writeRGBAToFile(filename.c_str(), width, height, &data.front());
  return Status::OK();
}

}  // namespace Debug
}  // namespace VideoStitch
