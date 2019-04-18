// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

// TODO move back to src/test/common once all tests are moved to GPU interface

#include "testing.hpp"

#include "libvideostitch/config.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include <util/pngutil.hpp>
#include <gpu/buffer.hpp>
#include <gpu/memcpy.hpp>
#include <gpu/uniqueBuffer.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace VideoStitch {
namespace Testing {

template <typename T>
class DeviceBuffer {
 public:
  /**
   * Creates a device buffer filled with @a value.
   * @param width Width.
   * @param height Height.
   * @param value value to fill the buffer with.
   */
  DeviceBuffer(int64_t width, int64_t height) : width(width), height(height) {
    auto potBuf = GPU::uniqueBuffer<T>((size_t)(width * height), "Test");
    ENSURE(potBuf.ok());
    buffer = potBuf.releaseValue();
  }

  /**
   * Fills the buffer with a single value.
   * @param value value to fill the buffer with.
   */
  void fill(T value) { fill(std::vector<T>((size_t)(width * height), value)); }

  /**
   * Fills the buffer with a vector's contents.
   * @param data contents. Must be of size width * height.
   */
  void fill(const std::vector<T>& data) {
    ENSURE_EQ(data.size(), (size_t)(width * height));
    ENSURE(GPU::memcpyBlocking(gpuBuf(), data.data()), "Could not fill buffer from vector");
  }

  /**
   * Fills the buffer with a host buffer's contents.
   * @param data contents. Must be of size width * height.
   */
  void fillData(const T* data) {
    ENSURE(GPU::memcpyBlocking(buffer.get(), data, buffer.byteSize()), "Could not fill buffer from buffer");
  }

  /**
   * Fills the buffer with a vector's contents.
   * @param data contents. Must be of size width * height * sizeof(T).
   */
  void fillData(const std::vector<unsigned char>& data) {
    ENSURE_EQ(data.size(), gpuBuf().byteSize());
    ENSURE(GPU::memcpyBlocking(gpuBuf(), (T*)data.data()), "Could not fill buffer from untyped vector");
  }

  /**
   * Reads back the buffer.
   * @param data On return, contains the contents of the buffer.
   */
  void readback(std::vector<T>& data) {
    data.clear();
    data.resize((size_t)(width * height));
    ENSURE(GPU::memcpyBlocking(data.data(), gpuBufConst(), gpuBuf().byteSize()), "Could not readback");
  }

  /**
   * Reads back the buffer.
   * @param data On return, contains the contents of the buffer.
   */
  void readbackData(std::vector<unsigned char>& data) const {
    data.clear();
    data.resize(gpuBuf().byteSize());
    ENSURE(GPU::memcpyBlocking((T*)data.data(), gpuBuf()), "Could not readbackData");
  }

  void ENSURE_BUF_EQ(const DeviceBuffer& other) const {
    std::vector<unsigned char> thisData;
    readbackData(thisData);
    std::vector<unsigned char> otherData;
    other.readbackData(otherData);
    ENSURE_ARRAY_EQ(thisData.data(), otherData.data(), width * height * sizeof(T));
  }

  /**
   * Creates a device buffer filled with the contents of @a content.
   * @param width Width.
   * @param height Height.
   * @param content vuffer of size width * height
   */
  DeviceBuffer(int64_t width, int64_t height, const T* content) : width(width), height(height) {
    auto potBuf = GPU::uniqueBuffer<T>(width * height, "Test");
    ENSURE(potBuf.ok());
    buffer = potBuf.releaseValue();
    ENSURE(GPU::memcpyBlocking(gpuBuf(), content, gpuBuf().byteSize()), "Could not initialize DeviceBuffer");
  }

  /**
   * Reads the raw contents of @a filename.
   * Checks that the size is correct and dies on error.
   */
  void readRawFromFile(const char* filename) {
    std::vector<T> data(width * height);
    std::ifstream ifs(filename, std::ios_base::in | std::ios_base::binary);
    ifs.read((char*)data.data(), width * height * sizeof(T));
    ENSURE(ifs.good());
    ENSURE(ifs.get() == EOF);
    ENSURE(GPU::memcpyBlocking(buffer, data.data(), buffer.byteSize()));
  }

  /**
   * Reads contents from png file @a filename.
   */
  void readPngFromFile(const char* filename) {
    std::vector<uint32_t> data((size_t)(width * height));
    Util::PngReader reader;
    ENSURE(reader.readRGBAFromFile(filename, width, height, data.data()));
    ENSURE(GPU::memcpyBlocking(buffer.borrow(), data.data(), buffer.borrow().byteSize()));
  }

  const int64_t width;
  const int64_t height;

  GPU::Buffer<T> gpuBuf() const { return buffer.borrow(); }

  GPU::Buffer<const T> gpuBufConst() const { return gpuBuf().as_const(); }

  /**
   * Intepret the buffer as an RGBA8888 buffer and write it to PNG.
   */
  void dumpToPng8888(const char* filename) const {
    std::vector<uint32_t> data((size_t)(width * height));
    ENSURE(GPU::memcpyBlocking(data.data(), buffer.borrow().as_const(), (size_t)(width * height * 4)));
    Util::PngReader reader;
    ENSURE(reader.writeRGBAToFile(filename, width, height, data.data()));
  }

 protected:
  GPU::UniqueBuffer<T> buffer;
  DeviceBuffer& operator=(const DeviceBuffer&);
};

/**
 * Additional methods to deal with packed color data.
 */
class PackedDeviceBuffer : public DeviceBuffer<uint32_t> {
 public:
  /**
   * Creates a device buffer filled with @a value.
   * @param width Width.
   * @param height Height.
   */
  PackedDeviceBuffer(int64_t width, int64_t height) : DeviceBuffer<uint32_t>(width, height) {}

  /**
   * Fills the buffer with a signle value.
   * @param value value to fill the buffer with.
   */
  void fill(unsigned char r, unsigned char g, unsigned char b) {
    DeviceBuffer<uint32_t>::fill(Image::RGBA::pack(r, g, b, 255));
  }

  /**
   * Reads the packed contents from png file @a filename.
   * Checks that the size is correct and dies on error.
   */
  void readPngFromFile(const char* filename) {
    std::vector<uint32_t> data((size_t)(width * height));
    Util::PngReader reader;
    ENSURE(reader.readRGBAFromFile(filename, width, height, data.data()));
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = Image::RGBA::pack(Image::RGBA::r(data[i]), Image::RGBA::g(data[i]), Image::RGBA::b(data[i]),
                                  Image::RGBA::a(data[i]));
    }
    ENSURE(GPU::memcpyBlocking(buffer.borrow(), data.data(), buffer.borrow().byteSize()));
  }

  void ENSURE_EQ(const PackedDeviceBuffer& other) const {
    std::vector<uint32_t> thisData((size_t)(width * height));
    ENSURE(GPU::memcpyBlocking(thisData.data(), buffer.borrow()));
    std::vector<uint32_t> otherData((size_t)(width * height));
    ENSURE(GPU::memcpyBlocking(otherData.data(), other.buffer.borrow()));
    ENSURE_ARRAY_EQ(thisData.data(), otherData.data(), (unsigned)width * (unsigned)height);
  }

  void ENSURE_NEQ(const PackedDeviceBuffer& other) const {
    std::vector<uint32_t> thisData((size_t)(width * height));
    ENSURE(GPU::memcpyBlocking(thisData.data(), buffer.borrow()));
    std::vector<uint32_t> otherData((size_t)(width * height));
    ENSURE(GPU::memcpyBlocking(otherData.data(), other.buffer.borrow()));
    ENSURE_ARRAY_NEQ(thisData.data(), otherData.data(), (unsigned)width * height);
  }

  void dumpToPng(const char* filename) const {
    std::vector<uint32_t> data((size_t)(width * height));
    ENSURE(GPU::memcpyBlocking(data.data(), buffer.borrow()));
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = Image::RGBA::pack(Image::RGBA::r(data[i]), Image::RGBA::g(data[i]), Image::RGBA::b(data[i]),
                                  Image::RGBA::a(data[i]));
    }
    Util::PngReader reader;
    ENSURE(reader.writeRGBAToFile(filename, width, height, data.data()));
  }
};

/**
 * A printer that can print RGBA210 values in decimal.
 */
class RGBA210Printer {
 public:
  explicit RGBA210Printer(uint32_t value) : value(value) {}

 private:
  friend std::ostream& operator<<(std::ostream& os, const RGBA210Printer& printer);
  uint32_t value;
};

inline std::ostream& operator<<(std::ostream& os, const RGBA210Printer& printer) {
  if (Image::RGB210::a(printer.value)) {
    os << "(" << std::setw(3) << std::setfill('0') << Image::RGB210::r(printer.value) << "," << std::setw(3)
       << std::setfill('0') << Image::RGB210::g(printer.value) << "," << std::setw(3) << std::setfill('0')
       << Image::RGB210::b(printer.value) << ")";
  } else {
    os << "(           )";
  }
  return os;
}

Core::PanoDefinition* ensureParsePanoDefinition(const std::string& ptv) {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  if (!parser->parseData(ptv)) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }
  Core::PanoDefinition* res = Core::PanoDefinition::create(parser->getRoot());
  ENSURE(res, "could not parse pano definition");
  return res;
}

inline GPU::Stream createTestedStream() {
  auto potStream = GPU::Stream::create();
  ENSURE(potStream.ok());
  return potStream.value();
}

GPU::PotentialUniqueBuffer<uint32_t> loadFile(const char* filename, int64_t& width, int64_t& height) {
  std::vector<unsigned char> tmp;
  if (!VideoStitch::Util::PngReader::readRGBAFromFile(filename, width, height, tmp)) {
    std::stringstream msg;
    msg << "Image '" << filename << "': failed to setup reader.";
    return GPU::PotentialUniqueBuffer<uint32_t>(Status{Origin::Input, ErrType::SetupFailure, msg.str()});
  }
  std::vector<uint32_t> buffer((size_t)(width * height));
  for (size_t i = 0; i < (size_t)(width * height); ++i) {
    buffer[i] = VideoStitch::Image::RGBA::pack(tmp[(size_t)(4 * i)], tmp[(size_t)(4 * i + 1)], tmp[(size_t)(4 * i + 2)],
                                               tmp[(size_t)(4 * i + 3)]);
  }
  auto devBuffer = GPU::uniqueBuffer<uint32_t>((size_t)(width * height), "Test");
  ENSURE(devBuffer.status());
  ENSURE(GPU::memcpyBlocking(devBuffer.borrow(), &buffer.front()));
  return devBuffer;
}

}  // namespace Testing
}  // namespace VideoStitch
