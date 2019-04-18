// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "exrutil.hpp"

#include "libvideostitch/logging.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4515)
#pragma warning(disable : 4244)
#endif

#ifdef __linux__
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#else
#include <openexr/ImfOutputFile.h>
#include <openexr/ImfChannelList.h>
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <sstream>

#define WRITE_DEPTH_AS_RGB 1

namespace VideoStitch {
namespace Util {

bool Exr::writeDepthToFile(const char* filename, int64_t width, int64_t height, const float* data) {
  try {
    Imf::Header header((int)width, (int)height);
#if WRITE_DEPTH_AS_RGB
    header.channels().insert("R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("B", Imf::Channel(Imf::FLOAT));
#else
    header.channels().insert("Z", Imf::Channel(Imf::FLOAT));
#endif

    Imf::OutputFile file(filename, header);

    Imf::FrameBuffer frameBuffer;

#if WRITE_DEPTH_AS_RGB
    frameBuffer.insert("R",                                 // name
                       Imf::Slice(Imf::FLOAT,               // type
                                  (char*)data,              // base
                                  sizeof(*data) * 1,        // xStride
                                  sizeof(*data) * width));  // yStride
    frameBuffer.insert("G",                                 // name
                       Imf::Slice(Imf::FLOAT,               // type
                                  (char*)data,              // base
                                  sizeof(*data) * 1,        // xStride
                                  sizeof(*data) * width));  // yStride
    frameBuffer.insert("B",                                 // name
                       Imf::Slice(Imf::FLOAT,               // type
                                  (char*)data,              // base
                                  sizeof(*data) * 1,        // xStride
                                  sizeof(*data) * width));  // yStride
#else
    frameBuffer.insert("Z",                                 // name
                       Imf::Slice(Imf::FLOAT,               // type
                                  (char*)data,              // base
                                  sizeof(*data) * 1,        // xStride
                                  sizeof(*data) * width));  // yStride
#endif

    file.setFrameBuffer(frameBuffer);
    file.writePixels((int)height);
  } catch (const std::exception& e) {
    std::stringstream msg;
    msg << "writeDepthFile(): " << e.what() << std::endl;
    Logger::get(Logger::Error) << msg.str();
    return false;
  }

  return true;
}

}  // namespace Util
}  // namespace VideoStitch
