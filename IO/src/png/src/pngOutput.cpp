// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pngOutput.hpp"
#include "detail/Png.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <sstream>
#include <iostream>

namespace VideoStitch {
namespace Output {
const char PngWriter::extension[] = "png";

PngWriter* PngWriter::create(Ptv::Value const* config, Plugin::VSWriterPlugin::Config const& run_time) {
  BaseConfig baseConfig;

  if (!baseConfig.parse(*config).ok()) {
    Logger::error("[PNG]") << "Cannot parse PNG output configuration" << std::endl;
    return nullptr;
  }

  bool writeAlphaChannel = false;
  if (Parse::populateBool("PngWriter", *config, "alpha", writeAlphaChannel, false) == Parse::PopulateResult_WrongType) {
    Logger::warning("[PNG]") << "PNG output configuration value for 'alpha' should be true or false" << std::endl;
  }

  const bool writeMonochromeDepth = !strcmp(baseConfig.strFmt, "depth-png");

  const PixelFormat pixelFormat = [=]() {
    if (writeMonochromeDepth) {
      if (writeAlphaChannel) {
        Logger::warning("[PNG]") << "PNG output with 'depth' and alpha channel unsupported. Ignoring 'alpha' directive."
                                 << std::endl;
      }
      return PixelFormat::Grayscale16;
    } else {
      if (writeAlphaChannel) {
        return PixelFormat::RGBA;
      } else {
        return PixelFormat::RGB;
      }
    }
  }();

  const int referenceFrame = readReferenceFrame(*config);
  return new PngWriter(baseConfig.baseName, run_time.width, run_time.height, run_time.framerate, pixelFormat,
                       referenceFrame, baseConfig.numberNumDigits);
}

bool PngWriter::handles(VideoStitch::Ptv::Value const* config) {
  bool l_return = false;
  BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    l_return = (!strcmp(baseConfig.strFmt, "png") || !strcmp(baseConfig.strFmt, "depth-png"));
  }
  return l_return;
}

void PngWriter::writeFrame(const std::string& filename, const char* data) {
  detail::Png png;
  switch (getPixelFormat()) {
    case PixelFormat::RGB:
      png.writeRGBToFile(filename.c_str(), getWidth(), getHeight(), data);
      break;
    case PixelFormat::RGBA:
      png.writeRGBAToFile(filename.c_str(), getWidth(), getHeight(), data);
      break;
    case PixelFormat::Grayscale16:
      png.writeMonochrome16ToFile(filename.c_str(), getWidth(), getHeight(), data);
      break;
    default:
      Logger::error("[PNG]") << "Unsupported pixel format " << getStringFromPixelFormat(getPixelFormat()) << std::endl;
      break;
  }
}

PngWriter::PngWriter(const char* baseName, uint64_t width, uint64_t height, FrameRate framerate,
                     PixelFormat pixelFormat, int referenceFrame, int numberedNumDigits)
    : Output(baseName),
      NumberedFilesWriter(baseName, (unsigned)width, (unsigned)height, framerate, pixelFormat, referenceFrame,
                          numberedNumDigits) {}

PngWriter::~PngWriter() {}
}  // namespace Output
}  // namespace VideoStitch
