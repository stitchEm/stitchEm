// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "io.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/parse.hpp"

extern "C" {
#include "jpeglib.h"
}

#include "jpgOutput.hpp"
#include <cassert>
#include <sstream>
#include <fstream>
#include <iostream>

namespace VideoStitch {
namespace Output {
const char JpgWriter::extension[] = "jpg";

Potential<JpgWriter> JpgWriter::create(Ptv::Value const* config, Plugin::VSWriterPlugin::Config run_time) {
  VideoStitch::Output::BaseConfig baseConfig;
  FAIL_CAUSE(baseConfig.parse(*config), Origin::Output, ErrType::InvalidConfiguration,
             "Could not parse JPEG output configuration");
  int referenceFrame = readReferenceFrame(*config);
  int jpgQuality = JPEG_WRITER_DEFAULT_QUALITY;
  if (Parse::populateInt("JpgWriter", *config, "quality", jpgQuality, false) == Parse::PopulateResult_WrongType) {
    return {Origin::Output, ErrType::InvalidConfiguration, "JPEG configuration is missing 'quality' setting"};
  }
  if (jpgQuality <= 0 || jpgQuality > 100) {
    return {Origin::Output, ErrType::InvalidConfiguration, "JPEG 'quality' must be a setting between 1 and 100"};
  }
  int numberNumDigits = 1;
  Parse::populateInt("JpgWriter", *config, "numbered_digits", numberNumDigits, false);

  JpgWriter* res = new JpgWriter(baseConfig.baseName, run_time.width, run_time.height, run_time.framerate, jpgQuality,
                                 referenceFrame, numberNumDigits);
  if (res && res->cinfo.dest) {
    return res;
  } else {
    delete res;
    return {Origin::Output, ErrType::OutOfResources, "Could not set up JPEG writer"};
  }
}

namespace {
class MemJpegDest : public jpeg_destination_mgr {
 public:
  static MemJpegDest* create(unsigned width, unsigned height) {
    MemJpegDest* res = new MemJpegDest(width, height);
    if (res && res->jpegData) {
      return res;
    } else {
      Logger::get(Logger::Error) << "Out of memory." << std::endl;
      delete res;
      return NULL;
    }
  }

  static void init_mem_destination(j_compress_ptr cinfo) {
    static_cast<MemJpegDest*>(cinfo->dest)->initMemDestination();
  }

  static boolean realloc(j_compress_ptr /*cinfo*/) {
    assert(false);
    return TRUE;
  }

  static void term_mem_destination(j_compress_ptr /*cinfo*/) {}

  void initMemDestination() {
    next_output_byte = jpegData;
    free_in_buffer = jpegDataSize;
  }

  size_t getDataSize() const { return jpegDataSize - free_in_buffer; }

  const char* getData() const { return (const char*)jpegData; }

  ~MemJpegDest() { delete[] jpegData; }

 private:
  MemJpegDest(unsigned width, unsigned height)
      : jpegDataSize(width * height * 3), jpegData(new (std::nothrow) unsigned char[jpegDataSize]) {
    // parent fields
    this->init_destination = MemJpegDest::init_mem_destination;
    this->empty_output_buffer = MemJpegDest::realloc;  // Should never be called.
    this->term_destination = MemJpegDest::term_mem_destination;
  }

  size_t jpegDataSize;
  unsigned char* jpegData;
};
}  // namespace

void JpgWriter::writeFrame(const std::string& filename, const char* data) {
  jpeg_start_compress(&cinfo, TRUE);
  // JPGTurbo is not really const-correct hence the nasty const cast.
  JSAMPROW rowPointer = const_cast<JSAMPROW>((const unsigned char*)data);
  for (int i = 0; i < (int)getHeight(); ++i) {
    jpeg_write_scanlines(&cinfo, &rowPointer, 1);
    rowPointer += cinfo.input_components * getWidth();
  }
  jpeg_finish_compress(&cinfo);
  // Put data in file
  FILE* hf = VideoStitch::Io::openFile(filename.c_str(), "wb");
  if (!hf) {
    Logger::get(Logger::Error) << "Cannot open file '" << filename << "' for writing." << std::endl;
    return;
  }
  fwrite(static_cast<MemJpegDest*>(cinfo.dest)->getData(), 1, static_cast<MemJpegDest*>(cinfo.dest)->getDataSize(), hf);
  fclose(hf);
}

JpgWriter::JpgWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int jpgQuality,
                     int referenceFrame, int numberedNumDigits)
    : Output(baseName),
      NumberedFilesWriter(baseName, width, height, framerate, PixelFormat::RGB, referenceFrame, numberedNumDigits) {
  // TODO: Use logger.
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, jpgQuality, TRUE);
  // Tell libjpeg to output to memory.
  cinfo.dest = MemJpegDest::create(width, height);
}

JpgWriter::~JpgWriter() {
  delete static_cast<MemJpegDest*>(cinfo.dest);
  cinfo.dest = NULL;
  jpeg_destroy_compress(&cinfo);
}

bool JpgWriter::handles(VideoStitch::Ptv::Value const* config) {
  bool l_return = false;
  BaseConfig baseConfig;
  if (baseConfig.parse(*config).ok()) {
    l_return = (!strcmp(baseConfig.strFmt, "jpg"));
  }
  return l_return;
}
}  // namespace Output
}  // namespace VideoStitch
