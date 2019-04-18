// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/config.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/parse.hpp"

#include "tiffOutput.hpp"
#include <sstream>
#include <fstream>
#include <iostream>

#include "tiffio.h"

namespace VideoStitch {
namespace Output {
const char TiffWriter::extension[] = "tif";

Potential<TiffWriter> TiffWriter::create(const Ptv::Value& config, const char* baseName, unsigned width,
                                         unsigned height, FrameRate framerate, int referenceFrame) {
  std::string compStr(TIFF_WRITER_DEFAULT_COMPRESSION);
  if (Parse::populateString("TiffWriter", config, "compression", compStr, false) == Parse::PopulateResult_WrongType) {
    return {Origin::Output, ErrType::InvalidConfiguration,
            "TIFF writer configuration 'compression' field must be a string"};
  }
  int compression = -1;
  if (compStr == "none") {
    compression = COMPRESSION_NONE;
  } else if (compStr == "lzw") {
    compression = COMPRESSION_LZW;
  } else if (compStr == "packbits") {
    compression = COMPRESSION_PACKBITS;
  } else if (compStr == "jpeg") {
    compression = COMPRESSION_JPEG;
  } else if (compStr == "deflate") {
    compression = COMPRESSION_ADOBE_DEFLATE;
  } else {
    return {Origin::Output, ErrType::InvalidConfiguration, "Invalid TIFF writer compression '" + compStr + "'"};
  }
  int numberNumDigits = 1;
  Parse::populateInt("JpgOutputWriter", config, "numbered_digits", numberNumDigits, false);

  return new TiffWriter(baseName, width, height, framerate, compression, referenceFrame, numberNumDigits);
}

void TiffWriter::writeFrame(const std::string& filename, const char* data) {
  TIFF* tif = TIFFOpen(filename.c_str(), "w");
  if (!tif) {
    Logger::get(Logger::Error) << "Cannot open file '" << filename << "' for writing." << std::endl;
    return;
  }
  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, getWidth());
  TIFFSetField(tif, TIFFTAG_IMAGELENGTH, getHeight());
  TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 4);
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, getHeight());

  TIFFSetField(tif, TIFFTAG_COMPRESSION, compression);
  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);  // RGBA??
  TIFFSetField(tif, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

  TIFFSetField(tif, TIFFTAG_XRESOLUTION, 150.0);
  TIFFSetField(tif, TIFFTAG_YRESOLUTION, 150.0);
  TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);

  // Write the information to the file
  TIFFWriteEncodedStrip(tif, 0, (tdata_t)data, (tsize_t)(getWidth() * getHeight() * 4));
  TIFFClose(tif);
}

TiffWriter::TiffWriter(const char* baseName, unsigned width, unsigned height, FrameRate framerate, int compression,
                       int referenceFrame, int numberedNumDigits)
    : Output(baseName),
      NumberedFilesWriter(baseName, width, height, framerate, PixelFormat::RGBA, referenceFrame, numberedNumDigits),
      compression(compression) {}

TiffWriter::~TiffWriter() {}
}  // namespace Output
}  // namespace VideoStitch
