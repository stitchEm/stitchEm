// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "io.hpp"

#include "libvideostitch/logging.hpp"

#include <fstream>
#include <ostream>
#include <vector>
#include <setjmp.h>

#include <jpeglib.h>

namespace VideoStitch {
namespace Input {

void my_output_message(j_common_ptr cinfo) {
  char buffer[JMSG_LENGTH_MAX];
  /* Create the message */
  (*cinfo->err->format_message)(cinfo, buffer);

  /* Send it to stderr, adding a newline */
  Logger::get(Logger::Error) << buffer << std::endl;
}

class JPGReader {
 public:
  JPGReader(const char* filename, VideoStitch::ThreadSafeOstream* err = NULL) : hf(NULL), width(0), height(0) {
    hf = VideoStitch::Io::openFile(filename, "rb");
    if (!hf) {
      if (err) {
        *err << "Cannot open file '" << filename << "' for reading." << std::endl;
      }
    } else {
      readHeader();
    }
  }

  ~JPGReader() {
    if (hf) {
      if (cinfo.output_scanline) {
        // otherwise libjpeg crashes - see VSA-1326
        jpeg_finish_decompress(&cinfo);
      }
      jpeg_destroy_decompress(&cinfo);
      fclose(hf);
    }
  }

  unsigned getWidth() const { return width; }

  unsigned getHeight() const { return height; }

  bool ok() const { return hf != NULL && width != 0 && height != 0; }

  /**
   * Fill in the given buffer with the next row (RGBRGBRGB).
   * @data must be large enough to hold one row.
   */
  bool getNextRow(unsigned char* data) {
    JSAMPROW rowArray[1];
    rowArray[0] = data;
    return (cinfo.output_scanline < cinfo.output_height) && jpeg_read_scanlines(&cinfo, rowArray, 1);
  }

 private:
  struct my_error_mgr {
    struct jpeg_error_mgr pub; /* "public" fields */
    jmp_buf setjmp_buffer;     /* for return to caller */
  };
  typedef struct my_error_mgr* my_error_ptr;

  /*
   * Here's the routine that will replace the standard error_exit method:
   */
  static void my_error_exit(j_common_ptr cinfo) {
    /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
    my_error_ptr myerr = (my_error_ptr)cinfo->err;

    /* Always display the message. */
    /* We could postpone this until after returning, if we chose. */
    (*cinfo->err->output_message)(cinfo);

    /* Return control to the setjmp point */
    longjmp(myerr->setjmp_buffer, 1);
  }

  void readHeader() {
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    jerr.pub.output_message = my_output_message;
    /* Establish the setjmp return context for my_error_exit to use. */
    if (setjmp(jerr.setjmp_buffer)) {
      return;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, hf);
    jpeg_read_header(&cinfo, TRUE);

    width = cinfo.image_width;
    height = cinfo.image_height;

    jpeg_start_decompress(&cinfo);
    // FIXME: make sure that we have RGB data
  }

 private:
  FILE* hf;
  unsigned width;
  unsigned height;
  struct jpeg_decompress_struct cinfo;
  struct my_error_mgr jerr;
};
}  // namespace Input
}  // namespace VideoStitch
