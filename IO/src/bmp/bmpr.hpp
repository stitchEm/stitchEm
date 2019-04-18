// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef BMPR_HPP_
#define BMPR_HPP_

#include "io.hpp"

#include <cassert>
#include <fstream>
#include <ostream>
#include <vector>

typedef struct {
  char* filename;
  void* buffer;
} cache_t;

#define MAX_CACHED_FILES (8)

static cache_t filesCache[MAX_CACHED_FILES] = {
    {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr},
    {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr},
};

void* cachedFile(const char* filename) {
  cache_t* free_entry = nullptr;

  for (int i = 0; i < MAX_CACHED_FILES; i++) {
    if (filesCache[i].filename) {
      if (strcmp(filesCache[i].filename, filename) == 0) {
        return (filesCache[i].buffer);
      }
    } else if (free_entry == nullptr) {
      free_entry = filesCache + i;
    }
  }
  assert(free_entry);
  free_entry->filename = (char*)malloc(strlen(filename) + 1);
  strcpy(free_entry->filename, filename);
  FILE* fd = VideoStitch::Io::openFile(filename, "rb");
  assert(fd);
  fseek(fd, 0, SEEK_END);
  size_t len = ftell(fd);
  fseek(fd, 0, SEEK_SET);
  free_entry->buffer = malloc(len);
  assert(free_entry->buffer);
// we don't care about the return of read
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
  fread(free_entry->buffer, sizeof(char), len, fd);
#pragma GCC diagnostic pop
  fclose(fd);
  return (free_entry->buffer);
}

namespace VideoStitch {
namespace Input {
class BMPReader {
 public:
  BMPReader(const char* filename, std::ostream*) : width(0), height(0), cury(0), bottomup(0), pitch(0) {
    buffer = (char*)cachedFile(filename);
    assert(buffer);
    (void)readHeader();
  }

  ~BMPReader() {}

  unsigned getWidth() const { return width; }

  unsigned getHeight() const { return height; }

  bool ok() const { return true; }

  /**
   * Fill in the given buffer with the next row (RGBRGBRGB).
   * @data must be large enough to hold one row.
   */
  bool getNextRow(unsigned char* data) {
    char* src = buffer + bmf.bfOffBits;
    if (cury < height) {
      if (!bottomup) {
        src += (height - (cury + 1)) * pitch;
      } else {
        src += cury * pitch;
      }
      memcpy(data, src, width * 3);
      cury++;
      return true;
    }
    return false;
  }

 private:
  int readHeader() {
    memcpy(&bmf, buffer, sizeof(struct bmpfileheader));
    memcpy(&bmi, buffer + sizeof(struct bmpfileheader), sizeof(struct bmpinfoheader));
    if ((bmf.bfType != 0x4D42) || (bmf.bfReserved1) || (bmf.bfReserved2) ||
        (bmi.biSize != sizeof(struct bmpinfoheader)) || (bmi.biPlanes != 1) || (bmi.biBitCount != 24) ||
        (bmi.biCompression)) {
      return (-1);
    }
    width = bmi.biWidth;
    bottomup = (int)(bmi.biHeight >> 31);
    height = (bmi.biHeight ^ bottomup) - bottomup;
    pitch = (size_t)(unsigned int)((((width * bmi.biPlanes * bmi.biBitCount) + 31) >> 3) & -4);
    cury = 0;
    return (0);
  }

 private:
  char* buffer;
  unsigned int width;
  unsigned int height;
  unsigned int cury;
  int bottomup;
  size_t pitch;

  struct __attribute__((__packed__)) bmpfileheader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
  } bmf;

  struct __attribute__((__packed__)) bmpinfoheader {
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
  } bmi;
};
}  // namespace Input
}  // namespace VideoStitch

#endif
