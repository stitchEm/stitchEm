// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __clang_analyzer__  // VSA-7043

#include "pnm.hpp"

namespace VideoStitch {
namespace Util {
bool PnmReader::read(const char *filename, int64_t &w, int64_t &h, std::vector<unsigned char> &data, std::ostream *err,
                     bool pad) {
  std::ifstream ifs(filename, std::ifstream::in | std::ifstream::binary);
  if (!ifs.good()) {
    if (err) {
      *err << "Cannot open file '" << filename << "' for reading." << std::endl;
    }
    return false;
  }

  if (!read(ifs, w, h, data, err, pad)) {
    return false;
  }

  ifs.close();
  return true;
}

bool PnmReader::read(std::ifstream &ifs, int64_t &w, int64_t &h, std::vector<unsigned char> &data, std::ostream *err,
                     bool pad) {
  data.resize(0);
  char buf[bufSize];
  ifs.getline(buf, bufSize);
  if (buf[0] != 'P') {
    if (err) {
      *err << "Not a PNM file." << std::endl;
    }
    return false;
  }
  switch (buf[1]) {
    case '1':
      return _read<AsciiPBM>(ifs, w, h, data, pad, err);
    case '2':
      return _read<AsciiPGM>(ifs, w, h, data, pad, err);
    case '3':
      return _read<AsciiPPM>(ifs, w, h, data, pad, err);
    case '4':
      return _read<BinPGM>(ifs, w, h, data, pad, err);  // FIXME
    case '5':
      return _read<BinPGM>(ifs, w, h, data, pad, err);
    case '6':
      return _read<BinPPM>(ifs, w, h, data, pad, err);
    default:
      if (err) {
        *err << "Not a PNM file." << std::endl;
      }
      return false;
  }
}

bool PnmReader::_readCommentWidthHeight(std::ifstream &ifs, int64_t &width, int64_t &height, std::ostream *err) {
  // read comments
  char buf[bufSize];
  while (ifs.peek() == '#') {
    ifs.getline(buf, bufSize);
  }
  ifs >> width;
  if (ifs.eof() || ifs.fail()) {
    if (err) {
      *err << "Cannot read PNM width." << std::endl;
    }
    return false;
  }
  ifs >> height;
  if (ifs.eof() || ifs.fail()) {
    if (err) {
      *err << "Cannot read PNM width." << std::endl;
    }
    return false;
  }
  int64_t depth;
  ifs >> depth;
  if (ifs.eof() || ifs.fail()) {
    if (err) {
      *err << "Cannot read PNM depth." << std::endl;
    }
    return false;
  }
  ifs.getline(buf, bufSize);  // read the remaining of the line (\n)
  return true;
}

template <PnmReader::PixType type>
bool PnmReader::_read(std::ifstream &ifs, int64_t &w, int64_t &h, std::vector<unsigned char> &data, bool pad,
                      std::ostream *err) {
  if (!_readCommentWidthHeight(ifs, w, h, err)) {
    return false;
  }
  data.reserve((size_t)(w * h * (3 + (int)pad)));
  for (int64_t i = 0; i < w * h; ++i) {
    _readPixel<type>(ifs, data);
    if (pad) {
      data.push_back(255U);
    }
  }
  return true;
}

template <>
void PnmReader::_readPixel<PnmReader::AsciiPBM>(std::ifstream &ifs, std::vector<unsigned char> &data) {
  unsigned v;
  ifs >> v;
  data.push_back(v ? 255U : 0);
  data.push_back(v ? 255U : 0);
  data.push_back(v ? 255U : 0);
}

template <>
void PnmReader::_readPixel<PnmReader::AsciiPGM>(std::ifstream &ifs, std::vector<unsigned char> &data) {
  unsigned v;
  ifs >> v;
  data.push_back((unsigned char)v);
  data.push_back((unsigned char)v);
  data.push_back((unsigned char)v);
}

template <>
void PnmReader::_readPixel<PnmReader::AsciiPPM>(std::ifstream &ifs, std::vector<unsigned char> &data) {
  unsigned v;
  ifs >> v;
  data.push_back((unsigned char)v);
  ifs >> v;
  data.push_back((unsigned char)v);
  ifs >> v;
  data.push_back((unsigned char)v);
}

template <>
void PnmReader::_readPixel<PnmReader::BinPGM>(std::ifstream &ifs, std::vector<unsigned char> &data) {
  unsigned char v;
  ifs.read((char *)&v, 1);
  data.push_back(v);
  data.push_back(v);
  data.push_back(v);
}

template <>
void PnmReader::_readPixel<PnmReader::BinPPM>(std::ifstream &ifs, std::vector<unsigned char> &data) {
  unsigned char v;
  ifs.read((char *)&v, 1);
  data.push_back(v);
  ifs.read((char *)&v, 1);
  data.push_back(v);
  ifs.read((char *)&v, 1);
  data.push_back(v);
}

std::ofstream *PpmWriter::openPpm(const char *filename, int64_t w, int64_t h, std::ostream *err) {
  std::ofstream *ofs = openGeneric(filename, err);
  if (ofs) {
    *ofs << "P6\n" << w << " " << h << "\n255\n";
  }
  return ofs;
}

std::ofstream *PpmWriter::openPam(const char *filename, int64_t w, int64_t h, std::ostream *err) {
  std::ofstream *ofs = openGeneric(filename, err);
  if (ofs) {
    *ofs << "P7\nWIDTH " << w << "\nHEIGHT " << h << "\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n";
  }
  return ofs;
}

std::ofstream *PpmWriter::openPgm(const char *filename, int64_t w, int64_t h, std::ostream *err) {
  std::ofstream *ofs = openGeneric(filename, err);
  if (ofs) {
    *ofs << "P5\n" << w << " " << h << "\n255\n";
  }
  return ofs;
}

std::ofstream *PpmWriter::openGeneric(const char *filename, std::ostream *err) {
  std::ofstream *ofs = new std::ofstream(filename, std::ios_base::out | std::ios_base::binary);
  if (!ofs->good()) {
    if (err) {
      *err << "Cannot open file '" << filename << "' for writing." << std::endl;
    }
    delete ofs;
    ofs = NULL;
  }
  return ofs;
}
}  // namespace Util
}  // namespace VideoStitch

#endif  // __clang_analyzer__
