// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "strutils.hpp"

#include "libvideostitch/logging.hpp"

#ifdef _MSC_VER
#include <codecvt>
#endif
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>

std::string _vs_put_time(const std::tm* tmb, const char* fmt) {
  char foo[256];
  return (0 < std::strftime(foo, sizeof(foo), fmt, tmb)) ? std::string(foo) : "";
}

namespace VideoStitch {
namespace Util {
std::unique_ptr<std::istream, IStreamDeleter> createIStream(const std::string& filename, std::ios_base::openmode mode) {
  std::unique_ptr<std::istream, IStreamDeleter> ifs;
#ifdef _MSC_VER
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  std::wstring wideFilename;
  try {
    wideFilename = converter.from_bytes(filename);
  } catch (std::range_error) {
    ifs.reset(new std::ifstream(wideFilename, mode));
    return ifs;
  }
  ifs.reset(new std::ifstream(filename, mode));
#else
  ifs.reset(new std::ifstream(filename, mode));
#endif
  return ifs;
}

bool splitOnce(const char* str, char delim, std::string* first, std::string* second) {
  for (const char* p = str; *p != '\0'; ++p) {
    if (*p == delim) {
      first->assign(str, p - str);
      second->assign(p + 1);
      return true;
    }
  }
  first->clear();
  second->clear();
  return false;
}

void split(const char* str, char delim, std::vector<std::string>* res) {
  const char* lastP = str;
  for (const char* p = str; *p != '\0'; ++p) {
    if (*p == delim) {
      res->push_back(std::string(lastP, p - lastP));
      lastP = p + 1;
    }
  }
  res->push_back(std::string(lastP));
}

void splitWithComma(const std::string& text, std::vector<std::string>& out) {
  size_t start = 0, end = 0;
  while ((end = text.find(',', start)) != std::string::npos) {
    out.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  out.push_back(text.substr(start));
}

std::string escapeStr(const std::string& in) {
  std::string res;
  for (std::string::const_iterator it = in.begin(); it != in.end(); ++it) {
    switch (*it) {
      case '"':
        res.push_back('\\');
        res.push_back('"');
        break;
      case '\\':
        res.push_back('\\');
        res.push_back('\\');
        break;
      case '\n':
        res.push_back('\\');
        res.push_back('n');
        break;
      case '\t':
        res.push_back('\\');
        res.push_back('t');
        break;
      case '\r':
        res.push_back('\\');
        res.push_back('r');
        break;
      case '\b':
        res.push_back('\\');
        res.push_back('b');
        break;
      case '\f':
        res.push_back('\\');
        res.push_back('f');
        break;
      default:
        res.push_back(*it);
        break;
    }
  }
  return res;
}

namespace {

/**
 * Decodes a hex value into a decimal value.
 * @param hex Two hex digits
 * @param out where the output is written.
 * @returns false on error.
 */
bool decodeHexChar(const char hex, int& out) {
  if ('0' <= hex && hex <= '9') {
    out = (int)(hex - '0');
    return true;
  } else if ('a' <= hex && hex <= 'f') {
    out = (int)(10 + hex - 'a');
    return true;
  }
  out = 0;
  return false;
}

/**
 * Decodes a hex-encoded codepoint.
 * @param hex Four hex digits
 * @param codepoint where the output is written.
 * @returns false on error.
 */
bool decodeAsciiToUnicode(const char* hex, uint16_t& codepoint) {
  codepoint = 0;
  int comp[4];
  for (int i = 0; i < 4; ++i) {
    if (!decodeHexChar(hex[i], comp[i])) {
      return false;
    }
  }
  codepoint = (uint16_t)((comp[0] << 12) | (comp[1] << 8) | (comp[2] << 4) | comp[3]);
  return true;
}
}  // namespace

bool unescapeStr(const std::string& in, std::string& res) {
  res.clear();
  for (std::string::const_iterator it = in.begin(); it != in.end(); ++it) {
    if (*it == '\\') {
      ++it;
      if (it == in.end()) {
        Logger::get(Logger::Error) << "Invalid escape sequence at end of string literal." << std::endl;
        res.clear();
        return false;
      }
      switch (*it) {
        case '"':
          res.push_back('"');
          break;
        case '\\':
          res.push_back('\\');
          break;
        case '/':
          res.push_back('/');
          break;
        case 'n':
          res.push_back('\n');
          break;
        case 't':
          res.push_back('\t');
          break;
        case 'r':
          res.push_back('\r');
          break;
        case 'b':
          res.push_back('\b');
          break;
        case 'f':
          res.push_back('\f');
          break;
        case 'u':  // Escaping in json is \uc3a9
          if (in.size() - (it - in.begin()) <= 4) {
            Logger::get(Logger::Error) << "Invalid JSON escape sequence at end of string literal." << std::endl;
            return false;
          } else {
            char hex[4];
            hex[0] = *(++it);
            hex[1] = *(++it);
            hex[2] = *(++it);
            hex[3] = *(++it);
            uint16_t unicode = 0;
            if (!(decodeAsciiToUnicode(hex, unicode))) {
              Logger::get(Logger::Error) << "Invalid JSON escape sequence '\\u" << hex[0] << hex[1] << hex[2] << hex[3]
                                         << "'" << std::endl;
              return false;
            }
            unicodeToUtf8(unicode, res);
          }
          break;
        default:
          Logger::get(Logger::Error) << "Invalid escape sequence '\\" << *it << "'" << std::endl;
          res.clear();
          return false;
      }
    } else {
      res.push_back(*it);
    }
  }
  return true;
}

bool parseHtmlColor(const std::string& str, uint32_t& color) {
  std::istringstream iss(str);
  if (str.size() == 6) {
    iss >> std::hex >> color;
    // Color is RGB, convert to ABGR solid.
    color = ((color & (uint32_t)0xff) << 16) | ((color & (uint32_t)0xff00)) | ((color & (uint32_t)0xff0000) >> 16) |
            (uint32_t)0xff000000;
    return true;
  } else if (str.size() == 8) {
    iss >> std::hex >> color;
    // Color is RGBA, convert to ABGR.
    color = ((color & (uint32_t)0xff) << 24) | ((color & (uint32_t)0xff00) << 8) | ((color & (uint32_t)0xff0000) >> 8) |
            ((color & (uint32_t)0xff000000) >> 24);
    return true;
  }
  return false;
}

void unicodeToUtf8(uint16_t codepoint, std::string& sink) {
  if (codepoint < 0x0080) {
    sink.push_back((char)codepoint);
  } else if (codepoint < 0x07ff) {
    sink.push_back((char)(0xc0 | ((codepoint & 0x07c0) >> 6)));
    sink.push_back((char)(0x80 | ((codepoint & 0x003f))));
  } else {
    sink.push_back((char)(0xe0 | ((codepoint & 0xf000) >> 12)));
    sink.push_back((char)(0x80 | ((codepoint & 0x0fc0) >> 6)));
    sink.push_back((char)(0x80 | ((codepoint & 0x003f))));
  }
}

}  // namespace Util
}  // namespace VideoStitch
