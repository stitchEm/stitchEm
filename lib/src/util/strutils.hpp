// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef STRUTILS_HPP_
#define STRUTILS_HPP_

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#ifdef _MSC_VER
#include <Windows.h>
// Workaround VC snprintf.
#define snprintf _snprintf
#else
#include <cstring>
#endif

// put_time is not defined in GCC
#ifdef __GNUC__
#include <ctime>
std::string _vs_put_time(const std::tm* tmb, const char* fmt);
#define vs_put_time _vs_put_time
#else
#include <iomanip>
#define vs_put_time std::put_time
#endif

namespace VideoStitch {
namespace Util {

/**
 * A deleter of non-stdio streams.
 */
struct IStreamDeleter {
  void operator()(std::istream* stream) {
    if (stream != &std::cin) {
      delete stream;
    }
  }
};

std::unique_ptr<std::istream, IStreamDeleter> createIStream(const std::string& filename,
                                                            std::ios_base::openmode mode = std::ios_base::in);

/**
 * @brief String prefix matching.
 * @param str The string where to look for prefix.
 * @param prefix The prefix to look for.
 * @returns true if prefix is a prefix of str.
 */
inline bool startsWith(const char* str, const char* prefix) { return !strncmp(str, prefix, strlen(prefix)); }

/**
 * @brief Split string in two substrings using delimiter. We split only on the first instance.
 * @param str The string to split.
 * @param delim The delimiter.
 * @param first Storage for the first part (without delimiter). Not NULL.
 * @param second Storage for the second part (without delimiter). Not NULL.
 * @returns true if the delimiter was found.
 */
bool splitOnce(const char* str, char delim, std::string* first, std::string* second);

/**
 * Split string in several substrings using a comma delimiter.
 */
void splitWithComma(const std::string& text, std::vector<std::string>& out);

/**
 * @brief Split string in several substrings using delimiter.
 * @param str The string to split.
 * @param delim The delimiter.
 * @param res Storage for the parts. Not NULL.
 */
void split(const char* str, char delim, std::vector<std::string>* res);

/**
 * Escape special characters in a string.
 * @param str The string to escape.
 * @return The escaped string.
 */
std::string escapeStr(const std::string& in);

/**
 * @brief Unescape special characters in a string.
 * @param str The string to unescape.
 * @param res The unescaped string.
 * @return false on error.
 */
bool unescapeStr(const std::string& in, std::string& res);

/**
 * @brief Parses a color in html format.
 * @param str Color, RRGGBB or RRGGBBAA.
 * @param color result, in ABGR 8bpc.
 * @returns true on success.
 */
bool parseHtmlColor(const std::string& str, uint32_t& color);

/**
 * @brief Codes a unicode character to utf-8.
 * @param codepoint Unicode codepoint.
 * @param sink Output sink for utf-8 bytes.
 */
void unicodeToUtf8(uint16_t codepoint, std::string& sink);

/**
 * A class that provides locale RAII on a stream. During the lifetime of this object, the stream will use then given
 * locale. Usage: void f(std::ostream& os) { os << 1.1 << std::endl;
 *   {
 *     UsingLocaleOnStream usingLocale(os, std::locale("C"));
 *     os << 2.2 << std::endl;
 *     os << 3.3 << std::endl;
 *   }
 *   os << 4.4 << std::endl;
 * }
 *
 * might output:
 * 1,1
 * 2.2
 * 3.3
 * 4,4
 */
class UsingCLocaleOnStream {
 public:
  /**
   * @param os The stream whose locale to set.
   */
  explicit UsingCLocaleOnStream(std::ostream& os) : savedOs(&os), savedLocale(os.imbue(std::locale("C"))) {}

  ~UsingCLocaleOnStream() { savedOs->imbue(savedLocale); }

 private:
  std::ostream* const savedOs;
  const std::locale savedLocale;
};

}  // namespace Util
}  // namespace VideoStitch
#endif
