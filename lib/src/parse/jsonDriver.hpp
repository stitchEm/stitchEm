// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "json.hpp"

#include "parser-generated/jsonParser.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

#include <map>
#include <sstream>

#define YY_DECL                                                                                           \
  VideoStitch::Parse::JsonParser::token_type yylex(VideoStitch::Parse::JsonParser::semantic_type* yylval, \
                                                   VideoStitch::Parse::JsonParser::location_type* yylloc, \
                                                   VideoStitch::Parse::JsonDriver& /*driver*/)

YY_DECL;

namespace VideoStitch {
namespace Parse {
class JsonDriver : public Ptv::Parser {
 public:
  JsonDriver(bool trace_parsing, bool trace_scanning);
  virtual ~JsonDriver();

  bool parse(const std::string& fileName);
  bool parseData(const std::string& data);

  std::string getErrorMessage() const;

  const Ptv::Value& getRoot() const;

 private:
  friend class JsonParser;
  bool scan_begin();
  void scan_end();
  void error(const location& l, const std::string& m);
  void error(const std::string& m);
  bool parseInternal();
  bool trace_scanning;
  bool trace_parsing;
  enum ParserSource { FromStdin, FromFile, FromData };
  ParserSource parserSource;
  std::string filename;           // if FromFile
  std::vector<char> dataToParse;  // if FromData
  void* dataBuffer;               // Lex buffer when parserSource == FromData;
  std::stringstream errStream;
  JsonValue* root;
};
}  // namespace Parse
}  // namespace VideoStitch
