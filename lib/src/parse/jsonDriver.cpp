// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "jsonDriver.hpp"

#include "parser-generated/jsonParser.hpp"
#include "util/strutils.hpp"

#include <cstring>
#include <fstream>
#include <streambuf>

namespace VideoStitch {

namespace Ptv {
Potential<Parser> Parser::create() { return Potential<Parser>(new Parse::JsonDriver(false, false)); }
}  // namespace Ptv

namespace Parse {

JsonDriver::JsonDriver(bool trace_parsing, bool trace_scanning)
    : trace_scanning(trace_scanning),
      trace_parsing(trace_parsing),
      parserSource(FromStdin),
      dataBuffer(nullptr),
      root(nullptr) {}

JsonDriver::~JsonDriver() { delete root; }

bool JsonDriver::parse(const std::string& fileName) {
  filename = fileName;
  if (filename.empty() || filename == "-") {
    // Only supports json.
    parserSource = FromStdin;
  } else {
    parserSource = FromFile;
    // Is this UBJson ?
    auto ifs = VideoStitch::Util::createIStream(filename, std::ios_base::in | std::ios_base::binary);
    PotentialJsonValue res = JsonValue::parseUBJson(static_cast<std::istream&>(*ifs));
    if (res.status().getCode() != ParseStatus::StatusCode::NotUBJsonFormat) {
      delete root;
      root = res.release();
      return res.ok();
    }
  }
  return parseInternal();
}

bool JsonDriver::parseData(const std::string& data) {
  dataToParse.resize(data.size() + 1);
  memcpy(dataToParse.data(), data.c_str(), data.size() + 1);
  {
    // Is this UBJson ?
    DataInputStream dataStream(data);
    PotentialJsonValue res = JsonValue::parseUBJson(dataStream);
    if (res.status().getCode() != ParseStatus::StatusCode::NotUBJsonFormat) {
      delete root;
      root = res.release();
      return res.ok();
    }
  }
  parserSource = FromData;
  return parseInternal();
}

bool JsonDriver::parseInternal() {
  errStream.clear();
  if (!scan_begin()) {
    return false;
  }
  JsonParser parser(*this);
  delete root;
  root = nullptr;
  parser.set_debug_level(trace_parsing);
  int res = parser.parse();
  scan_end();
  if (res != 0 || !root) {
    delete root;
    root = nullptr;
    return false;
  } else {
    return true;
  }
}

std::string JsonDriver::getErrorMessage() const { return errStream.str(); }

const Ptv::Value& JsonDriver::getRoot() const { return *root; }

void JsonDriver::error(const location& l, const std::string& m) { errStream << l << ": " << m << std::endl; }

void JsonDriver::error(const std::string& m) { errStream << m << std::endl; }
}  // namespace Parse
}  // namespace VideoStitch
