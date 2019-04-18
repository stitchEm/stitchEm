// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#define DOUBLE(config, param, value)                                                                    \
  double param = value;                                                                                 \
  if (Parse::populateDouble("RTMP", config, #param, param, false) == Parse::PopulateResult_WrongType) { \
    Logger::get(Logger::Error) << "RTMP : invalid " << #param << std::endl;                             \
    return nullptr;                                                                                     \
  }
#define INT(config, param, value)                                                                    \
  int param = value;                                                                                 \
  if (Parse::populateInt("RTMP", config, #param, param, false) == Parse::PopulateResult_WrongType) { \
    Logger::get(Logger::Error) << "RTMP : invalid " << #param << std::endl;                          \
    return nullptr;                                                                                  \
  }
#define BOOLE(config, param, value)                                                                   \
  bool param = value;                                                                                 \
  if (Parse::populateBool("RTMP", config, #param, param, false) == Parse::PopulateResult_WrongType) { \
    Logger::get(Logger::Error) << "RTMP : invalid " << #param << std::endl;                           \
    return nullptr;                                                                                   \
  }
#define STRING(config, param, value)                                                                    \
  std::string param = value;                                                                            \
  if (Parse::populateString("RTMP", config, #param, param, false) == Parse::PopulateResult_WrongType) { \
    Logger::get(Logger::Error) << "RTMP : invalid " << #param << std::endl;                             \
    return nullptr;                                                                                     \
  }
