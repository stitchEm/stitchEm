// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <array>
#include <map>
#include <utility>
#include <vector>

#include <iostream>

namespace VideoStitch {
namespace Testing {

template <typename KeyType, typename ValueType>
std::ostream &operator<<(std::ostream &stream, const std::map<KeyType, ValueType> &map) {
  for (auto &kv : map) {
    stream << kv.first << " --> " << kv.second << std::endl;
  }
  return stream;
}

template <typename FirstType, typename SecondType>
std::ostream &operator<<(std::ostream &stream, const std::pair<FirstType, SecondType> &pair) {
  stream << "(" << pair.first << ", " << pair.second << ")" << std::endl;
  return stream;
}

template <class ClassType, size_t size>
std::ostream &operator<<(std::ostream &stream, const std::array<ClassType, size> &array) {
  stream << "(";
  for (const auto &v : array) {
    stream << v << ",";
  }
  stream << ")" << std::endl;
  return stream;
}

template <class ClassType>
std::ostream &operator<<(std::ostream &stream, const std::vector<ClassType> &vector) {
  stream << "(";
  for (const auto &v : vector) {
    stream << v << ",";
  }
  stream << ")" << std::endl;
  return stream;
}

}  // namespace Testing
}  // namespace VideoStitch
