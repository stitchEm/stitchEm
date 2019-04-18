// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CRASHER_HPP
#define CRASHER_HPP

class Crasher {
 public:
  void crash() {
    char *str = nullptr;
    str[0] = 0;
  }
};

#endif  // CRASHER_HPP
