// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief fileExists checks the existence of a file.
 * @param filename The input full filename.
 * @return true if the file exists, false otherwise.
 */
bool fileExists(const std::string &filename) {
  std::ifstream file(filename.c_str());
  bool fileExists = file.good();
  file.close();
  return fileExists;
}
