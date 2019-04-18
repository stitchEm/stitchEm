// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <libvideostitch/ptv.hpp>
#include <libvideostitch/status.hpp>
#include <libvideostitch/parse.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <memory>

void printUsage(const std::string& argv0) { std::cerr << "usage: " << argv0 << " input.ptvb output.ptv" << std::endl; }

int main(int argc, char** argv) {
  if (argc < 3) {
    printUsage(std::string(argv[0]));
    return 1;
  }

  VideoStitch::Potential<VideoStitch::Ptv::Parser> parser(VideoStitch::Ptv::Parser::create());
  if (!parser.ok()) {
    std::cerr << "Error: could not create the parser" << std::endl;
    return 1;
  }

  if (!parser->parse(std::string(argv[1]))) {
    std::cerr << "Error: could not parse the input ptvb." << std::endl;
    return 1;
  }

  std::unique_ptr<VideoStitch::Ptv::Value> content(parser->getRoot().clone());
  std::ofstream outFile;
  outFile.open(std::string(argv[2]), std::ios_base::out);
  if (!outFile.is_open()) {
    std::cerr << "Error: could not open the output file " << argv[2] << std::endl;
    return 1;
  }

  content->printJson(outFile);
  outFile.close();
  return 0;
}
