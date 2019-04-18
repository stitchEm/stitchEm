// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <libvideostitch/logging.hpp>
#include <libvideostitch/ptv.hpp>
#include <libvideostitch/inputDef.hpp>
#include <libvideostitch/panoDef.hpp>
#include "version.hpp"

#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>

namespace {

#ifdef _MSC_VER
#include <io.h>
#include <sys/types.h>
#include <sys/stat.h>
const char dirSep = '\\';
#else
#include <unistd.h>
const char dirSep = '/';
#endif
void extractBasename(const std::string& infn, std::string& outfn) {
  for (int i = (int)infn.size() - 1; i > 0; --i) {
    if (infn[i] == dirSep) {
      // Something like: "toto.tata/titi" (no extension)
      outfn = infn;
      break;
    } else if (infn[i] == '.') {
      outfn = infn.substr(0, i);
      break;
    }
  }
}
void extractPath(const std::string& infn, std::string& outfn) {
  for (int i = (int)infn.size() - 1; i > 0; --i) {
    if (infn[i] == dirSep) {
      outfn = infn.substr(0, i + 1);
      return;
    }
  }
  outfn = "";
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
std::string detectExtractImagesPattern(const char* filename) {
  std::string fn(filename);
  if (fn.find('-') == std::string::npos)  // at least one dash
    return "";
  std::vector<std::string> lFn;
  split(filename, '.', &lFn);  //"source/A", "mp4-10", "jpg"
  if (lFn.size() < 3) return "";
  if (lFn[lFn.size() - 2].find('-') == std::string::npos)  // at least one dash
    return "";
  std::vector<std::string> lFnExt;
  split(lFn[lFn.size() - 2].c_str(), '-', &lFnExt);  //"source/A", {"mp4", "10"}, "jpg"
  if (lFnExt.size() != 2) return "";
  std::string frmNum = lFnExt[lFnExt.size() - 1];  //"10"
  int res;
  if (!(std::stringstream(frmNum) >> res))  // sanity check: is a number
    return "";
  lFn.pop_back();            // remove "jpg"
  lFn.pop_back();            // remove "mp4-10"
  lFn.push_back(lFnExt[0]);  // add "mp4"
  std::string srcFn;         //"source/A.mp4"
  for (size_t i = 0; i < lFn.size(); ++i) {
    srcFn += (i + 1 < lFn.size()) ? lFn[i] + "." : lFn[i];
  }
  return srcFn;
}

void printUsage(const char* execName, VideoStitch::ThreadSafeOstream& os) {
  os << "Usage: " << execName << " [options] -i <input.pto> [output.ptv]" << std::endl;
  os << "Options are:" << std::endl;
  os << "  -t <output_type>: Set the output type to <output_type> (default 'mp4')." << std::endl;
  os << "  -v <q|0|1|2|3|4> : Log level: quiet, error, warning, info (default), verbose, debug." << std::endl;
  os << "If output is not specified, the same name as input is used, with a 'ptv' extension." << std::endl;
}
}  // namespace

using VideoStitch::Logger;

int main(int argc, char** argv) {
  if (argc == 2 && !strcmp(argv[1], "--version")) {
    std::cout << "VideoStitch calibration import tool. Copyright (c) 2018 stitchEm" << std::endl;
    std::cout << "library version: " << LIB_VIDEOSTITCH_VERSION << std::endl;
    return 0;
  }
  Logger::readLevelFromArgv(argc, argv);

  std::string infn;
  std::string outfn;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '\0' && argv[i][1] != '\0' && argv[i][0] == '-') {
      switch (argv[i][1]) {
        case 'i':
          if (i >= argc - 1 || argv[i + 1][0] == '-') {
            Logger::get(Logger::Error) << "The -i option takes at least one parameter." << std::endl;
            printUsage(argv[0], Logger::get(Logger::Error));
            return 1;
          }
          if (!infn.empty()) {
            Logger::get(Logger::Error) << "Several input files: \"" << infn << "\" and \"" << argv[i + 1] << "\""
                                       << std::endl;
            printUsage(argv[0], Logger::get(Logger::Error));
            return 1;
          }
          infn = argv[++i];
          break;
        case 't':
          if (i >= argc - 1 || argv[i + 1][0] == '-') {
            Logger::get(Logger::Error) << "The -t option takes a parameter." << std::endl;
            printUsage(argv[0], Logger::get(Logger::Error));
            return 1;
          }
          break;
        default:
          printUsage(argv[0], Logger::get(Logger::Error));
          return 1;
      }
    } else if (outfn.empty()) {
      outfn = argv[i];
    } else {
      printUsage(argv[0], Logger::get(Logger::Error));
      return 1;
    }
  }

  if (infn.empty()) {
    Logger::get(Logger::Error) << "Error: no input file." << std::endl;
    printUsage(argv[0], Logger::get(Logger::Error));
    return 1;
  }
  if (outfn.empty()) {
    extractBasename(infn, outfn);
    outfn.append(".ptv");
  }

  VideoStitch::Potential<VideoStitch::Core::PanoDefinition> pano(VideoStitch::Core::PanoDefinition::parseFromPto(infn));
  if (!pano.ok()) {
    return 1;
  }

  // find if the input file pattern matches with an 'extractimages' extracted file name.
  // pattern: source/A.mp4, 10th frame -> source/A.mp4-10.jpg
  for (videoreaderid_t i = 0; i < pano->numVideoInputs(); ++i) {
    VideoStitch::Core::InputDefinition& iDef = pano->getVideoInput(i);
    std::string srcBaseFn = "";
    if (iDef.getReaderConfig().getType() == VideoStitch::Ptv::Value::STRING) {
      srcBaseFn = detectExtractImagesPattern(iDef.getReaderConfig().asString().c_str());
    }
    std::string srcFullFn;
    extractPath(outfn, srcFullFn);
    srcFullFn.append(srcBaseFn);
    if (std::ifstream(srcFullFn)) {
      iDef.setFilename(srcBaseFn);
    }
  }

  std::stringstream error;
  if (!pano->validate(error)) {
    Logger::get(Logger::Error) << error.str();
    return 1;
  }

  std::ofstream ofs(outfn.c_str(), std::ios_base::out);
  if (!ofs.is_open()) {
    Logger::get(Logger::Error) << "Error: cannot open '" << outfn << "' for writing." << std::endl;
    return 1;
  }

  Logger::get(Logger::Info) << "Boostrapping..." << std::endl;
  VideoStitch::Ptv::Value* root = pano.release()->serialize();
  root->printJson(ofs);
  delete root;
}
