// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
// VideoStitch Autocrop

// VideoStitch SDK
#include <libvideostitch/logging.hpp>
#include <libvideostitch/parse.hpp>
#include <libvideostitch/imageProcessingUtils.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/imgproc.hpp>

// System-dependant filesystem stuff.
#ifdef _MSC_VER
#include <opencv2/imgcodecs.hpp>
#include <direct.h>
#define chdir _chdir
#define snprintf _snprintf
#define getcwd _getcwd
#include <io.h>
#include <sys/types.h>
#include <sys/stat.h>
const char dirSep = '\\';
#else
#include <unistd.h>
const char dirSep = '/';
#endif

#ifdef _MSC_VER
#include <Windows.h>
#endif
#include <iostream>
#include <cassert>
#include <memory>

#ifndef _MSC_VER
#include <dirent.h>
#include <dlfcn.h>
#else
#include <libvideostitch/win32/dirent.h>
#endif

using namespace VideoStitch;

namespace {
/**
 * @brief Prints the executable usage.
 * @param execName Name of the executable (argv[0]).
 * @param os Output stream: e.g; std::cerr to output in the standard error stream.
 */
void printUsage(const char* execName, ThreadSafeOstream& os) {
  os << "Usage: " << execName << " -i <template> -o <output.json> [options]" << std::endl;
  os << "  -i <template.png>: can be an image file / or directory that contains all inputs for detection" << std::endl
#ifdef _MSC_VER
     << " Note that only PNG, JPEG, JPG files are supported." << std::endl
#else
     << " Note that only PNG file is supported." << std::endl
#endif
     << " This option can be used several times." << std::endl;
  os << "<output.json> the output json file to be written." << std::endl;
  os << "[options] are:" << std::endl;
  os << "  -d: Set this option to dump debug images." << std::endl;
  os << std::endl;
}

/**
 * @brief fileExists checks the existence of a file.
 * @param filename The input full filename.
 * @return true if the file exists, false otherwise.
 */
bool fileExists(const std::string& filename) {
  std::ifstream file(filename.c_str());
  if (file.good()) {
    file.close();
    return true;
  } else {
    file.close();
    return false;
  }
}

/**
 * Give a pointer on the file component of a full input filename.
 * @param input The input full filename.
 * @return the filename, belonging to the same char* as @input.
 */
const char* extractFilename(const char* input) {
  const char* file = input;
  // separator is '\' or '/'
  for (const char* p = input; *p != 0; ++p) {
    if (*p == '/' || *p == '\\') {
      file = p + 1;
    }
  }
  return file;
}

/**
 * List files with a certain extension in a directory.
 * @param directory: directory to list.
 * @param ext: the file prefix (may begin with '.').
 * @return A list of matching files.
 */
std::vector<std::string> listDirectory(const std::string& directory, const std::string& ext) {
  size_t extSize = ext.size();
  std::vector<std::string> filenames;
  DIR* dir = opendir(directory.c_str());
  if (dir) {
    /*print all the files and directories within directory*/
    struct dirent* ent = readdir(dir);
    while (ent) {
      std::string str(ent->d_name);
      ent = readdir(dir);
      if (str.size() > extSize) {
        std::string strLower = str;
        std::transform(strLower.begin(), strLower.end(), strLower.begin(), ::tolower);
        if (strLower.substr(strLower.size() - extSize) == ext) {
          filenames.push_back(directory + dirSep + str);
          continue;
        }
      }
    }
    closedir(dir);
  }
  return filenames;
}

Status readImage(const std::string& filename, int64_t& width, int64_t& height, std::vector<unsigned char>& data) {
  int channelCount;
  return VideoStitch::Util::ImageProcessing::readImage(filename, width, height, channelCount, data);
}

int extractArg(int argc, char** argv, std::vector<std::string>& inputFilenames, std::string& outputFilename,
               bool& outputDebugImage, std::string& algoSettingFilename) {
  // Check if the output folder/file is writable
  const std::string program = std::string(argv[0]);
  const size_t found = program.find_last_of("/\\");
  std::string executableDirectory = program.substr(0, found);
  auto& errLog = Logger::get(Logger::Error);
  char currentDir[1024];
  if (getcwd(currentDir, 1024)) {
    executableDirectory = (executableDirectory.empty() || executableDirectory == std::string(argv[0]))
                              ? std::string(currentDir)
                              : executableDirectory;
  }

  // Parse command line
  std::vector<std::string> inputDirs;
  outputDebugImage = false;
  outputFilename = "";
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '\0' && argv[i][1] != '\0' && argv[i][0] == '-') {
      switch (argv[i][1]) {
        case 'i': /* input-directory or input-file*/
          if (i >= argc - 1) {
            errLog << "The -i option takes a parameter." << std::endl;
            printUsage(argv[0], errLog);
            return -1;
          }
          ++i;
          while (argv[i][0] != '-') {
            inputDirs.push_back(argv[i]);
            i++;
            if (i == argc) {
              break;
            }
          }
          i--;
          break;
        case 'o': /* output-file name*/
          outputFilename = argv[++i];
          break;
        case 'd': /* output debug images*/
          outputDebugImage = true;
          break;
        case 's':
          algoSettingFilename = argv[++i];
          break;
        default:
          errLog << "No such option: " << argv[i] << std::endl << std::endl;
          printUsage(argv[0], errLog);
          return -1;
      }
    } else {
      errLog << "No such option: " << argv[i] << std::endl;
      printUsage(argv[0], errLog);
      return -1;
    }
  }

#ifdef _MSC_VER
  const std::vector<std::string> exts = {std::string("png"), std::string("jpeg"), std::string("jpg")};
#else
  const std::vector<std::string> exts = {std::string("png")};
#endif

  for (auto inputDir : inputDirs) {
    if (fileExists(inputDir)) {
      std::string strLower = inputDir;
      std::transform(strLower.begin(), strLower.end(), strLower.begin(), ::tolower);
      for (auto ext : exts) {
        if ((strLower.substr(strLower.size() - ext.size()) == ext)) {
          inputFilenames.push_back(inputDir);
          break;
        }
      }

    } else {
      for (auto ext : exts) {
        std::vector<std::string> newFilenames = listDirectory(inputDir, ext);
        inputFilenames.insert(inputFilenames.end(), newFilenames.begin(), newFilenames.end());
      }
    }
  }

  if (!inputFilenames.size()) {
    errLog << "Missing input file. Use -i for file or folder." << std::endl;
    printUsage(argv[0], errLog);
    return -1;
  }
  if (!outputFilename.size()) {
    errLog << "Missing output file. Use -o to specify the output file." << std::endl;
    printUsage(argv[0], errLog);
    return -1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> inputFilenames;
  std::string outputFilename;
  bool outputDebugImage;
  std::string algoSettingFilename;
  auto& errorLog = Logger::get(Logger::Error);

  // Extract command-line parameters
  if (extractArg(argc, argv, inputFilenames, outputFilename, outputDebugImage, algoSettingFilename) < 0) {
    return -1;
  }

  // Extract algorithm parameters
  std::unique_ptr<Ptv::Value> algoConfig = nullptr;
  if (algoSettingFilename.length() > 0) {
    auto& errLog = Logger::get(Logger::Error);
    Potential<Ptv::Parser> algoParser = Ptv::Parser::create();
    if (!algoParser->parse(algoSettingFilename)) {
      errLog << "Error: Cannot parse algos config PTV file: " << algoSettingFilename << std::endl;
      errLog << algoParser->getErrorMessage() << std::endl;
      return -1;
    }

    const Ptv::Value& algos = algoParser->getRoot();
    if (!(algos.has("algorithms") && algos.has("algorithms")->getType() == Ptv::Value::LIST)) {
      errLog << "Invalid algorithms file." << std::endl;
      return -1;
    }
    const std::vector<Ptv::Value*>& algoDefs = algos.has("algorithms")->asList();
    for (size_t i = 0; i < algoDefs.size(); ++i) {
      const Ptv::Value* algoDef = algoDefs[i];
      if (algoDef->has("name")->asString().compare("autocrop") == 0) {
        algoConfig.reset(algoDef->has("config")->clone());
        break;
      }
    }
  } else {
    algoConfig.reset(Ptv::Value::stringObject("fake"));
  }

  std::unique_ptr<Ptv::Value> jsonInputs(Ptv::Value::emptyObject());
  jsonInputs->asList();
  int success = 0;
  int fail = 0;
  for (auto filename : inputFilenames) {
    bool getCircle = true;
    // This likely is the visualization file --> skip it
    if (filename.find("_circle.png") != std::string::npos) {
      continue;
    }
    int64_t width, height;
    std::vector<unsigned char> data;
    readImage(filename, width, height, data);
    int x, y, radius;

    Status status = VideoStitch::Util::ImageProcessing::findCropCircle(
        (int)width, (int)height, &data[0], x, y, radius, algoConfig.get(), outputDebugImage ? &filename : nullptr);

    errorLog << "Input File '" << filename << "': ";
    if (status.ok()) {
      errorLog << "crop circle found!" << std::endl;
    } else {
      errorLog << "could not find crop circle!" << std::endl;
      getCircle = false;
    }

    Ptv::Value* res = Ptv::Value::emptyObject();
    res->push("reader_config", Ptv::Value::stringObject(filename));
    res->push("width", Ptv::Value::intObject(width));
    res->push("height", Ptv::Value::intObject(height));
    res->push("proj", Ptv::Value::stringObject(std::string("circular_fisheye_opt")));

    res->push("crop_left", Ptv::Value::intObject(x - radius));
    res->push("crop_right", Ptv::Value::intObject(x + radius));
    res->push("crop_top", Ptv::Value::intObject(y - radius));
    res->push("crop_bottom", Ptv::Value::intObject(y + radius));

    // Supplementary data
    res->push("center_x", Ptv::Value::intObject(x));
    res->push("center_y", Ptv::Value::intObject(y));
    res->push("radius", Ptv::Value::intObject(radius));
    if (outputDebugImage) {
      std::string outputFilePath = filename + "_circle.png";
      res->push("output_circle", Ptv::Value::stringObject(outputFilePath));
    }
    jsonInputs->asList().push_back(res);
    std::ofstream ofs(outputFilename);
    jsonInputs->printJson(ofs);
    if (ofs.fail()) {
      errorLog << "Could not write result to json file!" << std::endl;
      getCircle = false;
    }
    ofs.close();

    if (getCircle) {
      success++;
    } else {
      fail++;
    }
  }
  if (fail == 0) {
    std::cout << "Output Json file was written to: " << outputFilename << std::endl;
  } else if (success > 0) {
    std::cout << "Only " << success << " over " << success + fail << " results were written to: " << outputFilename
              << std::endl;
  } else {
    std::cout << "Failed to write result to: " << outputFilename << std::endl;
  }

  if (fail == 0) {
    return 0;
  } else {
    return -1;
  }
}
