// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "binaryCache.hpp"
#include "version.hpp"
#include <stdio.h>
#include <fstream>
#include "libvideostitch/dirutils.hpp"

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace VideoStitch {
namespace GPU {

const static std::string separator("/");
const static std::string gpuBackend("OpenCL");
const static std::string binaryExtension(".bin");

std::mutex mutexCreateDirectory;

// replace characters that could be invalid as path string
void nameSanitization(std::string& fileName) {
  for (char& ch : fileName) {
    if (!(isalpha(ch) || isdigit(ch) || (ch == '-'))) {
      ch = '_';
    }
  }
}

// returns cache Filename
PotentialValue<std::string> ProgramBinary::getFullFilePath(std::string devName, std::string driverVersion) {
  std::string version(std::string(LIB_VIDEOSTITCH_VERSION) + "-" + std::string(LIB_VIDEOSTITCH_BRANCH));
  // get the hash of the source
  auto hashed = std::hash<std::string>()(std::string(src, srcLength));
  std::string fileName = programName + std::string("-") + std::to_string(hashed);
  PotentialValue<std::string> potDirectoryName = Util::getVSCacheLocation();
  if (!potDirectoryName.ok()) {
    return potDirectoryName;
  }
  std::string directoryName = potDirectoryName.value();
  nameSanitization(version);
  nameSanitization(fileName);
  nameSanitization(devName);
  nameSanitization(driverVersion);

  mutexCreateDirectory.lock();

#if defined(_WIN32)
  _mkdir(directoryName.c_str());
  directoryName += separator + gpuBackend;
  _mkdir(directoryName.c_str());
  directoryName += separator + version;
  _mkdir(directoryName.c_str());
  directoryName += separator + devName;
  _mkdir(directoryName.c_str());
  directoryName += separator + driverVersion;
  _mkdir(directoryName.c_str());
#else
  mkdir(directoryName.c_str(), 0755);
  directoryName += separator + gpuBackend;
  mkdir(directoryName.c_str(), 0755);
  directoryName += separator + version;
  mkdir(directoryName.c_str(), 0755);
  directoryName += separator + devName;
  mkdir(directoryName.c_str(), 0755);
  directoryName += separator + driverVersion;
  mkdir(directoryName.c_str(), 0755);
#endif
  mutexCreateDirectory.unlock();

  const std::string fullFilePath = directoryName + separator + fileName + binaryExtension;
  return std::string(fullFilePath);
}

// Returns the binary for the program.
std::vector<unsigned char> ProgramBinary::getFinalBinary(cl_program& program) {
  size_t binary_size;
  clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size), &binary_size, NULL);
  std::vector<unsigned char> finalBinary(binary_size);
  unsigned char* binary_ptr = &finalBinary[0];
  clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char**), &binary_ptr, 0);

  return finalBinary;
}

/* Save the compiled program binary to a file for later reuse.  */
void ProgramBinary::cacheSave(cl_program& program, const char* devName, const char* driverVersion) {
  PotentialValue<std::string> potFileName = getFullFilePath(devName, driverVersion);
  if (!potFileName.ok()) {
    Logger::get(Logger::Warning) << "[OpenCL] Can not get Cache location " << std::endl;
    return;
  }
  std::string fname = potFileName.value();
  std::ofstream bfile(fname.c_str(), std::ios::binary);
  if (!bfile) {
    // unable to write into cache
    Logger::get(Logger::Warning) << "[OpenCL] Unable to store program binary into cache for program " << programName
                                 << std::endl;
    return;
  }
  std::vector<unsigned char> finalBinary = getFinalBinary(program);
  bfile.write((char*)finalBinary.data(), finalBinary.size());
  bfile.close();
}

/* Try to load the cached compiled program binary, returns true or false depending if found or not */
bool ProgramBinary::cacheLoad(cl_program& program, const cl_device_id& device_id, const char* devName,
                              const char* driverVersion, const OpenCLContext& context) {
  PotentialValue<std::string> fname = getFullFilePath(devName, driverVersion);
  if (!fname.ok()) {
    Logger::get(Logger::Warning) << "[OpenCL] Can not get Cache location " << std::endl;
    program = nullptr;
    return false;
  }
  std::ifstream bfile(fname.value().c_str(), std::ios::binary);
  if (!bfile) {
    // program not in cache
    program = nullptr;
    return false;
  }

  // get size of file:
  bfile.seekg(0, bfile.end);
  size_t binary_size = bfile.tellg();
  bfile.seekg(0, bfile.beg);

  // file size has to be checked before calling clcreateprogramwithbinary. Otherwize it can crash
  if (binary_size == 0) {
    // delete the empty cache file
    Logger::get(Logger::Warning) << "[OpenCL] Empty OpenCL cache file. Deleting it " << std::endl;
    remove(fname.value().c_str());
    program = nullptr;
    return false;
  }
  std::vector<unsigned char> finalBinary(binary_size);
  bfile.read((char*)finalBinary.data(), binary_size);
  bfile.close();
  cl_int err, binary_status;
  const unsigned char* data = finalBinary.data();
  program = clCreateProgramWithBinary(context, 1, &device_id, &binary_size, &data, &binary_status, &err);
  if (binary_status == CL_INVALID_BINARY) {
    // delete the invalid file
    Logger::get(Logger::Warning) << "[OpenCL] Invalid binary in cache. Deleting it " << std::endl;
    remove(fname.value().c_str());
    program = nullptr;
    return false;
  }
  if (err != CL_SUCCESS) {
    Logger::get(Logger::Warning) << "[OpenCL] Can not create program with binary in cache " << std::endl;
    program = nullptr;
    return false;
  }
  return true;
}

}  // namespace GPU
}  // namespace VideoStitch
