// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "context.hpp"

namespace VideoStitch {
namespace GPU {

class ProgramBinary {
 public:
  ProgramBinary(const std::string& programName, const char* src, const size_t srcLength)
      : programName(programName), src(src), srcLength(srcLength) {}

  bool cacheLoad(cl_program& program, const cl_device_id& device_id, const char* devName, const char* driverVersion,
                 const OpenCLContext& context);
  void cacheSave(cl_program& program, const char* devName, const char* driverVersion);

 private:
  const std::string programName;
  const char* src;
  const size_t srcLength;

  PotentialValue<std::string> getFullFilePath(std::string devName, std::string driverVersion);
  std::vector<unsigned char> getFinalBinary(cl_program& program);
};

}  // namespace GPU
}  // namespace VideoStitch
