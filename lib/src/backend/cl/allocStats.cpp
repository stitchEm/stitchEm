// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../common/allocStats.hpp"
#include "gpu/buffer.hpp"

namespace VideoStitch {

AllocStatsMap deviceStats("Device OpenCL");
AllocStatsMap hostStats("Host OpenCL");

void printBufferPoolStats() { deviceStats.print(std::cout); }

std::size_t getBufferPoolCurrentSize() { return deviceStats.bytesUsed(); }

std::vector<std::size_t> getBufferPoolCurrentSizeByDevices() { return deviceStats.bytesUsedByDevices(); }
}  // namespace VideoStitch
