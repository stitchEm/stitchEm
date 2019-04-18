// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/buffer.hpp"
#include "statefulReader.hxx"

namespace VideoStitch {
namespace Input {

template class StatefulReader<GPU::Buffer<unsigned char>>;

}
}  // namespace VideoStitch
