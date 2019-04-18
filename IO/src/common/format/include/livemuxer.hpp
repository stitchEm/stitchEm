// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "muxer.hpp"

#define LIVE_MAX_PACKETS 300
#define LIVE_TIMEOUT_MS 2000

namespace VideoStitch {
namespace Output {

class LiveMuxer : public Muxer {
 public:
  LiveMuxer(size_t index, const std::string& url, std::vector<AVEncoder>& codecs);
  virtual ~LiveMuxer();
  virtual void writeTrailer();
  virtual bool openResource(const std::string& url);
};

}  // namespace Output
}  // namespace VideoStitch
