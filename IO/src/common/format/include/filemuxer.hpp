// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "muxer.hpp"

namespace VideoStitch {
namespace Output {

class FileMuxer : public Muxer {
 public:
  explicit FileMuxer(size_t index, const std::string& format, const std::string& filename,
                     std::vector<AVEncoder>& codecs, const AVDictionary*);
  ~FileMuxer();
  virtual void writeTrailer();
  virtual bool openResource(const std::string& filename);

 private:
  bool MP4WebOptimizerInternal(const std::string&);
  bool reserved_moov;
};

}  // namespace Output
}  // namespace VideoStitch
