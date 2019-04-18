// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
namespace VideoStitch {
namespace IO {

template <typename T>
class VS_EXPORT Sink : public virtual T {
 public:
  explicit Sink();

  virtual ~Sink();
  virtual Status addSink(const Ptv::Value* /* config */, const mtime_t /* videoTimeStamp */,
                         const mtime_t /* audioTimeStamp */);

  virtual void removeSink();
};

}  // namespace IO
}  // namespace VideoStitch
