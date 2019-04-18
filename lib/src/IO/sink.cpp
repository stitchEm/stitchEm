// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/input.hpp"
#include "libvideostitch/sink.hpp"

namespace VideoStitch {
namespace IO {

template <typename T>
Sink<T>::Sink() : T(-1) {  // never called
}
template <typename T>
Sink<T>::~Sink() {}

template <typename T>
Status Sink<T>::addSink(const Ptv::Value* /* config */, const mtime_t /* videoTimeStamp */,
                        const mtime_t /* audioTimeStamp */) {
  return {Origin::Output, ErrType::UnsupportedAction, "Sink not implemented for this Reader"};
}

template <typename T>
void Sink<T>::removeSink() {}

template class Sink<Input::Reader>;

}  // namespace IO
}  // namespace VideoStitch
