// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stereoOutput.hpp"

#include "common/container.hpp"

namespace VideoStitch {
namespace Core {
/**
 * StereoOutput implementation
 */
template <typename Writer>
StitcherOutput<Writer>::StitcherOutput(Pimpl* pimplVal) : pimpl(pimplVal) {}

template <typename Writer>
StitcherOutput<Writer>::~StitcherOutput() {
  delete pimpl;
}

template <typename Writer>
bool StitcherOutput<Writer>::setRenderers(const std::vector<std::shared_ptr<PanoRenderer>>& r) {
  return pimpl->setRenderers(r);
}
template <typename Writer>
bool StitcherOutput<Writer>::addRenderer(std::shared_ptr<PanoRenderer> r) {
  return pimpl->addRenderer(r);
}
template <typename Writer>
bool StitcherOutput<Writer>::removeRenderer(const std::string& name) {
  return pimpl->removeRenderer(name);
}

template <typename Writer>
bool StitcherOutput<Writer>::setWriters(const std::vector<std::shared_ptr<Writer>>& w) {
  return pimpl->setWriters(w);
}
template <typename Writer>
bool StitcherOutput<Writer>::addWriter(std::shared_ptr<Writer> w) {
  return pimpl->addWriter(w);
}
template <typename Writer>
bool StitcherOutput<Writer>::removeWriter(const std::string& name) {
  return pimpl->removeWriter(name);
}
template <typename Writer>
bool StitcherOutput<Writer>::updateWriter(const std::string& name, const Ptv::Value& config) {
  return pimpl->updateWriter(name, config);
}

template class StitcherOutput<Output::StereoWriter>;

/**
 * StereoWriterPusher implementation
 */
StereoWriterPusher::StereoWriterPusher(size_t w, size_t /*h*/,
                                       const std::vector<std::shared_ptr<Output::StereoWriter>>& writersIn)
    : width(w) {
  setWriters(writersIn);
}

StereoWriterPusher::~StereoWriterPusher() {
  std::unique_lock<std::mutex> lk(writersLock);
  //  deleteAllValues(writers);
  writers.clear();
}

bool StereoWriterPusher::setWriters(const std::vector<std::shared_ptr<Output::StereoWriter>>& newWriters) {
  std::unique_lock<std::mutex> lk(writersLock);
  // delete the old writers
  //  deleteAllValues(writers);
  writers.clear();
  // register the new ones
  bool r = true;
  for (auto& w : newWriters) {
    auto res = writers.insert(std::make_pair(w->getName(), std::shared_ptr<Output::StereoWriter>(w)));
    if (!res.second) {
      r = false;
    }
  }
  // downsampling setup
  downsamplingFactors.clear();
  for (auto& w : writers) {
    assert((int)width % (int)w.second->getPanoWidth() ==
           0);  // This should have been caught in Writer::create. Crash in debug.
    downsamplingFactors[w.first] =
        (int)width % (int)w.second->getPanoWidth() == 0 ? (int)width / (int)w.second->getPanoWidth() : 1;
  }
  return r;
}

bool StereoWriterPusher::addWriter(std::shared_ptr<Output::StereoWriter> w) {
  std::unique_lock<std::mutex> lk(writersLock);
  auto res = writers.insert(std::make_pair(w->getName(), w));
  if (!res.second) {
    return false;
  }
  downsamplingFactors[w->getName()] =
      (int)width % (int)w->getPanoWidth() == 0 ? (int)width / (int)w->getPanoWidth() : 1;
  return true;
}

bool StereoWriterPusher::removeWriter(const std::string& name) {
  std::unique_lock<std::mutex> lk(writersLock);
  auto w = writers.find(name);
  if (w == writers.end()) {
    return false;
  }
  //  delete w->second;
  writers.erase(name);
  return true;
}

bool StereoWriterPusher::updateWriter(const std::string& name, const Ptv::Value& config) {
  std::unique_lock<std::mutex> lk(writersLock);
  auto w = writers.find(name);
  if (w == writers.end()) {
    return false;
  }
  w->second->updateConfig(config);
  return true;
}

void StereoWriterPusher::pushVideoToWriters(mtime_t /*date*/,
                                            std::pair<StereoFrameBuffer*, StereoFrameBuffer*> /*buffer*/) const {
  std::unique_lock<std::mutex> lk(writersLock);
  for (auto& w : writers) {
    // auto dsf = downsamplingFactors.find(w.first);
    switch (w.second->getExpectedOutputBufferType()) {
      case Host: {
        /* XXX TODO FIXME
        auto leftBuf = buffer.first->getBuffer(w.second->getPixelFormat(), dsf->second);
        auto rightBuf = buffer.second->getBuffer(w.second->getPixelFormat(), dsf->second);
        w.second->pushEye(date,
                          Output::LeftEye,
                          (const char *)leftBuf);
        w.second->pushEye(date,
                          Output::RightEye,
                          (const char *)rightBuf);
                          */
        break;
      }
      case Device: {
        // XXX TODO FIXME
        /*
        auto leftArr = buffer.first->getArray(w.second->getPixelFormat(), dsf->second);
        auto rightArr = buffer.second->getArray(w.second->getPixelFormat(), dsf->second);
        */
        break;
      }
    }
  }
}
}  // namespace Core
}  // namespace VideoStitch
