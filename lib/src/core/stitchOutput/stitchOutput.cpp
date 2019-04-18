// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stitchOutput.hpp"

#include "common/container.hpp"

#include "libvideostitch/overlay.hpp"

namespace VideoStitch {
namespace Core {
/**
 * ExtractOutput implementation
 */
ExtractOutput::ExtractOutput(Pimpl* pimplVal) : pimpl(pimplVal) {}

ExtractOutput::~ExtractOutput() { delete pimpl; }

bool ExtractOutput::setRenderers(const std::vector<std::shared_ptr<SourceRenderer>>& r) {
  return pimpl->setRenderers(r);
}
bool ExtractOutput::addRenderer(std::shared_ptr<SourceRenderer> r) { return pimpl->addRenderer(r); }
bool ExtractOutput::removeRenderer(const std::string& name) { return pimpl->removeRenderer(name); }

bool ExtractOutput::setWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>& w) {
  return pimpl->setWriters(w);
}
bool ExtractOutput::addWriter(std::shared_ptr<Output::VideoWriter> w) { return pimpl->addWriter(w); }
bool ExtractOutput::removeWriter(const std::string& name) { return pimpl->removeWriter(name); }
bool ExtractOutput::updateWriter(const std::string& name, const Ptv::Value& config) {
  return pimpl->updateWriter(name, config);
}

/**
 * StitchOutput implementation
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
void StitcherOutput<Writer>::setCompositor(const std::shared_ptr<GPU::Overlayer>& c) {
  pimpl->setCompositor(c);
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

template class StitcherOutput<Output::VideoWriter>;

/**
 * PanoWriterPusher implementation
 */
template <typename FrameBuffer>
WriterPusher<FrameBuffer>::WriterPusher(size_t w, size_t /*h*/,
                                        const std::vector<std::shared_ptr<Output::VideoWriter>>& writersIn)
    : width(w), compositor(nullptr) {
  setWriters(writersIn);
}

template <typename FrameBuffer>
WriterPusher<FrameBuffer>::~WriterPusher() {
  {
    std::lock_guard<std::mutex> lk(writersLock);
    writers.clear();
  }
  {
    std::lock_guard<std::mutex> lk(renderersLock);
    renderers.clear();
  }
}

template <typename FrameBuffer>
bool WriterPusher<FrameBuffer>::setRenderers(
    const std::vector<std::shared_ptr<typename FrameBuffer::Renderer>>& newRenderers) {
  std::lock_guard<std::mutex> lk(renderersLock);
  // delete the old renderers
  renderers.clear();
  // register the new ones
  bool res = true;
  for (auto& r : newRenderers) {
    if (!r) {
      continue;
    }
    auto p = renderers.insert(std::make_pair(r->getName(), std::shared_ptr<typename FrameBuffer::Renderer>(r)));
    if (!p.second) {
      res = false;
    }
  }
  return res;
}

template <typename FrameBuffer>
bool WriterPusher<FrameBuffer>::addRenderer(std::shared_ptr<typename FrameBuffer::Renderer> r) {
  std::lock_guard<std::mutex> lk(renderersLock);
  auto res = renderers.insert(std::make_pair(r->getName(), r));
  if (!res.second) {
    return false;
  }
  return true;
}

template <typename FrameBuffer>
bool WriterPusher<FrameBuffer>::removeRenderer(const std::string& name) {
  {
    std::lock_guard<std::mutex> lk(renderersLock);
    auto renderer = renderers.find(name);
    if (renderer == renderers.end()) {
      return false;
    }
    renderers.erase(name);
  }
  return true;
}

template <typename FrameBuffer>
void WriterPusher<FrameBuffer>::setCompositor(const std::shared_ptr<GPU::Overlayer>& c) {
  std::lock_guard<std::mutex> lk(compositorLock);
  compositor = c;
}

template <typename FrameBuffer>
bool WriterPusher<FrameBuffer>::setWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>& newWriters) {
  std::lock_guard<std::mutex> lk(writersLock);
  // delete the old writers
  writers.clear();
  // register the new ones
  bool r = true;
  for (auto& w : newWriters) {
    if (!w) {
      continue;
    }
    auto res = writers.insert(std::make_pair(w->getName(), w));
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

template <typename FrameBuffer>
bool WriterPusher<FrameBuffer>::addWriter(std::shared_ptr<Output::VideoWriter> w) {
  std::lock_guard<std::mutex> lk(writersLock);
  auto res = writers.insert(std::make_pair(w->getName(), w));
  if (!res.second) {
    return false;
  }
  downsamplingFactors[w->getName()] =
      (int)width % (int)w->getPanoWidth() == 0 ? (int)width / (int)w->getPanoWidth() : 1;
  return true;
}

template <typename FrameBuffer>
bool WriterPusher<FrameBuffer>::removeWriter(const std::string& name) {
  {
    std::lock_guard<std::mutex> lk(writersLock);
    auto writer = writers.find(name);
    if (writer == writers.end()) {
      return false;
    }
    writers.erase(name);
  }
  return true;
}

template <typename FrameBuffer>
bool WriterPusher<FrameBuffer>::updateWriter(const std::string& name, const Ptv::Value& config) {
  std::lock_guard<std::mutex> lk(writersLock);
  auto writer = writers.find(name);
  if (writer == writers.end()) {
    return false;
  }
  writer->second->updateConfig(config);
  return true;
}

template <typename FrameBuffer>
void WriterPusher<FrameBuffer>::pushVideoToWriters(mtime_t date, FrameBuffer* delegate) const {
  {
    std::lock_guard<std::mutex> lk(compositorLock);
    if (compositor) {
      compositor->attachContext();
      if (!delegate->getOpenGLSurface()) {
        assert(false);
      }
      delegate->getSurface()->accept(compositor, delegate->getOpenGLSurface(), date);
      delegate->pushOpenGLVideo();
      delegate->streamOpenGLSynchronize();
      compositor->detachContext();

      for (auto& r : renderers) {
        delegate->getOpenGLSurface()->accept(r.second, date);
      }
    } else {
      delegate->pushVideo();
      delegate->streamSynchronize();

      std::lock_guard<std::mutex> lk(renderersLock);
      for (auto& r : renderers) {
        delegate->getSurface()->accept(r.second, date);
      }
    }
  }

  std::lock_guard<std::mutex> lk(writersLock);
  for (auto& w : writers) {
    auto dsf = downsamplingFactors.find(w.first);
    assert(dsf != downsamplingFactors.end());
    Frame frame = delegate->getFrame(w.second->getPixelFormat(), w.second->getExpectedOutputBufferType(), dsf->second);
    frame.pts = date;
    w.second->pushVideo(frame);
  }
}

template <>
void WriterPusher<SourceFrameBuffer>::pushVideoToWriters(mtime_t date, SourceFrameBuffer* delegate) const {
  delegate->pushVideo();
  delegate->streamSynchronize();

  {
    std::lock_guard<std::mutex> lk(renderersLock);
    for (auto& r : renderers) {
      delegate->getSurface()->accept(r.second, date);
    }
  }

  std::lock_guard<std::mutex> lk(writersLock);
  for (auto& w : writers) {
    auto dsf = downsamplingFactors.find(w.first);
    assert(dsf != downsamplingFactors.end());
    Frame frame = delegate->getFrame(w.second->getPixelFormat(), w.second->getExpectedOutputBufferType(), dsf->second);
    frame.pts = date;
    w.second->pushVideo(frame);
  }
}

template class WriterPusher<PanoFrameBuffer>;
template class WriterPusher<SourceFrameBuffer>;
}  // namespace Core
}  // namespace VideoStitch
