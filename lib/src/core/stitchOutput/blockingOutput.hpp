// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "frameBuffer.hpp"
#include "stereoOutput.hpp"
#include "stitchOutput.hpp"

#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/controller.hpp"
#include "libvideostitch/output.hpp"

#include <memory>
#include <vector>

namespace VideoStitch {
namespace Core {

/**
 * A StitchOutput implementation that writes a video frame synchronously to a writer.
 *
 * Not thread safe.
 */
class BlockingSourceOutput : public ExtractOutput::Pimpl, WriterPusher<SourceFrameBuffer> {
 public:
  static Potential<BlockingSourceOutput> create(std::shared_ptr<SourceSurface> surface,
                                                const std::vector<std::shared_ptr<SourceRenderer>>& renderers,
                                                const std::vector<std::shared_ptr<Output::VideoWriter>>& writers,
                                                int source) {
    std::unique_ptr<BlockingSourceOutput> bso(new BlockingSourceOutput(surface, renderers, writers, source));
    FAIL_RETURN(GPU::useDefaultBackendDevice());
    Potential<SourceFrameBuffer> frameBuffer = SourceFrameBuffer::create(surface, writers);
    FAIL_RETURN(frameBuffer.status());
    bso->delegate = frameBuffer.release();
    return bso.release();
  }

  virtual ~BlockingSourceOutput() { delete delegate; }

  Status pushVideo(mtime_t date) override;

  virtual GPU::Surface& acquireFrame(mtime_t date, GPU::Stream& stream) override;

  virtual bool setRenderers(const std::vector<std::shared_ptr<SourceRenderer>>& r) override {
    return WriterPusher<SourceFrameBuffer>::setRenderers(r);
  }
  virtual bool addRenderer(std::shared_ptr<SourceRenderer> r) override {
    return WriterPusher<SourceFrameBuffer>::addRenderer(r);
  }
  virtual bool removeRenderer(const std::string& id) override {
    return WriterPusher<SourceFrameBuffer>::removeRenderer(id);
  }
  virtual bool setWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>& ws) override {
    return WriterPusher<SourceFrameBuffer>::setWriters(ws);
  }
  virtual bool addWriter(std::shared_ptr<Output::VideoWriter> w) override {
    return WriterPusher<SourceFrameBuffer>::addWriter(w);
  }
  virtual bool removeWriter(const std::string& id) override {
    return WriterPusher<SourceFrameBuffer>::removeWriter(id);
  }
  virtual bool updateWriter(const std::string& id, const Ptv::Value& config) override {
    return WriterPusher<SourceFrameBuffer>::updateWriter(id, config);
  }

 private:
  BlockingSourceOutput(std::shared_ptr<SourceSurface> surface,
                       const std::vector<std::shared_ptr<SourceRenderer>>& renderers,
                       const std::vector<std::shared_ptr<Output::VideoWriter>>& writers, int source)
      : Pimpl(surface->getWidth(), surface->getHeight(), source),
        WriterPusher<SourceFrameBuffer>(surface->getWidth(), surface->getHeight(), writers) {
    setRenderers(renderers);
  }

  SourceFrameBuffer* delegate;
};

class BlockingStitchOutput : public StitchOutput::Pimpl, WriterPusher<PanoFrameBuffer> {
 public:
  static Potential<BlockingStitchOutput> create(std::shared_ptr<PanoSurface> surface,
                                                const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                                                const std::vector<std::shared_ptr<Output::VideoWriter>>& writers) {
    std::unique_ptr<BlockingStitchOutput> bso(new BlockingStitchOutput(surface, renderers, writers));
    FAIL_RETURN(GPU::useDefaultBackendDevice());
    Potential<PanoFrameBuffer> frameBuffer = PanoFrameBuffer::create(surface, writers);
    FAIL_RETURN(frameBuffer.status());
    bso->delegate = frameBuffer.release();
    return bso.release();
  }

  virtual ~BlockingStitchOutput() { delete delegate; }

  Status pushVideo(mtime_t date) override;

  virtual PanoSurface& acquireFrame(mtime_t) override;

  virtual bool setRenderers(const std::vector<std::shared_ptr<PanoRenderer>>& r) override {
    return WriterPusher<PanoFrameBuffer>::setRenderers(r);
  }
  virtual bool addRenderer(std::shared_ptr<PanoRenderer> r) override {
    return WriterPusher<PanoFrameBuffer>::addRenderer(r);
  }
  virtual bool removeRenderer(const std::string& id) override {
    return WriterPusher<PanoFrameBuffer>::removeRenderer(id);
  }
  virtual void setCompositor(const std::shared_ptr<GPU::Overlayer>& c) override {
    WriterPusher<PanoFrameBuffer>::setCompositor(c);
  }
  virtual bool setWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>& ws) override {
    return WriterPusher<PanoFrameBuffer>::setWriters(ws);
  }
  virtual bool addWriter(std::shared_ptr<Output::VideoWriter> w) override {
    return WriterPusher<PanoFrameBuffer>::addWriter(w);
  }
  virtual bool removeWriter(const std::string& id) override { return WriterPusher<PanoFrameBuffer>::removeWriter(id); }
  virtual bool updateWriter(const std::string& id, const Ptv::Value& config) override {
    return WriterPusher<PanoFrameBuffer>::updateWriter(id, config);
  }

 private:
  BlockingStitchOutput(std::shared_ptr<PanoSurface> surface,
                       const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                       const std::vector<std::shared_ptr<Output::VideoWriter>>& writers)
      : Pimpl(surface->getWidth(), surface->getHeight()),
        WriterPusher<PanoFrameBuffer>(surface->getWidth(), surface->getHeight(), writers) {
    setRenderers(renderers);
  }

  PanoFrameBuffer* delegate;
};

/**
 * @brief A StereoOutput that writes a stereoscopic frame synchronously to a writer.
 *
 * Not thread safe.
 */
class BlockingStereoOutput : public StitcherOutput<Output::StereoWriter>::Pimpl, StereoWriterPusher {
 public:
  static Potential<BlockingStereoOutput> create(std::shared_ptr<PanoSurface> surf,
                                                const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                                                const std::vector<std::shared_ptr<Output::StereoWriter>>& writers);

  virtual ~BlockingStereoOutput() {
    delete left;
    delete right;
  }

  Status pushVideo(mtime_t date, Eye eye) override;

  virtual PanoSurface& acquireLeftFrame(mtime_t) override;
  virtual PanoSurface& acquireRightFrame(mtime_t) override;

  virtual bool setRenderers(const std::vector<std::shared_ptr<PanoRenderer>>& r) override {
    return StereoWriterPusher::setRenderers(r);
  }
  virtual bool addRenderer(std::shared_ptr<PanoRenderer> r) override { return StereoWriterPusher::addRenderer(r); }
  virtual bool removeRenderer(const std::string& id) override { return StereoWriterPusher::removeRenderer(id); }
  virtual bool setWriters(const std::vector<std::shared_ptr<Output::StereoWriter>>& ws) override {
    return StereoWriterPusher::setWriters(ws);
  }
  virtual bool addWriter(std::shared_ptr<Output::StereoWriter> w) override { return StereoWriterPusher::addWriter(w); }
  virtual bool removeWriter(const std::string& id) override { return StereoWriterPusher::removeWriter(id); }
  virtual bool updateWriter(const std::string& id, const Ptv::Value& config) override {
    return StereoWriterPusher::updateWriter(id, config);
  }

 protected:
  BlockingStereoOutput(std::shared_ptr<PanoSurface> surf, const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
                       const std::vector<std::shared_ptr<Output::StereoWriter>>& writers)
      : Pimpl(surf->getWidth(), surf->getHeight()), StereoWriterPusher(surf->getWidth(), surf->getHeight(), writers) {
    setRenderers(renderers);
  }

 private:
  StereoFrameBuffer* left;
  StereoFrameBuffer* right;
};
}  // namespace Core
}  // namespace VideoStitch
