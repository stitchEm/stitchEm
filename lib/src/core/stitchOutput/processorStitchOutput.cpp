// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/processorStitchOutput.hpp"

#include "stitchOutput.hpp"

#include "gpu/image/reduce.hpp"

namespace VideoStitch {
namespace Core {

namespace {

/**
 * A generic filter for MultiProcessorStitchOutput;
 */
class Functor {
 public:
  virtual ~Functor() {}

  /**
   * Returns the name of the output variable.
   * @param panoDevBuffer input buffer
   * @param devWork work buffer, size requiresDevWorkBuffer().
   * @param width Input width
   * @param input Input heigth.
   * @param value value object, for output.
   */
  virtual Status apply(GPU::Buffer<const uint32_t> panoDevBuffer, GPU::Buffer<char> devWork, const int64_t width,
                       const int64_t height, Ptv::Value* value) const = 0;

  /**
   * Returns the required memory size.
   * @param width Input width
   * @param input Input heigth.
   */
  virtual int64_t requiresDevWorkBuffer(int64_t width, int64_t height) const = 0;
};

class SumFunctor : public Functor {
 public:
  int64_t requiresDevWorkBuffer(int64_t width, int64_t height) const {
    return sizeof(uint32_t) * Image::getReduceWorkBufferSize(width * height);
  }

  Status apply(GPU::Buffer<const uint32_t> panoDevBuffer, GPU::Buffer<char> devWork, const int64_t width,
               const int64_t height, Ptv::Value* value) const {
    uint32_t sum = 0;
    const Status status = Image::reduceSumSolid(panoDevBuffer, devWork.as<uint32_t>(), width * height, sum);
    if (status.ok()) {
      value->get("sum")->asInt() = sum;
    }
    return status;
  }
};

class CountFunctor : public Functor {
 public:
  int64_t requiresDevWorkBuffer(int64_t width, int64_t height) const {
    return sizeof(uint32_t) * Image::getReduceWorkBufferSize(width * height);
  }

  Status apply(GPU::Buffer<const uint32_t> panoDevBuffer, GPU::Buffer<char> devWork, const int64_t width,
               const int64_t height, Ptv::Value* value) const {
    uint32_t count = 0;
    const Status status = Image::reduceCountSolid(panoDevBuffer, devWork.as<uint32_t>(), width * height, count);
    if (status.ok()) {
      value->get("count")->asInt() = count;
    }
    return status;
  }
};

/**
 * A ProcessorStitchOutput that applies a bunch of functors to the output;
 */
class MultiProcessorStitchOutput : public ProcessorStitchOutput {
 public:
  class Pimpl : public StitchOutput::Pimpl {
   public:
    Pimpl(size_t width, size_t height, const std::vector<Functor*>& functors)
        : StitchOutput::Pimpl(width, height), value(Ptv::Value::emptyObject()), functors(functors) {
      surf = OffscreenAllocator::createPanoSurface(width, height, "processor stitch output").release();
      int64_t maxDevWorkBufferSize = 0;
      for (size_t i = 0; i < functors.size(); ++i) {
        const int64_t size = functors[i]->requiresDevWorkBuffer(width, height);
        if (size > maxDevWorkBufferSize) {
          maxDevWorkBufferSize = size;
        }
      }
      devWork.alloc(maxDevWorkBufferSize, "MultiProcessorStitchOutput");
    }

    ~Pimpl() {
      delete value;
      for (size_t i = 0; i < functors.size(); ++i) {
        delete functors[i];
      }
      delete surf;
    }

    Status pushVideo(mtime_t /*frame*/) {
      // Reset the value.
      value->asNil();
      value->asObject();
      // Apply functors.
      for (size_t i = 0; i < functors.size(); ++i) {
        PROPAGATE_FAILURE_STATUS(functors[i]->apply(surf->pimpl->buffer, devWork.borrow(), width, height, value));
      }
      return Status::OK();
    }

    bool setRenderers(const std::vector<std::shared_ptr<PanoRenderer>>&) { return false; }
    bool addRenderer(std::shared_ptr<PanoRenderer>) { return false; }
    bool removeRenderer(const std::string&) { return false; }
    void setCompositor(const std::shared_ptr<GPU::Overlayer>&) {}
    bool setWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>&) { return false; }
    bool addWriter(std::shared_ptr<Output::VideoWriter>) { return false; }
    bool removeWriter(const std::string&) { return false; }
    bool updateWriter(const std::string&, const Ptv::Value&) { return false; }

    virtual PanoSurface& acquireFrame(mtime_t) { return *surf; }

   private:
    Ptv::Value* const value;
    std::vector<Functor*> functors;

    PanoSurface* surf;
    GPU::UniqueBuffer<char> devWork;  // temporary work buffer.

    friend class MultiProcessorStitchOutput;
  };

  MultiProcessorStitchOutput(size_t w, size_t h, const std::vector<Functor*>& functors)
      : ProcessorStitchOutput(new Pimpl(w, h, functors)) {}

  ~MultiProcessorStitchOutput() {}

  const Ptv::Value& getResult() const { return *static_cast<Pimpl*>(pimpl)->value; }
};
}  // namespace

ProcessorStitchOutput::ProcessorStitchOutput(Pimpl* pimpl) : StitchOutput(pimpl) {}

ProcessorStitchOutput::Spec::Spec() : sum(false), count(false) {}

ProcessorStitchOutput::Spec& ProcessorStitchOutput::Spec::withSum() {
  sum = true;
  return *this;
}

ProcessorStitchOutput::Spec& ProcessorStitchOutput::Spec::withCount() {
  count = true;
  return *this;
}

Potential<ProcessorStitchOutput> ProcessorStitchOutput::create(size_t w, size_t h, const Spec& spec) {
  std::vector<Functor*> functors;
  if (spec.sum) {
    functors.push_back(new SumFunctor());
  }
  if (spec.count) {
    functors.push_back(new CountFunctor());
  }
  if (functors.size()) {
    return new MultiProcessorStitchOutput(w, h, functors);
  }
  return Status{Origin::Stitcher, ErrType::ImplementationError, "No functor"};
}
}  // namespace Core
}  // namespace VideoStitch
