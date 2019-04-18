// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef FAKE_TRANSFORM_HPP_
#define FAKE_TRANSFORM_HPP_

#include "util.hpp"
#include "testing.hpp"

#include <core/rect.hpp>
#include <core/photoTransform.hpp>
#include <gpu/core1/transform.hpp>
#include <backend/cuda/deviceBuffer.hpp>
#include <backend/cuda/deviceStream.hpp>

namespace VideoStitch {
namespace Testing {

/**
 * A fake transform.
 */
class FakeTransform : public Core::Transform {
 public:
  /**
   * Creates a fake transform that expects an output of a given size and will fill it in with the provided data.
   * @param fakeOutputWidth Fake output width
   * @param fakeOutputHeight Fake output height
   * @param fakeOutputData The fake data. RGB210.
   */
  FakeTransform(int64_t fakeOutputWidth, int64_t fakeOutputHeight, const uint32_t* fakeOutputData)
      : Transform(),
        fakeOutputWidth(fakeOutputWidth),
        fakeOutputHeight(fakeOutputHeight),
        fakeOutputData(fakeOutputData),
        photo() {}

  Status computeZone(GPU::Buffer<uint32_t> /*devOut*/, const Core::PanoDefinition& /*pano*/,
                     const Core::InputDefinition& /*im*/, unsigned /*imId*/,
                     GPU::Buffer<const unsigned char> /*maskDevBuffer*/, GPU::Stream /*stream*/) const {
    return Status::OK();
  }
  Status cubemapMap(GPU::Buffer<uint32_t>, const PanoDefinition&, const InputDefinition&, videoreaderid_t,
                    GPU::Buffer<const unsigned char>, GPU::Stream) {
    return Status::OK();
  }

  Status mapBufferLookup(frameid_t, GPU::Buffer<uint32_t>, const GPU::Buffer<const float2>&, const Core::Rect&,
                         const Core::PanoDefinition&, const Core::InputDefinition&, GPU::Cached2DBuffer<const uint32_t>,
                         const GPU::ChannelFormatDescription&, GPU::Stream) const {
    return Status::OK();
  }

  Status mapBufferCoord(frameid_t, GPU::Buffer<float2>, const Core::Rect&, const Core::PanoDefinition&,
                        const Core::InputDefinition&, GPU::Stream) const {
    return Status::OK();
  }

  Status mapBuffer(int /*time*/, GPU::Buffer<uint32_t> devOut, const Core::Rect& outputBounds,
                   const Core::PanoDefinition& /*pano*/, const Core::InputDefinition& /*im*/,
                   GPU::Cached2DBuffer<const uint32_t> /*texDevice*/, const cudaChannelFormatDesc& /*channelDesc*/,
                   GPU::Stream stream) const {
    ENSURE_EQ(fakeOutputWidth, outputBounds.getWidth());
    ENSURE_EQ(fakeOutputHeight, outputBounds.getHeight());
    ENSURE(Cuda::cudaStatus(cudaMemcpyAsync(devOut.get(), fakeOutputData,
                                            (size_t)(fakeOutputWidth * fakeOutputHeight * sizeof(uint32_t)),
                                            cudaMemcpyHostToDevice, stream.get())));
    return Status::OK();
  }

  Status warpCubemap(frameid_t, GPU::Buffer<uint32_t>, const Rect&, GPU::Buffer<uint32_t>, const Rect&,
                     GPU::Buffer<uint32_t>, const Rect&, GPU::Buffer<uint32_t>, const Rect&, GPU::Buffer<uint32_t>,
                     const Rect&, GPU::Buffer<uint32_t>, const Rect&, const PanoDefinition&, const InputDefinition&,
                     GPU::Surface&, GPU::Stream) const {
    return Status::OK();
  }

  Status mapDistortion(int /*time*/, GPU::Buffer<unsigned char> devOut, const Core::Rect& outputBounds,
                       const Core::PanoDefinition& /*pano*/, const Core::InputDefinition& /*im*/,
                       GPU::Stream stream) const {
    ENSURE_EQ(fakeOutputWidth, outputBounds.getWidth());
    ENSURE_EQ(fakeOutputHeight, outputBounds.getHeight());
    ENSURE(Cuda::cudaStatus(cudaMemcpyAsync(devOut.get(), fakeOutputData,
                                            (size_t)(fakeOutputWidth * fakeOutputHeight * sizeof(unsigned char)),
                                            cudaMemcpyHostToDevice, stream.get())));
    return Status::OK();
  }

  class FakePhotoTransform : public Core::PhotoTransform {
    float3 mapPhotoInputToLinear(const Core::InputDefinition& /*im*/, Core::TopLeftCoords2 /*uv*/,
                                 float3 /*rgb*/) const {
      return make_float3(0.0, 0.0, 0.0);
    }

    float3 mapPhotoLinearToPano(float3 /*rgb*/) const { return make_float3(0.0, 0.0, 0.0); }
    float3 mapPhotoPanoToLinear(float3 /*rgb*/) const { return make_float3(0.0, 0.0, 0.0); }

    const char* getDevicePhotoParam() const { return NULL; }
  };

  const Core::PhotoTransform* getPhoto() const { return &photo; }

 private:
  const int64_t fakeOutputWidth;
  const int64_t fakeOutputHeight;
  const uint32_t* const fakeOutputData;
  const FakePhotoTransform photo;
};

}  // namespace Testing
}  // namespace VideoStitch

#endif
