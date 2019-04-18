// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputsMapCubemap.hpp"

#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/core1/transform.hpp"

#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/inputDef.hpp"

//#define READBACKSETUPIMAGE
//#define READBACKSETUPIMAGEPERINPUT

#if defined(READBACKSETUPIMAGE) || defined(READBACKSETUPIMAGEPERINPUT)
#ifdef _MSC_VER
static const std::string DEBUG_FOLDER = "";
#else
static const std::string DEBUG_FOLDER = "/tmp/inputs/";
#endif
#include "cuda/error.hpp"
#include "util/pngutil.hpp"
#include "util/pnm.hpp"
#include "util/debugUtils.hpp"
#include "image/unpack.hpp"
#endif

namespace VideoStitch {
namespace Core {

Potential<InputsMapCubemap> InputsMapCubemap::create(const PanoDefinition& pano) {
  std::unique_ptr<InputsMapCubemap> inputsMap;
  inputsMap.reset(new InputsMapCubemap(pano));
  Status status = inputsMap->allocateBuffers();
  if (status.ok()) {
    return Potential<InputsMapCubemap>(inputsMap.release());
  } else {
    return Potential<InputsMapCubemap>(status);
  }
}

InputsMapCubemap::InputsMapCubemap(const PanoDefinition& pano) : length(pano.getLength()) {}

InputsMapCubemap::~InputsMapCubemap() {}

Status InputsMapCubemap::allocateBuffers() {
  FAIL_RETURN(xPos.alloc(length * length, "Setup Buffer"));
  FAIL_RETURN(xNeg.alloc(length * length, "Setup Buffer"));
  FAIL_RETURN(yPos.alloc(length * length, "Setup Buffer"));
  FAIL_RETURN(yNeg.alloc(length * length, "Setup Buffer"));
  FAIL_RETURN(zPos.alloc(length * length, "Setup Buffer"));
  FAIL_RETURN(zNeg.alloc(length * length, "Setup Buffer"));
  return Status::OK();
}

Status InputsMapCubemap::compute(const std::map<readerid_t, Input::VideoReader*>& readers, const PanoDefinition& pano) {
  FAIL_RETURN(GPU::memsetToZeroBlocking(xPos.borrow(), length * length * 4));
  FAIL_RETURN(GPU::memsetToZeroBlocking(xNeg.borrow(), length * length * 4));
  FAIL_RETURN(GPU::memsetToZeroBlocking(yPos.borrow(), length * length * 4));
  FAIL_RETURN(GPU::memsetToZeroBlocking(yNeg.borrow(), length * length * 4));
  FAIL_RETURN(GPU::memsetToZeroBlocking(zPos.borrow(), length * length * 4));
  FAIL_RETURN(GPU::memsetToZeroBlocking(zNeg.borrow(), length * length * 4));

  for (auto reader : readers) {
    const InputDefinition& inputDef = pano.getInput(reader.second->id);
    const size_t bufferSize = (size_t)(inputDef.getWidth() * inputDef.getHeight());

    // Create mask buffer
    GPU::UniqueBuffer<unsigned char> maskDevBuffer;
    FAIL_RETURN(maskDevBuffer.alloc(bufferSize, "MaskSetup"));
    const unsigned char* data = inputDef.getMaskPixelDataIfValid();
    if (data && inputDef.deletesMaskedPixels()) {
      FAIL_RETURN(GPU::memcpyBlocking(maskDevBuffer.borrow(), data, bufferSize));
    } else {
      FAIL_RETURN(GPU::memsetToZeroBlocking(maskDevBuffer.borrow(), bufferSize));
    }

    // Update assigned pixels
    Transform* t = Transform::create(inputDef);
    FAIL_RETURN(t->cubemapMap(xPos.borrow(), xNeg.borrow(), yPos.borrow(), yNeg.borrow(), zPos.borrow(), zNeg.borrow(),
                              pano, inputDef, reader.second->id, maskDevBuffer.borrow(),
                              pano.getProjection() == PanoProjection::EquiangularCubemap, GPU::Stream::getDefault()));
    FAIL_RETURN(GPU::Stream::getDefault().synchronize());
    delete t;

#ifdef READBACKSETUPIMAGEPERINPUT
    {
      {
        const int64_t length = pano.getLength();
        GPU::Stream::getDefault().synchronize();
        {
          std::stringstream ss;
          ss << DEBUG_FOLDER << "setup_face+x_" << reader.second->id << ".png";
          Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), xPos.borrow_const(), length, length);
        }
        {
          std::stringstream ss;
          ss << DEBUG_FOLDER << "setup_face-x_" << reader.second->id << ".png";
          Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), xNeg.borrow_const(), length, length);
        }
        {
          std::stringstream ss;
          ss << DEBUG_FOLDER << "setup_face+y_" << reader.second->id << ".png";
          Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), yPos.borrow_const(), length, length);
        }
        {
          std::stringstream ss;
          ss << DEBUG_FOLDER << "setup_face-y_" << reader.second->id << ".png";
          Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), yNeg.borrow_const(), length, length);
        }
        {
          std::stringstream ss;
          ss << DEBUG_FOLDER << "setup_face+z_" << reader.second->id << ".png";
          Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), zPos.borrow_const(), length, length);
        }
        {
          std::stringstream ss;
          ss << DEBUG_FOLDER << "setup_face-z_" << reader.second->id << ".png";
          Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), zNeg.borrow_const(), length, length);
        }
      }
    }
#endif
  }

#ifdef READBACKSETUPIMAGE
  {
    {
      const int64_t width = pano.getWidth();
      GPU::Stream::getDefault().synchronize();
      {
        std::stringstream ss;
        ss << DEBUG_FOLDER << "setup_face+x.png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), xPos.borrow_const(), width, width);
      }
      {
        std::stringstream ss;
        ss << DEBUG_FOLDER << "setup_face-x.png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), xNeg.borrow_const(), width, width);
      }
      {
        std::stringstream ss;
        ss << DEBUG_FOLDER << "setup_face+y.png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), yPos.borrow_const(), width, width);
      }
      {
        std::stringstream ss;
        ss << DEBUG_FOLDER << "setup_face-y.png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), yNeg.borrow_const(), width, width);
      }
      {
        std::stringstream ss;
        ss << DEBUG_FOLDER << "setup_face+z.png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), zPos.borrow_const(), width, width);
      }
      {
        std::stringstream ss;
        ss << DEBUG_FOLDER << "setup_face-z.png";
        Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), zNeg.borrow_const(), width, width);
      }
    }
  }
#endif

  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
