// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "inputsMap.hpp"

#include "common/container.hpp"
#include "core/geoTransform.hpp"
#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/stream.hpp"
#include "gpu/core1/strip.hpp"
#include "gpu/core1/transform.hpp"
#include "mask/mergerMask.hpp"
#include "util/polylineEncodingUtils.hpp"

#include "libvideostitch/stereoRigDef.hpp"
#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/inputDef.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>

//#define READBACKSETUPIMAGE
//#define INPUTMAPS_PRECOMPUTED

#if defined(READBACKSETUPIMAGE) || defined(INPUTMAPS_PRECOMPUTED)
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

static const double OVERLAP = 0.25;

namespace VideoStitch {
namespace Core {

Potential<InputsMap> InputsMap::create(const PanoDefinition& pano) {
  std::unique_ptr<InputsMap> inputsMap;
  inputsMap.reset(new InputsMap(pano));
  Status status = inputsMap->allocateBuffers();
  if (status.ok()) {
    return Potential<InputsMap>(inputsMap.release());
  } else {
    return Potential<InputsMap>(status);
  }
}

InputsMap::InputsMap(const PanoDefinition& pano)
    : _boundedFrames(std::make_pair(std::numeric_limits<frameid_t>::max(), std::numeric_limits<frameid_t>::min())),
      _width(pano.getWidth()),
      _height(pano.getHeight()) {}

InputsMap::~InputsMap() {}

Status InputsMap::allocateBuffers() { return setupBuffer.alloc(_width * _height, "Setup Buffer"); }

Status InputsMap::compute(const std::map<readerid_t, Input::VideoReader*>& readers, const PanoDefinition& pano,
                          const bool loadingEnabled) {
  return compute(readers, pano, nullptr, LeftEye, loadingEnabled);
}

void computeStrip(float& min, float& max, double baseline, double rigradius, size_t countCameras) {
  double theta = 2.0 * M_PI / (double)countCameras;

  /*Compute disparity angle*/
  double disparityAngle = asin(0.5 * baseline / rigradius);
  double angularStripWidth = theta / 2.0;
  max = (float)(disparityAngle + (1.0 + OVERLAP) * angularStripWidth);
  min = (float)(disparityAngle - (1.0 + OVERLAP) * angularStripWidth);
}

std::pair<frameid_t, frameid_t> InputsMap::getBoundedFrameIds() const { return _boundedFrames; }

#ifndef VS_OPENCL
Status InputsMap::loadPrecomputedMap(const frameid_t frameId, const PanoDefinition& pano,
                                     const std::map<readerid_t, Input::VideoReader*>& readers,
                                     std::unique_ptr<MaskInterpolation::InputMaskInterpolation>& inputMaskInterpolation,
                                     bool& loaded) {
  loaded = false;
  if (!pano.getBlendingMaskEnabled()) {
    return Status::OK();
  }

  std::pair<frameid_t, frameid_t> boundedFrames = pano.getBlendingMaskBoundedFrameIds(frameId);
  std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>> inputIndexPixelData =
      pano.getInputIndexPixelDataIfValid(frameId);
  if (_boundedFrames.first == boundedFrames.first && _boundedFrames.second == boundedFrames.second &&
      frameId <= boundedFrames.second && frameId >= boundedFrames.first &&
      boundedFrames.first != boundedFrames.second) {
    // The bounded maps have been loaded, perform interpolation if needed
    loaded = true;
    if (pano.getBlendingMaskInterpolationEnabled() && inputMaskInterpolation.get() && inputIndexPixelData.size()) {
      FAIL_RETURN(inputMaskInterpolation->getInputsMap(pano, frameId, setupBuffer.borrow()));
    }
    return Status::OK();
  }

  if (inputIndexPixelData.size()) {
    if (inputMaskInterpolation.get()) {
      inputMaskInterpolation->deactivate();
    }
    if (!pano.getBlendingMaskInterpolationEnabled() || inputIndexPixelData.size() == 1 ||
        !inputMaskInterpolation.get()) {
      FAIL_RETURN(MergerMask::MergerMask::transformMasksFromEncodedInputToOutputSpace(
          pano, readers, inputIndexPixelData[0].second, setupBuffer.borrow()));
      Logger::get(Logger::Info) << "Precomputed map is loaded" << std::endl;
    } else {
      _boundedFrames = std::make_pair((frameid_t)boundedFrames.first, (frameid_t)boundedFrames.second);
      std::map<videoreaderid_t, std::vector<cv::Point>> point0s, point1s;
      Util::PolylineEncoding::polylineDecodePolygons(inputIndexPixelData[0].second, point0s);
      Util::PolylineEncoding::polylineDecodePolygons(inputIndexPixelData[1].second, point1s);
      Logger::get(Logger::Info) << "Bounded maps are loaded" << std::endl;
      FAIL_RETURN(
          inputMaskInterpolation->setupKeyframes(pano, _boundedFrames.first, point0s, _boundedFrames.second, point1s));
      FAIL_RETURN(inputMaskInterpolation->getInputsMap(pano, frameId, setupBuffer.borrow()));
    }

#ifdef INPUTMAPS_PRECOMPUTED
    {
      std::stringstream ss;
      ss.str("");
      ss << "precomputed-result " << frameId << ".png";
      Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), setupBuffer.borrow_const().as_const(), _cropped_width,
                                       _cropped_height);
    }
#endif
    loaded = true;
  }
  return Status::OK();
}
#endif

Status InputsMap::compute(const std::map<readerid_t, Input::VideoReader*>& readers, const PanoDefinition& pano,
                          const StereoRigDefinition* rigDef, Eye eye, const bool loadingEnabled) {
  /* The blending mask is computed in the output but stored in the input space.
   * There are two major advantages when storing it in the input space:
   * - The interpolation between frames will become easier in the input space
   * - The potential to call remap directly in the transformation stacks
   * However, storing the blending mask in the input space might result in a slightly different
   * from the mask when it was computed.
   */
  // Load the precomputed map if it is valid
  if (loadingEnabled) {
    bool loaded = false;
#ifndef VS_OPENCL
    std::unique_ptr<MaskInterpolation::InputMaskInterpolation> inputMaskInterpolation(nullptr);
    FAIL_RETURN(loadPrecomputedMap(0, pano, readers, inputMaskInterpolation, loaded));
#endif
    if (loaded) {
      return Status::OK();
    }
  }

  // Reset the boundedFrames to infinity, nothing was loaded
  _boundedFrames = std::make_pair(std::numeric_limits<frameid_t>::max(), std::numeric_limits<frameid_t>::min());
  // If no map was precomputed, now generate it on the fly
  FAIL_RETURN(GPU::memsetToZeroBlocking(setupBuffer.borrow(), _width * _height * 4));

  for (auto reader : readers) {
    const InputDefinition& inputDef = pano.getInput(reader.second->id);
    const GeometryDefinition& geometry = inputDef.getGeometries().at(0);
    const size_t bufferSize = (size_t)(inputDef.getWidth() * inputDef.getHeight());

    /*Create mask buffer*/
    GPU::UniqueBuffer<unsigned char> maskDevBuffer;
    FAIL_RETURN(maskDevBuffer.alloc(bufferSize, "MaskSetup"));

    /* Retrieve input mask and send it to the gpu */
    const unsigned char* data = inputDef.getMaskPixelDataIfValid();

    if (data && inputDef.deletesMaskedPixels()) {
      FAIL_RETURN(GPU::memcpyBlocking(maskDevBuffer.borrow(), data, bufferSize));
    } else {
      FAIL_RETURN(GPU::memsetToZeroBlocking(maskDevBuffer.borrow(), bufferSize));
    }

    /* Get an horizontal strip for stereoscopic videos */
    if (rigDef && rigDef->getGeometry() == StereoRigDefinition::Circular) {
      std::vector<int> inputs = (eye == LeftEye ? rigDef->getLeftInputs() : rigDef->getRightInputs());
      if (std::find(inputs.begin(), inputs.end(), reader.second->id) != inputs.end()) {
        TransformStack::GeoTransform* geoParams = TransformStack::GeoTransform::create(pano, inputDef);
        TransformGeoParams params(inputDef, geometry, pano);
        StereoRigDefinition::Orientation orientation = rigDef->getOrientation();

        float min = 0.0;
        float max = 1.0;

        const auto countCamera = inputs.size();
        double baseline = rigDef->getIPD();
        double radius = rigDef->getDiameter() / 2.0;

        computeStrip(min, max, baseline, radius, countCamera);

        // flip ?
        if ((eye == RightEye) !=
            (orientation == StereoRigDefinition::Landscape || orientation == StereoRigDefinition::Portrait)) {
          float tmp = min;
          min = -max;
          max = -tmp;
        }

        float2 inputScale = {(float)geometry.getHorizontalFocal(), (float)geometry.getVerticalFocal()};

        switch (orientation) {
          case StereoRigDefinition::Portrait:
          case StereoRigDefinition::Portrait_flipped:
            hStrip(maskDevBuffer.borrow(), inputDef.getWidth(), inputDef.getHeight(), min, max, inputDef.getFormat(),
                   (float)inputDef.getCenterX(geometry), (float)inputDef.getCenterY(geometry), params, inputScale,
                   GPU::Stream::getDefault());
            break;
          case StereoRigDefinition::Landscape:
          case StereoRigDefinition::Landscape_flipped:
            vStrip(maskDevBuffer.borrow(), inputDef.getWidth(), inputDef.getHeight(), min, max, inputDef.getFormat(),
                   (float)inputDef.getCenterX(geometry), (float)inputDef.getCenterY(geometry), params, inputScale,
                   GPU::Stream::getDefault());
        }
        delete geoParams;
      }
    }

    /* Update assigned pixels */
    Transform* t = Transform::create(inputDef);
    if (!t) {
      return {Origin::Stitcher, ErrType::SetupFailure,
              "Cannot create v1 transformation for input " + std::to_string(reader.second->id)};
    }
    FAIL_RETURN(t->computeZone(setupBuffer.borrow(), pano, inputDef, reader.second->id, maskDevBuffer.borrow(),
                               GPU::Stream::getDefault()));
    FAIL_RETURN(GPU::Stream::getDefault().synchronize());
    delete t;

#ifdef READBACKSETUPIMAGE
    {
      {
        const int64_t width = pano.getWidth();
        const int64_t height = pano.getHeight();
        GPU::Stream::getDefault().synchronize();
        {
          std::stringstream ss;
          ss << DEBUG_FOLDER << "setup-" << reader.second->id;
          if (eye == LeftEye) {
            ss << "left";
          } else {
            ss << "right";
          }
          ss << "-.png";
          Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), setupBuffer.borrow_const(), width, height);
        }
      }
    }
#endif
  }

  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
