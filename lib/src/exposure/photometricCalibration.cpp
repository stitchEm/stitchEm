// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "photometricCalibration.hpp"

#include "pointSampler.hpp"

#include "backend/common/imageOps.hpp"
#include "backend/common/vectorOps.hpp"

#include "core/controllerInputFrames.hpp"
#include "core/kernels/photoStack.cu"

#include "gpu/hostBuffer.hpp"
#include "gpu/vectorTypes.hpp"

#include "util/registeredAlgo.hpp"
#include "util/lmfit/lmmin.hpp"

#include "libvideostitch/emor.hpp"
#include "libvideostitch/inputDef.hpp"

#include <sstream>
#include <memory>

static const int PHOTOMETRIC_CALIBRATION_NUM_SOURCE_FRAMES = 10;

static const double HUBER_SIGMA = 5;

static const int NUM_RADIUS_BUCKETS = 10;

static const int NUM_LIGHTNESS_BUCKETS = 16;

// for debug purposes, set the calibrated exposure and white balance values
// in normal operation, we just set vignette and emor, ignore computerd exp, wb
// #define PHOTOMETRIC_CALIBRATION_OVERWRITE_EXP_WB

static const double PROGRESS_REPORT_POINT_SAMPLING = 5.0;
static const double PROGRESS_REPORT_FRAME_SAMPLING = 33.0;

namespace VideoStitch {
namespace Util {

RegisteredAlgo<PhotometricCalibrationAlgorithm> registered("photometric_calibration");

inline Status photometricCalibrationCancelled() {
  return {Origin::PhotometricCalibrationAlgorithm, ErrType::OperationAbortedByUser,
          "Photometric calibration cancelled"};
}

namespace {

class PhotometricParams {
  // careful, data layout not enforced by compiler anywhere
  // needs to be in sync with eval/computeInitialGuess/getNum...
 public:
  PhotometricParams(const double* params, videoreaderid_t numVideoInputs, size_t numFrames)
      : params(params), numVideoInputs(numVideoInputs), numFrames(numFrames){};

  double emor1() const { return params[numVideoInputs * numFrames * numVariablesPerInput + 0]; };
  double emor2() const { return params[numVideoInputs * numFrames * numVariablesPerInput + 1]; };
  double emor3() const { return params[numVideoInputs * numFrames * numVariablesPerInput + 2]; };
  double emor4() const { return params[numVideoInputs * numFrames * numVariablesPerInput + 3]; };
  double emor5() const { return params[numVideoInputs * numFrames * numVariablesPerInput + 4]; };

  // double vigCenterX() const { return params[numVideoInputs + 5];  }
  // double vigCenterY() const { return params[numVideoInputs + 6];  }

  // double vigCoeff0()  const { return params[numVideoInputs + 7];  }

  double vigCoeff1() const { return params[numVideoInputs * numFrames * numVariablesPerInput + 5]; }
  double vigCoeff2() const { return params[numVideoInputs * numFrames * numVariablesPerInput + 6]; }
  double vigCoeff3() const { return params[numVideoInputs * numFrames * numVariablesPerInput + 7]; }

  double exposureForInput(size_t idx, size_t frameID) const {
    return params[frameID * numVideoInputs * numVariablesPerInput + numVariablesPerInput * idx];
  }

  double2 whiteBalanceForInput(size_t idx, size_t frameID) const {
    return make_double2(
        (double)params[frameID * numVideoInputs * numVariablesPerInput + numVariablesPerInput * idx + 1],
        (double)params[frameID * numVideoInputs * numVariablesPerInput + numVariablesPerInput * idx + 2]);
  }

  size_t getNumberOfFrames() { return numFrames; }

  bool areValid() const {
    for (size_t idx = 0; idx < 8 + numVideoInputs * numFrames * numVariablesPerInput; ++idx) {
      if (VS_ISNAN(params[idx])) {
        return false;
      }
    }

    const double MAX_EMOR = 5;

    if (fabs(emor1()) > MAX_EMOR || fabs(emor2()) > MAX_EMOR || fabs(emor3()) > MAX_EMOR || fabs(emor4()) > MAX_EMOR ||
        fabs(emor5()) > MAX_EMOR) {
      return false;
    }

    const double MAX_EXP = 12;
    const double MAX_WB = 3.0;

    for (size_t frameID = 0; frameID < numFrames; ++frameID) {
      for (videoreaderid_t inp = 0; inp < numVideoInputs; ++inp) {
        if (fabs(exposureForInput(inp, frameID)) > MAX_EXP) {
          return false;
        }

        if (whiteBalanceForInput(inp, frameID).x > MAX_WB || whiteBalanceForInput(inp, frameID).x <= 0 ||
            whiteBalanceForInput(inp, frameID).y > MAX_WB || whiteBalanceForInput(inp, frameID).y <= 0) {
          return false;
        }
      }
    }

    const double MAX_VIG = 1.0;
    if (fabs(vigCoeff1()) > MAX_VIG || fabs(vigCoeff2()) > MAX_VIG || fabs(vigCoeff3()) > MAX_VIG) {
      return false;
    }

    return true;
  }

  void print() {
    std::cout << "photoParams - emor1: " << emor1() << std::endl;
    std::cout << "photoParams - emor2: " << emor2() << std::endl;
    std::cout << "photoParams - emor3: " << emor3() << std::endl;
    std::cout << "photoParams - emor4: " << emor4() << std::endl;
    std::cout << "photoParams - emor5: " << emor5() << std::endl;
    // std::cout << "photoParams - vigCenterX: " << vigCenterX() << std::endl;
    // std::cout << "photoParams - vigCenterY: " << vigCenterY() << std::endl;
    // std::cout << "photoParams - vigCoeff0: " << vigCoeff0() << std::endl;
    std::cout << "photoParams - vigCoeff1: " << vigCoeff1() << std::endl;
    std::cout << "photoParams - vigCoeff2: " << vigCoeff2() << std::endl;
    std::cout << "photoParams - vigCoeff3: " << vigCoeff3() << std::endl;

    for (size_t frameID = 0; frameID < numFrames; ++frameID) {
      std::cout << "At frame " << frameID << ":" << std::endl;
      for (videoreaderid_t inp = 0; inp < numVideoInputs; ++inp) {
        std::cout << "photoParams - exposure@" << inp << ": " << exposureForInput(inp, frameID) << std::endl;
        std::cout << "photoParams - whiteBalance @" << inp << " red: " << whiteBalanceForInput(inp, frameID).x
                  << " blue: " << whiteBalanceForInput(inp, frameID).y << std::endl;
      }
    }
  }

 private:
  const double* params;
  const videoreaderid_t numVideoInputs;
  const size_t numFrames;
  const int numVariablesPerInput = 3;
};

}  // namespace

/**
 * Lmmin problem for exposure stabilization.
 */
class PhotometricCalibrationProblem : public SolverProblem {
 public:
  /**
   * @param numParams Number of parameters to optimize.
   * @param pano Panorama definition
   * @param maxSampledPoints Stopping criterion: stops as soon as that many points have been drawn.
   * @param minPointsPerInput Stopping criterion: stops as soon at all inputs have at least that number of points.
   * @param neighbourhoodSize Size of the neighbourhood
   */
  PhotometricCalibrationProblem(const Core::PanoDefinition& pano, Algorithm::ProgressReporter* progress,
                                std::vector<size_t>& frameList, std::vector<PointPairAtTime>& selectedPoints);

  virtual ~PhotometricCalibrationProblem();

  static const int EMOR_PARAMS = 5;
  static const int VIGNETTE_PARAMS = 3;  // center is zero vigCoeff0 = 1

  int numParamsPerInput() const {
    return 3;  // exposure, white balance red, blue
  }

  int numParams() const {
    int pn = EMOR_PARAMS + VIGNETTE_PARAMS;
    pn += (int)pano.numVideoInputs() * numParamsPerInput() * (int)frameList.size();
    return pn;
  }

  int getNumValuesPerSample() const {
    return 2 * 3;  // 2 * (R, G, B) for each sample
  }

  int getNumAdditionalValues() const {
    return 1;  // monotony error
  }

  double exposureMultFromEv(double ev) const { return pow(2.0, ev); }

  void fillInitialParamVector(std::vector<double>& params, bool useCurrentProjectSettings) const {
    params.clear();

    bool u = useCurrentProjectSettings;
    for (size_t currentFrame : frameList) {
      for (videoreaderid_t inputIdx = 0; inputIdx < pano.numVideoInputs(); ++inputIdx) {
        params.push_back(u ? pano.getVideoInput(inputIdx).getExposureValue().at((int)currentFrame) : 0);
        params.push_back(u ? pano.getVideoInput(inputIdx).getRedCB().at((int)currentFrame) : 1);
        params.push_back(u ? pano.getVideoInput(inputIdx).getBlueCB().at((int)currentFrame) : 1);
      }
    }

    const Core::InputDefinition& firstVideoInput = pano.getVideoInput(0);

    if (u && firstVideoInput.getPhotoResponse() == Core::InputDefinition::PhotoResponse::EmorResponse) {
      params.push_back(firstVideoInput.getEmorA());
      params.push_back(firstVideoInput.getEmorB());
      params.push_back(firstVideoInput.getEmorC());
      params.push_back(firstVideoInput.getEmorD());
      params.push_back(firstVideoInput.getEmorE());
    } else {
      for (int emorIdx = 0; emorIdx < EMOR_PARAMS; emorIdx++) {
        params.push_back(0);
      }
    }

    if (u) {
      params.push_back(firstVideoInput.getVignettingCoeff1());
      params.push_back(firstVideoInput.getVignettingCoeff2());
      params.push_back(firstVideoInput.getVignettingCoeff3());
    } else {
      for (int vignIdx = 0; vignIdx < VIGNETTE_PARAMS; vignIdx++) {
        params.push_back(0);
      }
    }

    // sanity check of data layout
    assert((int)params.size() == numParams());
  }

  void computeInitialGuess(std::vector<double>& params) const {
    fillInitialParamVector(params, true);

    // fallback if our current values aren't valid
    PhotometricParams photoParams(params.data(), pano.numVideoInputs(), frameList.size());
    if (!photoParams.areValid()) {
      fillInitialParamVector(params, false);
    }
  }

 private:
  const Core::PanoDefinition& pano;
  std::vector<PointPairAtTime> selectedPoints;
  std::vector<double> diagonals;
  std::vector<size_t> frameList;
  Algorithm::ProgressReporter* progress;

  double inverseDemiDiagonalSquared(const Core::InputDefinition& im) const {
    if (im.hasCroppedArea()) {
      return 4.0f /
             (float)(im.getCroppedWidth() * im.getCroppedWidth() + im.getCroppedHeight() * im.getCroppedHeight());
    } else {
      return 4.0f / (float)(im.getWidth() * im.getWidth() + im.getHeight() * im.getHeight());
    }
  }

  void eval(const double* raw_params, int /*m_dat*/, double* fvec, const char* fFilter, int iterNum,
            bool* requestBreak) const {
    PhotometricParams photoParams(raw_params, pano.numVideoInputs(), frameList.size());

    if (!photoParams.areValid()) {
      for (int i = 0; i < getNumOutputValues(); ++i) {
        fvec[i] = std::numeric_limits<float>::max();
      }
      return;
    }

    Core::EmorResponseCurve rc(photoParams.emor1(), photoParams.emor2(), photoParams.emor3(), photoParams.emor4(),
                               photoParams.emor5());

    const int monotonicityError = rc.getMonotonyError();
    fvec[(size_t)getNumInputSamples() * 6] = (double)(monotonicityError * pano.numVideoInputs());

    for (size_t i = 0; i < (size_t)getNumInputSamples(); ++i) {
      std::pair<float3, float3> photometricError;
      if (!fFilter || fFilter[i]) {
        photometricError = evalPointPair(rc, photoParams, selectedPoints[i]);
      } else {
        photometricError = std::pair<float3, float3>(make_float3(0, 0, 0), make_float3(0, 0, 0));
      }

      fvec[6 * i + 0] = photometricError.first.x;
      fvec[6 * i + 1] = photometricError.first.y;
      fvec[6 * i + 2] = photometricError.first.z;

      fvec[6 * i + 3] = photometricError.second.x;
      fvec[6 * i + 4] = photometricError.second.y;
      fvec[6 * i + 5] = photometricError.second.z;
    }

    // we don't know how many steps it takes - be enthusiastic at the beginning, then grow slower with each step
    double progressEstimation = 1.0 - 1.0 / ((double)iterNum / numParams() + 1);
    if (progress && progress->notify("Calibrating vignette and camera response",
                                     PROGRESS_REPORT_FRAME_SAMPLING +
                                         progressEstimation * (100.0 - PROGRESS_REPORT_FRAME_SAMPLING))) {
      *requestBreak = true;
    }
  }

  /**
   * Compute vignetting:
   * vigMult = a0 + a1 * r + a2 * r^2 + a3 * r^3
   *         = a0 + r * (a1 + r * (a2 + r * a3))
   */
  double vignette(const Core::CenterCoords2& uv, double inverseDemiDiagonalSquared, double vigCoeff1, double vigCoeff2,
                  double vigCoeff3) const {
    const double dx = (double)uv.x - 0;  // vigCenter x, y = 0
    const double dy = (double)uv.y - 0;
    const double vigRadiusSquared = (dx * dx + dy * dy) * inverseDemiDiagonalSquared;
    double vigMult = vigRadiusSquared * vigCoeff3;
    vigMult += vigCoeff2;
    vigMult *= vigRadiusSquared;
    vigMult += vigCoeff1;
    vigMult *= vigRadiusSquared;
    vigMult += 1;  // vigCoeff0 = 1
    return vigMult;
  }

  double3 lookupEmor(const double3& rgb, const float* responseCurve) const {
    double3 l;
    l.x = 255 * Core::EmorPhotoCorrection::lookup((float)(rgb.x / 255), responseCurve);
    l.y = 255 * Core::EmorPhotoCorrection::lookup((float)(rgb.y / 255), responseCurve);
    l.z = 255 * Core::EmorPhotoCorrection::lookup((float)(rgb.z / 255), responseCurve);
    return l;
  }

  double huberError(double abserror) const {
    if (abserror > HUBER_SIGMA) {
      return sqrt(HUBER_SIGMA * (2 * abserror - HUBER_SIGMA));
    }
    return abserror;
  }

  float3 photometricError(const float3& rgbK, const float3& rgbL, const Core::CenterCoords2& coordsK,
                          const Core::CenterCoords2& coordsL, const Core::EmorResponseCurve& emor,
                          const PhotometricParams photoParams, size_t k, size_t l, size_t frameID, double iddsK,
                          double iddsL) const {
    // should have been checked already in eval()
    assert(photoParams.areValid());

    // we start with the value measured by the camera
    double3 B1_from_B2 = make_double3(rgbL.x, rgbL.y, rgbL.z);

    // apply inverse photometric corrections to rgbL
    B1_from_B2 = lookupEmor(B1_from_B2, emor.getInverseResponseCurve());

    double vignMult =
        vignette(coordsL, iddsL, photoParams.vigCoeff1(), photoParams.vigCoeff2(), photoParams.vigCoeff3());

    B1_from_B2 /= vignMult;

    // we want to compare the scene irradiance, we would need to inverse-apply the camera's exposure
    // worded differently: apply exposure correction, like on the photometric stack
    double exposureL = exposureMultFromEv(photoParams.exposureForInput(l, frameID));
    B1_from_B2 *= exposureL;

    double2 whiteBalanceL = photoParams.whiteBalanceForInput(l, frameID);

    // green wb == 1 to avoid ambiguity
    B1_from_B2.x /= whiteBalanceL.x;
    B1_from_B2.z /= whiteBalanceL.y;

    // at this point we have an estimate of the true scene radiance

    // expose the image by doing inverse exposure correction for 2nd image
    double2 whiteBalanceK = photoParams.whiteBalanceForInput(k, frameID);
    B1_from_B2.x *= whiteBalanceK.x;
    B1_from_B2.z *= whiteBalanceK.y;

    double exposureK = exposureMultFromEv(photoParams.exposureForInput(k, frameID));
    B1_from_B2 /= exposureK;

    // apply vignette, as lens would
    vignMult = vignette(coordsK, iddsK, photoParams.vigCoeff1(), photoParams.vigCoeff2(), photoParams.vigCoeff3());
    B1_from_B2 *= vignMult;

    // apply camera response
    B1_from_B2 = lookupEmor(B1_from_B2, emor.getResponseCurve());

    // how far are we off from what we measured at B1?
    float3 error;
    error.x = (float)huberError(fabs((double)rgbK.x - B1_from_B2.x));
    error.y = (float)huberError(fabs((double)rgbK.y - B1_from_B2.y));
    error.z = (float)huberError(fabs((double)rgbK.z - B1_from_B2.z));

    assert(!VS_ISNAN(error.x) && !VS_ISNAN(error.y) && !VS_ISNAN(error.z));

    return error;
  }

  /**
   * Evals a single point set.
   */
  std::pair<float3, float3> evalPointPair(Core::EmorResponseCurve& responseCurve, const PhotometricParams photoParams,
                                          const PointPairAtTime& pointPairAtTime) const {
    const PointPair& pointPair = pointPairAtTime.pointPair();

    if (!(pointPair.p_k->hasColor() && pointPair.p_l->hasColor())) {
      return std::pair<float3, float3>(make_float3(0, 0, 0), make_float3(0, 0, 0));
    } else {
      const videoreaderid_t k = pointPair.p_k->videoInputId();
      const videoreaderid_t l = pointPair.p_l->videoInputId();

      const Core::TopLeftCoords2 coordsK = pointPair.p_k->coords();
      const Core::TopLeftCoords2 coordsL = pointPair.p_l->coords();

      const Core::CenterCoords2 centerCoordsK =
          Core::CenterCoords2(coordsK, Core::TopLeftCoords2((float)pano.getVideoInput(k).getWidth() / 2.0f,
                                                            (float)pano.getVideoInput(k).getHeight() / 2.0f));

      const Core::CenterCoords2 centerCoordsL =
          Core::CenterCoords2(coordsL, Core::TopLeftCoords2((float)pano.getVideoInput(l).getWidth() / 2.0f,
                                                            (float)pano.getVideoInput(l).getHeight() / 2.0f));

      const double iddsK = diagonals[k];
      const double iddsL = diagonals[l];

      const float3 accRgbK = pointPair.p_k->color();
      const float3 accRgbL = pointPair.p_l->color();

      const float3 error_kl = photometricError(accRgbK, accRgbL, centerCoordsK, centerCoordsL, responseCurve,
                                               photoParams, k, l, pointPairAtTime.time(), iddsK, iddsL);

      const float3 error_lk = photometricError(accRgbL, accRgbK, centerCoordsL, centerCoordsK, responseCurve,
                                               photoParams, l, k, pointPairAtTime.time(), iddsL, iddsK);

      return std::pair<float3, float3>(error_kl, error_lk);
    }
  }

  int getNumInputSamples() const { return (int)selectedPoints.size(); }
};

PhotometricCalibrationProblem::~PhotometricCalibrationProblem() {}

// When sampling, we must make sure to have a single connected compunent.
// Else, it can become possible to optimize each groups of inputs individually and end up having them badly fit.
PhotometricCalibrationProblem::PhotometricCalibrationProblem(const Core::PanoDefinition& pano,
                                                             Algorithm::ProgressReporter* progress,
                                                             std::vector<size_t>& frameList,
                                                             std::vector<PointPairAtTime>& selectedPoints)
    : pano(pano), selectedPoints(selectedPoints), frameList(frameList), progress(progress) {
  for (videoreaderid_t inp = 0; inp < pano.numVideoInputs(); ++inp) {
    const Core::InputDefinition& inputDefinition = pano.getVideoInput(inp);
    double idds = inverseDemiDiagonalSquared(inputDefinition);
    diagonals.push_back(idds);
  }
}

PhotometricCalibrationBase::PhotometricCalibrationBase(const Ptv::Value* config)
    : maxSampledPoints(100000), minPointsPerInput(100), neighbourhoodSize(5) {
  if (config != NULL) {
    config->printJson(std::cout);

    const Ptv::Value* value = config->has("max_sampled_points");
    if (value && value->getType() == Ptv::Value::INT) {
      maxSampledPoints = (int)value->asInt();
      if (maxSampledPoints < 0) {
        maxSampledPoints = 0;
      }
    }
    value = config->has("min_points_per_input");
    if (value && value->getType() == Ptv::Value::INT) {
      minPointsPerInput = (int)value->asInt();
      if (minPointsPerInput < 0) {
        minPointsPerInput = 0;
      }
    }
    value = config->has("neighbourhood_size");
    if (value && value->getType() == Ptv::Value::INT) {
      neighbourhoodSize = (int)value->asInt();
      if (neighbourhoodSize < 0) {
        neighbourhoodSize = 0;
      }
    }
  }
}

using PointPairGradient = std::pair<PointPair*, int>;
using GradientVector = std::vector<PointPairGradient>;

Status selectPointsByGradient(Core::PanoDefinition* pano, const RadialPointSampler& pointSampler,
                              std::vector<GPU::HostBuffer<uint32_t>>& loadedFrames, int pairsPerInputPerRadiusBin) {
  const std::map<videoreaderid_t, std::map<int, std::vector<PointPair*>>>& pointVectors =
      pointSampler.getPointPairsByRadius();

  for (videoreaderid_t frameInputID = 0; frameInputID < (videoreaderid_t)loadedFrames.size(); frameInputID++) {
    auto it = pointVectors.find(frameInputID);
    if (it == pointVectors.cend()) {
      return Status{Origin::PhotometricCalibrationAlgorithm, ErrType::ImplementationError, "No input points"};
    }
    const std::map<int, std::vector<PointPair*>>& pointPairsByRadius = it->second;

    for (const auto& pointVector : pointPairsByRadius) {
      GradientVector gradientVector;
      gradientVector.reserve(pointVector.second.size());

      for (PointPair* pointPair : pointVector.second) {
        Point* p = NULL;
        if (pointPair->p_k->videoInputId() == frameInputID) {
          p = pointPair->p_k;
        } else if (pointPair->p_l->videoInputId() == frameInputID) {
          p = pointPair->p_l;
        } else {
          // we expect our points sorted by input
          // if this point pair doesn't belong to our current input, there's something seriously wrong
          return Status{Origin::PhotometricCalibrationAlgorithm, ErrType::ImplementationError, "Input points invalid"};
        }
        const int p_k_x = (int)p->coords().x;
        const int p_k_y = (int)p->coords().y;

        const Core::InputDefinition& input = pano->getVideoInput(frameInputID);

        auto getPixel = [&](int x, int y) -> uint32_t {
          x = std::min((int)input.getWidth() - 1, x);
          y = std::min((int)input.getHeight() - 1, y);
          x = std::max(0, x);
          y = std::max(0, y);
          return loadedFrames[frameInputID][y * input.getWidth() + x];
        };

        // TODO surely there's a better way
        auto diffPixel = [](uint32_t i, uint32_t j) -> int {
          int ir = (int)Image::RGBA::r(i);
          int ig = (int)Image::RGBA::g(i);
          int ib = (int)Image::RGBA::b(i);

          int jr = (int)Image::RGBA::r(j);
          int jg = (int)Image::RGBA::g(j);
          int jb = (int)Image::RGBA::b(j);

          return abs(ir - jr) + abs(ig - jg) + abs(ib - jb);
        };

        // Ignore any zero alpha (masked) pixels.
        uint32_t top = getPixel(p_k_x - 1, p_k_y);
        uint32_t bottom = getPixel(p_k_x + 1, p_k_y);
        uint32_t left = getPixel(p_k_x, p_k_y - 1);
        uint32_t right = getPixel(p_k_x, p_k_y + 1);

        int gradient = diffPixel(top, bottom) + diffPixel(left, right);
        gradientVector.push_back(PointPairGradient(pointPair, gradient));

        uint32_t v = loadedFrames[frameInputID][p_k_y * input.getWidth() + p_k_x];
        float3 rgb = make_float3((float)Image::RGBA::r(v), (float)Image::RGBA::g(v), (float)Image::RGBA::b(v));
        p->setColor(rgb);
      }

      sort(gradientVector.begin(), gradientVector.end(),
           [&](PointPairGradient i, PointPairGradient j) -> bool { return (i.second < j.second); });

      int elementsToChose = std::min(pairsPerInputPerRadiusBin, (int)gradientVector.size());

      for (auto gIt = gradientVector.begin(); gIt < gradientVector.begin() + elementsToChose; ++gIt) {
        gIt->first->choose();
      }
    }
  }

  return Status::OK();
}

Status selectPointsForCalibration(Core::PanoDefinition* pano,
                                  Core::ControllerInputFrames<PixelFormat::RGBA, uint32_t>* container,
                                  Algorithm::ProgressReporter* progress, const RadialPointSampler& pointSampler,
                                  const std::vector<size_t>& inputFrames, int pairsPerInputPerRadiusBin,
                                  std::vector<PointPairAtTime>& selectedPoints) {
  const std::vector<PointPair*>& sampledPointPairs = pointSampler.getPointPairs();

  std::vector<PointPairAtTime> globalSamples;

  int frameID = 0;
  for (size_t currentFrame : inputFrames) {
    std::stringstream ss;
    ss << "Sampling points from frame " << currentFrame;

    if (progress &&
        progress->notify(ss.str(), (double)frameID / (double)inputFrames.size() *
                                           (PROGRESS_REPORT_FRAME_SAMPLING - PROGRESS_REPORT_POINT_SAMPLING) +
                                       PROGRESS_REPORT_POINT_SAMPLING)) {
      return photometricCalibrationCancelled();
    }

    // Start a round of selecting a subset of the samples for the calibration
    for (PointPair* pointPair : sampledPointPairs) {
      pointPair->resetChoice();
    }

    FAIL_RETURN(container->seek((int)currentFrame));

    std::map<readerid_t, PotentialValue<GPU::HostBuffer<uint32_t>>> frames;

    container->load(frames);
    std::vector<GPU::HostBuffer<uint32_t>> succesfullyLoadedFrames;

    for (auto frame : frames) {
      if (frame.second.ok()) {
        succesfullyLoadedFrames.push_back(frame.second.value());
      } else {
        return frame.second.status();
      }
    }

    FAIL_RETURN(selectPointsByGradient(pano, pointSampler, succesfullyLoadedFrames, pairsPerInputPerRadiusBin));

    for (PointPair* pointPair : sampledPointPairs) {
      globalSamples.push_back(PointPairAtTime(pointPair, frameID));
    }

    frameID++;
  }

  int totalSelected = 0;

  std::vector<std::pair<std::vector<PointPairAtTime>, int>> pointPairsByLightness(NUM_LIGHTNESS_BUCKETS);
  for (PointPairAtTime& pointPairAtTime : globalSamples) {
    if (pointPairAtTime.pointPair().p_k->hasColor() && pointPairAtTime.pointPair().p_l->hasColor()) {
      float3 pairAvgColor = (pointPairAtTime.pointPair().p_k->color() + pointPairAtTime.pointPair().p_l->color());
      pairAvgColor /= 2;
      float lightness = (pairAvgColor.x + pairAvgColor.y + pairAvgColor.z) / 3;
      int lightnessID = (int)lightness / NUM_LIGHTNESS_BUCKETS;

      pointPairsByLightness[lightnessID].first.push_back(pointPairAtTime);
      if (pointPairAtTime.pointPair().shouldBeUsed()) {
        pointPairsByLightness[lightnessID].second++;
        totalSelected++;
      }
    }
  }

  selectedPoints.reserve(globalSamples.size());

  // try to get at least 2/3 of the average number of pointsamples in each lightness bucket, if possible
  int minPointsPerLighnessBucket = totalSelected / NUM_LIGHTNESS_BUCKETS * 2 / 3;

  for (auto& lightnessBucket : pointPairsByLightness) {
    int fillUnselected = minPointsPerLighnessBucket - lightnessBucket.second;

    for (PointPairAtTime& ppAt : lightnessBucket.first) {
      if (ppAt.pointPair().shouldBeUsed()) {
        selectedPoints.push_back(ppAt);
      } else if (fillUnselected-- > 0) {
        selectedPoints.push_back(ppAt);
      }
    }
  }

  selectedPoints.shrink_to_fit();

  // std::cout << "numSamples: " << globalSamples.size() << ", totalSelected: " << totalSelected << ", with filled
  // lightness: " << selectedPoints.size() << std::endl;

  return Status::OK();
}

// -------------------------- Offline algorithm -----------------------------

const char* PhotometricCalibrationAlgorithm::docString =
    "An algorithm that estimates camera parameters: radial vignetting and the EMoR camera response. The following "
    "parameters are available: "
    "{\n"
    "  \"max_sampled_points\": 100000      # Stopping criterion 1. We'll stop after drawing that many sample points.\n"
    "  \"min_points_per_input\": 80        # Stopping criterion 2. Each input shall have at least min_points_per_input "
    "samples.\n"
    "  \"neighbourhood_size\": 5           # Size of the neighbourhood to use to compute luminosity.\n"
    "  \"first_frame\": 0                  # Restriction in time: define start of sequence.\n"
    "  \"last_frame\": 0                   # Restriction in time: define end of sequence.\n"
    "}\n";

PhotometricCalibrationAlgorithm::PhotometricCalibrationAlgorithm(const Ptv::Value* config)
    : PhotometricCalibrationBase(config) {
  if (config != NULL) {
    config->printJson(std::cout);

    const Ptv::Value* value = config->has("first_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      firstFrame = (int)value->asInt();
      if (firstFrame < 0) {
        firstFrame = 0;
      }
    }
    value = config->has("last_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      lastFrame = (int)value->asInt();
      if (lastFrame < firstFrame) {
        lastFrame = firstFrame;
      }
    }
  }
}

Potential<Ptv::Value> PhotometricCalibrationAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                                             OpaquePtr**) const {
  std::vector<size_t> inputFrames;

  const int len = lastFrame - firstFrame;
  if (lastFrame == firstFrame || lastFrame == NO_LAST_FRAME) {
    inputFrames.push_back(firstFrame);
  } else {
    const int numFrames = std::min(len, PHOTOMETRIC_CALIBRATION_NUM_SOURCE_FRAMES);
    for (int i = 0; i < numFrames; ++i) {
      size_t frame = (size_t)ceilf((firstFrame + i * (float)len / (numFrames - 1)));
      inputFrames.push_back(frame);
    }
  }

  if (inputFrames.empty()) {
    return {Origin::PhotometricCalibrationAlgorithm, ErrType::InvalidConfiguration,
            "Invalid input frame sequence: [" + std::to_string(firstFrame) + ", " + std::to_string(lastFrame) + "]"};
  }

  if (progress && progress->notify("Sampling points in overlapping regions", 0.0)) {
    return photometricCalibrationCancelled();
  }

  // Parameters. Reuse the result from one iteration to the other as initial guess.
  auto container = Core::ControllerInputFrames<PixelFormat::RGBA, uint32_t>::create(pano);

  // with more and more input frames, reduce the number of points per frame
  // but not too much, as we have parameters (exp, wb) for each frame
  int minPointsPerInputFrame = (int)(minPointsPerInput / log2((double)inputFrames.size() + 1.0));
  RadialPointSampler pointSampler(*pano, maxSampledPoints, minPointsPerInputFrame, neighbourhoodSize,
                                  NUM_RADIUS_BUCKETS);

  // we try to have a even distribution of samples over the image's radius
  // usually there will be a couple of empty buckets in the image center (overlaps only on border zones)
  // so the buckets that contain any elements at all contain more then minPointsPerInput / NUM_RADIUS_BUCKETS
  // let's throw out at least half of them and only keep the ones that have a low gradient
  const int pointsPerBucket = minPointsPerInputFrame / NUM_RADIUS_BUCKETS / 2;

  // choose points for this calibration, and set their color
  std::vector<PointPairAtTime> chosenPoints;
  FAIL_RETURN(selectPointsForCalibration(pano, container.object(), progress, pointSampler, inputFrames, pointsPerBucket,
                                         chosenPoints));

  const auto problem = std::make_unique<PhotometricCalibrationProblem>(*pano, progress, inputFrames, chosenPoints);

  const auto solver = std::make_unique<LmminSolver<SolverProblem>>(*problem, nullptr, false);

  // We must make large steps for things to move, else the gradient will be zero.
  solver->getControl().epsilon = 0.01;

  // Parameters. Reuse the result from one iteration to the other as initial guess.
  std::vector<double> params;
  problem->computeInitialGuess(params);

  if (progress && progress->notify("Calibrating vignette and camera response", 33.0)) {
    return photometricCalibrationCancelled();
  }

  PhotometricParams p(params.data(), pano->numVideoInputs(), inputFrames.size());

  // Find the set of parameters that minimize spatial inconsitencies.
  if (solver->run(params)) {
    for (videoreaderid_t inp = 0; inp < pano->numVideoInputs(); ++inp) {
      pano->getVideoInput(inp).setRadialVignetting(1, p.vigCoeff1(), p.vigCoeff2(), p.vigCoeff3(), 0, 0);

#ifdef PHOTOMETRIC_CALIBRATION_OVERWRITE_EXP_WB
      Core::Curve* evCurve = new Core::Curve(p.exposureForInput(inp));
      pano->getVideoInput(inp).replaceExposureValue(evCurve);

      Core::Curve* wbRedCurve = new Core::Curve(p.whiteBalanceForInput(inp).x);
      pano->getVideoInput(inp).replaceRedCB(wbRedCurve);

      Core::Curve* wbGreenCurve = new Core::Curve(1.0);
      pano->getVideoInput(inp).replaceGreenCB(wbGreenCurve);

      Core::Curve* wbBlueCurve = new Core::Curve(p.whiteBalanceForInput(inp).y);
      pano->getVideoInput(inp).replaceBlueCB(wbBlueCurve);
#endif  // PHOTOMETRIC_CALIBRATION_OVERWRITE_EXP_WB

      pano->getVideoInput(inp).setEmorPhotoResponse(p.emor1(), p.emor2(), p.emor3(), p.emor4(), p.emor5());
    }
  } else {
    return {Origin::PhotometricCalibrationAlgorithm, ErrType::RuntimeError,
            "Unable to compute a usable photometric calibration.\n"
            "Please check that the geometric calibration of the camera array is correct and that there is enough "
            "overlap between the cameras.\n"
            "When selecting a sequence in the timeline, please try to include different lighting conditions.\n"
            "Exclude scenes with fast movement if possible."};
  }

  if (progress) {
    progress->notify("Done", 100.0);
  }
  return Status::OK();
}

}  // namespace Util
}  // namespace VideoStitch
