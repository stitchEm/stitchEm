// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sampledStabilization.hpp"

#include "exposureStabilize.hpp"
#include "pointSampler.hpp"

//#define RANSAC_EXPERIMENT
#ifdef RANSAC_EXPERIMENT
#include "ransac.hpp"
#else
#include "util/lmfit/lmmin.hpp"
#endif

#include "backend/common/imageOps.hpp"
#include "backend/common/vectorOps.hpp"

#include "common/container.hpp"
#include "core/controllerInputFrames.hpp"
#include "core/photoTransform.hpp"

#include "gpu/memcpy.hpp"
#include "gpu/surface.hpp"

#include "util/registeredAlgo.hpp"

#include "libvideostitch/curves.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Util {
namespace {
RegisteredAlgo<SampledStabilizationAlgorithm> registered("exposure_stabilize");
RegisteredAlgo<SampledStabilizationOnlineAlgorithm, true> registeredOnline("exposure_stabilize");

inline Status exposureAlgorithmCancelled() {
  return {Origin::ExposureAlgorithm, ErrType::OperationAbortedByUser, "Exposure stabilization cancelled"};
}

}  // namespace

// For a description of the algorithm and notations here:
// cd ../../doc; pdflatex exp_correction.tex; acroread exp_correction.pdf

/**
 * Lmmin problem for exposure stabilization.
 */
class SampledExposureStabilizationProblem : public ExposureStabilizationProblemBase {
 public:
  /**
   * @param numParams Number of parameters to optimize.
   * @param pano Panorama definition
   * @param maxSampledPoints Stopping criterion: stops as soon as that many points have been drawn.
   * @param minPointsPerInput Stopping criterion: stops as soon at all inputs have at least that number of points.
   * @param neighbourhoodSize Size of the neighbourhood
   * @param anchor Id of the anchor (zero-based).
   */
  SampledExposureStabilizationProblem(const Core::PanoDefinition& pano, int maxSampledPoints, int minPointsPerInput,
                                      int neighbourhoodSize, int anchor,
                                      ExposureStabilizationProblemBase::ParameterSetType parameterSetType);

  virtual ~SampledExposureStabilizationProblem();

  const std::vector<PointPair*>& getPointPairs() const { return pointSampler.getPointPairs(); }

  int getMinPointsInOneOutput() const { return pointSampler.getMinPointsInOneOutput(); }

  const Core::HostPhotoTransform* getPhotoTransform(videoreaderid_t k) const { return photoTransforms[k]; }

  int getNumConnectedComponents() const { return pointSampler.getNumConnectedComponents(); }

 protected:
  std::vector<Core::HostPhotoTransform*> photoTransforms;
  PointSampler pointSampler;

 private:
  void eval(const double* params, int /*m_dat*/, double* fvec, const char* fFilter, int /*iterNum*/, bool*) const {
    for (size_t i = 0; i < getPointPairs().size(); ++i) {
      if (!fFilter || fFilter[i]) {
        evalPointPair(params, getPointPairs()[i], fvec + 3 * i);
      } else {
        fvec[3 * i] = 0.0;
        fvec[3 * i + 1] = 0.0;
        fvec[3 * i + 2] = 0.0;
      }
    }
  }

  int getNumInputSamples() const { return (int)getPointPairs().size(); }

  /**
   * Evals a single point set.
   */
  void evalPointPair(const double* params, const PointPair* pointPair, double* res) const {
    if (!(pointPair->p_k->hasColor() && pointPair->p_l->hasColor())) {
      res[0] = 0.0;
      res[1] = 0.0;
      res[2] = 0.0;
    } else {
      if (!isValid(params)) {
        res[0] = std::numeric_limits<double>::max();
        res[1] = std::numeric_limits<double>::max();
        res[2] = std::numeric_limits<double>::max();
        return;
      }
      const videoreaderid_t k = pointPair->p_k->videoInputId();
      const videoreaderid_t l = pointPair->p_l->videoInputId();
      const float3 colorMultK = getVideoColorMult(params, k);
      const float3 colorMultL = getVideoColorMult(params, l);
      const float3 accRgbK = photoTransforms[k]->mapPhotoLinearToPano(
          photoTransforms[k]->mapPhotoCorrectLinear(colorMultK, pointPair->p_k->color()));
      const float3 accRgbL = photoTransforms[l]->mapPhotoLinearToPano(
          photoTransforms[l]->mapPhotoCorrectLinear(colorMultL, pointPair->p_l->color()));
      res[0] = (1.0 / 255.0) * (double)(accRgbK.x - accRgbL.x);
      res[1] = (1.0 / 255.0) * (double)(accRgbK.y - accRgbL.y);
      res[2] = (1.0 / 255.0) * (double)(accRgbK.z - accRgbL.z);
      // std::cout << *pointPair->p_k << " | " << *pointPair->p_l << " -> " << accRgbK.x << ", " << accRgbL.x <<
      // std::endl;
    }
  }
};

SampledExposureStabilizationProblem::~SampledExposureStabilizationProblem() { deleteAll(photoTransforms); }

// When sampling, we must make sure to have a single connected compunent.
// Else, it can become possible to optimize each groups of inputs individually and end up having them badly fit.
SampledExposureStabilizationProblem::SampledExposureStabilizationProblem(
    const Core::PanoDefinition& pano, int maxSampledPoints, int minPointsPerInput, int neighbourhoodSize, int anchor,
    ExposureStabilizationProblemBase::ParameterSetType parameterSetType)
    : ExposureStabilizationProblemBase(pano, anchor, parameterSetType),
      pointSampler(pano, maxSampledPoints, minPointsPerInput, neighbourhoodSize) {
  for (videoreaderid_t i = 0; i < pano.numVideoInputs(); ++i) {
    photoTransforms.push_back(Core::HostPhotoTransform::create(pano.getVideoInput(i)));
  }
}

SampledStabilizationBase::SampledStabilizationBase(const Ptv::Value* config)
    : maxSampledPoints(100000), minPointsPerInput(80), neighbourhoodSize(30), stabilizeWB(false) {
  if (config != NULL) {
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
    value = config->has("anchor");
    if (value && value->getType() == Ptv::Value::INT) {
      anchor = (int)value->asInt();
    }
    value = config->has("stabilize_wb");
    if (value && value->getType() == Ptv::Value::BOOL) {
      stabilizeWB = value->asBool();
    }
  }
}

namespace {
/**
 * Returns true if a value is nearly saturated.
 */
bool isNearlyBurnt(const float3& rgb) {
  return rgb.x < 10.0f || rgb.x > 245.0f || rgb.y < 10.0f || rgb.y > 245.0f || rgb.z < 10.0f || rgb.z > 245.0f;
}

#ifdef RANSAC_EXPERIMENT
class ExposureRansacSolver : public RansacSolver {
 private:
  // Ransac: For the problem to be at least constrained, we need: params.size() elements.
  // There are at least minPointsPerInput points per input, i.e. at least pano.numVideoInputs() * minPointsPerInput / 2
  // samples in total. We require two thirds of these to be consensual.
  ExposureRansacSolver(const SolverProblem& problem, int minSamplesForFit, int numIters, int minConsensusSamples,
                       bool debug = false)
      : RansacSolver(problem, params.size(), 100, (pano.numVideoInputs() * minPointsPerInput) / 3, debug) {}

  bool isConsensualSample(double* values) const {
    // values[0] is a difference in red value in [0;1].
    return values[0] * values[0] + values[1] * values[1] + values[2] * values[2] < 0.0001;
  }
};
#endif

Solver<SolverProblem>* createSolver(const SolverProblem& problem) {
#ifdef RANSAC_EXPERIMENT
  LmminSolver* solver = new LmminSolver(problem, NULL, false);
#else
  LmminSolver<SolverProblem>* solver = new LmminSolver<SolverProblem>(problem, NULL, false);
  // We must make large steps for things to move, else the gradient will be zero.
  solver->getControl().epsilon = 0.01;
  return solver;
#endif
}
}  // namespace

// -------------------------- Offline algorithm -----------------------------

const char* SampledStabilizationAlgorithm::docString =
    "An algorithm that minimizes photometric distorsions in space and time. The default configuration is: "
    "{\n"
    "  \"max_sampled_points\": 100000      # Stopping criterion 1. We'll stop after drawing that many sample points.\n"
    "  \"min_points_per_input\": 80        # Stopping criterion 2. Each input shall have at least min_points_per_input "
    "samples.\n"
    "  \"neighbourhood_size\": 5           # Size of the neighbourhood to use to compute luminosity.\n"
    "  \"first_frame\": 0                  # Restriction in time.\n"
    "  \"last_frame\": inf                 # Restriction in time.\n"
    "  \"time_step\": 60                   # Number of frames between two keyframes.\n"
    "  \"anchor\": 0                       # The input to use as anchor. If -1, anchor all inputs.\n"
    "  \"stabilize_wb\": false             # If true, also stabilizes white balance.\n"
    "  \"temporal\": false                 # If true, also stabilizes the global exposure / wb for temporal "
    "consistency.\n"
    "  \"preserve_outside\": false         # If true, use create keyframes on each side to preserve values ouside of "
    "the [first,last] range.\n"
    "  \"return_point_set\": false         # If true, returns the sampled point set.\n"
    "}\n";

SampledStabilizationAlgorithm::SampledStabilizationAlgorithm(const Ptv::Value* config)
    : SampledStabilizationBase(config),
      firstFrame(0),
      lastFrame(std::numeric_limits<int>::max()),
      timeStep(60),
      preserveOutside(false),
      returnPointSet(false) {
  if (config != NULL) {
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
    value = config->has("time_step");
    if (value && value->getType() == Ptv::Value::INT) {
      timeStep = (int)value->asInt();
      if (timeStep < 1) {
        timeStep = 1;
      }
    }
    value = config->has("temporal");
    if (value && value->getType() == Ptv::Value::BOOL) {
      temporalStabilization = value->asBool();
    }
    value = config->has("preserve_outside");
    if (value && value->getType() == Ptv::Value::BOOL) {
      preserveOutside = value->asBool();
    }
    value = config->has("return_point_set");
    if (value && value->getType() == Ptv::Value::BOOL) {
      returnPointSet = value->asBool();
    }
  }
}

Potential<Ptv::Value> SampledStabilizationAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                                           OpaquePtr**) const {
  if (progress && progress->notify("Sampling points", 0.0)) {
    return exposureAlgorithmCancelled();
  }

  Potential<SampledExposureStabilizationProblem> problem = createProblem(pano);
  FAIL_RETURN(problem.status());

  const std::unique_ptr<Solver<SolverProblem>> solver(createSolver(*problem.object()));

  // Parameters. Reuse the result from one iteration to the other as initial guess.
  std::vector<double> params;
  problem->computeInitialGuess(params);

  auto container = Core::ControllerInputFrames<PixelFormat::RGBA, uint32_t>::create(pano);
  FAIL_RETURN(container.status());

  for (int time = firstFrame; time < lastFrame; time += timeStep) {
    problem->setTime(time);

    FAIL_RETURN(container->seek(time));

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

    if (progress &&
        progress->notify("Stabilizing exposure", (100.0 * (time - firstFrame)) / (lastFrame - firstFrame + 1))) {
      return exposureAlgorithmCancelled();
    }

    sample(pano, succesfullyLoadedFrames, *problem.object());

    // Find the set of parameters that minimize spatial inconsitencies.
    std::vector<double> prevParams(params);  // Keep a copy in case we fail.
    {
      SIMPLEPROFILE_MS("solve");
      if (solver->run(params)) {
        problem->saveControlPoint(params);
      } else {
        params = prevParams;  // Reset to previous value.
        Logger::get(Logger::Verbose) << "Could not compute exposure for frame " << time << ", skipping." << std::endl;
      }
    }
  }

  if (!problem->injectSavedControlPoints(pano, preserveOutside, firstFrame, lastFrame)) {
    return Potential<Ptv::Value>(Status::OK());
  }

  Ptv::Value* returnValue = NULL;
  if (returnPointSet) {
    returnValue = Ptv::Value::emptyObject();
    std::vector<Ptv::Value*>& pointPairs = returnValue->get("homographies")->asList();
    for (std::vector<PointPair*>::const_iterator it = problem->getPointPairs().begin();
         it != problem->getPointPairs().end(); ++it) {
      Ptv::Value* pointPair = Ptv::Value::emptyObject();
      pointPairs.push_back(pointPair);
      Ptv::Value* point = Ptv::Value::emptyObject();
      point->get("x")->asDouble() = (*it)->p_k->coords().x;
      point->get("y")->asDouble() = (*it)->p_k->coords().y;
      point->get("input")->asInt() = (*it)->p_k->videoInputId();
      pointPair->asList().push_back(point);
      point = Ptv::Value::emptyObject();
      point->get("x")->asDouble() = (*it)->p_l->coords().x;
      point->get("y")->asDouble() = (*it)->p_l->coords().y;
      point->get("input")->asInt() = (*it)->p_l->videoInputId();
      pointPair->asList().push_back(point);
    }
  }

  if (progress) {
    progress->notify("Done", 100.0);
  }
  return returnValue ? Potential<Ptv::Value>(returnValue) : Potential<Ptv::Value>(Status::OK());
}

// -------------------------- Online algorithm -----------------------------

const char* SampledStabilizationOnlineAlgorithm::docString =
    "An algorithm that minimizes photometric distorsions in space and time. The default configuration is: "
    "{\n"
    "  \"max_sampled_points\": 100000      # Stopping criterion 1. We'll stop after drawing that many sample points.\n"
    "  \"min_points_per_input\": 80        # Stopping criterion 2. Each input shall have at least min_points_per_input "
    "samples.\n"
    "  \"neighbourhood_size\": 5           # Size of the neighbourhood to use to compute luminosity.\n"
    "  \"anchor\": 0                       # The input to use as anchor. If -1, anchor all inputs.\n"
    "  \"stabilize_wb\": false             # If true, also stabilizes white balance.\n"
    "}\n";

void clearBuffers(std::vector<GPU::HostBuffer<uint32_t>>& buffers) {
  for (auto buffer : buffers) {
    buffer.release();
  }
}

const std::unordered_map<std::string, std::pair<const Core::Curve& (Core::InputDefinition::*)(void)const,
                                                void (Core::InputDefinition::*)(Core::Curve*)>>
    SampledStabilizationOnlineAlgorithm::functionMap = {
        {"exposureValue", {&Core::InputDefinition::getExposureValue, &Core::InputDefinition::replaceExposureValue}},
        {"redCB", {&Core::InputDefinition::getRedCB, &Core::InputDefinition::replaceRedCB}},
        {"greenCB", {&Core::InputDefinition::getGreenCB, &Core::InputDefinition::replaceGreenCB}},
        {"blueCB", {&Core::InputDefinition::getBlueCB, &Core::InputDefinition::replaceBlueCB}}};

const mtime_t SampledStabilizationOnlineAlgorithm::InterpolationDurationMultiplier = 1000000;

SampledStabilizationOnlineAlgorithm::SampledStabilizationOnlineAlgorithm(const Ptv::Value* config)
    : SampledStabilizationBase(config), interpolationFixationFrames(5) {
  double runInterval = 0.6;
  VideoStitch::Parse::populateDouble("Ptv", *config, "run_interval", runInterval, false);
  int interpolationPercent = 50;
  VideoStitch::Parse::populateInt("Ptv", *config, "interpolation_interval_percent", interpolationPercent, false);
  VideoStitch::Parse::populateInt("Ptv", *config, "safety_margin_frames", interpolationFixationFrames, false);

  interpolationDuration = mtime_t((interpolationPercent / 100.) * InterpolationDurationMultiplier * runInterval);
}

Potential<Ptv::Value> SampledStabilizationOnlineAlgorithm::onFrame(
    Core::PanoDefinition& pano, std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& frames, mtime_t date,
    FrameRate frameRate, Util::OpaquePtr** /*ctx*/) {
  auto algorithmStartTime = std::chrono::steady_clock::now();
  auto stitcherStartFrame = frameRate.timestampToFrame(date);

  auto preservedCurves = preserveCurves(pano, stitcherStartFrame);

  if (frames.empty()) {
    return {Origin::ExposureAlgorithm, ErrType::InvalidConfiguration, "No input frames"};
  }

  Potential<SampledExposureStabilizationProblem> problem = createProblem(&pano);
  FAIL_RETURN(problem.status());

  problem->setTime(stitcherStartFrame);

  const std::unique_ptr<Solver<SolverProblem>> solver(createSolver(*problem.object()));

  std::vector<double> params;
  problem->computeInitialGuess(params);
  PROPAGATE_FAILURE_STATUS(processFrames(frames, pano, problem));

  // Find the set of parameters that minimize spatial inconsistencies.
  SIMPLEPROFILE_MS("solve");
  if (solver->run(params)) {
    problem->constantControlPoint(params);
    // This may seem redundant here as we then will replace result with new curves, but the thing is that here we don't
    // have access to new splines So pano is used as a transfer vehicle for the data.
    problem->injectSavedControlPoints(&pano, false, 0, 0);

    auto newDate = date + std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                                                algorithmStartTime)
                              .count();
    auto algorithmFinishFrame = frameRate.timestampToFrame(newDate) + interpolationFixationFrames;

    updateInputCurves(pano, preservedCurves, algorithmFinishFrame,
                      algorithmFinishFrame + frameRate.timestampToFrame(interpolationDuration));

    return Potential<Ptv::Value>(Status::OK());
  } else {
    return {Origin::ExposureAlgorithm, ErrType::RuntimeError,
            "Unable to compute a uniform exposure for the panorama.\n"
            "Please check that the geometric calibration of the camera array is correct and that there is enough "
            "overlap between the cameras.\n"
            "Exposure compensation will work best on static scenes with little movement."};
  }
}

Status SampledStabilizationOnlineAlgorithm::processFrames(
    const std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& frames, Core::PanoDefinition& pano,
    const Potential<SampledExposureStabilizationProblem>& problem) {
  std::vector<GPU::HostBuffer<uint32_t>> inputBuffers;

  // Copy the host buffers
  for (auto frame : frames) {
    auto hostBuffer =
        GPU::HostBuffer<uint32_t>::allocate(frame.second.width() * frame.second.height(), "Exposure Stabilization");
    if (hostBuffer.ok()) {
      const Status copyStatus = GPU::memcpyBlocking(hostBuffer.value().hostPtr(), frame.second);
      if (copyStatus.ok()) {
        inputBuffers.push_back(hostBuffer.value());
      } else {
        clearBuffers(inputBuffers);

        // Logger::get(Logger::Error) << "Exposure host error: " << Status::getErrorMessage(copyStatus.code()) <<
        // std::endl; return Potential<Ptv::Value>(copyStatus);
        return Status(Origin::ExposureAlgorithm, ErrType::OutOfResources,
                      "Can't allocate host memory for exposure stabilization", hostBuffer.status());
      }
    } else {
      clearBuffers(inputBuffers);
      return Status(Origin::ExposureAlgorithm, ErrType::RuntimeError,
                    "Can't copy host memory for exposure stabilization", hostBuffer.status());
    }
  }

  sample(&pano, inputBuffers, *problem.object());

  // Release the host buffers
  clearBuffers(inputBuffers);

  return Status();
}

std::unordered_map<std::string, std::vector<Core::Spline*>> SampledStabilizationOnlineAlgorithm::preserveCurves(
    const Core::PanoDefinition& panorama, frameid_t frame) {
  std::unordered_map<std::string, std::vector<Core::Spline*>> result;

  for (const auto& curveFunctions : functionMap) {
    for (const auto& input : panorama.getVideoInputs()) {
      result[curveFunctions.first].push_back(
          Core::Spline::point(frame, (input.get().*curveFunctions.second.first)().at(frame)));
    }
  }

  return result;
}

void SampledStabilizationOnlineAlgorithm::updateInputCurves(
    Core::PanoDefinition& panorama, std::unordered_map<std::string, std::vector<Core::Spline*>> preservedCurves,
    frameid_t algorithmFinishFrame, frameid_t interpolationFinishFrame) {
  for (const auto& curveFunctions : functionMap) {
    size_t counter = 0;
    for (auto& input : panorama.getVideoInputs()) {
      auto curveValue = (input.get().*curveFunctions.second.first)().at(0);
      auto& spline = preservedCurves[curveFunctions.first][counter];
      // At the beginning here spline is just a point, so finishFrame is ok here
      spline->cubicTo(algorithmFinishFrame, spline->at(algorithmFinishFrame))
          ->cubicTo(interpolationFinishFrame, curveValue)
          ->cubicTo(interpolationFinishFrame + interpolationFixationFrames, curveValue);
      (input.get().*curveFunctions.second.second)(new Core::Curve(spline));
      counter++;
    }
  }
}

void SampledStabilizationBase::sample(Core::PanoDefinition* pano, std::vector<GPU::HostBuffer<uint32_t>>& frames,
                                      SampledExposureStabilizationProblem& problem) const {
  SIMPLEPROFILE_MS("read");

  for (videoreaderid_t inputID = 0; inputID < (videoreaderid_t)frames.size(); inputID++) {
    auto frame = frames[inputID];
    /*{
      const Core::InputDefinition& input = pano->getVideoInput(pFrame->first);
      Util::PngReader writer;
      std::stringstream ss;
      ss << "expo-" << pFrame->first << ".png";
      writer.writeRGBAToFile(ss.str().c_str(), input.getWidth(), input.getHeight(), (void*)pFrame->second);
    }*/
    for (std::vector<PointPair*>::const_iterator it = problem.getPointPairs().begin();
         it != problem.getPointPairs().end(); ++it) {
      Point* p = NULL;
      if ((*it)->p_k->videoInputId() == inputID) {
        p = (*it)->p_k;
      } else if ((*it)->p_l->videoInputId() == inputID) {
        p = (*it)->p_l;
      } else {
        continue;
      }
      const int p_k_x = (int)p->coords().x;
      const int p_k_y = (int)p->coords().y;
      float3 accRgb = make_float3(0.0f, 0.0f, 0.0f);
      int numAcc = 0;
      const Core::InputDefinition& input = pano->getVideoInput(inputID);
      for (int y = std::max(p_k_y - neighbourhoodSize, 0);
           y <= std::min(p_k_y + neighbourhoodSize, (int)input.getHeight() - 1); ++y) {
        for (int x = std::max(p_k_x - neighbourhoodSize, 0);
             x <= std::min(p_k_x + neighbourhoodSize, (int)input.getWidth() - 1); ++x) {
          // Ignore any zero alpha (masked) pixels.
          uint32_t v = frame[y * input.getWidth() + x];
          if (Image::RGBA::a(v) != 0) {
            const float3 rgb =
                make_float3((float)Image::RGBA::r(v), (float)Image::RGBA::g(v), (float)Image::RGBA::b(v));
            if (!isNearlyBurnt(rgb)) {
              // Disable points that are over/underexposed.
              accRgb += rgb;
              ++numAcc;
            }
          }
        }
      }
      if (numAcc > 0) {
        const float3 a = (1.0f / (float)numAcc) * accRgb;
        const float3 c = problem.getPhotoTransform(inputID)->mapPhotoInputToLinear(input, p->coords(), a);
        p->setColor(c);
      } else {
        p->setNoColor();
      }
    }
  }
}

Potential<SampledExposureStabilizationProblem> SampledStabilizationBase::createProblem(
    Core::PanoDefinition* pano) const {
  auto problem = std::make_unique<SampledExposureStabilizationProblem>(
      *pano, maxSampledPoints, minPointsPerInput, neighbourhoodSize, anchor,
      stabilizeWB ? ExposureStabilizationProblemBase::WBParameterSet
                  : ExposureStabilizationProblemBase::EvParameterSet);

  // Make sure we have at least 1 point per input.
  if (problem->getMinPointsInOneOutput() == 0) {
    return {Origin::ExposureAlgorithm, ErrType::RuntimeError,
            "Unable to perform an exposure compensation. At least one input does not have a sufficiently large "
            "overlapping area with its neighbors."};
  }
  if (problem->getNumConnectedComponents() > 1) {
    return {Origin::ExposureAlgorithm, ErrType::RuntimeError,
            "Unable to perform an exposure compensation. There are too few overlapping areas between the inputs."};
  }
  return problem.release();
}

}  // namespace Util
}  // namespace VideoStitch
