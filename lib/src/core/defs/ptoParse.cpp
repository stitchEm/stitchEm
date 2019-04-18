// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// PTO doc:
// http://wiki.panotools.org/PTStitcher
// http://wiki.panotools.org/PTOptimizer

#ifndef __clang_analyzer__  // VSA-7043

#include "core/defs/panoInputDefsPimpl.hpp"
#include "core/transformGeoParams.hpp"
#include "util/base64.hpp"
#include "util/strutils.hpp"
#include "libvideostitch/logging.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sstream>
#include <clocale>

namespace VideoStitch {
namespace Core {

namespace {
/**
 * A context to pts context data.
 */
struct PtsContext {
  PtsContext()
      : useCameraCurve(false),
        hasGlobalExposure(false),
        wbr(0.0),
        wbb(0.0),
        eev(0.0),
        hasLastDummyImage(false),
        hasLastImage(false),
        lastWidth(0),
        lastHeight(0),
        hasLastExposureParams(false),
        lastExposureParams{},
        hasVigParams(false) {}

  // Global
  bool useCameraCurve;
  bool hasGlobalExposure;
  double wbr;  // white balance, red
  double wbb;  // white balance, blue
  double eev;
  // If the pts has dummy images, these are their ids.
  std::vector<size_t> dummyImageIds;
  double emorParams[5];

  // Per-input
  bool hasLastDummyImage;
  bool hasLastImage;
  int lastWidth;
  int lastHeight;
  char lastFilename[4096];
  bool hasLastExposureParams;
  std::array<double, 4> lastExposureParams;  // wbred, wbblue, flare, eev
  bool hasVigParams;
  double vigParams[5];
  std::string sourceMask;
};

double local_atof(const char* str) {
  double val;
  std::istringstream istr(str);
  istr.imbue(std::locale("C"));
  istr >> val;
  return val;
}

}  // namespace

Potential<PanoDefinition> PanoDefinition::Pimpl::mergeCalibrationIntoPano(const PanoDefinition* calibration,
                                                                          const PanoDefinition* pano) {
  if (calibration->numVideoInputs() != pano->numVideoInputs()) {
    std::stringstream msg;
    msg << "Cannot apply the calibration to the current panorama configuration. The calibration has ";
    msg << calibration->numVideoInputs() << " video inputs, the current panorama has " << pano->numVideoInputs();
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, msg.str()};
  }

  auto newPano = Potential<PanoDefinition>(pano->clone());

  for (videoreaderid_t i = 0; i < calibration->numVideoInputs(); ++i) {
    const Core::InputDefinition& calibInput = calibration->getVideoInputs()[i].get();
    Core::InputDefinition& newInput = newPano->getVideoInputs()[i].get();

    newInput.setMaskData(calibInput.getMaskData());
    newInput.setCropLeft(calibInput.getCropLeft());
    newInput.setCropRight(calibInput.getCropRight());
    newInput.setCropTop(calibInput.getCropTop());
    newInput.setCropBottom(calibInput.getCropBottom());
    newInput.setFormat(calibInput.getFormat());
    newInput.setUseMeterDistortion(calibInput.getUseMeterDistortion());

    GeometryDefinition def = calibInput.getGeometries().at(0);
    /*Temporary hack, horizontal focal actually contains fov*/
    /*Look for HFOV in InputDefinition::parseFromPtoLine switch for more infos about this hack*/
    /*Convert fovs to focals*/
    double fov = def.getHorizontalFocal();
    double focal = TransformGeoParams::computeHorizontalScale(newInput, fov);
    def.setHorizontalFocal(focal);
    newInput.replaceGeometries(new GeometryDefinitionCurve(def));
  }

  return newPano;
}

Potential<PanoDefinition> PanoDefinition::parseFromPto(const std::string& filename, const PanoDefinition* sourcePano) {
  std::unique_ptr<std::istream, VideoStitch::Util::IStreamDeleter> ifs(&std::cin);
  if (!(filename[0] == '-' && filename[1] == '\0')) {
    ifs = VideoStitch::Util::createIStream(filename);
  }

  if (!ifs->good()) {
    return Status(Origin::Stitcher, ErrType::InvalidConfiguration, "Cannot open file '" + filename + "' for reading.");
  }

  /*Change the numeric locale with locale("C") for avoiding the comma decimal separator issue */
  std::setlocale(LC_NUMERIC, "C");

  // TODO: parse 'm' line
  // m g1 i5 f0 m2 p0.00784314

  std::unique_ptr<PanoDefinition> pano(new PanoDefinition());

  PtsContext ptsContext;

  /*
  #-outputfile E:\VS\VideoStitch-assets\mappingPts01\PTGui-soft.jpg
  #-jpegparameters 100 0
  #-tiffparameters 8bit packbits alpha_assoc
  #-vignettingparams
  */

  // std::vector<Calibration::ControlPoint> controlPoints;
  // parse image lines
  std::string line;
  for (unsigned l = 0; !ifs->eof() && !ifs->fail(); ++l) {
    std::getline(*ifs, line);
    {
      // Remove windows eol.
      size_t len = line.size();
      if (len > 1 && line[len - 1] == '\r') {
        line[len - 1] = '\0';
      }
    }
    char* lineBuf = const_cast<char*>(line.c_str());
    switch (lineBuf[0]) {
      case 'o':
        // Pts uses o to mean i. But in that case there should be a started image.
        if (!(ptsContext.hasLastImage || ptsContext.hasLastDummyImage)) {
          Logger::get(Logger::Verbose) << "Ignored unimplemented '" << line[0] << "' entry at line " << l << "."
                                       << std::endl;
          break;
        }
      // fallthrough
      case 'i': {
        Potential<InputDefinition> input(InputDefinition::parseFromPtoLine(lineBuf, pano->pimpl->inputs));
        FAIL_RETURN(input.status());
        ptsContext.hasLastDummyImage = false;
        if (ptsContext.hasLastImage) {
          input->setWidth(ptsContext.lastWidth);
          input->setHeight(ptsContext.lastHeight);
          input->setFilename(ptsContext.lastFilename);
          ptsContext.hasLastImage = false;
        }
        if (ptsContext.hasGlobalExposure || ptsContext.hasLastExposureParams) {
          double logRedCB = ptsContext.wbr + ptsContext.lastExposureParams[0];
          double logBlueCB = ptsContext.wbb + ptsContext.lastExposureParams[1];
          input->replaceRedCB(new Curve(1.0 / pow(2.0, logRedCB)));
          input->replaceBlueCB(new Curve(1.0 / pow(2.0, logBlueCB)));
          // Green is computed automatically
          input->replaceGreenCB(new Curve(1.0 / pow(2.0, -(logRedCB + logBlueCB))));
          input->replaceExposureValue(new Curve(ptsContext.lastExposureParams[3]));
          pano->replaceExposureValue(new Curve(ptsContext.eev));
          ptsContext.hasLastExposureParams = false;
        }
        if (ptsContext.hasVigParams) {
          input->pimpl->vignettingCoeff0 = 1.0;
          input->pimpl->vignettingCoeff1 = ptsContext.vigParams[0];
          input->pimpl->vignettingCoeff2 = ptsContext.vigParams[1];
          input->pimpl->vignettingCoeff3 = ptsContext.vigParams[2];
          input->pimpl->vignettingCenterX = ptsContext.vigParams[3];
          input->pimpl->vignettingCenterY = ptsContext.vigParams[4];
        }
        if (ptsContext.useCameraCurve) {
          input->pimpl->emorA = ptsContext.emorParams[0];
          input->pimpl->emorB = ptsContext.emorParams[1];
          input->pimpl->emorC = ptsContext.emorParams[2];
          input->pimpl->emorD = ptsContext.emorParams[3];
          input->pimpl->emorE = ptsContext.emorParams[4];
          input->pimpl->photoResponse = InputDefinition::PhotoResponse::InvEmorResponse;
        }
        input->pimpl->maskData = Util::base64Decode(ptsContext.sourceMask);
        ptsContext.sourceMask.clear();
        pano->pimpl->inputs.push_back(input.release());
        break;
      }
      case 'p': {
        FAIL_RETURN(pano->readParams(lineBuf));
        break;
      }
      case '#':
        // TrX TrY TrZ j0 Va Vb Vc Vd Vx Vy Vm
        if (Util::startsWith(lineBuf, "#-")) {
          // read pts instructions.
          if (Util::startsWith(lineBuf, "#-cameracurve")) {
            // Equivalent to Ra Rb Rc Rd Re.
            sscanf(lineBuf, "#-cameracurve %lf %lf %lf %lf %lf", &ptsContext.emorParams[0], &ptsContext.emorParams[1],
                   &ptsContext.emorParams[2], &ptsContext.emorParams[3], &ptsContext.emorParams[4]);
          } else if (Util::startsWith(lineBuf, "#-pmoptcameracurvemode")) {
            // TODO: handle the several modes, but everything seems to boil down to the same thing.
            ptsContext.useCameraCurve = true;
          } else if (Util::startsWith(lineBuf, "#-wbexposure")) {
            // Format is Er Eb Eev. 0 means no correction.
            sscanf(lineBuf, "#-wbexposure %lf %lf %lf", &ptsContext.wbr, &ptsContext.wbb, &ptsContext.eev);
            ptsContext.hasGlobalExposure = true;
          } else if (Util::startsWith(lineBuf, "#-dummyimage")) {
            ptsContext.dummyImageIds.push_back(pano->numInputs());
            ptsContext.hasLastDummyImage = true;
          } else if (Util::startsWith(lineBuf, "#-imgfile")) {
            ptsContext.hasLastImage = (sscanf(lineBuf, "#-imgfile %i %i \"%[^\"]\"", &ptsContext.lastWidth,
                                              &ptsContext.lastHeight, ptsContext.lastFilename) == 3);
          } else if (Util::startsWith(lineBuf, "#-viewpoint")) {
            std::vector<double> lastViewpoint(5, 0);
            bool hasLastViewpoint =
                (sscanf(lineBuf, "#-viewpoint %lf %lf %lf %lf %lf", &lastViewpoint[0], &lastViewpoint[1],
                        &lastViewpoint[2], &lastViewpoint[3], &lastViewpoint[4]) == 5);
            if (hasLastViewpoint && (lastViewpoint[0] != 0.0 || lastViewpoint[1] != 0.0 || lastViewpoint[2] != 0.0 ||
                                     lastViewpoint[3] != 0.0 || lastViewpoint[4] != 0.0)) {
              return {Origin::PanoramaConfiguration, ErrType::UnsupportedAction,
                      "Cannot import this PTGui file: the support for PTGui viewpoint correction has been deprecated."};
            }
          } else if (Util::startsWith(lineBuf, "#-vignettingparams")) {
            ptsContext.hasVigParams =
                (sscanf(lineBuf, "#-vignettingparams %lf %lf %lf %lf %lf", &ptsContext.vigParams[0],
                        &ptsContext.vigParams[1], &ptsContext.vigParams[2], &ptsContext.vigParams[3],
                        &ptsContext.vigParams[4]) == 5);
          } else if (Util::startsWith(lineBuf, "#-exposureparams")) {
            ptsContext.hasLastExposureParams =
                (sscanf(lineBuf, "#-exposureparams %lf %lf %lf %lf", &ptsContext.lastExposureParams[0],
                        &ptsContext.lastExposureParams[1], &ptsContext.lastExposureParams[2],
                        &ptsContext.lastExposureParams[3]) == 4);
          } else if (Util::startsWith(lineBuf, "#-sourcemask")) {
            ptsContext.sourceMask = line.substr(13);
          } else {
            Logger::get(Logger::Verbose) << "Ignoring pts option lineBuf '" << lineBuf << "'" << std::endl;
          }
        } else if (Util::startsWith(lineBuf, "# PTGui Trial Project File")) {
          return {Origin::PanoramaConfiguration, ErrType::UnsupportedAction,
                  "Cannot import from encrypted PTGui trial project files. Please export your project files using a "
                  "full version of PTGui."};
        }
        break;
      case '\0':
        break;
      // empty line, ignore
      case 'c': {
        // control points
        // int index0, index1;
        // float x0, y0, x1, y1;
        // sscanf(lineBuf, "c n%d N%d x%f y%f X%f Y%f", &index0, &index1, &x0, &y0, &x1, &y1);
        // const int64_t w0 = pano->getInput(index0).getWidth();
        // const int64_t h0 = pano->getInput(index0).getHeight();
        // const int64_t w1 = pano->getInput(index1).getWidth();
        // const int64_t h1 = pano->getInput(index1).getHeight();
        // controlPoints.push_back(Calibration::ControlPoint((Calibration::Node)index0, (Calibration::Node)index1,
        //                        x0 - (float)w0 / 2.0f, y0 - (float)h0 / 2.0f, x1 - (float)w1 / 2.0f, y1 - (float)h1
        //                        / 2.0f, -1, -1.0));
        break;
      }
      case 'v':
        break;
      // variable to optimize, ignore
      case '\n':
        break;
      // empty line, ignore
      case '\r':
        // empty line, ignore
        break;
      default:
        Logger::get(Logger::Verbose) << "Ignored unimplemented '" << line[0] << "' entry at line " << l << "."
                                     << std::endl;
        break;
    }
  }

  for (std::vector<size_t>::const_reverse_iterator it = ptsContext.dummyImageIds.rbegin();
       it != ptsContext.dummyImageIds.rend(); ++it) {
    InputDefinition* dummy = pano->pimpl->inputs[*it];
    delete dummy;
    pano->pimpl->inputs.erase(pano->pimpl->inputs.begin() + *it);
  }

  if (sourcePano != nullptr) {
    return PanoDefinition::Pimpl::mergeCalibrationIntoPano(pano.get(), sourcePano);
  }

  return Potential<PanoDefinition>(pano.release());
}

namespace {
enum IState {
  WhiteSpace,
  Name,
  Width,
  Height,
  IFormat,
  RedCB,
  BlueCB,
  ExposureValue,
  EmorA,
  EmorB,
  EmorC,
  EmorD,
  EmorE,
  ResponseType,
  LensDistA,
  LensDistB,
  LensDistC,
  CenterX,
  CenterY,
  ShearA,
  ShearB,
  VA,
  VB,
  VC,
  VD,
  VX,
  VY,
  TrX,
  TrY,
  TrZ,
  Pitch,
  Roll,
  Yaw,
  HFOV,
  Vm,
  U,
  CropLeft,
  CropRight,
  CropTop,
  CropBottom,
  FileName,
  TimeOffset,
  Stack,
  TranslationPlaneYaw,
  TranslationPlanePitch,
};
}

#define GETINTPARAM(name)                                                   \
  if (useSame) {                                                            \
    int prevId = atoi(sym);                                                 \
    if (prevId < 0 || prevId >= (int)prevInputs.size()) {                   \
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, \
              "No such previous input '" + std::string(sym) + "'"};         \
    }                                                                       \
    newInput->pimpl->name = prevInputs[prevId]->pimpl->name;                \
  } else {                                                                  \
    newInput->pimpl->name = atoi(sym);                                      \
  }

#define GETINTPARAM2(getter, setter)                                        \
  if (useSame) {                                                            \
    int prevId = atoi(sym);                                                 \
    if (prevId < 0 || prevId >= (int)prevInputs.size()) {                   \
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, \
              "No such previous input '" + std::string(sym) + "'"};         \
    }                                                                       \
    newInput->setter(prevInputs[prevId]->getter());                         \
  } else {                                                                  \
    newInput->setter(atoi(sym));                                            \
  }

#define GETFLOATPARAM(name)                                                 \
  if (useSame) {                                                            \
    int prevId = atoi(sym);                                                 \
    if (prevId < 0 || prevId >= (int)prevInputs.size()) {                   \
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, \
              "No such previous input '" + std::string(sym) + "'"};         \
    }                                                                       \
    newInput->pimpl->name = prevInputs[prevId]->pimpl->name;                \
  } else {                                                                  \
    newInput->pimpl->name = local_atof(sym);                                \
  }

#define GETFLOATPARAM2(getter, setter)                                      \
  if (useSame) {                                                            \
    int prevId = atoi(sym);                                                 \
    if (prevId < 0 || prevId >= (int)prevInputs.size()) {                   \
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, \
              "No such previous input '" + std::string(sym) + "'"};         \
    }                                                                       \
    setter(prevInputs[prevId]->getter());                                   \
  } else {                                                                  \
    setter(local_atof(sym));                                                \
  }

#define GETCURVEPARAM(name)                                                 \
  if (useSame) {                                                            \
    int prevId = atoi(sym);                                                 \
    if (prevId < 0 || prevId >= (int)prevInputs.size()) {                   \
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, \
              "No such previous input '" + std::string(sym) + "'"};         \
    }                                                                       \
    newInput->replace##name(prevInputs[prevId]->get##name().clone());       \
  } else {                                                                  \
    newInput->replace##name(new Core::Curve(local_atof(sym)));              \
  }

#define SETWHITESPACESTATE \
  state = WhiteSpace;      \
  sym = NULL;              \
  break

namespace {
int tagToInt(const char* tag, size_t tagLen) {
  switch (tagLen) {
    case 1:
      return tag[0];
    case 2:
      return 256 * tag[0] + tag[1];
    case 3:
      return 256 * 256 * tag[0] + 256 * tag[1] + tag[2];
    default:
      return -1;
  }
}

Status photoResponseFromInt(int v, InputDefinition::PhotoResponse& response) {
  switch (v) {
    case 0:
      response = InputDefinition::PhotoResponse::EmorResponse;
      return Status::OK();
    case 1:
      response = InputDefinition::PhotoResponse::LinearResponse;
      return Status::OK();
    case 2:
      response = InputDefinition::PhotoResponse::GammaResponse;
      return Status::OK();
    case 3:
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
              "Loading camera response from file is currently not supported"};
    case 4:
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
              "ICC camera responses are currently not supported."};
    default:
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
              "Unknown camera response type '" + std::to_string(v) + "'"};
  }
}
}  // namespace

Potential<ReaderInputDefinition> ReaderInputDefinition::parseFromPtoLine(
    char* line, const std::vector<ReaderInputDefinition*>& prevInputs) {
  /*
    i w3456 h2304 f0 Eb1 Eev0 Er1 Ra0 Rb0 Rc0 Rd0 Re0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 a0 b-0.0127103 c0 d0 e0 g0 p2.19134
    r-5.76063 t123 v61.2111 y-47.9372  Vm5 u10 n"/home/clem/photos/panoramas/pano7/img_6845.jpg"
  */
  std::unique_ptr<ReaderInputDefinition> newInput(new ReaderInputDefinition());

  const char* const lpEnd = line + strlen(line);
  const char* sym = NULL;

  bool useSame = false;  // parameters can reference other parameters with "=<imageId>" syntax, e.g. "v=0"

  IState state = WhiteSpace;
  for (char* lp = line + 1; lp < lpEnd + 1; ++lp) {
    switch (state) {
      case WhiteSpace:
        if (*lp == ' ' || *lp == '\t' || *lp == '\0' || *lp == '\r' || *lp == '\n') {
          continue;
        } else {
          // begin reading a name
          state = Name;
          sym = lp;
        }
        break;
      case Name:
        // names are [a-zA-Z_]
        if (!(('a' <= *lp && *lp <= 'z') || ('A' <= *lp && *lp <= 'Z') || (*lp == '_'))) {
          size_t len = lp - sym;
          switch (tagToInt(sym, len)) {
            case 'w':
              state = Width;
              break;
            case 'h':
              state = Height;
              break;
            case 'n':
              state = FileName;
              break;
            case 'T' * 256 + 'O':
              state = TimeOffset;
              break;
            default: {
              std::stringstream msg;
              msg << "Invalid parameter '" << std::string(sym, len) << "' in image line. Context:" << std::endl;
              msg << "  " << line << std::endl;
              return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, msg.str()};
            }
          }
          useSame = false;
          if (state == FileName) {
            if (*lp != '"') {
              std::stringstream msg;
              msg << "Invalid string parameter in image line. Context:" << std::endl;
              msg << "  " << line << std::endl;
              return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, msg.str()};
            } else {
              sym = lp + 1;
            }
          } else if (*lp == '=') {
            sym = lp + 1;
            useSame = true;
          } else {
            sym = lp;
          }
        }
        break;
      case FileName:
        // FIXME: handle escaping
        if (*lp == '"') {
          newInput->setFilename(std::string(sym, lp - sym));
          state = WhiteSpace;
          sym = NULL;
        }
        break;
      default:
        //[0-9.eE]
        if (!(('0' <= *lp && *lp <= '9') || *lp == 'e' || *lp == 'E' || *lp == '.' || *lp == '-')) {
          char c = *lp;
          *lp = 0;
          switch (state) {
            case Width:
              GETINTPARAM2(getWidth, setWidth);
              SETWHITESPACESTATE;
            case Height:
              GETINTPARAM2(getHeight, setHeight);
              SETWHITESPACESTATE;
            case TimeOffset:
              GETINTPARAM2(getFrameOffset, setFrameOffset);
              SETWHITESPACESTATE;
            case WhiteSpace:
            case Name:
            case FileName:
            default:
              return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                      "Invalid state for ReaderInputDefinition. Context: " + std::string(line)};
          }
          *lp = c;
        }
        break;
    }
  }

  return Potential<ReaderInputDefinition>(newInput.release());
}

Potential<InputDefinition> InputDefinition::parseFromPtoLine(char* line,
                                                             const std::vector<InputDefinition*>& prevInputs) {
  /*
    i w3456 h2304 f0 Eb1 Eev0 Er1 Ra0 Rb0 Rc0 Rd0 Re0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 a0 b-0.0127103 c0 d0 e0 g0 p2.19134
    r-5.76063 t123 v61.2111 y-47.9372  Vm5 u10 n"/home/clem/photos/panoramas/pano7/img_6845.jpg"
  */
  std::unique_ptr<InputDefinition> newInput(new InputDefinition());

  const char* const lpEnd = line + strlen(line);
  const char* sym = NULL;

  bool useSame = false;  // parameters can reference other parameters with "=<imageId>" syntax, e.g. "v=0"

  GeometryDefinition def;

  IState state = WhiteSpace;
  for (char* lp = line + 1; lp < lpEnd + 1; ++lp) {
    switch (state) {
      case WhiteSpace:
        if (*lp == ' ' || *lp == '\t' || *lp == '\0' || *lp == '\r' || *lp == '\n') {
          continue;
        } else {
          // begin reading a name
          state = Name;
          sym = lp;
        }
        break;
      case Name:
        // names are [a-zA-Z_]
        if (!(('a' <= *lp && *lp <= 'z') || ('A' <= *lp && *lp <= 'Z') || (*lp == '_'))) {
          size_t len = lp - sym;
          switch (tagToInt(sym, len)) {
            case 'w':
              state = Width;
              break;
            case 'h':
              state = Height;
              break;
            case 'f':
              state = IFormat;
              break;
            case 'a':
              state = LensDistA;
              break;
            case 'b':
              state = LensDistB;
              break;
            case 'c':
              state = LensDistC;
              break;
            case 'd':
              state = CenterX;
              break;
            case 'e':
              state = CenterY;
              break;
            case 'g':
              state = ShearA;
              break;
            case 't':
              state = ShearB;
              break;
            case 'p':
              state = Pitch;
              break;
            case 'r':
              state = Roll;
              break;
            case 'y':
              state = Yaw;
              break;
            case 'v':
              state = HFOV;
              break;
            case 'u':
              state = U;
              break;
            case 'C':
            case 'S':
              state = CropLeft;
              break;
            case 'n':
              state = FileName;
              break;
            case 'j':
              state = Stack;
              break;
            case 'E' * 256 + 'r':
              state = RedCB;
              break;
            case 'E' * 256 + 'b':
              state = BlueCB;
              break;
            case 'R' * 256 + 'a':
              state = EmorA;
              break;
            case 'R' * 256 + 'b':
              state = EmorB;
              break;
            case 'R' * 256 + 'c':
              state = EmorC;
              break;
            case 'R' * 256 + 'd':
              state = EmorD;
              break;
            case 'R' * 256 + 'e':
              state = EmorE;
              break;
            case 'R' * 256 + 't':
              state = ResponseType;
              break;
            case 'V' * 256 + 'a':
              state = VA;
              break;
            case 'V' * 256 + 'b':
              state = VB;
              break;
            case 'V' * 256 + 'c':
              state = VC;
              break;
            case 'V' * 256 + 'd':
              state = VD;
              break;
            case 'V' * 256 + 'x':
              state = VX;
              break;
            case 'V' * 256 + 'y':
              state = VY;
              break;
            case 'V' * 256 + 'm':
              state = Vm;
              break;
            case 'V' * 256 + 'f':
              return {Origin::PanoramaConfiguration, ErrType::UnsupportedAction,
                      "Flatfield vignetting correction (Vf) is currently not supported"};
            case 'T' * 256 + 'O':
              state = TimeOffset;
              break;
            case 'E' * 256 * 256 + 'e' * 256 + 'v':
              state = ExposureValue;
              break;
            case 'T' * 256 * 256 + 'r' * 256 + 'X':
              state = TrX;
              break;
            case 'T' * 256 * 256 + 'r' * 256 + 'Y':
              state = TrY;
              break;
            case 'T' * 256 * 256 + 'r' * 256 + 'Z':
              state = TrZ;
              break;
            case 'T' * 256 * 256 + 'p' * 256 + 'y':
              state = TranslationPlaneYaw;
              break;
            case 'T' * 256 * 256 + 'p' * 256 + 'p':
              state = TranslationPlanePitch;
              break;
            default: {
              std::stringstream msg;
              msg << "Invalid parameter '" << std::string(sym, len) << "' in image line. Context:" << std::endl;
              msg << "  " << line << std::endl;
              return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, msg.str()};
            }
          }
          useSame = false;
          if (state == FileName) {
            if (*lp != '"') {
              std::stringstream msg;
              msg << "Invalid string parameter in image line. Context:" << std::endl;
              msg << "  " << line << std::endl;
              return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, msg.str()};
            } else {
              sym = lp + 1;
            }
          } else if (*lp == '=') {
            sym = lp + 1;
            useSame = true;
          } else {
            sym = lp;
          }
        }
        break;
      case FileName:
        // FIXME: handle escaping
        if (*lp == '"') {
          newInput->setFilename(std::string(sym, lp - sym));
          state = WhiteSpace;
          sym = NULL;
        }
        break;
      default:
        //[0-9.eE]
        if (!(('0' <= *lp && *lp <= '9') || *lp == 'e' || *lp == 'E' || *lp == '.' || *lp == '-')) {
          char c = *lp;
          *lp = 0;
          switch (state) {
            case Width:
              GETINTPARAM2(getWidth, setWidth);
              SETWHITESPACESTATE;
            case Height:
              GETINTPARAM2(getHeight, setHeight);
              SETWHITESPACESTATE;
            case IFormat:
              if (useSame) {
                newInput->pimpl->format = prevInputs[atoi(sym)]->pimpl->format;
              } else if (!fromPTFormat(sym, &newInput->pimpl->format)) {
                std::stringstream msg;
                msg << "Unsupported format '" << sym << "' in image line. Context:" << std::endl;
                msg << "  " << line << std::endl;
                return {Origin::PanoramaConfiguration, ErrType::UnsupportedAction, msg.str()};
              }
              SETWHITESPACESTATE;
            case RedCB:
              GETCURVEPARAM(RedCB);
              SETWHITESPACESTATE;
            case BlueCB:
              GETCURVEPARAM(BlueCB);
              SETWHITESPACESTATE;
            case ExposureValue:
              GETCURVEPARAM(ExposureValue);
              SETWHITESPACESTATE;
            case EmorA:
              GETFLOATPARAM(emorA);
              SETWHITESPACESTATE;
            case EmorB:
              GETFLOATPARAM(emorB);
              SETWHITESPACESTATE;
            case EmorC:
              GETFLOATPARAM(emorC);
              SETWHITESPACESTATE;
            case EmorD:
              GETFLOATPARAM(emorD);
              SETWHITESPACESTATE;
            case EmorE:
              GETFLOATPARAM(emorE);
              SETWHITESPACESTATE;
            case ResponseType:
              FAIL_CAUSE(photoResponseFromInt(atoi(sym), newInput->pimpl->photoResponse), Origin::PanoramaConfiguration,
                         ErrType::InvalidConfiguration, "Unsupported photo response. Context: " + std::string(line));
              SETWHITESPACESTATE;
            case LensDistA:
              GETFLOATPARAM2(getGeometries().at(0).getDistortA, def.setDistortA);
              SETWHITESPACESTATE;
            case LensDistB:
              GETFLOATPARAM2(getGeometries().at(0).getDistortB, def.setDistortB);
              SETWHITESPACESTATE;
            case LensDistC:
              GETFLOATPARAM2(getGeometries().at(0).getDistortC, def.setDistortC);
              SETWHITESPACESTATE;
            case CenterX:
              GETFLOATPARAM2(getGeometries().at(0).getCenterX, def.setCenterX);
              SETWHITESPACESTATE;
            case CenterY:
              GETFLOATPARAM2(getGeometries().at(0).getCenterY, def.setCenterY);
              SETWHITESPACESTATE;
            case ShearA:
              if (local_atof(sym) != 0.0) {
                Logger::get(Logger::Warning) << "Warning: No support for shear, ignoring." << std::endl;
              }
              SETWHITESPACESTATE;
            case ShearB:
              if (local_atof(sym) != 0.0) {
                Logger::get(Logger::Warning) << "Warning: No support for shear, ignoring." << std::endl;
              }
              SETWHITESPACESTATE;
            case VA:
              GETFLOATPARAM(vignettingCoeff0);
              SETWHITESPACESTATE;
            case VB:
              GETFLOATPARAM(vignettingCoeff1);
              SETWHITESPACESTATE;
            case VC:
              GETFLOATPARAM(vignettingCoeff2);
              SETWHITESPACESTATE;
            case VD:
              GETFLOATPARAM(vignettingCoeff3);
              SETWHITESPACESTATE;
            case VX:
              GETFLOATPARAM(vignettingCenterX);
              SETWHITESPACESTATE;
            case VY:
              GETFLOATPARAM(vignettingCenterY);
              SETWHITESPACESTATE;
            case TrX:
              SETWHITESPACESTATE;
            case TrY:
              SETWHITESPACESTATE;
            case TrZ:
              SETWHITESPACESTATE;
            case TranslationPlaneYaw:
              GETFLOATPARAM(huginTranslationPlaneYaw);
              if (fabs(newInput->pimpl->huginTranslationPlaneYaw) > 0.0001) {
                Logger::get(Logger::Warning)
                    << "Warning: No support for Hugin's translation plane Yaw (Tpy)." << std::endl;
              }
              SETWHITESPACESTATE;
            case TranslationPlanePitch:
              GETFLOATPARAM(huginTranslationPlanePitch);
              if (fabs(newInput->pimpl->huginTranslationPlanePitch) > 0.0001) {
                Logger::get(Logger::Warning)
                    << "Warning: No support for Hugin's translation plane Pitch (Tpp)." << std::endl;
              }
              SETWHITESPACESTATE;
            case Pitch:
              GETFLOATPARAM2(getGeometries().at(0).getPitch, def.setPitch);
              SETWHITESPACESTATE;
            case Roll:
              GETFLOATPARAM2(getGeometries().at(0).getRoll, def.setRoll);
              SETWHITESPACESTATE;
            case Yaw:
              GETFLOATPARAM2(getGeometries().at(0).getYaw, def.setYaw);
              SETWHITESPACESTATE;
            case HFOV:
              /*Temporary hack, we store it there*/
              GETFLOATPARAM2(getGeometries().at(0).getHorizontalFocal, def.setHorizontalFocal);
              SETWHITESPACESTATE;
            case Vm:
              // radial vignetting is default
              SETWHITESPACESTATE;
            case U:
              // GETFLOATPARAM(u);
              SETWHITESPACESTATE;
            case CropLeft:
              GETINTPARAM(cropLeft);
              state = CropRight;
              sym = lp + 1;
              break;
            case CropRight:
              GETINTPARAM(cropRight);
              state = CropTop;
              sym = lp + 1;
              break;
            case CropTop:
              GETINTPARAM(cropTop);
              state = CropBottom;
              sym = lp + 1;
              break;
            case CropBottom:
              GETINTPARAM(cropBottom);
              SETWHITESPACESTATE;
            case TimeOffset:
              GETINTPARAM2(getFrameOffset, setFrameOffset);
              SETWHITESPACESTATE;
            case Stack:
              GETINTPARAM(stack);
              SETWHITESPACESTATE;
            case WhiteSpace:
            case Name:
            case FileName:
              return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                      "Invalid state for InputDefinition. Context: " + std::string(line)};
          }
          *lp = c;
        }
        break;
    }
  }

  if (def.getHorizontalFocal() == 0.0) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Horizontal field of view not specified"};
  }

  newInput->replaceGeometries(new VideoStitch::Core::GeometryDefinitionCurve(def));

  return Potential<InputDefinition>(newInput.release());
}

namespace {
enum PState {
  PWhiteSpace,
  PName,
  PFormat,
  PWidth,
  PHeight,
  PExposure,
  PHFOV,
  POptions,
  PCropLeft,
  PCropRight,
  PCropTop,
  PCropBottom,
  PR,  // LDR/HDR. Ignore.
  PIgnored
};
}

Status PanoDefinition::readParams(char* line) {
  /*
    p f2 w13116 h4342 v254  E0 R0 n"TIFF_m c:NONE r:CROP"
  */

  const char* const lpEnd = line + strlen(line);
  const char* sym = NULL;
  bool needsDownRotation = false;

  PState state = PWhiteSpace;
  for (char* lp = line + 1; lp < lpEnd + 1; ++lp) {
    switch (state) {
      case WhiteSpace:
        if (*lp == ' ' || *lp == '\t' || *lp == '\0' || *lp == '\r' || *lp == '\n') {
          continue;
        } else {
          // begin reading a name
          state = PName;
          sym = lp;
        }
        break;
      case PName:
        // names are [a-zA-Z]
        if (!(('a' <= *lp && *lp <= 'z') || ('A' <= *lp && *lp <= 'Z') || (*lp == '_'))) {
          size_t len = lp - sym;
          if (len == 1) {
            switch (sym[0]) {
              case 'w':
                state = PWidth;
                break;
              case 'h':
                state = PHeight;
                break;
              case 'f':
                state = PFormat;
                break;
              case 'v':
                state = PHFOV;
                break;
              case 'E':
                state = PExposure;
                break;
              case 'R':
                state = PR;
                break;  // LDR(0) /HDR(1). Ignore.
              case 'n':
                state = POptions;
                break;
              case 'S':
                state = PCropLeft;
                break;
              case 'u':
              // state = PFeathering; break; Ignore.
              case 'k':
              case 'b':
              case 'd':
                Logger::get(Logger::Verbose)
                    << "Ignoring parameter '" << std::string(sym, len) << "' in panomara line. Context:" << std::endl;
                Logger::get(Logger::Verbose) << "  " << line << std::endl;
                state = PIgnored;
                break;
              default: {
                std::stringstream msg;
                msg << "Invalid parameter '" << std::string(sym, len) << "' in panomara line. Context:" << std::endl;
                msg << "  " << line << std::endl;
                return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, msg.str()};
              }
            }
          } else {
            // Try to parse a projection (for PTS).
            Core::PanoProjection format;
            if (fromPTSFormat(std::string(sym, len), &format)) {
              // Hack to transform fstereographic_down into fstereographic + rotation
              if (std::string(sym, len) == "fstereographic_down") {
                needsDownRotation = true;
              }
              state = PWhiteSpace;
              sym = NULL;
              pimpl->projection = format;
            } else {
              std::stringstream msg;
              msg << "invalid parameter '" << std::string(sym, len) << "' in panomara line. Context:" << std::endl;
              msg << "  " << line << std::endl;
              return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, msg.str()};
            }
          }
          if (state == POptions) {
            if (*lp != '"') {
              std::stringstream msg;
              msg << "invalid string parameter in panomara line. Context:" << std::endl;
              msg << "  " << line << std::endl;
              return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, msg.str()};
            } else {
              sym = lp + 1;
            }
          } else {
            sym = lp;
          }
        }
        break;
      case POptions:
        // FIXME: handle escaping
        if (*lp == '"') {
          // throw it away
          state = PWhiteSpace;
          sym = NULL;
        }
        break;
      default:
        //[0-9.eE]
        if (!(('0' <= *lp && *lp <= '9') || *lp == 'e' || *lp == 'E' || *lp == '.' || *lp == '-')) {
          char c = *lp;
          *lp = 0;
          switch (state) {
            case PWidth:
              setWidth(atoi(sym));
              state = PWhiteSpace;
              sym = NULL;
              break;
            case PHeight:
              setHeight(atoi(sym));
              state = PWhiteSpace;
              sym = NULL;
              break;
            case PFormat: {
              Core::PanoProjection format;
              if (!fromPTFormat(sym, &format)) {
                std::stringstream msg;
                msg << "Unsupported format '" << sym << "' in pano line. Context:" << std::endl;
                msg << "  " << line << std::endl;
                return {Origin::PanoramaConfiguration, ErrType::UnsupportedAction, msg.str()};
              }
              pimpl->projection = format;
              state = PWhiteSpace;
              sym = NULL;
              break;
            }
            case PExposure:
              replaceExposureValue(new Core::Curve(local_atof(sym)));
              state = PWhiteSpace;
              sym = NULL;
              break;
            case PHFOV:
              setHFOV(local_atof(sym));
              state = PWhiteSpace;
              sym = NULL;
              break;
            case PCropLeft:
              Logger::get(Logger::Warning) << "Warning: No support for cropping, ignoring." << std::endl;
              state = PCropRight;
              sym = lp + 1;
              break;
            case PCropRight:
              Logger::get(Logger::Warning) << "Warning: No support for cropping, ignoring." << std::endl;
              state = PCropTop;
              sym = lp + 1;
              break;
            case PCropTop:
              Logger::get(Logger::Warning) << "Warning: No support for cropping, ignoring." << std::endl;
              state = PCropBottom;
              sym = lp + 1;
              break;
            case PCropBottom:
              Logger::get(Logger::Warning) << "Warning: No support for cropping, ignoring." << std::endl;
              state = PWhiteSpace;
              sym = NULL;
              break;
            case PR:  // LDR/HDR. Ignore.
              Logger::get(Logger::Warning) << "Warning: No support for HDR output, ignoring." << std::endl;
              state = PWhiteSpace;
              sym = NULL;
              break;
            case PIgnored:
              state = PWhiteSpace;
              sym = NULL;
              break;
            default: {
              std::stringstream msg;
              msg << "Invalid state. Context:" << std::endl;
              msg << "  " << line << std::endl;
              return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, msg.str()};
            }
          }
          *lp = c;
        }
        break;
    }
  }

  if (getHFOV() == 0.0) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Horizontal field of view not specified"};
  }
  if (getWidth() == 0) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Image width not specified"};
  }
  if (getHeight() == 0) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Image height not specified"};
  }

  if (needsDownRotation) {
    // FIXME: should be: add 90 degrees, not set constant.
    replaceGlobalOrientation(new QuaternionCurve(Quaternion<double>::fromEulerZXY(0.0, M_PI / 2.0, 0.0)));
  }
  return Status::OK();
}

/*
#-pathseparator \
#-encoding utf8
#-fileversion 40
#-previewwidth 1132
#-previewheight 566
#-vfov 180
#-resolution 300
#-fixaspect 1
#-ccdcrop 1
#-hasbeenoptimized 1
#-hvcpmode 1
#-morphmode 2
#-psdparameters 8bit packbits layered
#-qtvrparameters 800 600 1 1000 70 0 0 -180 180 0 -90 90 90 10 120 1
#-honorexiforientation 1
#-exrparameters noalpha
#-hdroutputhdrblended
#-hdroutputtonemapped
#-hdrfileformat hdr
#-hdrmethod fuse
#-hdrpsdparameters float none layered
#-tonemapsettings 0 0 0.5 1 0 20 0 0 2 0.27 0.67 0.06
#-fusesettings 0.5 0 0.2 0 0
#-pmoptexposuremode auto
#-pmoptvignettingmode enabled
#-pmoptwbmode disabled
#-pmoptflaremode disabled
#-pmoptcameracurvemode auto
#-blendweight 100 100 100
#-optviewpoint 000
#-colorcorrectlayers
#-useexif1
#-batchbuilder_useexif 0
#-stitcher ptgui
#-blender ptgui
#-blenderfeather 1
#-optimizer ptgui
#-interpolator default
#-autocpdone
#-imgrotate444
#-cpinactive
#-imginactive
#-linktoprevious
#-previewinactive
#-outputcrop 0 1 0 1
#-morphcp
#-nooptcp
#-alignsettings_generatecp 1
#-alignsettings_optimize 1
#-alignsettings_optimizeprealign 1
#-alignsettings_straighten 1
#-alignsettings_fit 1
#-alignsettings_chooseprojection 1
#-alignsettings_setoptimumsize 1
#-alignsettings_limitsize 500
#-alignsettings_optimizeexposure 0
#-hdrsettings_defaultlinkmode nolink
#-hdrsettings_donotask 0
#-batchsettings_align 0
#-batchsettings_stitch 1
#-batchsettings_stitchonlyifcontrolpoints 1
#-defaultprojectfilenamemode firstsourceimage
#-defaultprojectfilename_custom " Panorama"
#-defaultprojectfoldermode sourcefolder
#-defaultprojectfolder_custom ""
#-defaultpanoramafilenamemode asproject
#-defaultpanoramafilename_custom ""
#-defaultpanoramafoldermode projectfolder
#-defaultpanoramafolder_custom ""
#-userelativesourceimagepaths 1
#-optimizeraskreinitialize 1
#-applytemplate_lens 1
#-applytemplate_imageparams 1
#-applytemplate_crop 1
#-applytemplate_mask 1
#-applytemplate_panoramasettings 1
#-applytemplate_projectsettings 1
#-applytemplate_optimizer 1
#-globalcrop 0 0 0 0 0 -0.0046875 0.001041666666666667 0.3151041666666667
#-theoreticalhfov -1
#-rect_compression_x 0
#-rect_compression_y 0
#-cylindrical_compression_y 0
#-transverse_cylindrical_compression_x 0
#-vedutismo_compression_x 1
#-transverse_vedutismo_compression_y 1
#-stereographic_compression 1
#-rectifisheye_compression 1
*/

}  // namespace Core
}  // namespace VideoStitch

#endif  // __clang_analyzer__
