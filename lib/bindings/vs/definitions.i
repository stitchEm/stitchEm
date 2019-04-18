%include <attribute.i>
%include <std_string.i>
%include <stdint.i>

#define VS_EXPORT

/* Parse the header file to generate wrappers.
 * ORDER MATTERS A LOT HERE, beware of dependencies!!! */

%ignore *::operator=; //Ignore operator = warning
%warnfilter(314, 320, 325, 401, 503, 509); //Ignoring warnings. Check SWIG doc for more details

%include "../../include/libvideostitch/config.hpp"
%include "../../include/libvideostitch/object.hpp"
%include "../../include/libvideostitch/status.hpp"
%include "../../include/libvideostitch/frame.hpp"
%template(PotentialDouble) VideoStitch::PotentialValue<double>;

%include "../../include/libvideostitch/algorithm.hpp"

%include "extensions.i"

%ignore VideoStitch::Core::ResponseCurve::TotalLutSize;
%ignore VideoStitch::Core::ResponseCurve::LutSize;
%include "../../include/libvideostitch/emor.hpp"

%include "../../include/libvideostitch/matrix.hpp"
%template(Vector) VideoStitch::Vector3<double>;
%template(Matrix) VideoStitch::Matrix33<double>;

%include "../../include/libvideostitch/quaternion.hpp"
%template(Quat) VideoStitch::Quaternion<double>;


%include "../../include/libvideostitch/curves.hpp"
%template(Point) VideoStitch::Core::PointTemplate<double>;
%template(QuaternionPoint) VideoStitch::Core::PointTemplate<VideoStitch::Quaternion<double> >;
%template(Splines) VideoStitch::Core::SplineTemplate<double>;
%template(QuaternionSplines) VideoStitch::Core::SplineTemplate<VideoStitch::Quaternion<double> >;
%template(Curve) VideoStitch::Core::CurveTemplate<double>;
%template(QuaternionCurve) VideoStitch::Core::CurveTemplate<VideoStitch::Quaternion<double> >;

%include "../../include/libvideostitch/geometryDef.hpp"
%attribute(VideoStitch::Core::GeometryDefinition, double, lens_dist_A, getDistortA, setDistortA);
%attribute(VideoStitch::Core::GeometryDefinition, double, lens_dist_B, getDistortB, setDistortB);
%attribute(VideoStitch::Core::GeometryDefinition, double, lens_dist_C, getDistortC, setDistortC);
%attribute(VideoStitch::Core::GeometryDefinition, double, lens_dist_center_x, getCenterX, setCenterX);
%attribute(VideoStitch::Core::GeometryDefinition, double, lens_dist_center_y, getCenterY, setCenterY);
%attribute(VideoStitch::Core::GeometryDefinition, double, yaw, getYaw, setYaw);
%attribute(VideoStitch::Core::GeometryDefinition, double, pitch, getPitch, setPitch);
%attribute(VideoStitch::Core::GeometryDefinition, double, roll, getRoll, setRoll);

%include "../../include/libvideostitch/readerInputDef.hpp"
%include "../../include/libvideostitch/inputDef.hpp"
%attribute(VideoStitch::Core::InputDefinition, int, width, getWidth);
%attribute(VideoStitch::Core::InputDefinition, int, height, getHeight);
%attribute(VideoStitch::Core::InputDefinition, int, cropped_width, getCroppedWidth);
%attribute(VideoStitch::Core::InputDefinition, int, cropped_height, getCroppedHeight);
%attribute(VideoStitch::Core::InputDefinition, int, crop_left, getCropLeft, setCropLeft);
%attribute(VideoStitch::Core::InputDefinition, int, crop_right, getCropRight, setCropRight);
%attribute(VideoStitch::Core::InputDefinition, int, crop_top, getCropTop, setCropTop);
%attribute(VideoStitch::Core::InputDefinition, int, crop_bottom, getCropBottom, setCropBottom);
%attribute(VideoStitch::Core::InputDefinition, double, emor_a, getEmorA, setEmorA);
%attribute(VideoStitch::Core::InputDefinition, double, emor_b, getEmorB, setEmorB);
%attribute(VideoStitch::Core::InputDefinition, double, emor_c, getEmorC, setEmorC);
%attribute(VideoStitch::Core::InputDefinition, double, emor_d, getEmorD, setEmorD);
%attribute(VideoStitch::Core::InputDefinition, double, emor_e, getEmorE, setEmorE);
%attribute(VideoStitch::Core::InputDefinition, double, gamma, getGamma, setGamma);
%attribute(VideoStitch::Core::InputDefinition, double, vignetting_0, getVignettingCoeff0, setVignettingCoeff0);
%attribute(VideoStitch::Core::InputDefinition, double, vignetting_1, getVignettingCoeff1, setVignettingCoeff1);
%attribute(VideoStitch::Core::InputDefinition, double, vignetting_2, getVignettingCoeff2, setVignettingCoeff2);
%attribute(VideoStitch::Core::InputDefinition, double, vignetting_3, getVignettingCoeff3, setVignettingCoeff3);
%attribute(VideoStitch::Core::InputDefinition, double, vignetting_center_x, getVignettingCenterX, setVignettingCenterX);
%attribute(VideoStitch::Core::InputDefinition, double, vignetting_center_y, getVignettingCenterY, setVignettingCenterY);
%attribute(VideoStitch::Core::InputDefinition, double, vignetting_center_y, getVignettingCenterY, setVignettingCenterY);
%attribute(VideoStitch::Core::InputDefinition, int, frame_offset, getFrameOffset, setFrameOffset);
%attribute(VideoStitch::Core::InputDefinition, double, synchro_cost, getSynchroCost, setSynchroCost);

%include "../../include/libvideostitch/projections.hpp"

%include "../../include/libvideostitch/controlPointListDef.hpp"

%include "../../include/libvideostitch/panoDef.hpp"
%attribute(VideoStitch::Core::PanoDefinition, int, width, getWidth, setWidth);
%attribute(VideoStitch::Core::PanoDefinition, int, height, getHeight, setHeight);
%attribute(VideoStitch::Core::PanoDefinition, double, hfov, getHFOV, setHFOV);
%attribute(VideoStitch::Core::PanoDefinition, double, vfov, getVFOV, setVFOV);

%include "../../include/libvideostitch/panoramaDefinitionUpdater.hpp"
%include "../../include/libvideostitch/audioPipeDef.hpp"

%ignore VideoStitch::Audio::Samples::clone;
%include "../../include/libvideostitch/audio.hpp"

%include "../../include/libvideostitch/parse.hpp"
%template(PotentialParser) VideoStitch::Potential<VideoStitch::Ptv::Parser>;

// Because swig doesent know anything about function otherwise and will leak memory,
// but this way it seems to work with it just fine -_-
namespace std {
template <typename T>
class function{};
}

%template(StringPayloadFunction) std::function<void(const std::string&)>;
%template(PreventLeakFunction) std::function< VideoStitch::Potential< VideoStitch::Core::PanoDefinition >
(VideoStitch::Core::PanoDefinition const &) >;

%include "../../include/libvideostitch/orah/imuStabilization.hpp"
%include "../../include/libvideostitch/orah/exposureData.hpp"

%include <std_map.i>
namespace std {
%template(ExposureMap) map<videoreaderid_t, VideoStitch::Metadata::Exposure>;
%template(WhiteBalanceMap) map<videoreaderid_t, VideoStitch::Metadata::WhiteBalance>;
%template(ToneCurveMap) map<videoreaderid_t, VideoStitch::Metadata::ToneCurve>;
}

%include "../../include/libvideostitch/input.hpp"

%template(MetadataReadStatus) VideoStitch::Result<VideoStitch::Input::MetadataReader::MetadataReadStatusCode>;
%template(PotentialReader) VideoStitch::Potential<VideoStitch::Input::Reader>;
