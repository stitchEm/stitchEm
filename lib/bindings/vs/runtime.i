%include <stdint.i>
%include <std_string.i>
%include <std_vector.i>
%include <std_shared_ptr.i>

#define VS_EXPORT

/* Parse the header file to generate wrappers.
 * ORDER MATTERS A LOT HERE, beware of dependencies!!! */

/* Not implemented yet */
%ignore VideoStitch::Output::StereoWriter::getExpectedFrameSizeFor;
%ignore VideoStitch::Output::StereoWriter::getExpectedFrameSize;
%ignore VideoStitch::Audio::AudioPath::addSink;
%ignore VideoStitch::Audio::AudioPath::AudioPath;
%ignore VideoStitch::Audio::AudioPath::setSinks;
%ignore VideoStitch::Audio::AudioPath::removeSink;

%include "../../include/libvideostitch/config.hpp"
%include "../../include/libvideostitch/logging.hpp"
%include "../../include/libvideostitch/status.hpp"
%include "../../include/libvideostitch/audio.hpp"
%include "../../include/libvideostitch/allocator.hpp"
%template(PotentialSourceSurface) VideoStitch::Potential<VideoStitch::Core::SourceSurface>;
%template(PotentialPanoSurface) VideoStitch::Potential<VideoStitch::Core::PanoSurface>;
%template(PotentialSourceOpenGLSurface) VideoStitch::Potential<VideoStitch::Core::SourceOpenGLSurface>;
%template(PotentialPanoOpenGLSurface) VideoStitch::Potential<VideoStitch::Core::PanoOpenGLSurface>;
%template(SourceRendererPtr) std::shared_ptr<VideoStitch::Core::SourceRenderer>;
%template(PanoRendererPtr) std::shared_ptr<VideoStitch::Core::PanoRenderer>;
%template(SourceRendererPtrVector) std::vector<std::shared_ptr<VideoStitch::Core::SourceRenderer>>;
%template(PanoRendererPtrVector) std::vector<std::shared_ptr<VideoStitch::Core::PanoRenderer>>;

%include "../../include/libvideostitch/ptv.hpp"
%template(PtvValuePtrVector) std::vector<VideoStitch::Ptv::Value*>;

%ignore VideoStitch::PotentialValue::PotentialValue(T&&);

%ignore VideoStitch::GPU::operator<<;
%include "../../include/libvideostitch/frame.hpp"

%ignore VideoStitch::Core::operator<<;
%include "../../include/libvideostitch/gpu_device.hpp"

%include "../../include/libvideostitch/outputEventManager.hpp"
%include "../../include/libvideostitch/output.hpp"
%template(Writer) VideoStitch::Potential<VideoStitch::Output::Output>;

%include "../../include/libvideostitch/input.hpp"
%include "../../include/libvideostitch/inputFactory.hpp"
%include "../../include/libvideostitch/imageMergerFactory.hpp"
%template(PotentialMergerFactory) VideoStitch::Potential<VideoStitch::Core::ImageMergerFactory>;

%include "../../include/libvideostitch/imageWarperFactory.hpp"
%template(PotentialWarperFactory) VideoStitch::Potential<VideoStitch::Core::ImageWarperFactory>;

%include "../../include/libvideostitch/imageFlowFactory.hpp"
%template(PotentialFlowFactory) VideoStitch::Potential<VideoStitch::Core::ImageFlowFactory>;

%include "../../include/libvideostitch/stitchOutput.hpp"
%template(StitchOutput) VideoStitch::Core::StitcherOutput<VideoStitch::Output::VideoWriter>;
%template(PotentialExtractOutput) VideoStitch::Potential<VideoStitch::Core::ExtractOutput>;
%template(PotentialStitchOutput) VideoStitch::Potential<VideoStitch::Core::StitcherOutput<VideoStitch::Output::VideoWriter>>;
%template(PotentialOnlineAlgorithm) VideoStitch::Potential<VideoStitch::Util::OnlineAlgorithm>;
%template(ExtractOutputPtrVector) std::vector<VideoStitch::Core::ExtractOutput*>;
%template(StitchOutputPtrVector) std::vector<VideoStitch::Core::StitchOutput*>;


%include "../../include/libvideostitch/controller.hpp"

/* StitcherController inherits getFrameRate from InputController 
 * But Swig doesn't seem to catch that (because it's virtual inheritance)
 * Workaround: extend Controller with new method, let C++ compiler do the virtual inheritance resolving
 */
%extend VideoStitch::Core::StitcherController {
  FrameRate getFrameRateFromInputController() const override {
    return $self->getFrameRate();
  }
}

%template(Controller) VideoStitch::Core::StitcherController<VideoStitch::Core::StitchOutput, VideoStitch::Core::PanoDeviceDefinition>;
%template(PotentialController) VideoStitch::Potential<VideoStitch::Core::Controller, VideoStitch::Core::ControllerDeleter<VideoStitch::Core::Controller>>;
%template(OutputWriterPtrVector) std::vector<std::shared_ptr<VideoStitch::Core::StitcherController< VideoStitch::Core::StitcherOutput< VideoStitch::Output::VideoWriter >,PanoDeviceDefinition >::Output::Writer>>;

