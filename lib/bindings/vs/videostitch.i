%module(directors="1") vs


/* Beware, SWIG doesn't like its own proxy objects in template declaration.
 * When declaring proxy objects like that, take care of using
 * the exact same type than the functions supposed to return
 * it/ take it as argument, not a SWIG alias!
 * Spell everything out!
 */

%feature ("flatnested");
//Some naming conflicts
%rename (AudioReaderSpec) VideoStitch::Input::AudioReader::Spec;
%rename (MetadataSpec) VideoStitch::Input::MetadataReader::Spec;

%feature("autodoc", "1");

// generate directors for all virtual methods in class
%feature("director") VideoStitch::Core::AlgorithmOutput::Listener;
%feature("director") AlgorithmListenerGIL;
%feature("director") CppCallback;
%feature("director") RendererFunctor;

%feature("valuewrapper") VideoStitch::Potential;

%include "vs_doxygen.i"

%include "definitions.i"
%include "runtime.i"

#define VS_EXPORT

%include "../../include/libvideostitch/plugin.hpp"

/* What's in this block is copied verbatim to the
 * wrapper file. */
%{
#include "../../include/libvideostitch/output.hpp"
#include "../../include/libvideostitch/algorithm.hpp"
#include "../../include/libvideostitch/allocator.hpp"
#include "../../include/libvideostitch/audio.hpp"
#include "../../include/libvideostitch/controller.hpp"
#include "../../include/libvideostitch/curves.hpp"
#include "../../include/libvideostitch/emor.hpp"
#include "../../include/libvideostitch/geometryDef.hpp"
#include "../../include/libvideostitch/imageMergerFactory.hpp"
#include "../../include/libvideostitch/imageWarperFactory.hpp"
#include "../../include/libvideostitch/imageFlowFactory.hpp"
#include "../../include/libvideostitch/readerInputDef.hpp"
#include "../../include/libvideostitch/inputDef.hpp"
#include "../../include/libvideostitch/inputFactory.hpp"
#include "../../include/libvideostitch/input.hpp"
#include "../../include/libvideostitch/logging.hpp"
#include "../../include/libvideostitch/matrix.hpp"
#include "../../include/libvideostitch/object.hpp"
#include "../../include/libvideostitch/outputEventManager.hpp"
#include "../../include/libvideostitch/output.hpp"
#include "../../include/libvideostitch/controlPointListDef.hpp"
#include "../../include/libvideostitch/audioPipeDef.hpp"
#include "../../include/libvideostitch/panoDef.hpp"
#include "../../include/libvideostitch/panoramaDefinitionUpdater.hpp"
#include "../../include/libvideostitch/parse.hpp"
#include "../../include/libvideostitch/frame.hpp"
#include "../../include/libvideostitch/plugin.hpp"
#include "../../include/libvideostitch/postprocessor.hpp"
#include "../../include/libvideostitch/preprocessor.hpp"
#include "../../include/libvideostitch/processorStitchOutput.hpp"
#include "../../include/libvideostitch/profile.hpp"
#include "../../include/libvideostitch/projections.hpp"
#include "../../include/libvideostitch/ptv.hpp"
#include "../../include/libvideostitch/quaternion.hpp"
#include "../../include/libvideostitch/rigDef.hpp"

#include "../../include/libvideostitch/stitchOutput.hpp"
#include "../../include/libvideostitch/orah/imuStabilization.hpp"

#define MS_NO_COREDLL

using namespace VideoStitch;
using namespace VideoStitch::Core;
using namespace VideoStitch::Output;
using namespace VideoStitch::Util;
using namespace VideoStitch::Audio;
using namespace VideoStitch::Stab;
%}
