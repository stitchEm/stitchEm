from os import path as osp
import gc

import defaults
import vs
import errors
import logging

from utils.ptv import PTV
from utils.settings_manager import SETTINGS
from video_modes import VIDEO_MODES
from calibration_filter import CALIBRATION_FILTER
from clientmessenger import CLIENT_MESSENGER
from system import audio_device_manager


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ProjectManager(object):
    """
    Class to manage project files
    """

    AUDIO_NOAUDIO = "noaudio"
    AUDIO_INPUTS_SUPPORTING_DELAY = ["portaudio"]
    AUDIO_PRESET_FOLDER = "audio"
    AUDIO_INPUT_PRESET = "{}_input.preset"
    AUDIO_PIPELINE_PRESET = "audio_pipe.preset"
    AMB_DEC_PRESET = "ambisonic-fuma-decoding.preset"
    PORTAUDIO_INPUT_PTV = PTV.from_file(osp.join(defaults.SYSTEM_PRESETS_DIR_PATH,
                                             AUDIO_PRESET_FOLDER,
                                             AUDIO_INPUT_PRESET.format("portaudio")))

    OVERLAYLOGO_PRESET_FOLDER = "overlay"
    OVERLAYLOGO_INPUT_PRESET = "overlay_logo.preset"


    def __init__(self, project_file_path, project_save_path=None):
        vs.setDefaultBackendDevice(0)
        self.calibration_string = None
        self.resolution = SETTINGS.resolution

        self.delay_adjusts = {}
        #listing the audio sources will update the delays
        self.list_audio_sources()

        self.project_file_path = project_file_path
        self.project_save_path = project_save_path if project_save_path else project_file_path
        self.reset(False)

        self.audio_pipe = None

        if SETTINGS.enable_ev_compensation and SETTINGS.enable_metadata_processing:
            raise errors.ConfigurationError('Conflict: both exposure algorithm and metadata processing are enabled')

    def update_audio_delays(self):
        for source in self.audio_sources:
            if self.audio_sources[source]["input"] in self.AUDIO_INPUTS_SUPPORTING_DELAY\
                    and source not in self.delay_adjusts:
                self.delay_adjusts[source] = SETTINGS.audio_delays.get(source, 0.0)

    @property
    def panorama(self):
        if self.controller:
            return self.controller.getPano()

    def terminate(self):
        self.cleanup()

    def cleanup(self):
        self.merger_conf = None
        self.merger = None
        self.controller = None

    def reset(self, force_default):
        if (force_default):
            self.project_file_path = defaults.SYSTEM_DEFAULT_PTV_PATH
        logger.info('Loading ' + self.project_file_path)

        self.cleanup()
        try:
            ptv = PTV.from_file(self.project_file_path)
        except errors.ParserFileNotFound:
            logger.info("current configuration {} could not be loaded "
                         "Switching to default configuration.".format(self.project_file_path))
            return self.reset(True)
        except errors.ParserError as error:
            logger.warn("current configuration {} seems corrupted : {} "
                         "Switching to default configuration.".format(self.project_file_path, error))
            CLIENT_MESSENGER.send_event(defaults.RESET_TO_DEFAULT_MESSAGE)
            return self.reset(True)

        # TODO : merge data from saved panoramas

        self.ptv = ptv
        self._configure_video_debug(ptv)
        self._configure_audio(ptv)
        self._configure_overlay(ptv)
        
        ptv["pano"]["width"] = VIDEO_MODES[self.resolution]["width"]
        ptv["pano"]["height"] = VIDEO_MODES[self.resolution]["height"]

        self.ptv = ptv

        if self.calibration_string:
            self.apply_calibration(self.calibration_string)
        else:
            self.project_config = ptv.to_config()

    def apply_calibration(self, ptv_string):
        self.calibration_string = ptv_string
        calibration_ptv = PTV.from_string(ptv_string)
        calibration_ptv.filter(calibration_ptv, CALIBRATION_FILTER)
        self.ptv.merge(calibration_ptv["pano"], self.ptv["pano"])
        self.project_config = self.ptv.to_config()

    def _configure_audio(self, ptv):

        if self.audio_source is None:
            return

        audio_pipe_ptv = PTV.from_file(osp.join(defaults.SYSTEM_PRESETS_DIR_PATH,
                                                self.AUDIO_PRESET_FOLDER,
                                                self.AUDIO_PIPELINE_PRESET))

        current_source_data = self.audio_sources[self.audio_source]
        if (self.audio_layout):
            chanelLayout = vs.getChannelLayoutFromString(self.audio_layout)
            adjust_nb_channels = vs.getNbChannelsFromChannelLayout(chanelLayout)\
                if current_source_data["input_adjust_layout"] else None
        else:
            adjust_nb_channels = None
        #for the moment support only simple cases with all channels from one reader
        adjust_nb_channels_reader_id = None
        #sources relies on types, there is one inputs preset per type
        for audio_input in set(self.audio_sources[source_name]["input"] for source_name in self.audio_sources):
            audio_input_ptv = PTV.from_file(osp.join(defaults.SYSTEM_PRESETS_DIR_PATH,
                                                     self.AUDIO_PRESET_FOLDER,
                                                     self.AUDIO_INPUT_PRESET.format(audio_input)))
            if audio_input == current_source_data["input"]:
                readers = audio_input_ptv["inputs"]
                for reader_index, reader in enumerate(readers):
                    if "reader_config" in reader.keys():
                        # set number of channels if needed
                        if adjust_nb_channels:
                            adjust_nb_channels_reader_id = reader_index
                            reader["reader_config"]["audio_channels"] = adjust_nb_channels
                        # if selected source is of type portaudio, set the name of the source
                        if audio_input == "portaudio" and "type" in reader["reader_config"].keys()\
                                and reader["reader_config"]["type"] == "portaudio":
                            reader["reader_config"]["name"] = current_source_data["device"]

            ptv.merge(audio_input_ptv, ptv["pano"])

        audio_pipe_ptv["audio_selected"] = current_source_data["input"]
        # adjust input channel sources if needed
        if adjust_nb_channels_reader_id is not None:
            for audio_pipe_input in audio_pipe_ptv["audio_pipe"]["audio_inputs"]:
                if audio_pipe_input["name"] == current_source_data["input"]:
                    audio_pipe_input["sources"] = []
                    audio_pipe_input["layout"] = self.audio_layout
                    for channel_index in range(adjust_nb_channels):
                        audio_pipe_input["sources"].append({
                            "reader_id": adjust_nb_channels_reader_id,
                            "channel" : channel_index })

        ptv.merge(audio_pipe_ptv)

    def _configure_overlay(self, ptv):
        overlay_input_ptv  = PTV.from_file(osp.join(defaults.SYSTEM_PRESETS_DIR_PATH,
                                                    self.OVERLAYLOGO_PRESET_FOLDER,
                                                    self.OVERLAYLOGO_INPUT_PRESET))

        for input in overlay_input_ptv["overlays"]:
            inputname = input["reader_config"]
            input["reader_config"] = osp.join(defaults.SYSTEM_PRESETS_DIR_PATH,
                                              self.OVERLAYLOGO_PRESET_FOLDER,
                                              inputname)
                                        
        if SETTINGS.with_logo:
            ptv.merge(overlay_input_ptv, ptv["pano"])
            
    def _configure_video_debug(self, ptv):
        if SETTINGS.procedural:
            for input in ptv["pano"]["inputs"]:
                input["reader_config"] = {
                    "type": "procedural",
                    "name": "frameNumber",
                    "color": "ff0000"}

        ptv["pano"]["inputs"] = ptv["pano"]["inputs"][0:SETTINGS.input_size]

    def _init_panorama(self):
        panorama_configuration = self.project_config.has("pano")
        if not panorama_configuration:
            raise errors.StitcherError('Parsing: No pano found in PTV')
        panorama = vs.PanoDefinition_create(panorama_configuration)
        if not panorama:
            raise errors.StitcherError('Cannot create pano definition')
        panorama.thisown = 1
        return panorama

    def _init_audiopipe(self):
        audiopipe_configuration = self.project_config.has("audio_pipe")
        if not audiopipe_configuration:
            logger.warning('Parsing: No audio pipe found in PTV. So create a default one.')
            self.audio_pipe = vs.AudioPipeDefinition_createDefault()
        else:
            self.audio_pipe = vs.AudioPipeDefinition_create(audiopipe_configuration)
            # init the delay if current inputs supports it
            current_source_data = self.audio_sources[self.audio_source]
            if current_source_data["input"] in self.AUDIO_INPUTS_SUPPORTING_DELAY:
                adjusted_delay = SETTINGS.audio_base_delay + self.delay_adjusts[self.audio_source]
                adjusted_delay = min(adjusted_delay, self.audio_pipe.getMaxDelayValue())
                adjusted_delay = max(0.0, adjusted_delay)
                self.audio_pipe.setDelay(current_source_data["input"], adjusted_delay)
            amb_dec = PTV.from_file(osp.join(defaults.SYSTEM_PRESETS_DIR_PATH, self.AUDIO_PRESET_FOLDER, self.AMB_DEC_PRESET))
            self.audio_pipe.setAmbDecodingCoef(amb_dec.to_config())

        if not self.audio_pipe:
            raise errors.StitcherError('Cannot create an audio pipe.')

        self.audio_pipe.thisown = 1

    def get_save_panorama(self):
        """
        Prepare config before saving it to disk
        :return:
        """
        result = self.panorama.clone()
        result.thisown = 1
        result.resetExposure()

        return result

    def create_controller(self):
        self.cleanup()
        gc.collect()
        self.merger_conf = self.project_config.has("merger")
        if not self.merger_conf:
            raise errors.StitcherError('Parsing: No merger found in PTV')

        self.merger = vs.ImageMergerFactory_createMergerFactory(self.merger_conf)
        self.merger.thisown = 1
        if not self.merger.ok():
            raise errors.StitcherError('Cannot create merger')

        self.warper = vs.ImageWarperFactory_createWarperFactory(self.project_config.has("warper"))
        self.warper.thisown = 1

        self.flow = vs.ImageFlowFactory_createFlowFactory(self.project_config.has("flow"))
        self.flow.thisown = 1

        input_factory = vs.DefaultReaderFactory(0, -1)
        input_factory.thisown = 0  # Pass the ownership to controller

        self._init_audiopipe()

        controller = vs.createController(self._init_panorama(),
                                        self.merger.object(),
                                        self.warper.object(),
                                        self.flow.object(),
                                        input_factory,
                                        self.audio_pipe)
        if not controller.ok():
            raise errors.StitcherError('Cannot create controller:' + str(controller.status().getErrorMessage()))

        # set audio source
        if self.audio_source:
            current_source_data = self.audio_sources[self.audio_source]
            controller.setAudioInput(current_source_data["input"])
            # set the audio preprocessor Orah Audio Sync
            controller.setAudioPreProcessor('OrahAudioSync', 0)

        controller.enableMetadataProcessing(SETTINGS.enable_metadata_processing)
        controller.enableStabilization(SETTINGS.enable_stabilization)
        controller.setUserOrientation( vs.Quat(
            SETTINGS.orientation_quaternion[0],
            SETTINGS.orientation_quaternion[1],
            SETTINGS.orientation_quaternion[2],
            SETTINGS.orientation_quaternion[3]
        ));

        self.controller = controller
        self._check_pano_size()
        return controller

    def update(self, name, ptv_object, auto_save=None):
        self.project_config.push(name, ptv_object)

        auto_save = auto_save if auto_save else SETTINGS.save_algorithm_results
        if auto_save:
            self.save()

    def save(self, output_file=None):
        """Save current configuration to file"""
        output_file = self.project_save_path if output_file is None else output_file
        with open(output_file, mode="w") as project_file:
            project_file.write(self.project_config.getJsonStr())

    def get_panorama_size(self):
        return self.ptv["pano"]["width"], self.ptv["pano"]["height"]

    def get_preview_size(self):
        return self.ptv["pano"]["width"] / 2, self.ptv["pano"]["height"] / 2

    def get_num_inputs(self):
        return len(self.ptv["pano"]["inputs"])

    def get_input_size(self):
        return self.ptv["pano"]["inputs"][0]["width"], self.ptv["pano"]["inputs"][0]["height"]

    def get_crop(self, number):
        input_def = self.panorama.getInput(number)
        return input_def.getCroppedWidth(), input_def.getCroppedHeight()

    def _check_pano_size(self):
        """check that the panorama size from the project manager is fitting our available video modes
        """
        found_video_mode = None
        width, height = self.get_panorama_size()
        for video_mode in VIDEO_MODES.keys():
            if VIDEO_MODES[video_mode]["width"] == width and VIDEO_MODES[video_mode]["height"] == height:
                found_video_mode = video_mode
                break
        if not found_video_mode:
            raise errors.StitcherError("Video mode specified in the project is not valid or not supported")

    def get_video_modes(self):
        return VIDEO_MODES

    def _check_video_mode(self, name):
        if not name in VIDEO_MODES:
            raise errors.StitcherVideoModeInvalidError("video mode {} is invalid".format(name))

    def list_audio_sources(self):
        self.audio_sources = audio_device_manager.list_sources()
        if SETTINGS.current_audio_source != self.AUDIO_NOAUDIO:
            if SETTINGS.current_audio_source not in self.audio_sources:
                SETTINGS.current_audio_source = defaults.DEFAULT_OPTIONS["current_audio_source"]
            self.audio_source = str(SETTINGS.current_audio_source)

            if SETTINGS.current_audio_layout not in self.audio_sources[self.audio_source]["layouts"]:
                SETTINGS.current_audio_layout = self.audio_sources[self.audio_source]["layouts"][0]
            self.audio_layout = str(SETTINGS.current_audio_layout)
        else:
            self.audio_source = None
            self.audio_layout = None

        self.update_audio_delays()

        return self.audio_sources

    def get_audio_source(self):
        return self.audio_source if self.audio_source else "null"

    def get_audio_layout(self):
        return self.audio_layout if self.audio_layout else "null"

    def set_audio_source(self, source, layout):
        if source not in self.audio_sources.keys():
            raise errors.AudioInvalidSourceError
        if layout not in self.audio_sources[source]["layouts"]:
            raise errors.AudioInvalidSourceError
        SETTINGS.current_audio_source = source
        SETTINGS.current_audio_layout = layout
        self.audio_source = SETTINGS.current_audio_source
        self.audio_layout = SETTINGS.current_audio_layout

    def get_audio_delay(self, source):
        if source not in self.audio_sources.keys():
            raise errors.AudioInvalidSourceError("audio source {} is invalid".format(source))
        if self.audio_sources[source]["input"] not in self.AUDIO_INPUTS_SUPPORTING_DELAY:
            raise errors.AudioDelaySourceError("audio source {} has no adjustable delay".format(source))
        return self.delay_adjusts[source]

    def set_audio_delay(self, source, delay_s):
        if source not in self.audio_sources.keys():
            raise errors.AudioInvalidSourceError("audio source {} is invalid".format(source))
        if self.audio_sources[source]["input"] not in self.AUDIO_INPUTS_SUPPORTING_DELAY:
            raise errors.AudioDelaySourceError("audio source {} has no adjustable delay".format(source))
        self.delay_adjusts[source] = delay_s
        SETTINGS.audio_delays = self.delay_adjusts;
        if self.audio_pipe:
            adjusted_delay = delay_s + SETTINGS.audio_base_delay
            if not 0 <= adjusted_delay <= self.audio_pipe.getMaxDelayValue():
                raise errors.AudioInvalidDelayValueError(
                    "audio delay value {} (adjusted : {}) is invalid".format(delay_s, adjusted_delay))
            self.audio_pipe.setDelay(self.audio_sources[source]["input"], adjusted_delay)
            self.controller.applyAudioProcessorParam(self.audio_pipe)

    def set_resolution(self, resolution):
        self._check_video_mode(resolution)
        SETTINGS.resolution = resolution
        self.resolution = resolution

    def reset_exposure(self):
        pano_copy_no_exposure = self.get_save_panorama()
        status = self.controller.updatePanorama(pano_copy_no_exposure)
        if not status.ok():
            raise errors.StitcherError("Could not reset exposure values")

        self.update("pano", pano_copy_no_exposure.serialize())

    def start_metadata_processing(self):
        if self.controller:
            self.controller.enableMetadataProcessing(True)

    def stop_metadata_processing(self):
        if self.controller:
            self.controller.enableMetadataProcessing(False)

    def start_imu_stabilization(self):
        SETTINGS.enable_stabilization = True
        # reset user orientation value as it is done automatically in the library
        SETTINGS.orientation_quaternion = [1.0, 0.0, 0.0, 0.0]
        if self.controller:
            self.controller.enableStabilization(True)

    def stop_imu_stabilization(self):
        SETTINGS.enable_stabilization = False
        # reset user orientation value as it is done automatically in the library
        SETTINGS.orientation_quaternion = [1.0, 0.0, 0.0, 0.0]
        if self.controller:
            self.controller.enableStabilization(False)

    def is_imu_stabilization_enabled(self):
        return SETTINGS.enable_stabilization

    def set_stabilization_low_pass_filter(self, param):
        if self.controller:
            stab = self.controller.getStabilizationIMU()
            if stab:
                stab.setLowPassIIR(param)

    def set_stabilization_fusion_factor(self, param):
        if self.controller:
            stab = self.controller.getStabilizationIMU()
            if stab:
                stab.setFusionFactor(param)

    def get_user_orientation_quaternion(self):
        return SETTINGS.orientation_quaternion;

    def set_user_orientation_quaternion(self, qList):
        SETTINGS.orientation_quaternion = qList;
        q = vs.Quat(qList[0], qList[1], qList[2], qList[3])
        if self.controller:
            self.controller.setUserOrientation(q)

    def update_user_orientation_quaternion(self, qList):
        q = vs.Quat(qList[0], qList[1], qList[2], qList[3])
        if self.controller:
            self.controller.updateUserOrientation(q)
            new_quat = self.controller.getUserOrientation()
            SETTINGS.orientation_quaternion = [ new_quat.getQ(0), new_quat.getQ(1), new_quat.getQ(2), new_quat.getQ(3) ]

    def get_user_orientation_ypr(self):
        qList = SETTINGS.orientation_quaternion;
        q = vs.Quat(qList[0], qList[1], qList[2], qList[3])
        return q.toEuler_py()

    def set_user_orientation_ypr(self, yaw, pitch, roll):
        q = vs.Quat.fromEulerZXY(yaw, pitch, roll)
        SETTINGS.orientation_quaternion = [q.getQ0(), q.getQ1(), q.getQ2(), q.getQ3()]
        if self.controller:
            self.controller.setUserOrientation(q)

    def update_user_orientation_ypr(self, yaw, pitch, roll):
        q = vs.Quat.fromEulerZXY(yaw, pitch, roll)
        if self.controller:
            self.controller.updateUserOrientation(q)
            new_quat = self.controller.getUserOrientation()
            SETTINGS.orientation_quaternion = [ new_quat.getQ(0), new_quat.getQ(1), new_quat.getQ(2), new_quat.getQ(3) ]

    def reset_user_orientation(self):
        SETTINGS.orientation_quaternion = [1.0, 0.0, 0.0, 0.0]
        if self.controller:
            self.controller.resetUserOrientation()

    def get_horizon_leveling_quaternion(self):
        if self.controller:
            stab = self.controller.getStabilizationIMU()
            if stab:
                q = stab.computeHorizonLeveling().conjugate()
                return [ q.getQ0(), q.getQ1(), q.getQ2(), q.getQ3() ]

    def enable_gyro_bias_cancellation(self):
        if self.controller:
            stab = self.controller.getStabilizationIMU()
            if stab:
                stab.enableGyroBiasCorrection()

    def disable_gyro_bias_cancellation(self):
        if self.controller:
            stab = self.controller.getStabilizationIMU()
            if stab:
                stab.disableGyroBiasCorrection()

    def is_gyro_bias_cancellation_valid(self):
        if self.controller:
            stab = self.controller.getStabilizationIMU()
            if stab:
                return stab.isGyroBiasCorrectionValid()
        return False

    def get_gyro_bias_cancellation_status(self):
        if self.controller:
            stab = self.controller.getStabilizationIMU()
            if stab:
                return stab.getGyroBiasStatus()
        return None

    def get_latency(self):
        if self.controller:
            return self.controller.getLatency()
        return -1
