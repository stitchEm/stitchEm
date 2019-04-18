from concurrent.futures.thread import ThreadPoolExecutor

from tornado.concurrent import run_on_executor

import errors
from API.handlers import APIHandler
from API.schema import api
import math


class AlgorithmAPI(APIHandler):
    """
    Rest interface for running algorithms
    """

    executor = ThreadPoolExecutor(1)

    def __init__(self, extra):
        super(AlgorithmAPI, self).__init__(extra)
        self.stitcher = extra["video_stitcher"]
        self.project_manager = extra["project_manager"]
        self.camera = extra["camera"]


    @api(name="StartCalibration",
         endpoint="algorithm.start_calibration",
         description="Start Calibration algorithm with given parameters",
         parameters={
             "type": ["object", "null"],
             "properties":
                 {
                     "incremental": {
                         "type": "boolean",
                         "description": "True if you want to reuse previous control points. "
                                        "False if you want to start from scratch"},
                 }
         },
         errors=[errors.AlgorithmError, errors.StitcherError]
         )
    @run_on_executor
    def start_calibration(self, parameters):
        if not self.stitcher.is_Running():
            raise errors.AlgorithmRunningError("Calibration algorithm cannot run while stitcher is not running.")
        if not self.camera.rig_parameters:
            raise errors.AlgorithmRunningError(
                "Camera rig parameters are not available. Calibration algorithm cannot run without them.")
        if self.stitcher.algorithm_manager:
            incremental = parameters.get("incremental", False) if parameters is not None else False
            self.stitcher.algorithm_manager.start_calibration(self.camera.rig_parameters, incremental)

    @api(name="ResetCalibration",
         endpoint="algorithm.reset_calibration",
         description="Reset Calibration setting to factory default",
         errors=[errors.AlgorithmError, errors.StitcherError]
         )
    @run_on_executor
    def reset_calibration(self, parameters=None):
        if not self.stitcher.is_Running():
            raise errors.AlgorithmRunningError("Calibration algorithm cannot run while stitcher is not running.")
        if not self.camera.rig_parameters:
            raise errors.AlgorithmRunningError(
                "Camera rig parameters are not available. Calibration algorithm cannot run without them.")
        if self.stitcher.algorithm_manager:
            self.stitcher.algorithm_manager.reset_calibration(self.camera.rig_parameters)

    @api(name="StartIMUStabilization",
         endpoint="algorithm.start_imu_stabilization",
         description="Start IMU stabilization",
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def start_imu_stabilization(self, parameters=None):
        self.project_manager.start_imu_stabilization()

    @api(name="StopIMUStabilization",
         endpoint="algorithm.stop_imu_stabilization",
         description="Stop IMU stabilization",
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def stop_imu_stabilization(self, parameters=None):
        self.project_manager.stop_imu_stabilization()

    @api(name="IsIMUStabilizationEnabled",
         endpoint="algorithm.is_imu_stabilization_enabled",
         description="Get IMU Stabilization status",
         errors=[errors.StitcherError],
         result={
             "type": "object",
             "properties":
                 {
                     "enabled": {
                         "type": "boolean",
                         "description": "true if IMU stabilization is enabled, false otherwise"
                     }
                 }
         }
         )
    @run_on_executor
    def is_imu_stabilization_enabled(self, parameters=None):
        return { "enabled": self.project_manager.is_imu_stabilization_enabled()}

    @api(name="SetStabilizationLowPassFilter",
         endpoint="algorithm.set_stabilization_low_pass_filter",
         description="Set IMU Stabilization low pass filter",
         parameters={
             "type": "object",
             "properties":
                 {
                     "iir_low_pass": {
                         "type": "number",
                         "exclusiveMinimum": 0,
                         "maximum": 1,
                         "description": "Ratio of Nyquist frequency. Needs to be a floating number in ]0.0 ; 1.0]"
                     },
                 }
         },
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def set_stabilization_low_pass_filter(self, parameters):
        iirLowPass = parameters.get("iir_low_pass", 0.3)
        self.project_manager.set_stabilization_low_pass_filter(iirLowPass)

    @api(name="SetStabilizationFusionFactor",
         endpoint="algorithm.set_stabilization_fusion_factor",
         description="Set IMU Stabilization fusion factor",
         parameters={
             "type": "object",
             "properties":
                 {
                     "fusion_factor": {
                         "type": "number",
                         "minimum": 0,
                         "maximum": 1,
                         "description": "The fusion factor, takes a float value between 0 (gyro only) and 1 (accelerometer only)"
                     },
                 }
         },
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def set_stabilization_fusion_factor(self, parameters):
        fusionFactor = parameters.get("fusion_factor", 0.09)
        self.project_manager.set_stabilization_fusion_factor(fusionFactor)

    @api(name="GetUserOrientationQuaternion",
         endpoint="algorithm.get_user_orientation_quaternion",
         description="Get user orientation as a quaternion.",
         result={
             "type": "object",
             "properties":
             {
                 "quaternion": {
                     "$ref": "Quaternion"
                 }
             }
         },
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def get_user_orientation_quaternion(self, parameters=None):
        return { "quaternion": self.project_manager.get_user_orientation_quaternion() }

    @api(name="GetUserOrientationYawPitchRoll",
         endpoint="algorithm.get_user_orientation_ypr",
         description="Get user orientation as yaw, pitch, roll in degrees.",
         result={
             "type": "object",
             "properties":
             {
                 "yaw": {
                     "type": "number",
                     "description": "Yaw angle expressed in degrees"
                 },
                 "pitch": {
                     "type": "number",
                     "description": "Pitch angle expressed in degrees"
                 },
                 "roll": {
                     "type": "number",
                     "description": "Roll angle expressed in degrees"
                 },
             }
         },
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def get_user_orientation_ypr(self, parameters=None):
        ypr = self.project_manager.get_user_orientation_ypr()
        #only keep two decimals
        yaw, pitch, roll = round(ypr.yaw * 180. / math.pi, 2), \
                           round(ypr.pitch * 180. / math.pi, 2), \
                           round(ypr.roll * 180. / math.pi, 2)
        return { "yaw": yaw, "pitch": pitch, "roll": roll }

    @api(name="SetUserOrientationQuaternion",
         endpoint="algorithm.set_user_orientation_quaternion",
         description="Set user orientation using a quaternion. The values must be passed as 4 elements list of floats",
         parameters={
             "type": "object",
             "properties":
             {
                 "quaternion": {
                     "$ref": "Quaternion"
                 }
             },
             "required": ["quaternion"]
         },
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def set_user_orientation_quaternion(self, parameters):
        qList = parameters.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        self.project_manager.set_user_orientation_quaternion(qList)

    @api(name="SetUserOrientationYawPitchRoll",
         endpoint="algorithm.set_user_orientation_ypr",
         description="Set user orientation using Yaw/Pitch/Roll in degrees",
         parameters={
             "type": "object",
             "properties":
             {
                 "yaw": {
                     "type": "number",
                     "description": "Yaw angle expressed in degrees"
                 },
                 "pitch": {
                     "type": "number",
                     "description": "Pitch angle expressed in degrees"
                 },
                 "roll": {
                     "type": "number",
                     "description": "Roll angle expressed in degrees"
                 },
             },
             "required": ["yaw", "pitch", "roll"]
         },
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def set_user_orientation_ypr(self, parameters):
        yaw = parameters.get("yaw", 0.0)
        pitch = parameters.get("pitch", 0.0)
        roll = parameters.get("roll", 0.0)

        yaw, pitch, roll = yaw * math.pi / 180., pitch * math.pi / 180., roll * math.pi / 180.

        self.project_manager.set_user_orientation_ypr(yaw, pitch, roll)

    @api(name="UpdateUserOrientationQuaternion",
         endpoint="algorithm.update_user_orientation_quaternion",
         description="Update the user orientation using a quaternion",
         parameters={
             "type": "object",
             "properties":
             {
                 "quaternion": {
                     "$ref": "Quaternion"
                 }
             },
             "required": ["quaternion"]
         },
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def update_user_orientation_quaternion(self, parameters):
        qList = parameters.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        qList = self.project_manager.update_user_orientation_quaternion(qList)

    @api(name="UpdateUserOrientationYawPitchRoll",
         endpoint="algorithm.update_user_orientation_ypr",
         description="Update user orientation using Yaw/Pitch/Roll in degrees",
         parameters={
             "type": "object",
             "properties":
                 {
                     "yaw": {
                         "type": "number",
                         "description": "Yaw angle expressed in degrees"
                     },
                     "pitch": {
                         "type": "number",
                         "description": "Pitch angle expressed in degrees"
                     },
                     "roll": {
                         "type": "number",
                         "description": "Roll angle expressed in degrees"
                     },
                 },
             "required": ["yaw", "pitch", "roll"]
         },
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def update_user_orientation_ypr(self, parameters):
        yaw = parameters.get("yaw", 0.0)
        pitch = parameters.get("pitch", 0.0)
        roll = parameters.get("roll", 0.0)

        yaw, pitch, roll = yaw * math.pi / 180., pitch * math.pi / 180., roll * math.pi / 180.
        self.stitcher.project_manager.update_user_orientation_ypr(yaw, pitch, roll)

    @api(name="ResetUserOrientation",
         endpoint="algorithm.reset_user_orientation",
         description="Reset User orientation",
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def reset_user_orientation(self, parameters=None):
        self.stitcher.project_manager.reset_user_orientation()

    @api(name="GetHorizonLevelingQuaternion",
         endpoint="algorithm.get_horizon_leveling_quaternion",
         description="Get the horizon leveling quaternion",
         result={
             "type": "object",
             "properties":
                 {
                     "horizon_leveling": {
                         "$ref": "Quaternion"
                     }
                 }
         },
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def get_horizon_leveling_quaternion(self, parameters=None):
        qList = [1.0, 0.0, 0.0, 0.0]
        if self.stitcher.project_manager:
            qList = self.stitcher.project_manager.get_horizon_leveling_quaternion()
        return {"horizon_leveling": qList}


    @api(name="EnableGyroscopeBiasCancellation",
         endpoint="algorithm.enable_gyroscope_bias_cancellation",
         description="Enable gyroscope bias computation and cancellation",
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def enable_gyroscope_bias_cancellation(self, parameters=None):
        if self.stitcher.project_manager:
            self.stitcher.project_manager.enable_gyro_bias_cancellation()


    @api(name="DisableGyroscopeBiasCancellation",
         endpoint="algorithm.disable_gyroscope_bias_cancellation",
         description="Disable gyroscope bias computation and cancellation",
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def disable_gyroscope_bias_cancellation(self, parameters=None):
        if self.stitcher.project_manager:
            self.stitcher.project_manager.disable_gyro_bias_cancellation()


    @api(name="IsGyroscopeBiasCancellationValid",
         endpoint="algorithm.is_gyroscope_bias_cancellation_valid",
         description="Get IMU gyroscope bias cancellation simple status",
         result={"type": "boolean"},
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def is_gyroscope_bias_cancellation_valid(self, parameters=None):
        if self.stitcher.project_manager:
            return self.stitcher.project_manager.is_gyro_bias_cancellation_valid()
        return False


    @api(name="GetGyroscopeBiasCancellationDetailedStatus",
         endpoint="algorithm.get_gyroscope_bias_cancellation_detailed_status",
         description="Get IMU gyroscope bias cancellation detailed status",
         result={"type": ["integer", "null"]},
         errors=[errors.StitcherError]
         )
    @run_on_executor
    def get_gyroscope_bias_cancellation_detailed_status(self, parameters=None):
        if self.stitcher.project_manager:
            return self.stitcher.project_manager.get_gyro_bias_cancellation_status()
        return None
