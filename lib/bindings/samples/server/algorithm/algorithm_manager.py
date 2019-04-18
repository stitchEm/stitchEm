import threading
from blinker import signal
import logging
import errors

import vs
from calibration import CalibrationRunner
from exposure_compensation import ExposureCompensationRunner
import algorithm_scheduler


class AlgorithmManager(object):
    """
    Handles API requests for running algorithms, maintains information about active algorithm and maintains queue of
    algorithms to run.
    """

    algorithm_map = {algorithm.name: algorithm for algorithm in (CalibrationRunner, ExposureCompensationRunner)}

    def __init__(self, stitcher_controller):

        self.algorithm_scheduler = algorithm_scheduler.AlgorithmScheduler()

        self.running_algorithms = set()

        self.algo_lock = threading.RLock()

        self.extract_output_list = vs.ExtractOutputPtrVector()
        self.extract_output_managed = []
        for i in range(stitcher_controller.getPano().numInputs()):
            input = stitcher_controller.getPano().getInput(i)
            if not input.getIsVideoEnabled():
                continue
            surf = vs.OffscreenAllocator_createSourceSurface(input.width, input.height, "Algorithm")
            if not surf.ok():
              raise errors.AlgorithmError('Cannot run algorithm:' + str(surf.status().getErrorMessage()))
            dontmemleak = vs.sourceSurfaceSharedPtr(surf.release())
            # SWIG create a proxy object with an empty deleter
            # when passing directly the pointer to the vector object :(
            # DON'T TRY TO FACTORIZE THE PREVIOUS LINE OR MEMLEAK
            extract_output = stitcher_controller.createBlockingExtractOutput(i, dontmemleak, None, None)
            if not extract_output.status().ok():
                raise errors.AlgorithmError("Cannot create AsyncExtractOutput")

            # To preserve ownership over outputs
            self.extract_output_managed.append(extract_output)
            self.extract_output_list.push_back(extract_output.object())

        self._connect_signals()

    def _connect_signals(self):
        signal("algorithm_completed").connect(self._on_algorithm_completed)

    def _on_algorithm_completed(self, sender):
        with self.algo_lock:
            self.running_algorithms.remove(sender)
            self.algorithm_scheduler.reschedule(sender)

    def get_next_algorithm(self, panorama):
        with self.algo_lock:
            scheduled_algorithm = self.algorithm_scheduler.get_next_algorithm()
            if not scheduled_algorithm:
                return None, vs.ExtractOutputPtrVector()

            self.running_algorithms.add(scheduled_algorithm)

            return scheduled_algorithm.get_algo_output(panorama), self.extract_output_list

    def cancel_running_algorithms(self):
        with self.algo_lock:
            for algorithm in self.running_algorithms:
                algorithm.cancel()
            self.running_algorithms.clear()

    def stop(self):
        """
        Remove everything scheduled
        Abandon current results
        :return:
        """

    @classmethod
    def create_algorithm(cls, name, *args, **kwargs):
        return cls.algorithm_map[name](*args, **kwargs)

    # Specific algorithms management
    def start_calibration(self, calibration_preset, incremental=True):
        logging.info("Start calibration algorithm. Incremental: " + str(incremental))
        self.algorithm_scheduler.schedule(self.create_algorithm(
            CalibrationRunner.name,
            calibration_preset,
            incremental=incremental))

    def reset_calibration(self, calibration_preset):
        logging.info("Reset calibration")
        self.algorithm_scheduler.schedule(self.create_algorithm(
            CalibrationRunner.name,
            calibration_preset,
            reset=True))

    def start_exposure_compensation(self):
        logging.info("Start exposure compensation algorithm.")
        self.algorithm_scheduler.schedule(self.create_algorithm(ExposureCompensationRunner.name))

    def stop_exposure_compensation(self):
        logging.info("Stop exposure compensation algorithm.")
        self.algorithm_scheduler.unschedule(ExposureCompensationRunner.name)
