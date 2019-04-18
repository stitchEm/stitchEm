from blinker import signal

import logging
import vs
import errors
from clientmessenger import CLIENT_MESSENGER


class AlgorithmRunner(vs.Listener):
    repeat = False
    delay = 0
    name = "GenericAlgorithm"

    output_garbage = []
    """ We need to keep cancelled algorithm outputs alive. Destroying them would imply blocking the whole program because
        we are joining the algo thread in the destructor.
        Output object are not big, and we're not cancelling too often,
        so we're fine with this 'dirty' solution for now. """

    def __init__(self, name):
        super(AlgorithmRunner, self).__init__()

        self.config_name = name

        self.online_algorithm = None
        self.algorithm_output = None
        self.listener_wrapper = None
        self.panorama = None
        self.last_callback = None

    def get_algo_output(self, panorama):
        # Todo: investigate OnlineAlgorithm opaque pointer (shared calibration context? / last parameter)

        # Clone a panorama and acquire ownership.
        self.panorama = panorama.clone()
        self.panorama.thisown = 1

        self._update_config(panorama)

        self.online_algorithm = vs.OnlineAlgorithm_create(self.config_name, self.config.to_config())

        if not self.online_algorithm.status().ok():
            CLIENT_MESSENGER.send_error(
                errors.AlgorithmError("Online algorithm \"{}\" creation failed".format(self.config_name)))
            return None

        self.listener_wrapper = vs.AlgorithmListenerGIL(self.this)
        self.algorithm_output = vs.AlgorithmOutput(self.online_algorithm.release(), panorama,
                                                   vs.toListener(self.listener_wrapper.this), None)

        return self.algorithm_output

    def cancel(self):
        """
        Ignore the results of the algorithm when they arrive
        """
        self.algorithm_output.cancel()
        self.output_garbage.append(self.algorithm_output)
        self.last_callback = None
        return self

    def _update_config(self, panorama):
        """
        Algorithm-specific configuration update.
        :param panorama: Current pano definition
        """

    # Listener interface implementation

    def onPanorama(self, panorama):
        """
        Called when the algorithm is successfully completed.
        We want to:
        Reset the stitcher panorama.
        Notify user
        :param panorama: result of the algorithm
        """
        signal("algorithm_running_success").send(self, panorama=panorama, output=self.algorithm_output)
        if self.last_callback != self.onPanorama:
            CLIENT_MESSENGER.send_event("{}_algorithm_success".format(self.name))

        # Note. This signal will ultimately lead to destruction of self
        signal("algorithm_completed").send_async(self)
        self.last_callback = self.onPanorama

    def onError(self, status):
        if self.last_callback != self.onError:
            logging.error("Running of algorithm \"{}\" failed".format(self.name))
            CLIENT_MESSENGER.send_error(errors.AlgorithmRunningError(str(status.getErrorMessage())))

        # Note. This signal will ultimately lead to destruction of self
        signal("algorithm_completed").send_async(self)
        self.last_callback = self.onError