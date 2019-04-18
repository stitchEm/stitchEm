import utils.async
import vs


class CppCallback(vs.CppCallback):
    def __init__(self, to_call):
        super(CppCallback, self).__init__()

        self.to_call = to_call

    def __call__(self, payload):
        #defer to avoid deadlock
        utils.async.defer(self.to_call, payload)
