import os
from os import path as osp
import time

try:
    from matplotlib import pyplot
except ImportError:
    matplotlib = None

import psutil

ASSETS = os.environ.get("VIDEOSTITCH_ASSETS")

class VsProcess(psutil.Process):

    def __init__(self, name):
        res = None
        for process in psutil.process_iter():
            if process.name() == name:
                if res is None:
                    res = process
                else:
                    raise Exception("multiple processes {}".format(name))
        if res is None:
            raise Exception("process {} not found".format(name))
        else:
            super(VsProcess, self).__init__(res.pid)

    def try_cpu_ram(self):
        cpu = None
        ram = None
        try:
            cpu = self.get_cpu_percent()
            ram = self.get_memory_percent()
        except psutil.NoSuchProcess:
            pass
        return cpu, ram

    def monitor(self, interval=1):
        cpu = []
        ram = []
        if self.try_cpu_ram() is not None:
            start = time.time()
            intermediate = start
            while True:
                while time.time() - intermediate < interval:
                    time.sleep(0.01)
                intermediate += interval
                tmpcpu, tmpram = self.try_cpu_ram()
                if tmpcpu is None or tmpram is None:
                    break
                cpu += [tmpcpu]
                ram += [tmpram]
        return cpu, ram

def exec_and_monitor(program, args, verbose=False, **kwargs):
    start = time.time()
    if not verbose:
        args += " >> vscmd.log 2>&1"
    os.spawnv(os.P_NOWAIT, '/bin/sh', ['sh', '-c', " ".join([program, args])])
    time.sleep(0.01)
    process = VsProcess(osp.basename(program))
    cpu, ram = process.monitor(**kwargs)
    return cpu, ram, time.time() - start

def exec_and_plot(program, args, verbose=False, **kwargs):
    cpu, ram, _ = exec_and_monitor(program, args, verbose, **kwargs)
    xtime = range(len(cpu))
    pyplot.plot(xtime, cpu, 'g-', xtime, ram, 'b-')
    pyplot.show()

