import subprocess
import psutil

try:
    from pynvml import *
    nvmlPY = "OK"
except ImportError:
    nvmlPY = None

def getCpuInfo():
    """Returns CPU/s information.

    Notes:
        - All values set to zero when called the first time
        - Values stat updating on the second call
        - This can be changed by specifying an interval. In that case the
          calls will be blocking
    """
    return {
        'usage': psutil.cpu_percent(interval=None, percpu=True),
        'global_usage': psutil.cpu_percent(interval=None),
        'cpu_times': vars(psutil.cpu_times()),
    }

def getCudaInfo():
    """Returns CUDA device/s information
    """
    def get_attribute(att):
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu={}'.format(att),
             '--format=csv,noheader,nounits'])

    mem_used = get_attribute("memory.used")
    mem_total = get_attribute("memory.total")
    mem_per = int(mem_used) * 100 / int(mem_total)

    if nvmlPY is not None:
        nvmlInit()
        try:
           handle = nvmlDeviceGetHandleByIndex(0)
           (enc_util, ssize) = nvmlDeviceGetEncoderUtilization(handle)
        except NVMLError as err:
           enc_util = err
        nvmlShutdown()
    else:
        enc_util = "-1"

    return {
        'memory.used': mem_used,
        'memory.free': get_attribute("memory.free"),
        'memory.total': mem_total,
        'memory.percent': mem_per,
        'utilization.perf_state': get_attribute("pstate"),
        'utilization.gpu': get_attribute("utilization.gpu"),
        'utilization.enc': enc_util,
        'utilization.memory': get_attribute("utilization.memory"),
        'name': get_attribute("gpu_name"),
    }


def getMemoryInfo():
    """Returns information of the memory modules in the system
    """
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    vmem = vars(vmem)
    swap = vars(swap)

    return {
        'virtual': vmem,
        'swap': swap,
    }
