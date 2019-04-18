from os import path as osp
import re
from subprocess import Popen
import sys

from behave import when, then, step_matcher # pylint: disable=E0611

from pyvs.vsperf import exec_and_monitor, ASSETS

import csv
import massedit

step_matcher('re')

CURDIR = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))

# Utils {{{
def csv_profile_result(path):
    with open(path) as cvs_file:
        data = csv.reader(cvs_file)
        rownum = 0
        sumtime = 0
        sumact = 0
        for row in data:
            if rownum == 0:
                header = row
            else:
                colnum = 0
                time = 0
                activity = 0
                for col in row:
                    #removing the first rows that does not contains timing results
                    if (colnum == 0) and ((col.find("==") != -1) or (col.find("Time") != -1)
                    or (col.find("%") != -1) or (col.find("No kernels") != -1)):
                        break;
                    if colnum == 0:
                        activity = float(col)
                    if colnum == 1:
                        time = float(col)
                    colnum += 1
                if colnum == 0:
                    continue;
                # removing CPU/GPU memcpy that are concurrent to kernel processing (if they were asynchronous)
                if (col.find("CUDA memcpy HtoD") == -1) and (col.find("CUDA memcpy DtoH") == -1):
                    sumtime += time
                    sumact += activity
            rownum += 1
    return sumtime

#}}}
# Given {{{


# }}}
# When {{{

@when(r'I launch (?P<tool>\S+) with (?P<ptv>\S+\.ptv)')
def when_lauch_cmd(ctx, tool, ptv):
    args = " ".join(["-i", osp.join(ASSETS, ptv),
                     "-p", ctx.plugins,
                    ])
    if ctx.plugins_gpu is not None:
        args += " -p" + ctx.plugins_gpu
    ctx.cpu, ctx.ram, ctx.time = exec_and_monitor(
        osp.join(osp.join(ctx.studio, tool)),
        args,)

@when(r'I profile (?P<tool>\S+) with (?P<ptv>\S+\.ptv)')
def when_lauch_cmd(ctx, tool, ptv):
    csvfile = osp.join(CURDIR, "reports", re.sub('\.ptv$', '', osp.basename(ptv)) + '.csv')   
    cmd = ["nvprof",
           "--print-gpu-summary -u s -f --csv --log-file", osp.join(csvfile),
           osp.join(osp.join(ctx.studio, tool)),
           "-i", osp.join(ASSETS, ptv),
           "-p", ctx.plugins,
          ]
    if ctx.plugins_gpu is not None:
        cmd += ["-p", ctx.plugins_gpu]
    try:
        proc = Popen(" ".join(cmd), shell=True)
    except OSError as err:
        raise Exception("{}\n{}".format(" ".join(cmd),err))
    ctx.cmd = " ".join(cmd)
    ctx.cmd_output = proc.communicate()
    ctx.time = csv_profile_result(csvfile)

@when(r'I launch (?P<tool>\S+) with procedural (?P<ptv>\S+\.ptv)')
def when_lauch_cmd(ctx, tool, ptv):
    args = " ".join(["-i", osp.join(ASSETS, ptv),
                     "-p", ctx.plugins,
                     "-l", "99",
                    ])
    if ctx.plugins_gpu is not None:
        args += " -p" + ctx.plugins_gpu
    ctx.cpu, ctx.ram, ctx.time = exec_and_monitor(
        osp.join(osp.join(ctx.studio, tool)),
        args,
        )

@when(r'I profile (?P<tool>\S+) with procedural (?P<ptv>\S+\.ptv)')
def when_lauch_cmd(ctx, tool, ptv):
    with open(osp.join(ASSETS, ptv), "r") as sources:
        lines = sources.readlines()
    with open("tmp.ptv", "w") as sources:
        for line in lines:
            sources.write(re.sub(r'procedural:frameNumber', 'procedural:profiling', line))
    csvfile = osp.join(CURDIR, "reports", re.sub('\.ptv$', '', osp.basename(ptv)) + '.csv')
    cmd = ["nvprof",
           "--print-gpu-summary -u s -f --csv --log-file", osp.join(csvfile),
           osp.join(osp.join(ctx.studio, tool)),
           "-i", "tmp.ptv",
           "-p", ctx.plugins, "-l", "99",
          ]
    if ctx.plugins_gpu is not None:
        cmd += ["-p", ctx.plugins_gpu]
    try:
        proc = Popen(" ".join(cmd), shell=True)
    except OSError as err:
        raise Exception("{}\n{}".format(" ".join(cmd), err))
    ctx.cmd = " ".join(cmd)
    ctx.cmd_output = proc.communicate()
    ctx.time = csv_profile_result(csvfile)
# }}}
# Then {{{

@then(r'I expect it to take less than (?P<nb>\S+) ' +\
      r'(?P<unit>second|minute|hour)s?')
def then_check_time(ctx, nb, unit):
    unit_dict = {'second' : 1, 'minute' : 60, 'hour' : 3600}
    nb = float(nb) * unit_dict[unit]
    assert ctx.time < nb, "execution took too long: {}s".format(ctx.time)

# }}}
