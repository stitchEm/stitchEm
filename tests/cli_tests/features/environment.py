import os
from os import path as osp
import shutil
import sys
from uuid import getnode as get_mac
try:
    from matplotlib import pyplot
except ImportError:
    pass

BASE_DIR = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.join(BASE_DIR, "..", ".."))

from pyvs.vsfiles import remove_if_exist, REPO_DIR, reset_to_empty_folder
from pyvs.vsperf import ASSETS
from pyvs import helpers

REPORTS = osp.join(BASE_DIR, "..", "reports")
PICTURES = osp.join(BASE_DIR, "..", "pictures")
BRANCH_PICS = osp.join(PICTURES, "branch")
REFS = osp.join(PICTURES, "refs")

def before_all(ctx):
    os.environ["DISPLAY"] = ":0.0"
    ctx.utils = osp.join(BASE_DIR, "..", "etc")
    ctx.data = osp.join(ctx.utils, "data")
    reset_to_empty_folder(ctx.data)
    ctx.pictures = []
    cfg = ctx.config.userdata.get("CFG", "release")
    repo_all = osp.join(REPO_DIR, "..")

    bin_path = ctx.config.userdata.get("BIN")
    if not bin_path:
        ctx.studio = osp.join(repo_all, "bin", "x64", cfg)
    else:
        ctx.studio = osp.join(bin_path)

    if sys.platform == "darwin":
        ctx.plugins = osp.join(ctx.studio, "CorePlugins")
        ctx.plugins_gpu = None
        ctx.vahplugins = osp.join(ctx.studio, "VahanaPlugins")
    else:
        ctx.plugins = osp.join(ctx.studio, "core_plugins")

        ctx.vahplugins = osp.join(ctx.studio, "vahana_plugins")
        if sys.platform == "linux2":
            ctx.plugins_gpu = None
        else:
            plugin = "core_plugins_" + ctx.config.userdata.get("GPU", "CUDA")
            ctx.plugins_gpu = osp.join(ctx.studio, plugin)
    ctx.mac = get_mac()

def before_scenario(ctx, scenario):
    ctx.rtmp_flow = None

def after_scenario(ctx, scenario):
    if "perf" in scenario.tags:
        name = scenario.name.replace(" ", "_")
        xtime = range(len(ctx.cpu))
        pyplot.plot(xtime, ctx.cpu, 'g-')
        pyplot.savefig(osp.join(REPORTS, "_".join([name, "cpu.pdf"])))
        pyplot.close()
        pyplot.plot(xtime, ctx.ram, 'g-')
        pyplot.savefig(osp.join(REPORTS, "_".join([name, "ram.pdf"])))
        pyplot.close()
    if ctx.rtmp_flow is not None:
        helpers.stop_vs_cmd()
        ctx.rtmp_flow.join()
        ctx.rtmp_flow = None

def after_all(ctx):
    if not os.path.exists(PICTURES):
        os.makedirs(PICTURES)
    if not os.path.exists(BRANCH_PICS):
        os.makedirs(BRANCH_PICS)
    if not os.path.exists(REFS):
        os.makedirs(REFS)
    for pic, ref in ctx.pictures:
        file_name_base = osp.basename(pic)
        file_name = osp.join(BRANCH_PICS, file_name_base)
        ref_name = osp.join(REFS, file_name_base)
        remove_if_exist(file_name)
        remove_if_exist(ref_name)
        try:
            os.rename(pic, file_name)
        except OSError:
            pass
        try:
            shutil.copyfile(ref, ref_name)
        except OSError:
            pass

