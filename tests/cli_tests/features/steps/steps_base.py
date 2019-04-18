import json
import os
from os import path as osp
from random import choice
import re
import shutil
from subprocess import PIPE, Popen, check_call, check_output
import time
import threading
import urllib2
import sys

from behave import given, when, then # pylint: disable=E0611

from pyvs import helpers
from pyvs.helpers import LONG_FFPROBESIZE

from pyvs.vsperf import ASSETS
from pyvs.vsfiles import remove_if_exist, extend_path

if ASSETS is None:
    raise Exception("env variable VIDEOSTITCH_ASSETS is not set")

ALGO_FILE = osp.join(ASSETS, "algo.json")
EXPOSURE_OUTPUT = osp.join(ASSETS, "videoformat01", "res.ptv")
RTMP_INPUT_PTV = osp.join(ASSETS, "rtmp", "rtmp_input.ptv")
INTERNAL_RTMP_SERVER = "10.0.0.175"

if sys.platform == "win32":
    # name conflict with system tool,
    # see e.g. https://savage.net.au/ImageMagick/html/install-convert.html
    IMAGEMAGICK_CONVERT = "im-convert"
else:
    IMAGEMAGICK_CONVERT = "convert"

# Utils {{{

def normalize(path):
    return osp.join(ASSETS, path)

def remove_output_file(path):
    try:
        os.remove(path)
    except OSError:
        raise Exception("the output file {} does not exist".format(path))

def check_json(path):
    with open(path, 'r') as f:
        data = f.read()
    try:
        json.loads(data)
    except ValueError:
        return False
    return True

def json_file_to_dict(path, ndigits=None):
    def fcn(x):
        return round(float(x), int(ndigits))
    with open(path, 'r') as f:
        data = f.read()
    if ndigits:
        return json.loads(data, parse_float=fcn)
    else:
        return json.loads(data)

def integrity_check(path):
    try:
        proc = Popen(["avprobe", path],
                     stdout=PIPE,
                     stderr=PIPE,
                    )
    except OSError:
        raise Exception("avprobe is not in your PATH")
    return proc.communicate(), proc.returncode

def alignment_check(path):
    try:
        proc = Popen(["ffprobe", "-of", "json", "-show_streams",
                      "-count_frames", path],
                     stdout=PIPE,
                     stderr=PIPE,
                    )
    except OSError:
        raise Exception("ffprobe is not in your PATH")
    return proc.communicate(), proc.returncode

def atomic_check(path):
    try:
        proc = Popen(["AtomicParsley", path, "-T"],
                     stdout=PIPE,
                     stderr=PIPE,
                    )
    except OSError:
        raise Exception("AtomicParsley is not in your PATH")
    return proc.communicate(), proc.returncode

#}}}
# Given {{{

@given('I use {file_name:S} for synchronization')
@given('I use {file_name:S} for exposure')
@given('I use {file_name:S} for photometric calibration')
@given('I use {file_name:S} for calibration presets maker')
@given('I use {file_name:S} for calibration presets application')
@given('I use {file_name:S} for calibration deshuffling')
@given('I use {file_name:S} for epipolar')
@given('I use {file_name:S} for calibration')
@given('I use {file_name:S} for scoring')
@given('I use {file_name:S} for mask')
@given('I use {file_name:S} for autocrop')
def given_exposure(ctx, file_name):
    shutil.copy(osp.join(ctx.utils, file_name), ALGO_FILE)

@when('I start the RTMP flow')
@given('There is an RTMP flow')
def given_rtmp_started(ctx):
    rtmp_tpl = osp.join(ctx.utils, "assets_ptv", "rtmp", "rtmp.tpl")
    # generate a PTV with the correct address
    with open(rtmp_tpl, "r") as f:
        text = f.read()
    text = text.replace("##ADDRESS##", "rtmp://{}:1935/live/{}".format(
        INTERNAL_RTMP_SERVER, ctx.mac))
    with open(RTMP_INPUT_PTV, "w") as f:
        f.write(text)
    args = [
        "-i", RTMP_INPUT_PTV,
        "-p", ctx.vahplugins,
        "-f", "0",
        "-l", "999",
        ]
    cmd = osp.join(ctx.studio, "videostitch-cmd")
    # Fork a videostitch-cmd in the background
    proc = Popen([cmd] + args)
    ctx.rtmp_flow = threading.Thread(target=proc.communicate)
    ctx.rtmp_flow.start()
    time.sleep(5)

@given('I generated {ptv:S} with {tpl:S}')
def given_generate_ptv(ctx, ptv, tpl):
    tpl = osp.join(ctx.utils, "assets_ptv", tpl)
    ptv = osp.join(ctx.data, ptv)
    with open(tpl, "r") as f:
        text = f.read()
    text = text.replace("##ADDRESS##", "rtmp://{}:1935/live/{}".format(
        INTERNAL_RTMP_SERVER, ctx.mac))
    with open(ptv, "w") as f:
        f.write(text)

# }}}
# When {{{

def when_launch_control_cmd(ctx, control, tool, ptv, args, generated=False, from_repo=False):
    if from_repo:
        destination = osp.join(ctx.utils, "assets_ptv", ptv)
    else:
        destination = osp.join(ASSETS, ptv)
        if tool in ["videostitch-cmd", "undistort", "depth"]:
            try:
                shutil.copy(osp.join(ctx.utils, "assets_ptv", ptv),
                            destination)
            except IOError:
                if not generated:
                    raise
    cmd = [
        osp.join(ctx.studio, tool),
        "-i", destination,
        "-p", ctx.plugins,
        "-p", ctx.vahplugins,
        ] + args.split()

    if control is not None:
        cmd = control.split() + cmd
    if ctx.plugins_gpu is not None:
        cmd += ["-p", ctx.plugins_gpu]
    try:
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    except OSError:
        raise Exception("{} not installed in {}".format(tool, ctx.studio))
    ctx.cmd = " ".join(cmd)
    ctx.cmd_output = proc.communicate()
    print(cmd)
    print(ctx.cmd_output)
    ctx.res = proc.returncode
    os.remove(destination)

@when('I launch {tool:S} with {ptv:S} and "{args}"')
def when_launch_cmd(ctx, tool, ptv, args, generated=False, from_repo=False):
    when_launch_control_cmd(ctx, None, tool, ptv, args, generated, from_repo)

@when('I launch {tool:S} with {ptv:S} and "{args}" in the background')
def when_launch_cmd_background(ctx, tool, ptv, args, from_repo=True):
    ctx.cmd_background = threading.Thread(
        target=when_launch_cmd,
        args=(ctx, tool, ptv, args, False, from_repo)
        )
    ctx.cmd_background.start()

def when_launch_control_cmd_background(ctx, control, tool, ptv, args, from_repo=True):
    ctx.cmd_background = threading.Thread(target=when_launch_control_cmd, args=(ctx, control, tool, ptv, args, False, from_repo))
    ctx.cmd_background.start()

@when('I launch videostitch-cmd with generated {ptv:S} and "{args}"')
def when_launch_cmd_generated(ctx, ptv, args):
    when_launch_cmd(ctx, "videostitch-cmd", ptv, args, generated=True)

@when('I launch videostitch-cmd with {ptv:S} from repo and "{args}"')
def when_launch_cmd_repo(ctx, ptv, args):
    when_launch_cmd(ctx, "videostitch-cmd", ptv, args, from_repo=True)

@when('I launch {tool:S} for calibration with {ptv:S} and "{args}"')
@when('I launch {tool:S} for synchronization with {ptv:S}')
@when('I launch {tool:S} for exposure with {ptv:S}')
@when('I launch {tool:S} for synchronization with {ptv:S} and "{args}"')
@when('I launch {tool:S} for exposure with {ptv:S} and "{args}"')
@when('I launch {tool:S} for photometric calibration with {ptv:S} and "{args}"')
@when('I launch {tool:S} for calibration presets maker with {ptv:S} and "{args}"')
@when('I launch {tool:S} for calibration presets application with {ptv:S} and "{args}"')
@when('I launch {tool:S} for calibration deshuffling with {ptv:S} and "{args}"')
@when('I launch {tool:S} for epipolar with {ptv:S} and "{args}"')
@when('I launch {tool:S} for scoring with {ptv:S} and "{args}"')
@when('I launch {tool:S} for mask with {ptv:S} and "{args}"')
@when('I launch {tool:S} for autocrop with {ptv:S} and "{args}"')
def when_launch_algo(ctx, tool, ptv, args=""):
    args += "--apply_algos {} ".format(ALGO_FILE)
    when_launch_cmd(ctx, tool, ptv, args)
    os.remove(ALGO_FILE)

@when('I launch {tool:S} without arguments')
def when_launch_empty_cmd(ctx, tool):
    try:
        proc = Popen(
            [osp.join(osp.join(ctx.studio, tool))],
            stdout=PIPE,
            stderr=PIPE,
            )
    except OSError:
        raise Exception("{} not installed in {}".format(tool, ctx.studio))
    ctx.output = proc.communicate()
    ctx.res = proc.returncode

@when('I launch videostitch-cmd with "{args}"')
def when_launch_cmd_without_ptv(ctx, args):
    cmd = [
        osp.join(osp.join(ctx.studio, "videostitch-cmd")),
        "-i", " ",
        "-p", ctx.plugins,
        ] + args.split()
    ctx.start_time = time.time()
    if ctx.plugins_gpu is not None:
        cmd += ["-p", ctx.plugins_gpu]
    try:
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    except OSError:
        raise Exception("videostitch-cmd not installed in {}".format(
            ctx.studio))
    ctx.cmd = " ".join(cmd)
    ctx.cmd_output = proc.communicate()
    print(cmd)
    print(ctx.cmd_output)
    ctx.res = proc.returncode

def gen_ptv_with_param(template, name, value):
    reg = '(\\[{}\\])'.format(name)
    p = re.compile(reg)
    ptv_gen = p.sub(value, template)
    return ptv_gen

@when('I test and check videostitch-cmd with "{audio_codec}" and "{sampling_rate}" and "{sample_format}" and "{channel_layout}" and "{audio_bitrate}"')
def when_launch_and_check_audio_conf(
        ctx, audio_codec, sampling_rate, sample_format, channel_layout,
        audio_bitrate):
    template_ptv = osp.join(ctx.utils, "assets_ptv", "videoformat01",
                            "template_audio_output.ptv")
    template = open(template_ptv,"r").read()
    codecs = audio_codec.split(" ")
    rates = sampling_rate.split(" ")
    formats = sample_format.split(" ")
    layouts = channel_layout.split(" ")
    bitrates = audio_bitrate.split(" ")

    print_str = "Test codec {} and rate {} and format {} and layout {} and bitrate {}"
    output = osp.join("videoformat01","output.mp4")

    for codec in codecs:
        for rate in rates:
            for sp_format in formats:
                for layout in layouts:
                    for bitrate in bitrates:
                        ptv_gen = gen_ptv_with_param(template, "audio_codec",
                                                     '"{}"'.format(codec))
                        ptv_gen = gen_ptv_with_param(ptv_gen, "sampling_rate",
                                                     str(rate))
                        ptv_gen = gen_ptv_with_param(ptv_gen, "sample_format",
                                                     '"{}"'.format(sp_format))
                        ptv_gen = gen_ptv_with_param(ptv_gen, "audio_bitrate",
                                                     str(bitrate))
                        ptv_gen = gen_ptv_with_param(
                            ptv_gen, "channel_layout", '"{}"'.format(layout))
                        ptv_file = osp.join(
                            ctx.utils, "assets_ptv", "videoformat01",
                            "test_audio_output.ptv")
                        with open(ptv_file, "w") as f:
                            f.write(ptv_gen)
                        ptv_relative_path = osp.join("videoformat01",
                                                     "test_audio_output.ptv")
                        when_launch_cmd(ctx, "videostitch-cmd",
                                        ptv_relative_path, "-f 0 -l 200")
                        print(print_str.format(codec, rate, sp_format, layout,
                                               bitrate))
                        then_the_field_equal(ctx, output, "sampling_rate",
                                             rate)
                        then_the_field_equal(ctx, output, "channel_layout",
                                             layout)
                        then_the_field_equal(ctx, output, "sample_format",
                                             sp_format)

@when('I compare {output:S} with {ref:S}')
def when_compare_picture(ctx, output, ref):
    ref = normalize(ref)
    output = normalize(output)
    try:
        proc = Popen(["compare",
                      "-metric",
                      "MAE",
                      ref,
                      output,
                      osp.join(ctx.data, "null"),
                     ],
                     stdout=PIPE,
                     stderr=PIPE,
                    )
    except OSError:
        raise Exception("compare is not in your PATH")
    ctx.output = proc.communicate()[1]
    ctx.res = proc.returncode
    ctx.pictures.append([output, ref])

@when('I replace transparency with a red background in {output:S}')
def when_compare_picture(ctx, output):
    output = normalize(output)
    try:
        proc = Popen([IMAGEMAGICK_CONVERT,
                      "-background",
                      "red",
                      "-alpha",
                      "remove",
                      output,
                      output
                     ],
                     stdout=PIPE,
                     stderr=PIPE,
                    )
    except OSError:
        raise Exception(IMAGEMAGICK_CONVERT + " is not in your PATH. Expecting convert.exe to be renamed to im-convert.exe on Windows.")
    ctx.output = proc.communicate()[1]
    ctx.res = proc.returncode

@when('I analyze score of {output:S}')
def when_analyze_score(ctx, output):
    output = osp.join(ASSETS, output)
    with open(output, 'r') as f:
        data = f.read()
    try:
        data = json.loads(data)
    except ValueError:
        assert 0, "the ouput ptv is invalid"
    if "score" not in data[0]:
        assert 0, "no score in ptv"
    ctx.output= data[0]["score"]

@when('I analyze uncovered_ratio of {output:S}')
def when_analyze_uncovered_ratio(ctx, output):
    output = osp.join(ASSETS, output)
    with open(output, 'r') as f:
        data = f.read()
    try:
        data = json.loads(data)
    except ValueError:
        assert 0, "the ouput ptv is invalid"
    if "uncovered_ratio" not in data[0]:
        assert 0, "no score in ptv"
    ctx.output= data[0]["uncovered_ratio"]

@when('I check {output:S} integrity with avprobe')
def when_check_avprobe(ctx, output):
    """This removes the output file after"""
    output = normalize(output)
    ctx.output, ctx.res = integrity_check(output)
    remove_output_file(output)

@when('I check files {wildcard:S} integrity with avprobe')
def when_check_multiple_avprobe(ctx, wildcard):
    """This removes the output file after"""
    wildcard = normalize(wildcard)
    ctx.output = []
    ctx.res = []
    for path in extend_path(wildcard):
        r1, r2 = integrity_check(path)
        ctx.output.append(r1)
        ctx.res.append(r2)
        remove_output_file(path)

@when('I check files {output:S} {fformat:S} streams alignment with avprobe')
def when_check_alignment_avprobe(ctx, output, fformat):
    wildcard = normalize("{}-*.{}".format(output,fformat))
    for i in xrange(len(extend_path(wildcard))-2):
        path = normalize("{}-{}.{}".format(output,i+1,fformat))
        assert osp.isfile(path), "the file {} does not exist".format(path)
        r1, _ = alignment_check(path)
        joutput = json.loads(r1[0])
        start0 = float(joutput['streams'][0]['start_time'])
        start1 = float(joutput['streams'][1]['start_time'])
        duration0 = float(joutput['streams'][0]['duration'])
        duration1 = float(joutput['streams'][1]['duration'])

        print_str = "the file {} streams start_time are not aligned: {} <> {}"
        assert abs(start0 - start1) < 0.03, print_str.format(path, start0,
                                                             start1)

        print_str = "the file {} streams duration are not aligned: {} <> {}"
        assert abs(duration0 - duration1 + start0 - start1) < 0.05,\
            print_str.format(path, duration0 + start0, duration1 + start1)

        print_str = "the file {} did not decode the expected number of frames for stream {} : {} <> {}"
        for k in xrange(len(joutput['streams'])):
            stream =joutput['streams'][k]
            nb_frames = stream['nb_frames']
            nb_read_frames = stream['nb_read_frames']
            assert nb_frames == nb_read_frames, print_str.format(
                path, k, nb_frames, nb_read_frames)

@when('I rename {inputf} to {outputf}')
def when_rename_file(ctx, inputf, outputf):
    remove_if_exist(normalize(outputf))
    os.rename(normalize(inputf), normalize(outputf))

@when('I launch autocrop-cmd with input {input_picture:S} and output {output:S}')
def when_launch_autocrop_cmd(ctx, input_picture, output):
    cmd = [
        osp.join(osp.join(ctx.studio, "autocrop-cmd")),
        "-i", osp.join(ASSETS, input_picture),
        "-o", osp.join(ASSETS, output),
        "-d"
        ]
    try:
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
    except OSError:
        raise Exception("autocrop-cmd not installed in {}".format(ctx.studio))
    ctx.cmd = " ".join(cmd)
    ctx.cmd_output = proc.communicate()
    ctx.res = proc.returncode

@when('I wait the RTMP flow to stop')
def when_stop_rtmp_flow(ctx):
    ctx.rtmp_flow.join()
    ctx.rtmp_flow = None

@when('I compare video {generated:S} to {ref:S}')
def when_compare_video(ctx, generated, ref):
    generated = osp.join(ASSETS, generated)
    ref = osp.join(ASSETS, ref)
    ctx.log = "psnr.log"
    cmd = [
        "ffmpeg",
        "-i", generated,
        "-i", ref,
        "-lavfi",
        "psnr='stats_file={}'".format(ctx.log),
        "-f", "null", "-",
        ]
    check_call(cmd, stdout=PIPE, stderr=PIPE)

# }}}
# Then {{{

@then('I expect the command to {res:S}')
def cmd_should_have_res(ctx, res):
    if res == "fail":
        assert ctx.res, "the command should have failed:\n{}\n{}".format(
            ctx.cmd_output, ctx.cmd)
    elif res == "succeed":
        print_str = "the command should have succeeded (returned {}):\n{}\n{}"
        assert not ctx.res, print_str.format(ctx.res, ctx.cmd_output, ctx.cmd)
    else:
        raise Exception("Wrong usage of step")

@then('I expect the command to {res:S} and {stdx:S} contains "{keyword}"')
def the_cmd_should_have_res_keyword(ctx, res, stdx, keyword):
    if stdx == "stdout":
        stdx = 0
    elif stdx == "stderr":
        stdx = 1
    else:
        raise Exception("Wrong usage of step")
    if res == "fail":
        assert ctx.res, "the command should have failed:\n{}\n{}".format(
            ctx.cmd_output, ctx.cmd)
    elif res == "succeed":
        assert not ctx.res,\
            "the command should have succeeded:\n{}\n{}".format(
                ctx.cmd_output, ctx.cmd)
    else:
        raise Exception("Wrong usage of step")
    assert keyword in ctx.cmd_output[stdx],\
        "the captured stdout does not contain keyword: {}".format(keyword)

@then('I expect the command to fail with code {code:d}')
def cmd_should_have_code(ctx, code):
    assert ctx.res == code, "not the expected exit code expected " +\
            "{}, received {}".format(code, ctx.res)

@then('I expect the comparison to succeed')
@then('I expect the comparison error to be less than {error:f}')
def then_comparison_ok(ctx, error=0.05):
    res = re.sub(r'.*\((.*)\)', r'\1', ctx.output)
    try:
        float(res)
    except ValueError:
        assert 0, "the comparison failed: {}\n{}\n{}".format(
            res, ctx.cmd, ctx.cmd_output)
    assert float(res) < error, "error is too big: {}".format(res)

@then('mse_avg is under {error:f}')
def then_video_compare(ctx, error):
    with open(ctx.log, "r") as f:
        lines = f.readlines()
    for line in lines:
        line_ok = False
        for word in line.split():
            if "mse_avg" in word:
                line_ok = True
                rerror = word.split(":")[-1]
                assert float(rerror) <= error, "error is too big :{}".format(
                    line)
                break
        assert line_ok, "parsing error : {}".format(line)
    remove_if_exist(ctx.log)

@then('I expect the score to be more than {error:f}')
def then_score_ok(ctx, error):
    res = ctx.output
    assert float(res) >= error, "error is too big: {}".format(res)

@then('I expect the full coverage to be {target:S}')
def check_full_coverage(ctx, target):
    res = ctx.output
    if target == "true":
        assert float(res) == float(0.0),\
            "uncovered ratio {} is not 0, full coverage is false".format(res)
    elif target == "false":
        assert float(res) != float(0.0),\
            "uncovered ratio {} is 0, full coverage is true".format(res)
    else:
        raise Exception("Wrong usage of step")

@then('{wildcard:S} is a single file')
def when_check_single_file(ctx, wildcard):
    """This DOES NOT remove the output file after"""
    wildcard = normalize(wildcard)
    assert len(extend_path(wildcard)) == 1,\
        "number of video files is invalid, just one is expected : {}".format(
            wildcard)

@then('I check {output:S} faststart with AtomicParsley')
def when_check_fast_start_atomic(ctx, output):
    output = normalize(output)
    ctx.output, ctx.res = atomic_check(output)
    res = str(ctx.output).split("Atom ")
    moov = -1
    mdat = -1
    i = 0
    for s in res:
        if s.startswith("moov"):
            moov = i
        if s.startswith("mdat"):
            mdat = i
        i = i + 1
    assert moov > -1, "no moov Atom in output\n{}".format(ctx.output)
    print_str = "moov Atom {} after mdat Atom {} in output\n{}"
    assert moov < mdat, print_str.format(moov, mdat, ctx.output)

@then('I check {output:S} Atom {atom:S} with AtomicParsley')
def when_check_atomic(ctx, output, atom):
    output = normalize(output)
    ctx.output, ctx.res = atomic_check(output)
    res = str(ctx.output).split("Atom ")
    i = 0
    for s in res:
        if s.startswith(atom):
            break
        i = i + 1
    assert i != res.__len__(), "no {} atom in output \n{}".format(
        atom, ctx.output)

@then('I check {output:S} no Atom {atom:S} with AtomicParsley')
def when_check_no_atomic(ctx, output, atom):
    output = normalize(output)
    ctx.output, ctx.res = atomic_check(output)
    res = str(ctx.output).split("Atom ")
    i = 0
    for s in res:
        if s.startswith(atom):
            break
        i = i + 1
    assert i == res.__len__(), "{} atom in output \n{}".format(
        atom, ctx.output)

@then('The video is OK')
def then_video_ok(ctx):
    assert not ctx.res, "video is invalid : {}".format(ctx.output)

@then('The videos are OK')
def then_videos_ok(ctx):
    for i in xrange(len(ctx.res)):
        assert not ctx.res[i], "video is invalid : {}".format(ctx.output[i])

@then('The exposure output ptv is valid')
def then_exposure_output_valid(ctx):
    assert check_json(EXPOSURE_OUTPUT), "the output ptv is invalid"

@then('The JSON output {output_file:S} is valid')
@then('The photometric output {output_file:S} is valid')
def then_json_output_valid(ctx, output_file):
    assert_str = "the output ptv is invalid"
    assert check_json(osp.join(ASSETS, output_file)), assert_str

@then(r'The exposure RGB score in {output:S} is less than {diffr:d}, {diffg:d}, {diffb:d}')
def then_exposure_output_score(ctx, output, diffr, diffg, diffb):
    output = osp.join(ASSETS, output)
    with open(output, 'r') as f:
        data = f.read()

    remove_output_file(output)

    try:
        data = json.loads(data)[0]
    except ValueError:
        raise Exception("the ouput ptv is invalid: {}".format(data))
    assert_str = "ptv doesn't contain valid exposure score: {}".format(data)
    assert_bool = "diff_red" in data and "diff_green" in data and\
        "diff_blue" in data
    assert assert_bool, assert_str
    assert_str = "Expected red exposure score < {}, got: {}".format(
        diffr, data["diff_red"])
    assert data["diff_red"] < int(diffr), assert_str
    assert_str = "Expected green exposure score < {}, got: {}".format(
        diffg, data["diff_green"])
    assert data["diff_green"] < int(diffg), assert_str
    assert_str = "Expected blue exposure score < {}, got: {}".format(
        diffb, data["diff_blue"])
    assert data["diff_blue"] < int(diffb), assert_str


@then('The synchronization output "{file_name:S}" is valid')
def then_synchro_output_valid(ctx, file_name):
    assert check_json(normalize(file_name)), "the output ptv is invalid"

@then('The synchronization output "{file_name:S}" is consistent with "{ref:S}"')
@then('The synchronization output "{file_name:S}" is consistent with "{ref:S}" within {nb_frame:d} frames')
@then('The synchronization output "{file_name:S}" is consistent with "{ref:S}" within {nb_frame:d} frame')
def then_synchro_value_correct(ctx, file_name, ref, nb_frame=10):
    output = json_file_to_dict(normalize(file_name))
    ref = json_file_to_dict(osp.join(ctx.utils, ref))
    for ref_input in ref["inputs"]:
        in_offset = ref_input["frame_offset"]
        for out_input in output["inputs"]:
            out_offset = out_input["frame_offset"]
            if ref_input["reader_config"] == out_input["reader_config"]:
                assert abs(in_offset - out_offset) <= nb_frame, "wrong output"
                break

@then('The calibration cost of output "{file_name:S}" is consistent with "{ref:S}"')
def the_calibration_cost_correct(ctx, file_name, ref):
    output = json_file_to_dict(normalize(file_name))
    ref = json_file_to_dict(osp.join(ctx.utils, ref))

    calibration_cost = output.get("calibration_cost")
    if not calibration_cost:
        calibration_cost = output["pano"]["calibration_cost"]

    calibration_cost_ref = ref.get("calibration_cost")
    if not calibration_cost_ref:
        calibration_cost_ref = ref["pano"]["calibration_cost"]

    assert abs(calibration_cost -  calibration_cost_ref) <= 150,\
        "wrong output %f <> %f" % (calibration_cost, calibration_cost_ref)

@then('The translations of output "{file_name:S}" are consistent with "{ref:S}" for the first input')
def the_calibration_translations_correct(ctx, file_name, ref):
    output = json_file_to_dict(normalize(file_name))
    ref = json_file_to_dict(osp.join(ctx.utils, ref))
    in_x = ref["inputs"][0]["geometries"]["translation_x"]
    in_y = ref["inputs"][0]["geometries"]["translation_y"]
    in_z = ref["inputs"][0]["geometries"]["translation_z"]
    out_x = output["inputs"][0]["geometries"]["translation_x"]
    out_y = output["inputs"][0]["geometries"]["translation_y"]
    out_z = output["inputs"][0]["geometries"]["translation_z"]
    assert_str = "wrong output {} <> {}"
    assert abs(in_x - out_x) <= 0.001, assert_str.format(in_x, out_x)
    assert abs(in_y - out_y) <= 0.001, assert_str.format(in_y, out_y)
    assert abs(in_z - out_z) <= 0.001, assert_str.format(in_z, out_z)

@then('The file size of {output:S} is below {filesize:d} bytes')
def then_check_filesize(ctx, output, filesize):
    for eachfile in extend_path(normalize(output)):
        assert os.path.getsize(eachfile) < filesize,\
            "{} size {} is above {} limit".format(
                eachfile, os.path.getsize(eachfile), filesize)

@then(r'I expect {file_name:S} is the same as {ref:S} with {ndigits:S} digits after the decimal point for float')
def then_check_json_files_equal(ctx, file_name, ref, ndigits):
    output_json = json_file_to_dict(normalize(file_name), ndigits)
    ref_json = json_file_to_dict(osp.join(ctx.utils, "assets_ptv", ref),
                                 ndigits)
    assert output_json == ref_json, "{}\n\n\n{}".format(output_json, ref_json)
    remove_output_file(normalize(file_name))

@then(r'I expect the geometries of {file_name:S} are the same as {ref:S}')
def then_check_json_geometries_equal(ctx, file_name, ref):
    output_json = json_file_to_dict(normalize(file_name))
    ref_json = json_file_to_dict(osp.join(ctx.utils, "assets_ptv", ref))
    ref_pano = ref_json["pano"]["inputs"]
    out_pano = output_json["pano"]["inputs"]
    for ref_input, out_input in zip(ref_pano, out_pano):
        ref_geo = ref_input["geometries"]
        out_geo = out_input["geometries"]
        assert ref_geo == out_geo, "{}\n\n{}".format(ref_geo, out_geo)
    remove_output_file(normalize(file_name))

@then(r'I expect the input readers and stack orders of {file_name:S} are the same as {ref:S}')
def then_check_json_input_readers_equal(ctx, file_name, ref):
    output_json = json_file_to_dict(normalize(file_name))
    ref_json = json_file_to_dict(osp.join(ctx.utils, "assets_ptv", ref))
    ref_pano = ref_json["pano"]["inputs"]
    out_pano = output_json["pano"]["inputs"]
    for ref_input, out_input in zip(ref_pano, out_pano):
        ref_reader = ref_input["reader_config"]
        out_reader = out_input["reader_config"]
        ref_stack = ref_input["stack_order"]
        out_stack = out_input["stack_order"]
        assert ref_reader == out_reader, "{}\n\n{}".format(
            ref_reader, out_reader)
        assert ref_stack == out_stack, "{}\n\n{}".format(out_stack, ref_stack)
    remove_output_file(normalize(file_name))

@then('I check the audio bitrate of {filename:S} to be equal to {bitrate:d}')
def then_the_audio_bit_rate_equal(ctx, filename, bitrate):
    filename = osp.join(ASSETS, filename)
    ffprobe_output=helpers.get_ffprobe_audio_outputs("100000", filename)
    assert ffprobe_output.has_key("streams")
    result = ffprobe_output["streams"][0]["bit_rate"]
    result = float(result)/1000.
    print("audio bitrate measured = {} kb/s different from the expected {} kb/s".format(
        result, bitrate))
    tolerance=bitrate*0.05
    assert ((bitrate-tolerance)<=result)  & (result <= (bitrate+tolerance))

@then('I check the video effective_bitrate of the recorded video file for {duration:d} seconds is {bitrate:d} with precision of {precision:g}')
def then_the_videofile_bit_rate_equal(ctx, duration, bitrate, precision):
    effective_bitrate = helpers.get_effective_bitrate(
        LONG_FFPROBESIZE,
        ctx.strem_file_path,
        duration)
    res = (1.0 - float(precision)) * int(bitrate) < int(effective_bitrate)
    assert res, "expected value {}, but got {}".format(bitrate,
                                                       effective_bitrate)
    res = (1.0 + float(precision)) * int(bitrate) > int(effective_bitrate)
    assert res, "expected value {}, but got {}".format(bitrate,
                                                       effective_bitrate)

@then('I check the video effective_bitrate of the recorded video file for {duration:d} seconds is {order:S} than {bitrate:d}')
def then_the_videofile_bit_rate_equal(ctx, duration, order, bitrate):
    effective_bitrate = helpers.get_effective_bitrate(
        LONG_FFPROBESIZE,
        ctx.strem_file_path,
        duration)
    if (str(order) == "higher"):
      assert int(bitrate) < int(effective_bitrate), "expected more than {}, but got {}".format(bitrate, effective_bitrate)
    elif(str(order) == "lower"):
      assert int(bitrate) > int(effective_bitrate), "expected less than {}, but got {}".format(bitrate, effective_bitrate)
    elif(str(order) == "around"):
      res = 0.95 * int(bitrate) < int(effective_bitrate)
      assert res, "expected value {}, but got {}".format(bitrate,
                                                         effective_bitrate)
      res = 1.05 * int(bitrate) > int(effective_bitrate)
      assert res, "expected value {}, but got {}".format(bitrate,
                                                         effective_bitrate)
    else:
      raise Exception("wrong comparator used {}. use lower/higher/around".format(order))

@then('I check the {field:S} of {filename:S} to be equal to {value:S}')
def then_the_field_equal(ctx, filename, field, value):
    filename = osp.join(ASSETS, filename)
    ffprobe_output=helpers.get_ffprobe_audio_outputs("100000", filename)
    assert ffprobe_output.has_key("streams")
    key = field
    if field == "sampling_rate":
        key = "sample_rate"
    elif field == "sample_format":
        key = "sample_fmt"
    result = ffprobe_output["streams"][0][key]
    print("{} resulted {} different from the expected {}".format(
        field, result, value))
    assert (result == value)

@then('I check the video {field:S} of {filename:S} to be equal to {value:S}')
def then_the_video_field_equal(ctx, filename, field, value):
    filename = osp.join(ASSETS, filename)
    ffprobe_output=helpers.get_ffprobe_video_outputs("100000", filename)
    assert ffprobe_output.has_key("streams")
    key = field
    result = ffprobe_output["streams"][0][key]
    print("{} resulted {} different from the expected {}".format(
        field, result, value))
    assert (result == value)

@then('The background process was successful')
def then_background_process_sucess(ctx):
    ctx.cmd_background.join()
    assert ctx.res == 0, "background process failed (code {})".format(ctx.res)

@then('I record the audio output during {duration:d} seconds in {wavfile:S}')
def then_record_audio_of_rtmp(ctx, duration, wavfile):
    output_filename = osp.join(ASSETS, wavfile)
    stream = ctx.rtmp_stream
    helpers.record_audio_sample_from_broadcasting(
        stream, output_filename, duration)
    ctx.cmd_background.join()
    assert ctx.res == 0, "background process failed (code {})".format(ctx.res)


@then(u'I expect the output audio channel map of {filename:S} to be "{channel_map}"')
def then_check_channel_map(ctx, filename, channel_map):
    channel_map = channel_map.split(" ")
    c_map = []
    for m in channel_map:
        c_map.append(int(m))

    wavfile = osp.join(ASSETS, filename)
    f0 = 44100. / 2. / 512.
    resulted_channel_map = helpers.get_audio_channel_map(wavfile, f0)
    assert(len(resulted_channel_map) == len(c_map))
    print('expected channel map {}'.format(c_map))
    print('resulted channel map {}'.format(resulted_channel_map))
    assert(resulted_channel_map == c_map)

@then(u'I wait for {duration:d} seconds')
def then_wait(ctx, duration):
    time.sleep(duration)


@given(u'the RTMP stream is started with bandwidth limited to {bw:d} with {ptv:S} and "{args}"')
def given_limited_rtmp_started_with(ctx, bw, ptv, args):
    random_seed = "".join(choice('0123456789') for i in range(6))
    gen_file = osp.join(osp.join(ptv.split('/')[0], 'gen-rtmp-{}.ptv'.format(
        random_seed)))
    template_ptv = osp.join(ctx.utils, 'assets_ptv', ptv)
    ctx.stream_name = 'sinegen{}'.format(random_seed)
    ctx.rtmp_stream = 'rtmp://{}/audio_map_test/{}'.format(
        INTERNAL_RTMP_SERVER, ctx.stream_name)
    with open(osp.join(ctx.utils, 'assets_ptv', gen_file), 'w') as gen_ptv:
        with open(template_ptv,"r") as template_file:
            template = template_file.read()
            template = template.replace('##ADDRESS##', ctx.rtmp_stream)
            gen_ptv.write(template)

    if bw is None:
        control = None
    else:
        control = 'trickle -s -u {}'.format(bw)
    when_launch_control_cmd_background(ctx, control, 'videostitch-cmd', gen_file, args,
                                       from_repo=False)

@given(u'the RTMP stream is started with {ptv:S} and "{args}"')
def given_rtmp_started_with(ctx, ptv, args):
    given_limited_rtmp_started_with(ctx, None, ptv, args)

@then('I copy the file from the wowza server')
def then_rtmp_copy_stream(ctx):
    dwn_link = 'http://{}:1900/{}.mp4'.format(INTERNAL_RTMP_SERVER,
                                              ctx.stream_name)
    dwn_link = dwn_link.replace(" ", "")
    ctx.strem_file_path = osp.join(ASSETS, 'audio_channel_map/stream.mp4')
    ctx.wav_file_path = osp.join(ASSETS, 'audio_channel_map/output.wav')
    rsp = urllib2.urlopen(dwn_link)
    with open(ctx.strem_file_path,'wb') as f:
        f.write(rsp.read())

@then('I strip the audio from the recorded video file')
def then_strip_wav_from_file(ctx):
    helpers.get_wave_from_video_file(ctx.strem_file_path, ctx.wav_file_path)
    os.remove(ctx.strem_file_path)

@then('I expect program compilation to take less than {timeout:d} seconds')
def then_check_opencl_cache(ctx, timeout):
    execution_time = time.time() - ctx.start_time
    assert_str = "execution took too long: {}s".format(execution_time)
    assert execution_time < float(timeout), assert_str

@then('I expect the number of frames of the recorded video file to be {order:S} than {nb:d}')
def then_nb_order_frame(ctx, order, nb):
    args = ["ffprobe", "-select_streams", "v", "-show_streams", ctx.strem_file_path]
    output = check_output(args)
    res = re.search(r'.*nb_frames=(\d+).*', output)
    if res:
        res = res.group(1)
    else:
        raise Exception("something went wrong with ffmpeg {}".format(output))
    if (str(order) == "higher"):
      assert int(nb) < int(res), "expected more than {}, but got {}".format(nb, res)
    elif(str(order) == "lower"):
      assert int(nb) > int(res), "expected less than {}, but got {}".format(nb, res)
    else:
      raise Exception("wrong comparator used {}. use lower or higher".format(order))

@then('I expect the number of frames of {file_name:S} to be {nb:d}')
def then_nb_frame(ctx, file_name, nb):
    file_name = osp.join(ASSETS, file_name)
    args = ["ffprobe", "-select_streams", "v", "-show_streams", file_name]
    output = check_output(args)
    res = re.search(r'.*nb_frames=(\d+).*', output)
    if res:
        res = res.group(1)
    else:
        raise Exception("something went wrong with ffmpeg {}".format(output))
    assert int(res) == int(nb), "wrong number of frames {} != {}".format(
        res, nb)

# }}}
