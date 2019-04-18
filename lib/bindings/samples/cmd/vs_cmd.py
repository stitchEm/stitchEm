#!/usr/bin/python

import vs
import sys
from optparse import OptionParser
from os import path as osp

def main():
    usage = "vs_cmd.py PTV PATH_TO_PLUGINS"
    parser = OptionParser(usage)
    (_, args) = parser.parse_args()
    if len(args) != 2:
        parser.error(usage)
    ptv, path = args

    # Use arguments here
    vs.Logger_setLevel(4)

    vs.loadPlugins(path)
    parser = vs.Parser_create()
    if not parser.parse(ptv):
        raise Exception('could not parse the configuration')
    pano = vs.PanoDefinition_create(parser.getRoot().has("pano"))
    merger = vs.ImageMergerFactory_createMergerFactory(
        parser.getRoot().has("merger"))
    warper = vs.ImageWarperFactory_createWarperFactory(
        parser.getRoot().has("warper"))
    flow = vs.ImageFlowFactory_createFlowFactory(
        parser.getRoot().has("flow"))
    print("Panorama size: {}x{}".format(pano.width, pano.height))
    output = parser.getRoot().has("output")
    first_frame = 0
    last_frame = 100
    input_factory = vs.DefaultReaderFactory(first_frame, last_frame)
    controller = vs.createController(vs.PanoDeviceDefinition(), pano, merger.object(), wraper.object(), flow.object(), input_factory)
    writer = vs.create(
        output, "test", pano.width, pano.height)
    device = vs.StitcherDevice()
    device.device = 0
    stitcher = controller.createStitcher(device)
    stitch_output = controller.createAsyncStitchOutput(
        device, writer.object().getVideoWriter())
    status = stitcher.stitch(stitch_output.object())
    print("CAN I HAZ STITCH ? {}".format(status.ok()))

if __name__ == "__main__":
    main()
