# Retrieve metadata from an Orah 4i
# Connects to an already running Orah 4i under IP_ADDR
# Video data is ignored, decoder is mock
#
# Saves the data to a .csv file every NUM_ROWS_BLOCK_TO_WRITE received samples
#
# Plots the data continuously, until Ctrl+C is hit

import datetime
import gc
import json
import time

import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib.ticker

import pandas as pd

import vs

# how many times per second to check for new metadata
REFRESH_RATE = 30

# how many seconds of data to plot at a time
PLOT_SECONDS_X = 30

# Orah 4i address, needs to be started already
IP_ADDR = "10.0.0.53"

# CSV chunk to write a time
NUM_ROWS_BLOCK_TO_WRITE = 20

SESSION_ID = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
EXPOSURE_CSV = SESSION_ID + "-exposure.csv"
TONE_CURVE_CSV = SESSION_ID + "-tone_curve.csv"

ERROR_STATUS = vs.MetadataReader.MetadataReadStatusCode_ErrorWithStatus
EOF = vs.MetadataReader.MetadataReadStatusCode_EndOfFile
TRY_AGAIN = vs.MetadataReader.MetadataReadStatusCode_MoreDataAvailable


def make_ptv_config(board_id, ip_addr):
    return {
        "group": 0,
        "reader_config": {
            "type": "rtmp",
            "name": "rtmp://" + ip_addr + ":1935/inputs/" +
                    str(board_id) + "_0",
            "decoder": "mock",
            "audio_samplerate": 44100,
            "audio_channels": 2,
            "audio_sample_depth": "s16",
            "frame_rate": {
                "num": 30,
                "den": 1,
            },
        },
        "width": 1920,
        "height": 1440,
        "proj": "ff_fisheye",
        "response": "gamma"
    }


def create_metadata_reader(reader_factory, board_id):

    config = make_ptv_config(board_id, IP_ADDR)
    config_ptv = json.dumps(config)

    parser = vs.Parser_create()
    parser.parseData(config_ptv)
    input_def = vs.InputDefinition_create(parser.getRoot())

    reader = reader_factory.create(2, input_def)
    assert(reader.ok())

    metadata_reader = reader.getMetadataReader()
    assert(metadata_reader)

    reader.disown()
    return metadata_reader


def create_readers():
    """
    Create two libvideostitch RTMP readers on IP_ADDR's inputs
    Readers will be created for the 0_0 and 1_0 streams, where
    metadata is expected.
    """
    factory = vs.DefaultReaderFactory(0, -1)
    vs.loadPlugins("VahanaPlugins")

    return (create_metadata_reader(factory, 0),
            create_metadata_reader(factory, 1))


def time_normalized_data(data_frame):
    """
    Convert the time stamps in data_frame to relative time to the time of
    the first received sample.

    Returns a normalized copy of the data,
    Returns x_min and x_max for plotting
    """
    df_time_normal = data_frame.copy()
    df_time_normal["time"] -= df_time_normal["time"].min()

    # use seconds
    df_time_normal["time"] /= 1000. * 1000.

    # plot PLOT_SECONDS_X at a time, scroll old data if collecting more
    xmax = max(PLOT_SECONDS_X, df_time_normal["time"].max())
    xmin = xmax - PLOT_SECONDS_X

    return df_time_normal, xmin, xmax


def plot_exposure_data(data_frame,
                       iso_axes, shutter_time_axes, ev_axes, ev_diff_axes):
    """
    Interactive plot of ISO, shutter time, EV
    and the maximum EV diff to other sensors
    """

    if len(data_frame.index) == 0:
        return

    df_time_normal, xmin, xmax = time_normalized_data(data_frame)

    # drop data that won't be displayed to speed up pivot
    # 5 second buffer to fillna
    df_time_normal = df_time_normal[df_time_normal["time"] >= xmin - 5]

    multi_index = df_time_normal.pivot(index="time", columns="camera_id")
    multi_index = multi_index.fillna(method="ffill")

    iso_axes.clear()
    multi_index["iso"].plot(ax=iso_axes, logy=True, legend=False)
    iso_axes.set_xlim(xmin, xmax)
    iso_axes.set_title("ISO")
    iso_axes.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    iso_axes.set_yticks([100, 200, 400, 800, 1600, 3200, 6400])
    iso_axes.set_yticks([], minor=True)

    shutter_time_axes.clear()
    multi_index["shutterTime"].plot(ax=shutter_time_axes)
    shutter_time_axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    shutter_time_axes.set_xlim(xmin, xmax)
    shutter_time_axes.set_title("Shutter Time")
    shutter_time_axes.set_ylabel("seconds")
    shutter_time_axes.set_ylim(0, 0.04)

    ev = multi_index["ev"]

    ev_axes.clear()
    ev.plot(ax=ev_axes, legend=False)
    ev_axes.set_xlim(xmin, xmax)
    ev_axes.set_title("ISO x Shutter Time")
    ev_axes.set_ylim(0, 10)

    def max_ev_diff_fn(ev_series):
        return ev.subtract(ev_series, axis=0).abs().max(axis=1)

    ev_diff = ev.apply(max_ev_diff_fn)

    ev_diff_axes.clear()
    ev_diff.plot(ax=ev_diff_axes, legend=False)
    ev_diff_axes.set_xlim(xmin, xmax)
    ev_diff_axes.set_title("Max EV Difference")
    ev_diff_axes.set_ylabel("EV")
    ev_diff_axes.set_ylim(0, 8)

    plt.pause(0.0001)


def plot_tc_data(data_frame, curve_axes, midpoint_axes):
    """
    Interactive plot of the last received tone curves and the
    value of the tone curves at x=127.5 (midpoint)
    """

    if len(data_frame.index) == 0:
        return

    df_time_normal, xmin, xmax = time_normalized_data(data_frame)

    by_time = df_time_normal.set_index("time")
    last_tone_curves = by_time.groupby("camera_id").tail(1)

    # drop time
    last_tone_curves = last_tone_curves.set_index("camera_id")

    # tone curves are tuple objects, let's make them a series to plot
    as_series = last_tone_curves.curve.apply(lambda c: pd.Series(c)).T

    curve_axes.clear()
    as_series.plot(ax=curve_axes, legend=False)
    curve_axes.set_xlim(0, 256)
    curve_axes.set_xticks([0, 64, 128, 196, 256])
    curve_axes.set_title("Current tone curve")
    curve_axes.set_yticks([0, 256, 512, 768, 1024])
    curve_axes.set_yticks([], minor=True)

    # reduce curves to middle value
    def curve_midpoint(curve):
        return (curve[128] + curve[127]) / 2.
    by_time.curve = by_time.curve.apply(curve_midpoint)
    midpoint = by_time.pivot(columns="camera_id").fillna(method="ffill")

    midpoint_axes.clear()
    midpoint.plot(ax=midpoint_axes, legend=False)
    midpoint_axes.set_xlim(xmin, xmax)
    midpoint_axes.set_title("Tone Curve midpoint")
    midpoint_axes.set_ylim(512, 768)
    midpoint_axes.set_yticks([512, 576, 640, 704, 768])

    plt.pause(0.0001)


def write_to_disk(data_frame, last_written_idx, file_name):
    """
    Write the data_frame to `file_name`, or append to it if already present.
    May not write anything if fewer than NUM_ROWS_BLOCK_TO_WRITE were added
    to data_frame since last write.
    Returns new last_written_idx.
    """

    # is there any data to write yet?
    if len(data_frame.index) == 0:
        return -1

    if last_written_idx < 0:
        data_frame.to_csv(file_name)
        return data_frame.index[-1]
    elif data_frame.index[-1] - NUM_ROWS_BLOCK_TO_WRITE > last_written_idx:
        data_to_append = data_frame[last_written_idx + 1:]
        with open(file_name, 'a') as f:
            data_to_append.to_csv(f, header=False)
        return data_frame.index[-1]

    return last_written_idx


def read_exposure(metadata_reader):
    """
    Query metadata_reader for exposure data
    Stores the data in a DataFrame
    Returns whether more data is available
    (should be called again)
    Returns data frame
    """

    exposure_map = vs.ExposureMap()

    exposure_status = metadata_reader.readExposure(exposure_map)

    new_data = pd.DataFrame()

    if exposure_status.getCode() == ERROR_STATUS:
        print(exposure_status.getStatus().getErrorMessage())
        return False, new_data

    if exposure_status.getCode() == EOF:
        print("EndOfFile reached on board " + str(board_id))

    for (camera_id, exposure) in exposure_map.iteritems():
        value = {"time": exposure.timestamp,
                 "camera_id": camera_id,
                 "iso": exposure.iso,
                 "shutterTime": exposure.shutterTime,
                 "shutterTimeMax": exposure.shutterTimeMax,
                 "ev": exposure.computeEv()}

        new_data = pd.concat((new_data, pd.DataFrame([value])),
                             ignore_index=True)

    request_more = (exposure_status.getCode() == TRY_AGAIN)

    return request_more, new_data


def read_tone_curves(metadata_reader):
    """
    Query metadata_reader for tone curve data
    Stores the data in a DataFrame
    Returns whether more data is available
    (should be called again)
    Returns data frame
    """

    tc_map = vs.ToneCurveMap()

    tc_status = metadata_reader.readToneCurve(tc_map)

    new_data = pd.DataFrame()

    if tc_status.getCode() == ERROR_STATUS:
        print(tc_status.getStatus().getErrorMessage())
        return False, new_data

    for (camera_id, tone_curve) in tc_map.iteritems():
        value = {"time": tone_curve.timestamp,
                 "camera_id": camera_id,
                 "curve": tone_curve.curveAsArray()}

        new_data = pd.concat((new_data, pd.DataFrame([value])),
                             ignore_index=True)

    if tc_status.getCode() == EOF:
        print("EndOfFile reached on board " + str(board_id))

    request_more = (tc_status.getCode() == TRY_AGAIN)
    return request_more, new_data


if __name__ == "__main__":

    readers = create_readers()

    # don't care about warnings on late stitcher - there is no stitcher
    vs.Logger.setLevel(vs.Logger.Error)

    # set up pyplot figure
    matplotlib.style.use('ggplot')
    # use interactive mode
    plt.ion()

    fig, axes = plt.subplots(3, 2)
    plt.subplots_adjust(wspace=0.5, hspace=0.8)

    # collect metadata from all sensors into one data frame
    exposure_data = pd.DataFrame()
    # tone curves
    tc_data = pd.DataFrame()

    # CSV chunks
    exp_last_written_idx = -1
    tc_last_written_idx = -1

    try:

        # loop forever until interrupted with Ctrl+C

        while True:

            start_time = time.time()

            data_update = False

            for board_id in (0, 1):
                more_exp_data = True
                more_tc_data = True
                while more_exp_data or more_tc_data:

                    more_exp_data, exposure = read_exposure(readers[board_id])
                    if not exposure.empty:
                        exposure_data = pd.concat((exposure_data, exposure),
                                                  ignore_index=True)
                        data_update = True

                    more_tc_data, tc = read_tone_curves(readers[board_id])
                    if not tc.empty:
                        tc_data = pd.concat((tc_data, tc),
                                            ignore_index=True)
                        data_update = True

            if data_update:

                exp_last_written_idx = write_to_disk(exposure_data,
                                                     exp_last_written_idx,
                                                     EXPOSURE_CSV)
                tc_last_written_idx = write_to_disk(tc_data,
                                                    tc_last_written_idx,
                                                    TONE_CURVE_CSV)

                # continuously updating plot
                plot_exposure_data(exposure_data,
                                   axes[0][0], axes[0][1],
                                   axes[1][0], axes[1][1])
                plot_tc_data(tc_data, axes[2][0], axes[2][1])

            # run loops at REFRESH_RATE
            wait_until = start_time + 1. / REFRESH_RATE
            wait_time = wait_until - time.time()
            if (wait_time > 0):
                plt.pause(wait_time)

    except KeyboardInterrupt:
        # ensure the readers have stopped before shutting down
        readers = None
        gc.collect()
