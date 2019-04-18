from collections import OrderedDict

# list of supported video modes and, for each, resolution, default target bitrate for recording, supported profiles and bitrate limits
VIDEO_MODES = OrderedDict ([
    ("4K DCI", {
        "width": 4096,
        "height": 2048,
        "previewDownsamplingFactor": 2,
        "recording" : { "bitrate": 60000},
        "profiles" : {
            "baseline": {"min_bitrate": 500, "max_bitrate": 50000},
            "main": {"min_bitrate": 500, "max_bitrate": 50000},
            "high": {"min_bitrate": 500, "max_bitrate": 50000}
        }
    }),
    ("4K UHD", {
        "width": 3840,
        "height": 1920,
        "previewDownsamplingFactor": 2,
        "recording" : { "bitrate": 60000},
        "profiles" : {
            "baseline": {"min_bitrate": 500, "max_bitrate": 50000},
            "main": {"min_bitrate": 500, "max_bitrate": 50000},
            "high": {"min_bitrate": 500, "max_bitrate": 50000}
        }
    }),
    ("2.8K", {
        "width": 2880,
        "height": 1440,
        "previewDownsamplingFactor": 2,
        "recording" : { "bitrate": 45000},
        "profiles" : {
            "baseline": {"min_bitrate": 500, "max_bitrate": 50000},
            "main": {"min_bitrate": 500, "max_bitrate": 50000},
            "high": {"min_bitrate": 500, "max_bitrate": 50000}
        }
    }),
    ("2K", {
        "width": 2048,
        "height": 1024,
        "previewDownsamplingFactor": 2,
        "recording" : { "bitrate": 30000},
        "profiles" : {
            "baseline": {"min_bitrate": 500, "max_bitrate": 50000},
            "main": {"min_bitrate": 500, "max_bitrate": 50000},
            "high": {"min_bitrate": 500, "max_bitrate": 50000}
        }
    }),
    ("HD", {
        "width": 1920,
        "height": 960,
        "previewDownsamplingFactor": 2,
        "recording" : { "bitrate": 30000},
        "profiles" : {
            "baseline": {"min_bitrate": 500, "max_bitrate": 50000},
            "main": {"min_bitrate": 500, "max_bitrate": 50000},
            "high": {"min_bitrate": 500, "max_bitrate": 50000}
        }
    })
])
