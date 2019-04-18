## Supported platforms
* Linux
* Windows
* macOS


## Configuration
```
{
  "group" : 1,
  "reader_config" : {
    "type" : "portaudio",
    "audio_delay" : 500000,
    "audio_channels" : 2,
    "audio_sample_rate" : 44100
  },
...
  "video_enabled" : false,
  "audio_enabled" : true
}
```

* add an input of type 'portaudio' to your ptv with audio enabled & video disabled.
* `name`: the name of the portaudio device. The plugin can list device names. You can use Vahana VR to find out the devices on your system.
* The input group parameter must be different from that of other inputs in your ptv configuration.
* `audio_delay`: delay in µseconds, in range 0 < audio_delay < 5000000. (500000 = 0.5s) The delay is set staticly at stitcher setup.
* `audio_channels`: nb of audio channels, default is 2 (stereo) & currently the only supported config.
* `audio_sample_rate`: supported & default values depend on your portaudio device. After setup, the plugin logs its configuration, including the sampling rate used.









