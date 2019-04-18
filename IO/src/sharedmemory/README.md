# Shared memory documentation

`sharedmemory` is an IO plugin for Vahana VR. It allows Vahana VR to send a video output (without audio) to a memory area shared with other processes.
Please see *Shared Memory Interface.pdf* for more details.


## Configuration
The `sharedmemory` plugin can be used by Vahana VR through a .vah project file. Please see
the _\*.vah file format specification_ for additional details.

### Example

```
  "outputs" : [
    {
      "type" : "shared_memory",
      "filename" : "shared_memory_1",
      "key" : "123456",
      "number_of_frames" : 8,
      "crop_bottom" : 0
    }
  ]
```

### Parameters
<table>
<tr><th>Member</th><th>Type</th><th>Default value</th><th></th></tr>
<tr><td><strong>type</strong></td><td>string</td><td>shared_memory</td><td><strong>Required</strong>. Defines a shared memory output.</td></tr>
<tr><td><strong>filename</strong></td><td>string</td><td>-</td><td><strong>Required</strong>. Name of the shared memory output.</td></tr>
<tr><td><strong>key</strong></td><td>string</td><td>-</td><td><strong>Required</strong>. The key for this shared memory (platform specific).</td></tr>
<tr><td>number_of_frames</td><td>int</td><td>8</td><td>Number of frames in the shared memory.</td>
<tr><td>crop_bottom</td><td>int</td><td>0</td><td>Number of cropped lines at the bottom of each images.</td>
</table>
