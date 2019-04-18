// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ximea_reader.hpp"
#include "ximea_unpack.hpp"

#include "libvideostitch/gpu_cuda.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <algorithm>

using namespace VideoStitch;
using namespace Input;

namespace {
// Sensor constants: see xiB_QuickStartGuide section "6.4 Bandwitch control".
const int sensorWidth = 5120;
const int sensorHeight = 3840;
const int bitsPerPixels = 12;
const double marginError = 0.99;
const int defaultChannels = 16;  // Number of channels employed
const int halfDefaultChannels = 8;
const int defaultSensorOperatingFrequency = 480, minSensorOperatingFrequency = 120,
          maxSensorOperatingFrequency = 480;  // MHz (bit frequency of one data channel from sensor)

// See xiB_QuickStartGuide section "6.5 Gain and Black Level Offset". These parameters are sensor specific. They should
// be based on sensor calibration data.
const int defaultGain = 43, minGain = 1, maxGain = 63;
const int defaultBlackLevelOffset = -55;

// See acquisition.cpp in the xiApiViewver project
const int headerSize = 512;
const int imageHeaderSize = headerSize * 8;

std::string xiReturnToString(const xi_return_e e) {
  switch (e) {
    case XI_OK:
      return "Function call succeeded";
    case XI_INVALID_HANDLE:
      return "Invalid handle";
    case XI_READREG:
      return "Register read error";
    case XI_WRITEREG:
      return "Register write error";
    case XI_FREE_RESOURCES:
      return "Freeing resources error";
    case XI_FREE_CHANNEL:
      return "Freeing channel error";
    case XI_FREE_BANDWIDTH:
      return "Freeing bandwidth error";
    case XI_READBLK:
      return "Read block error";
    case XI_WRITEBLK:
      return "Write block error";
    case XI_NO_IMAGE:
      return "No image";
    case XI_TIMEOUT:
      return "Timeout";
    case XI_INVALID_ARG:
      return "Invalid arguments supplied";
    case XI_NOT_SUPPORTED:
      return "Not supported";
    case XI_ISOCH_ATTACH_BUFFERS:
      return "Attach buffers error";
    case XI_GET_OVERLAPPED_RESULT:
      return "Overlapped result";
    case XI_MEMORY_ALLOCATION:
      return "Memory allocation error";
    case XI_DLLCONTEXTISNULL:
      return "DLL context is NULL";
    case XI_DLLCONTEXTISNONZERO:
      return "DLL context is non zero";
    case XI_DLLCONTEXTEXIST:
      return "DLL context exists";
    case XI_TOOMANYDEVICES:
      return "Too many devices connected";
    case XI_ERRORCAMCONTEXT:
      return "Camera context error";
    case XI_UNKNOWN_HARDWARE:
      return "Unknown hardware";
    case XI_INVALID_TM_FILE:
      return "Invalid TM file";
    case XI_INVALID_TM_TAG:
      return "Invalid TM tag";
    case XI_INCOMPLETE_TM:
      return "Incomplete TM";
    case XI_BUS_RESET_FAILED:
      return "Bus reset error";
    case XI_NOT_IMPLEMENTED:
      return "Not implemented";
    case XI_SHADING_TOOBRIGHT:
      return "Shading too bright";
    case XI_SHADING_TOODARK:
      return "Shading too dark";
    case XI_TOO_LOW_GAIN:
      return "Gain is too low";
    case XI_INVALID_BPL:
      return "Invalid bad pixel list";
    case XI_BPL_REALLOC:
      return "Bad pixel list realloc error";
    case XI_INVALID_PIXEL_LIST:
      return "Invalid pixel list";
    case XI_INVALID_FFS:
      return "Invalid Flash File System";
    case XI_INVALID_PROFILE:
      return "Invalid profile";
    case XI_INVALID_CALIBRATION:
      return "Invalid calibration";
    case XI_INVALID_BUFFER:
      return "Invalid buffer";
    case XI_INVALID_DATA:
      return "Invalid data";
    case XI_TGBUSY:
      return "Timing generator is busy";
    case XI_IO_WRONG:
      return "Wrong operation open/write/read/close";
    case XI_ACQUISITION_ALREADY_UP:
      return "Acquisition already started";
    case XI_OLD_DRIVER_VERSION:
      return "Old version of device driver installed to the system";
    case XI_GET_LAST_ERROR:
      return "To get error code please call GetLastError function";
    case XI_CANT_PROCESS:
      return "Data can't be processed";
    case XI_ACQUISITION_STOPED:
      return "Acquisition has been stopped It should be started before GetImage";
    case XI_ACQUISITION_STOPED_WERR:
      return "Acquisition has been stopped with error";
    case XI_INVALID_INPUT_ICC_PROFILE:
      return "Input ICC profile missed or corrupted";
    case XI_INVALID_OUTPUT_ICC_PROFILE:
      return "Output ICC profile missed or corrupted";
    case XI_DEVICE_NOT_READY:
      return "Device not ready to operate";
    case XI_SHADING_TOOCONTRAST:
      return "Shading too contrast";
    case XI_ALREADY_INITIALIZED:
      return "Mobile already initialized";
    case XI_NOT_ENOUGH_PRIVILEGES:
      return "Application doesn't enough privileges (one or more app)";
    case XI_NOT_COMPATIBLE_DRIVER:
      return "Installed driver not compatible with current software";
    case XI_TM_INVALID_RESOURCE:
      return "TM file was not loaded successfully from resources";
    case XI_DEVICE_HAS_BEEN_RESETED:
      return "Device has been reseted, abnormal initial state";
    case XI_NO_DEVICES_FOUND:
      return "No Devices Found";
    case XI_RESOURCE_OR_FUNCTION_LOCKED:
      return "Resource(device) or function locked by mutex";
    case XI_UNKNOWN_PARAM:
      return "Unknown parameter";
    case XI_WRONG_PARAM_VALUE:
      return "Wrong parameter value";
    case XI_WRONG_PARAM_TYPE:
      return "Wrong parameter type";
    case XI_WRONG_PARAM_SIZE:
      return "Wrong parameter size";
    case XI_BUFFER_TOO_SMALL:
      return "Input buffer too small";
    case XI_NOT_SUPPORTED_PARAM:
      return "Parameter info not supported";
    case XI_NOT_SUPPORTED_PARAM_INFO:
      return "Parameter info not supported";
    case XI_NOT_SUPPORTED_DATA_FORMAT:
      return "Data format not supported";
    case XI_READ_ONLY_PARAM:
      return "Read only parameter";
    case XI_WRITE_ONLY_PARAM:
      return "Write only parameter";
    case XI_BANDWIDTH_NOT_SUPPORTED:
      return "This camera does not support currently available bandwidth";
    case XI_RESOURCE_IN_USE:
      return "The resource is already in use";
    case XI_EVENT_KILLED:
      return "Event waiting was terminated by kill";
    case XI_LENS_NOT_DETECTED:
      return "Lens does not respond to Poll Busy command";
    case XI_LENS_NOT_ENABLED:
      return "Lens was not enabled";
    case XI_ERR_UNKNOWN_FILE:
      return "File or partition not found";
    case XI_ERR_SIZE_OF_DATA_OVERFLOW:
      return "Size of data to be copied is larger than internal buffer (eg Set_FileAccessBuffer)";
    case XI_ERR_FILE_NOT_SELECTED:
      return "No file is selected (FAL)";
    case XI_ERR_FILE_ALREADY_OPENED:
      return "File is already open";
    case XI_ERR_FILE_ALREADY_CLOSED:
      return "File is already closed";
    case XI_ERR_FILE_INVALID_OPEN_MODE:
      return "File is opened in wrong mode for this operation";
    case XI_ERR_FILE_INVALID_PARTITION:
      return "Partition table not found - file can not be used";
    case XI_ERR_FILE_NOT_OPENED:
      return "File is not opened";
    case XI_ERR_FILE_KEY_INVALID:
      return "File key is invalid";
    case XI_ERR_FILE_IS_LOCKED_FOR_WRITE:
      return "File is locked for write";
    case XI_ERR_LOCAL_FILE_OPEN_ERROR:
      return "Local file for read/write can not be open";
    case XI_ERR_FEATURE_NOT_FOUND:
      return "Requested feature not found";
    case XI_ERR_USERSET_NOT_FOUND:
      return "User set is not defined";
    case XI_ERR_USERSET_PARSING_ERR:
      return "User set parsing finished with error";
    case XI_ERR_USERSET_UNKNOWN_VAR_TYPE:
      return "User set value type is not implemented";
    default:
      return "Unknown error";
  }
}
}  // namespace

XimeaReader* XimeaReader::create(const Ptv::Value* config, const int64_t width, const int64_t height) {
  xi4Camera* device = nullptr;

  // Get the configuration
  std::string name;
  if (Parse::populateString("Ximea reader", *config, "name", name, true) != VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error)
        << "Error: Ximea device name couldn't be retrieved. Please, give a device name (\"name\") to your input."
        << std::endl;
    goto shutdown;
  }

  double fps;
  if (Parse::populateDouble("Ximea reader", *config, "fps", fps, true) != VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Error: Ximea framerate couldn't be retrieved on the device " << name
                               << ". Please, give a framerate (\"fps\") to your input." << std::endl;
    goto shutdown;
  }

  int exposure;
  if (Parse::populateInt("Ximea reader", *config, "exposure", exposure, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Error) << "Error: Ximea exposure couldn't be retrieved on the device " << name
                               << ". Please, give an exposure (\"exposure\" in microseconds) to your input."
                               << std::endl;
    goto shutdown;
  }

  int channels = defaultChannels;
  if (Parse::populateInt("Ximea reader", *config, "channels", channels, false) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Info) << "Error: Ximea PCIe used channels number (\"channels\") can be set on the device "
                              << name << ": " << defaultChannels << " by default, " << halfDefaultChannels
                              << " on slower systems." << std::endl;
  } else if (channels != defaultChannels && channels != halfDefaultChannels) {
    Logger::get(Logger::Error) << "Error: Ximea device channels number is not correct on the device " << name
                               << ". Please, give a correct channels number (\"channels\"): " << defaultChannels
                               << " by default, " << halfDefaultChannels << " on slower systems." << std::endl;
    goto shutdown;
  }

  double sensorOperatingFrequency = 0;
  if (Parse::populateDouble("Ximea reader", *config, "frequency", sensorOperatingFrequency, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Info) << "Ximea device frequency (\"frequency\" in MHz) can be set on the device " << name
                              << ". If it's not, the approximate maximum frequency will be used: calculated from "
                                 "\"fps\", \"width\", \"height\" and \"channels\". The default value is "
                              << defaultSensorOperatingFrequency << " MHz." << std::endl;
  } else if (sensorOperatingFrequency <= 0) {
    Logger::get(Logger::Error) << "Error: Ximea sensor operating frequency is not correct on the device " << name
                               << ". Please, give a sensor operating frequency (\"frequency\") larger than 0."
                               << std::endl;
    goto shutdown;
  }

  int gain;
  if (Parse::populateInt("Ximea reader", *config, "gain", gain, false) == VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Info) << "Error: Ximea device gain (\"gain\") can be set between " << minGain << " and "
                              << maxGain << " on the device " << name << ". If it's not, the default value ("
                              << defaultGain << ") will be used." << std::endl;
    gain = defaultGain;
  } else if (gain < minGain || gain > maxGain) {
    Logger::get(Logger::Error) << "Error: Ximea device gain is not correct on the device " << name
                               << ". Please, give a correct gain (\"gain\") between " << minGain << " and " << maxGain
                               << "." << std::endl;
    goto shutdown;
  }

  int blackLevelOffset = defaultBlackLevelOffset;
  if (Parse::populateInt("Ximea reader", *config, "black_level_offset", blackLevelOffset, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Info) << "Ximea device black level offset (\"black_level_offset\") can be set on the device "
                              << name << ". The default value (" << defaultBlackLevelOffset << ") will be used."
                              << std::endl;
  }

  double lensFStop = 0.;
  Parse::PopulateResult lensFStopPopulate = Parse::populateDouble("Ximea reader", *config, "f-stop", lensFStop, false);

  // Get the device
  DWORD id, devicesCount = 0;
  xiGetNumberDevices(&devicesCount);
  if (devicesCount == 0) {
    Logger::get(Logger::Error) << "Error: Ximea could not list any device." << std::endl;
    goto shutdown;
  }

  for (id = 0; id < devicesCount; id++) {
    Logger::get(Logger::Info) << "Ximea: found the device " << name << "." << std::endl;
    if (std::stoi(name) == id) {
      Logger::get(Logger::Debug) << "Ximea: found the suitable device " << name << "." << std::endl;
      break;
    }
  }
  if (std::stoi(name) != id) {
    Logger::get(Logger::Error) << "Error: Ximea could not found any suitable device for " << name << "." << std::endl;
    goto shutdown;
  }

  // Check if the device is connected
  int serial;
  DWORD serial_s = 4;
  XI_PRM_TYPE serial_t = XI_PRM_TYPE::xiTypeInteger;
  xi_return_e r = xiGetDeviceInfo(id, XI_PRM_DEVICE_SERIAL_NUMBER, &serial, &serial_s, &serial_t);
  if (r != XI_OK || serial == -1) {
    Logger::get(Logger::Error) << "Error: Ximea could not retrieve the device " << name
                               << ". Please check if the device is connected and available to capture." << std::endl;
    goto shutdown;
  }

  // Open the device
  try {
    device = new xi4Camera;
    device->Open(id, XI_DEVICE_OPEN_INDEX);
  } catch (xi_return_e& e) {
    Logger::get(Logger::Error) << "Error: Ximea could not open the device " << name << ". Returned code is " << e
                               << " \"" << xiReturnToString(e) << "\"." << std::endl;
    goto shutdown;
  }

  // Configure bandwitch control
  try {
    double calculatedSensorOperatingFrequency = fps * width * height * bitsPerPixels * marginError / channels;
    calculatedSensorOperatingFrequency /= 1000000;  // MHz
    calculatedSensorOperatingFrequency = (std::max)(
        minSensorOperatingFrequency, (std::min)(maxSensorOperatingFrequency, (int)calculatedSensorOperatingFrequency));
    Logger::get(Logger::Info) << "Ximea: approximate maximum sensor operating frequency calculated on the device "
                              << name << " is " << calculatedSensorOperatingFrequency << " MHz." << std::endl;
    if (sensorOperatingFrequency <= 0) {
      sensorOperatingFrequency = calculatedSensorOperatingFrequency;
    }
    device->SetParamFloat(XI_PRM_FREQ, (float)sensorOperatingFrequency);

    device->SetParamInt(XI_PRM_OUTPUT_MODE,
                        (channels == halfDefaultChannels) ? 1 : 0);  // For CMV20000 0 = 16 channels, 1 = 8 channels
  } catch (xi_return_e& e) {
    Logger::get(Logger::Error)
        << "Error: Ximea could not configure the bandwitch control (frequency and channels) of the device " << name
        << ". Returned code is " << e << " \"" << xiReturnToString(e) << "\"." << std::endl;
    goto shutdown;
  }

  // Set ROI
  try {
    const int offsetX = 0, offsetY = 0;
    device->SetParamInt(XI_PRM_OFFSET_X, 0);
    device->SetParamInt(XI_PRM_OFFSET_Y, 0);
    device->SetParamInt(XI_PRM_WIDTH, (int)width);
    device->SetParamInt(XI_PRM_HEIGHT, (int)height);
    Logger::get(Logger::Info) << "Ximea: set the origin of the Region Of Interest on the device " << name << " to ("
                              << offsetX << "," << offsetY << ") and the image resolution to " << width << "x" << height
                              << "." << std::endl;
  } catch (xi_return_e& e) {
    Logger::get(Logger::Error) << "Error: Ximea could not configure the Region Of Intereset on the device " << name
                               << ". Returned code is " << e << " \"" << xiReturnToString(e) << "\"." << std::endl;
    goto shutdown;
  }

  // Set Gain, Black Level Offset and Exposure
  try {
    device->SetParamInt(XI_PRM_GAIN, gain);
  } catch (xi_return_e& e) {
    Logger::get(Logger::Error) << "Error: Ximea could not configure the gain of the device " << name
                               << ". Returned code is " << e << " \"" << xiReturnToString(e) << "\"." << std::endl;
    goto shutdown;
  }
  try {
    device->SetParamInt(XI_PRM_BLACK_LEVEL_OFFSET, blackLevelOffset);
  } catch (xi_return_e& e) {
    Logger::get(Logger::Error) << "Error: Ximea could not configure the black level of the device " << name
                               << ". Returned code is " << e << " \"" << xiReturnToString(e) << "\"." << std::endl;
    goto shutdown;
  }
  try {
    device->SetParamInt(XI_PRM_EXPOSURE, exposure);
  } catch (xi_return_e& e) {
    Logger::get(Logger::Error) << "Error: Ximea could not configure the exposure of the device " << name
                               << ". Returned code is " << e << " \"" << xiReturnToString(e) << "\"." << std::endl;
    goto shutdown;
  }

  // Configure the lens f-stop aperture
  float minLensFStop, maxLensFStop;
  try {
    device->SetParamInt(XI_PRM_LC_ENABLE, 1);
    minLensFStop = device->GetParamFloat(XI_PRM_LC_FSTOP_MIN_VAL);
    maxLensFStop = device->GetParamFloat(XI_PRM_LC_FSTOP_MAX_VAL);
  } catch (xi_return_e& e) {
    Logger::get(Logger::Error) << "Error: Ximea could not configure the lens of the device " << name
                               << ". Returned code is " << e << " \"" << xiReturnToString(e) << "\"." << std::endl;
    goto capture;
  }
  if (minLensFStop == 0 || maxLensFStop == 0) {  // Values are set to 0 if incorrectly detected
    Logger::get(Logger::Error) << "Error: Ximea could not configure the lens of the device " << name
                               << ": incorrect lens parameters detected." << std::endl;
    goto capture;
  }

  if (lensFStopPopulate != VideoStitch::Parse::PopulateResult_Ok) {
    Logger::get(Logger::Info) << "Ximea lens' f-stop value (\"f-stop\") can be set on the device " << name
                              << ". The detected lens' f-stop range is between f/" << minLensFStop << " and f/"
                              << maxLensFStop << ". The minimum value (f/" << minLensFStop << ") will be used."
                              << std::endl;
    lensFStop = minLensFStop;
  } else if (lensFStop < minLensFStop || lensFStop > maxLensFStop) {
    Logger::get(Logger::Error) << "Ximea lens' f-stop value (\"f-stop\") is not correct on the device " << name
                               << ". The detected lens' f-stop range on the device " << name << " is between f/"
                               << minLensFStop << " and f/" << maxLensFStop << "." << std::endl;
    goto shutdown;
  }

  try {
    device->SetParamFloat(XI_PRM_LC_FSTOP_VAL, (float)lensFStop);
    Logger::get(Logger::Debug) << "Ximea lens' f-stop value set to f/" << device->GetParamFloat(XI_PRM_LC_FSTOP_VAL)
                               << " on the device " << name << std::endl;
  } catch (xi_return_e& e) {
    Logger::get(Logger::Error) << "Error: Ximea could not configure the lens' f-stop value on the device " << name
                               << ". Returned code is " << e << " \"" << xiReturnToString(e) << "\"." << std::endl;
    goto shutdown;
  }

  // Start capture
capture:
  try {
    device->FlushQueue(XI_ACQ_QUEUE_ALL_DISCARD);
    device->BuffersAllocate();
    device->BuffersQueue();

    device->SetParamInt(XI_PRM_ACQUISITION_FRAME_COUNT, 0);
    device->StartAcquisition();
  } catch (xi_return_e& e) {
    Logger::get(Logger::Error) << "Error: Ximea could not start the capture on the device " << name
                               << ". Returned code is " << e << " \"" << xiReturnToString(e) << "\"." << std::endl;
    goto shutdown;
  }

  // Set capture parameters
  int timeout = (int)(1000 / fps);
  int64_t frameDataSize = (width * height * 3) / 2;  // Expects mono 12 bits

  return new XimeaReader(width, height, fps, frameDataSize, name, device, timeout);

shutdown:
  if (device) {
    delete device;
  }
  return nullptr;
}

XimeaReader::XimeaReader(const int64_t width, const int64_t height, double fps, int64_t frameDataSize, std::string name,
                         xi4Camera* device, int timeout)
    : StatefulReader<unsigned char*>(width, height, frameDataSize, Unknown, fps, 0,
                                     VideoStitch::Input::Reader::NO_LAST_FRAME, NULL, 0, Audio::SamplingRate::SR_NONE,
                                     Audio::SamplingDepth::SD_NONE),
      name(name),
      device(device),
      timeout(timeout),
      currFrame(-1) {}

XimeaReader::~XimeaReader() {
  std::lock_guard<std::mutex> lk(mutex);
  device->StopAcquisition();
  device->FlushQueue(XI_ACQ_QUEUE_ALL_DISCARD);
  device->EventFlush(device->GetNewImageEventHandle());
  device->BuffersDeAllocate();
  device->CloseDevice();
  delete device;
}

bool XimeaReader::handles(const Ptv::Value* config) {
  return config && config->has("type") && (config->has("type")->asString() == "ximea");
}

Status XimeaReader::readFrame(int& frameId, unsigned char* data, Audio::Samples& /*audio*/) {
  std::lock_guard<std::mutex> lk(mutex);

  xi_return_e e = device->WaitForNextImage(timeout);
  if (e == XI_TIMEOUT) {
    return VideoStitch::Status::OK();
  }
  if (e != XI_OK) {
    Logger::get(Logger::Error) << "Error: Ximea device " << name << " couldn't reads frame, code " << e << "."
                               << "Lost frames: " << device->GetParamInt(XI_PRM_ACQUISITION_FRAME_LOST)
                               << " underrun frames: " << device->GetParamInt(XI_PRM_ACQUISITION_FRAME_UNDERRUN) << "."
                               << std::endl;
    return VideoStitch::Status(ReaderError);
  }

  EVENT_NEW_BUFFER* bufferInfo = device->GetNewImageEventBufferInfo();
  memcpy(data, (unsigned char*)(bufferInfo->UserPointer) + imageHeaderSize, getFrameDataSize());
  device->QueueBuffer(bufferInfo->BufferHandle);

  frameId = ++currFrame;
  return VideoStitch::Status::OK();
}

Status XimeaReader::readFrameAudioOnly(Audio::Samples& /*audio*/) { return VideoStitch::Status::OK(); }

Status XimeaReader::seekFrame(unsigned /*targetFrame*/) { return Status::OK(); }

Status XimeaReader::unpackDevBuffer(const GPU::Buffer<uint32_t>& dst, const GPU::Buffer<const unsigned char>& src,
                                    GPU::Stream& stream) const {
  unsigned char* const* frameBuffer = getCurrentDeviceData();
  const unsigned char* srcRaw = VideoStitch::GPU::getRawGPUPointer(src);
  cudaStream_t cudaStream = VideoStitch::GPU::getCudaStream(stream);
  unpackMono12p(*frameBuffer, srcRaw, getWidth(), getHeight(), cudaStream);
  VideoStitch::GPU::Buffer<const unsigned char>& frameBufferRef =
      VideoStitch::GPU::createBufferReference<const unsigned char>(*frameBuffer, getWidth() * getHeight());
  Status unpackStatus =
      Reader::unpackDevBuffer(VideoStitch::Bayer_RGGB, dst, frameBufferRef, getSpec().width, getSpec().height, stream);
  VideoStitch::GPU::destroyBufferReference(frameBufferRef);
  return unpackStatus;
}

Status XimeaReader::perThreadInit() {
  unsigned char* frameBuffer;
  cudaMalloc((void**)&frameBuffer, getWidth() * getHeight());
  cudaMemset(frameBuffer, 0, getWidth() * getHeight());
  return setCurrentDeviceData(frameBuffer);
}

void XimeaReader::perThreadCleanup() {
  unsigned char* const* frameBuffer = getCurrentDeviceData();
  cudaFree(*frameBuffer);
  StatefulBase::perThreadCleanup();
}

int XimeaReader::getIntegerValue(const char* key, int defaultValue) const {
  if (strcmp(key, "exposure") == 0) {
    return device->GetParamInt(XI_PRM_EXPOSURE);
  } else {
    return defaultValue;
  }
}

void XimeaReader::setIntegerValue(const char* key, int value) {
  if (strcmp(key, "exposure") == 0) {
    device->SetParamInt(XI_PRM_EXPOSURE, value);
  }
}
