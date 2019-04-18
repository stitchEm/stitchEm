/*****************************************************************************************************************************************
 * From the xiB_SP_Public_V2_62 package. Added an include guard, updated the fhandler macro (l.15), and the
 *XICAMERA_BUFF_COUNT (l.31) . *
 *****************************************************************************************************************************************/

#ifndef XI_CAMERA_H_
#define XI_CAMERA_H_

#include <xiApi.h>
#include <stdio.h>

// camera class
// each camera function finishes normally with return
// if error occours - it is thrown to caller by CPP throw/catch mechanism

#define fhandler(f)      \
  {                      \
    xi_return_e ret = f; \
    if (ret != XI_OK) {  \
      throw(ret);        \
    }                    \
  }

#define XICAMERA_BUFF_SIZE 32768000
#define XICAMERA_BUFF_COUNT \
  4  // We only want one buffer for Vahana VR for the lowest latency. Plus we are not using the scatter-gather
     // functionality it provides.

class xi4Camera {
 public:
  xi4Camera();
  ~xi4Camera();
  void Open(IN DWORD DevId, IN XI_DEVICE_OPEN DevIdType) { fhandler(xiOpenDevice(DevId, DevIdType, &handle)); }
  void SetCameraHandle(HANDLE h) { handle = h; }
  HANDLE GetCameraHandle() { return handle; }
  void CloseDevice();
  void StartAcquisition() { xiStartAcquisition(handle); }
  void StopAcquisition() { xiStopAcquisition(handle); }
  void SetParam(const char* prm, void* val, DWORD size, XI_PRM_TYPE type) {
    fhandler(xiSetParam(handle, prm, val, size, type);)
  }
  void GetParam(const char* prm, void* val, DWORD* size, XI_PRM_TYPE* type) {
    fhandler(xiGetParam(handle, prm, val, size, type);)
  }
  void AnnounceBuffer(void* pBuffer, int iSize, void* pPrivate, BUFFER_HANDLE* phBuffer) {
    fhandler(xiAnnounceBuffer(handle, pBuffer, iSize, pPrivate, phBuffer);)
  }
  void AllocAndAnnounceBuffer(int iSize, void* pPrivate, BUFFER_HANDLE* phBuffer) {
    fhandler(xiAllocAndAnnounceBuffer(handle, iSize, pPrivate, phBuffer);)
  }
  void FlushQueue(UINT32 iOperation) { fhandler(xiFlushQueue(handle, iOperation)); }
  void QueueBuffer(BUFFER_HANDLE hBuffer) { fhandler(xiQueueBuffer(handle, hBuffer)); }
  void RevokeBuffer(BUFFER_HANDLE hBuffer, void** ppBuffer, void** ppPrivate) {
    fhandler(xiRevokeBuffer(handle, hBuffer, ppBuffer, ppPrivate));
  }

  //**********************************************************************************************
  // Event Functions
  //**********************************************************************************************

  void RegisterEvent(XI_EVENT_TYPE iEventID, EVENT_HANDLE* phEvent) {
    fhandler(xiRegisterEvent(handle, iEventID, phEvent));
  }
  void UnRegisterEvent(XI_EVENT_TYPE iEventID) { fhandler(xiUnRegisterEvent(handle, iEventID)); }
  xi_return_e EventGetData(EVENT_HANDLE hEvent, void* pBuffer, UINT32* piSize, UINT32 iTimeout) {
    xi_return_e result = xiEventGetData(handle, hEvent, pBuffer, piSize, iTimeout);
    return result;
  }
  void EventGetDataInfo(EVENT_HANDLE hEvent, const void* pInBuffer, UINT32 iInSize, UINT32 iInfoCmd, UINT32* piType,
                        void* pOutBuffer, UINT32* piOutSize) {
    fhandler(xiEventGetDataInfo(handle, hEvent, pInBuffer, iInSize, iInfoCmd, piType, pOutBuffer, piOutSize));
  }
  void EventFlush(EVENT_HANDLE hEvent) { fhandler(xiEventFlush(handle, hEvent)); }
  void EventGetInfo(EVENT_HANDLE hEvent, XI_EVENT_INFO iInfoCmd, UINT32* pBuffer) {
    fhandler(xiEventGetInfo(handle, hEvent, iInfoCmd, pBuffer));
  }
  void EventKill(EVENT_HANDLE hEvent) { fhandler(xiEventKill(handle, hEvent)); }

  /*-----------------------------------------------------------------------------------*/
  // Set device parameter
  void SetParamInt(const char* prm, int val) { fhandler(xiSetParamInt(handle, prm, val)); }
  void SetParamFloat(const char* prm, float val) { fhandler(xiSetParamFloat(handle, prm, val)); }
  void SetParamString(const char* prm, void* val, DWORD size) { fhandler(xiSetParamString(handle, prm, val, size)); }
  /*-----------------------------------------------------------------------------------*/
  // Get device parameter
  int GetParamInt(const char* prm) {
    int val = 0;
    fhandler(xiGetParamInt(handle, prm, &val));
    return val;
  }
  void GetParamInt64(const char* prm, INT64* val) { fhandler(xiGetParamInt64(handle, prm, val)); }
  float GetParamFloat(const char* prm) {
    float val = 0;
    fhandler(xiGetParamFloat(handle, prm, &val));
    return val;
  }
  void GetParamString(const char* prm, void* val, DWORD size) { fhandler(xiGetParamString(handle, prm, val, size)); }

  /*-----------------------------------------------------------------------------------*/
  // image metadata
  UINT32 GetImageMetaDataInt(XI_Image_Data image, XI_IMAGE_METADATA_ID metadata_id) {
    UINT32 value = 0;
    fhandler(xiGetImageMetaData(image, metadata_id, &value));
    return value;
  }

  // new image event
  xi_return_e WaitForNextImage(int timeout_ms);
  HANDLE GetNewImageEventHandle() { return NewBufferEvent; }
  EVENT_NEW_BUFFER* GetNewImageEventBufferInfo() { return &NewBufferInfo; }
  // camera class buffering
  void BuffersQueue(int queue_count = XICAMERA_BUFF_COUNT);
  void BuffersAllocate();
  void BuffersDeAllocate();

 private:
  // camera handle
  HANDLE handle;
  // buffering
  BUFFER_HANDLE hBuffer[XICAMERA_BUFF_COUNT];
  void* Buffer_Dat[XICAMERA_BUFF_COUNT];
  void* Buffer_Private[XICAMERA_BUFF_COUNT];

  // events
  HANDLE NewBufferEvent;
  EVENT_NEW_BUFFER NewBufferInfo;
  bool buffers_allocated;
};

#endif  // XI_CAMERA_H_
