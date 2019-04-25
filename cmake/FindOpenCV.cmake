if(WINDOWS)
  find_debug_and_optimized_library(OpenCV_CORE_LIBS      "opencv2/Debug" "opencv_core310d"       "opencv2/Release" "opencv_core310")
  find_debug_and_optimized_library(OpenCV_CALIB_LIBS     "opencv2/Debug" "opencv_calib3d310d"    "opencv2/Release" "opencv_calib3d310")
  find_debug_and_optimized_library(OpenCV_FEATURES_LIBS  "opencv2/Debug" "opencv_features2d310d" "opencv2/Release" "opencv_features2d310")
  find_debug_and_optimized_library(OpenCV_IMGCODECS_LIBS "opencv2/Debug" "opencv_imgcodecs310d" "opencv2/Release" "opencv_imgcodecs310")
  find_debug_and_optimized_library(OpenCV_IMGPROC_LIBS   "opencv2/Debug" "opencv_imgproc310d"    "opencv2/Release" "opencv_imgproc310")
  find_debug_and_optimized_library(OpenCV_FLANN_LIBS     "opencv2/Debug" "opencv_flann310d"      "opencv2/Release" "opencv_flann310")
  find_debug_and_optimized_library(OpenCV_VIDEO_LIBS     "opencv2/Debug" "opencv_video310d"      "opencv2/Release" "opencv_video310")
  set(OpenCV_LIBRARIES ${OpenCV_CORE_LIBS} ${OpenCV_CALIB_LIBS} ${OpenCV_FEATURES_LIBS} ${OpenCV_IMGPROC_LIBS} ${OpenCV_FLANN_LIBS} ${OpenCV_VIDEO_LIBS})
  set(OpenCV_INCLUDE_DIRS ${CMAKE_EXTERNAL_DEPS}/include/opencv2)
endif()


if(LINUX OR ANDROID)
  if(LINUX)
    find_library(OpenCV_CORE_LIBS NAMES "opencv_core")
    find_library(OpenCV_CALIB_LIBS NAMES "opencv_calib3d")
    find_library(OpenCV_FEATURES_LIBS NAMES "opencv_features2d")
    find_library(OpenCV_IMGPROC_LIBS NAMES "opencv_imgproc")
    find_library(OpenCV_FLANN_LIBS NAMES "opencv_flann")
    find_library(OpenCV_VIDEO_LIBS NAMES "opencv_video")
  else()
    find_library(OpenCV_CORE_LIBS NAMES "opencv_core" PATHS ${CMAKE_EXTERNAL_DEPS}/lib  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(OpenCV_CALIB_LIBS NAMES "opencv_calib3d" PATHS ${CMAKE_EXTERNAL_DEPS}/lib  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(OpenCV_FEATURES_LIBS NAMES "opencv_features2d" PATHS ${CMAKE_EXTERNAL_DEPS}/lib  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(OpenCV_IMGPROC_LIBS NAMES "opencv_imgproc" PATHS ${CMAKE_EXTERNAL_DEPS}/lib  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(OpenCV_FLANN_LIBS NAMES "opencv_flann" PATHS ${CMAKE_EXTERNAL_DEPS}/lib  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(OpenCV_VIDEO_LIBS NAMES "opencv_video" PATHS ${CMAKE_EXTERNAL_DEPS}/lib  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    set(OpenCV_INCLUDE_DIRS ${CMAKE_EXTERNAL_DEPS}/include/opencv2)
  endif()

  set(OpenCV_LIBRARIES
      ${OpenCV_CALIB_LIBS}
      ${OpenCV_FEATURES_LIBS}
      ${OpenCV_FLANN_LIBS}
      ${OpenCV_VIDEO_LIBS}
      ${OpenCV_IMGPROC_LIBS}
      ${OpenCV_CORE_LIBS})
endif()

if(ANDROID)
  find_library(OpenCV_HAL_LIBS NAMES "opencv_hal" PATHS ${CMAKE_EXTERNAL_DEPS}/lib  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  set(OpenCV_LIBRARIES
      ${OpenCV_LIBRARIES}
      ${OpenCV_HAL_LIBS})
endif()

if(APPLE)
  if(APPLE_MACPORTS)
    find_package(OpenCV REQUIRED core calib3d features2d imgproc flann video PATHS /opt/local NO_DEFAULT_PATH)
  elseif(APPLE_BREW)
    find_package(OpenCV REQUIRED core calib3d features2d imgproc flann video PATHS "/usr/local/opt/opencv@3" "/usr/local" NO_DEFAULT_PATH)
  else()
    find_package(OpenCV REQUIRED core calib3d features2d imgproc flann video PATHS ${CMAKE_EXTERNAL_DEPS}/lib/opencv2/share/OpenCV NO_DEFAULT_PATH)
  endif()
endif()
