// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef GUICONSTANTS_HPP
#define GUICONSTANTS_HPP

#include <QString>
#include <QDir>
#include <QDesktopServices>
#include <QApplication>

/************************************
 * user data drive:/Users/MyUser/Vahana VR
 *   |-Projects
 *   |-Recordigns
 *   |-Snapshots
 *
 * application data drive:/ProgramData/Vahana VR
 *   logs
 *   ini file
 *
 * application files drive:/Program Files/VideoStitch/Vahana VR
 *   bins
 *   libs
 *   |-vahana_plugins
 *   |-core_plugins
 *   |-core_plugins_cuda
 *   |-core_plugins_opencl
 *   |-imageformats
 *   |-platforms
 *
 ************************************/

// Main Configurations and directories

// Main files
static const QString& FONT_FILE(":/assets/fonts/OpenSans-Light.ttf");
static const QString& QSS_FILE_VAHANA(":/resources/vahana-vr.qss");
static const QString& QSS_FILE_COMMON(":/style/vs_common.qss");
static const QString& QSS_VARIABLE_FILE(":/resources/style_variables.ini");
static const QString& QSS_COMMON_VARIABLE_FILE(":/style/common_style_variables.ini");

static const QString& DATE_TIME_FORMAT("dd-MM-yy HH:mm:ss");

QString getUserDataPath();
QString getRecordingsPath();
QString getProjectsPath();
QString getSnapshotsPath();

#ifdef __APPLE__
static const QString& LIVE_PLUG_PATH("../VahanaPlugins");
#else
static const QString& LIVE_PLUG_PATH("vahana_plugins");
#endif

// General
static const unsigned int COMPONENT_WIDTH(200);
static const unsigned int COMPONENT_HEIGHT(40);

// Main tabs
static const unsigned int TAB_SIZE(100);
static const unsigned int SCROLL_DISP(30);
static const unsigned int TABS_MARGIN(60);
static const unsigned int LINEEDIT_HEIGHT(40);

// Configurations
static const int CONFPANEL_MAX_COLS(2);
static const int CONFPANEL_MAX_ROWS(2);
static const int CONFPANEL_COL_BORDER(30);
static const int MAXIMUM_FILE_HISTORY(6);

// Configuration Output Input
static const int ITEM_HEIGHT(70);
static const int ITEM_VERTICAL_BORDER(4);
static const int INPUT_COLUMNS(1);

// File list
static const int FILE_MAX_COLS(1);

// Simple round button
static const int BUTTON_SIDE(70);
static const int BUTTON_SPACING(15);

// File extensions
static const QString& PTV_FILE(".ptv");
static const QString& VAH_FILE(".vah");

// Generic Dialog
static const int DIALOG_WIDTH(400);
static const int DIALOG_HEIGHT(300);
static const int BACKGROUND_RIGHT_MARGIN(50);

// Exposure
static const int EXPO_TIMEOUT(2000);

// Dynamic calibration
static const int CALIBRATION_INTERPOLATION_TIMEOUT(40);
static const int CALIBRATION_NUM_STEPS_INTERPOLATION(100);

// Snapshot
static const int SNAP_SHOW_DIALOG(500);

// Transparent background
static const int BLUR_FACTOR(200);

// Output default values
QString getDefaultOutputFileName();
static const QString DEFAULT_RTMP("rtmp://localhost:1935/live/stream");
static const std::string DEFAULT_CODEC("h264");
static const QString DEFAULT_PRESET("medium");
static const QString DEFAULT_PROFILE("baseline");
static const unsigned int DEFAULT_B_FRAMES(2);
static const unsigned int DEFAULT_QUALITY_BALANCE(20);
static const unsigned int DEFAULT_DOWNSAMPLING(1);
static const unsigned int DEFAULT_PANO_HEIGHT(1080);
static const unsigned int DEFAULT_PANO_WIDTH(1920);
static const unsigned int DEFAULT_AUDIO_BITRATE(128);  // In Kb
static const unsigned int DEFAULT_TARGET_USAGE(4);

static const int STATUS_ICON_SIZE(40);
static const int ICON_SIZE(35);
static const char CHECKED_ICON[] = ":/live/icons/assets/icon/live/check.png";

// Configuration Outputs
static const QString& OUTPUT_ACTIVE(QT_TRANSLATE_NOOP("Output state", "Active"));
static const QString& OUTPUT_INACTIVE(QT_TRANSLATE_NOOP("Output state", "Inactive"));
static const int TO_BITS(1000);

// Youtube
static const int YOUTUBE_ICON_HEIGHT(70);
static const int YOUTUBE_PRIVACY_WIDTH(20);

#endif  // GUICONSTANTS_HPP
