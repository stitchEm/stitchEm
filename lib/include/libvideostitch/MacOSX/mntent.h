// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
// Fake brut force replacement for mntent in BSD based systems.

//! @cond Doxygen_Suppress

#ifndef MNTENT_HPP_
#define MNTENT_HPP_

#include <stdio.h>

struct mntent {
  char *mnt_fsname;
  char *mnt_dir;
  char *mnt_type;
  char *mnt_opts;
  int mnt_freq;
  int mnt_passno;
};

#define setmntent(x, y) ((FILE *)0x1)
inline void endmntent(FILE *x) {}
struct mntent *getmntent(FILE *fp);

#include "mntent.c"

#endif

//! @endcond
