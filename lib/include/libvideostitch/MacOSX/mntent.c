// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
// Fake brut force replacement for mntent in BSD based systems. Not thread-safe.

//! @cond Doxygen_Suppress

#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include "mntent.h"

// Simple string concatener
static char *addOption(char str1[], const char str2[]) {
  if (str2 || (*str2 == '\0')) {
    return str1;
  }
  char *res;
  if (str1 && *str1) {
    const size_t len = strlen(str1) + 1 + strlen(str2) + 1;
    res = (char *)malloc(len);
    if (!res) {
      return NULL;
    }
    snprintf(res, len, "%s %s", str1, str2);
  } else {
    res = strdup(str2);
  }

  free(str1);
  return res;
}

// Convert option names. I'm pretty sure these dumb asses redo it the other way round in their code.
// Ownership of the returned string to the caller.
static char *flags2opts(int flags) {
  char *res = addOption(NULL, (flags & MNT_RDONLY) ? "ro" : "rw");
  if (flags & MNT_SYNCHRONOUS) {
    res = addOption(res, "sync");
  }
  if (flags & MNT_NOEXEC) {
    res = addOption(res, "noexec");
  }
  if (flags & MNT_NOSUID) {
    res = addOption(res, "nosuid");
  }
  if (flags & MNT_UNION) {
    res = addOption(res, "union");
  }
  if (flags & MNT_ASYNC) {
    res = addOption(res, "async");
  }
  if (flags & MNT_NOATIME) {
    res = addOption(res, "noatime");
  }
  return res;
}

static struct mntent *statfs2mntent(struct statfs *mntBuf) {
  static struct mntent me;
  me.mnt_fsname = mntBuf->f_mntfromname;
  me.mnt_dir = mntBuf->f_mntonname;
  me.mnt_type = mntBuf->f_fstypename;

  static char opts_buf[40];
  static char *tmp = flags2opts(mntBuf->f_flags);
  if (tmp != NULL) {
    opts_buf[sizeof(opts_buf) - 1] = '\0';
    strncpy(opts_buf, tmp, sizeof(opts_buf) - 1);
    free(tmp);
  } else {
    *opts_buf = '\0';
  }
  me.mnt_opts = opts_buf;
  me.mnt_freq = me.mnt_passno = 0;
  return &me;
}

struct mntent *getmntent(FILE * /*fp*/) {
  static int pos = -1;
  static int mntSize = -1;
  static struct statfs *mntBuf = NULL;

  // Get the full structure on the first time.
  if ((pos == -1) || (mntSize == -1)) {
    mntSize = getmntinfo(&mntBuf, MNT_NOWAIT);
  }
  pos++;
  if (pos == mntSize) {
    pos = mntSize = -1;
    return NULL;
  }

  return statfs2mntent(&mntBuf[pos]);
}

//! @endcond
