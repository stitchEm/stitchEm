// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

/* Workaround hack to fix a link problem when generating a shared lib
   on linux. */
extern "C" {
void *__dso_handle = 0;
}
