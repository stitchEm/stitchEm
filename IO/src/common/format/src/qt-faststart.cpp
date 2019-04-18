// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

/* This code is a copy of the qtfaststart utility.
 * "This file is placed in the public domain. Use the program however you see fit."
 */

/*
 * qt-faststart.c, v0.2
 * by Mike Melanson (melanson@pcisys.net)
 * This file is placed in the public domain. Use the program however you
 * see fit.
 *
 * This utility rearranges a Quicktime file such that the moov atom
 * is in front of the data, thus facilitating network streaming.
 *
 * To compile this program, start from the base directory from which you
 * are building Libav and type:
 *  make tools/qt-faststart
 * The qt-faststart program will be built in the tools/ directory. If you
 * do not build the program in this manner, correct results are not
 * guaranteed, particularly on 64-bit platforms.
 * Invoke the program with:
 *  qt-faststart <infile.mov> <outfile.mov>
 *
 * Notes: Quicktime files can come in many configurations of top-level
 * atoms. This utility stipulates that the very last atom in the file needs
 * to be a moov atom. When given such a file, this utility will rearrange
 * the top-level atoms by shifting the moov atom from the back of the file
 * to the front, and patch the chunk offsets along the way. This utility
 * presently only operates on uncompressed moov atoms.
 */

#include "io.hpp"
#include "libvideostitch/logging.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <cmath>

#ifdef __MINGW32__
#define fseeko(x, y, z) fseeko64(x, y, z)
#define ftello(x) ftello64(x)
#elif defined(_WIN32)
#define fseeko(x, y, z) _fseeki64(x, y, z)
#define ftello(x) _ftelli64(x)
#endif

#define BE_16(x) ((((uint8_t *)(x))[0] << 8) | ((uint8_t *)(x))[1])

#define BE_32(x) \
  ((((uint8_t *)(x))[0] << 24) | (((uint8_t *)(x))[1] << 16) | (((uint8_t *)(x))[2] << 8) | ((uint8_t *)(x))[3])

#define BE_64(x)                                                                       \
  (((uint64_t)(((uint8_t *)(x))[0]) << 56) | ((uint64_t)(((uint8_t *)(x))[1]) << 48) | \
   ((uint64_t)(((uint8_t *)(x))[2]) << 40) | ((uint64_t)(((uint8_t *)(x))[3]) << 32) | \
   ((uint64_t)(((uint8_t *)(x))[4]) << 24) | ((uint64_t)(((uint8_t *)(x))[5]) << 16) | \
   ((uint64_t)(((uint8_t *)(x))[6]) << 8) | ((uint64_t)((uint8_t *)(x))[7]))

#define BE_FOURCC(ch0, ch1, ch2, ch3)                                                                                \
  ((uint32_t)(unsigned char)(ch3) | ((uint32_t)(unsigned char)(ch2) << 8) | ((uint32_t)(unsigned char)(ch1) << 16) | \
   ((uint32_t)(unsigned char)(ch0) << 24))

#define QT_ATOM BE_FOURCC
/* top level atoms */
#define FREE_ATOM QT_ATOM('f', 'r', 'e', 'e')
#define JUNK_ATOM QT_ATOM('j', 'u', 'n', 'k')
#define MDAT_ATOM QT_ATOM('m', 'd', 'a', 't')
#define MOOV_ATOM QT_ATOM('m', 'o', 'o', 'v')
#define TRAK_ATOM QT_ATOM('t', 'r', 'a', 'k')
#define PNOT_ATOM QT_ATOM('p', 'n', 'o', 't')
#define SKIP_ATOM QT_ATOM('s', 'k', 'i', 'p')
#define WIDE_ATOM QT_ATOM('w', 'i', 'd', 'e')
#define PICT_ATOM QT_ATOM('P', 'I', 'C', 'T')
#define FTYP_ATOM QT_ATOM('f', 't', 'y', 'p')
#define UUID_ATOM QT_ATOM('u', 'u', 'i', 'd')
#define MDIA_ATOM QT_ATOM('m', 'd', 'i', 'a')
#define HDLR_ATOM QT_ATOM('h', 'd', 'l', 'r')
#define MINF_ATOM QT_ATOM('m', 'i', 'n', 'f')
#define STBL_ATOM QT_ATOM('s', 't', 'b', 'l')
#define STSD_ATOM QT_ATOM('s', 't', 's', 'd')
#define MP4A_ATOM QT_ATOM('m', 'p', '4', 'a')
#define SA3D_ATOM QT_ATOM('S', 'A', '3', 'D')

#define MDIA_VIDE QT_ATOM('v', 'i', 'd', 'e')
#define MDIA_SOUN QT_ATOM('s', 'o', 'u', 'n')

#define CMOV_ATOM QT_ATOM('c', 'm', 'o', 'v')
#define STCO_ATOM QT_ATOM('s', 't', 'c', 'o')
#define CO64_ATOM QT_ATOM('c', 'o', '6', '4')

#define ATOM_PREAMBLE_SIZE 8
#define COPY_BUFFER_SIZE 1024000

// ffcc8263-f855-4a93-8814-587a02521fdd
static const unsigned char uuid_value[] = {0xff, 0xcc, 0x82, 0x63, 0xf8, 0x55, 0x4a, 0x93,
                                           0x88, 0x14, 0x58, 0x7a, 0x02, 0x52, 0x1f, 0xdd};

static const unsigned char spherical_xml[] =
    "<?xml version=\"1.0\"?>\n"
    "<rdf:SphericalVideo"
    " xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\""
    " xmlns:GSpherical=\"http://ns.google.com/videos/1.0/spherical/\">\n"
    "  <GSpherical:Spherical>true</GSpherical:Spherical>\n"
    "  <GSpherical:Stitched>true</GSpherical:Stitched>\n"
    "  <GSpherical:StitchingSoftware>VideoStitch Studio"
    "</GSpherical:StitchingSoftware>\n"
    "  <GSpherical:ProjectionType>equirectangular"
    "</GSpherical:ProjectionType>\n"
    "</rdf:SphericalVideo>";

void set_atom_size(unsigned char *atom, uint64_t new_size) {
  atom[0] = (new_size >> 24) & 0xFF;
  atom[1] = (new_size >> 16) & 0xFF;
  atom[2] = (new_size >> 8) & 0xFF;
  atom[3] = (new_size >> 0) & 0xFF;
}

uint64_t find_atom_type(const uint64_t u_atom, uint64_t atom_size, uint32_t atom_type) {
  const unsigned char *atom = (const unsigned char *)u_atom;
  // find sub-element atom by parsing through the current atom
  for (uint64_t i = 4; i < atom_size - 4;) {
    if (atom_type == (uint32_t)BE_32(&atom[i])) {
      VideoStitch::Logger::debug("libavoutput") << "atom " << atom[i] << atom[i + 1] << atom[i + 2] << atom[i + 3]
                                                << " of size " << BE_32(&atom[i - 4]) << std::endl;
      return u_atom + i - 4;
    } else {
      i += BE_32(&atom[i - 4]);
    }
  }
  return 0;
}

#define CHECK_ATOM(res, position, offset, type)                                                           \
  {                                                                                                       \
    res = find_atom_type(position + offset, BE_32(position) - offset, type);                              \
    if (res == 0) {                                                                                       \
      VideoStitch::Logger::warning("libavoutput") << "ATOM of type " << type << "not found" << std::endl; \
      return false;                                                                                       \
    }                                                                                                     \
  }

void insert_video_spatial_metadata(unsigned char **cur_pos_src, unsigned char **cur_pos_dst,
                                   const uint64_t trak_atom_position) {
  // we don't want to copy the NUL-termination to the uuid atom
  uint32_t xml_size = (uint32_t)sizeof(spherical_xml) - 1;

  // 'uuid' name, atom size info
  uint32_t uuid_header_size = 4 + 4;
  uint32_t uuid_size = uuid_header_size + (uint32_t)sizeof(uuid_value) + xml_size;

  uint64_t trak_atom_size = BE_32(trak_atom_position);

  // copy everything up to the current end of trak
  uint64_t copy_size = trak_atom_position + trak_atom_size - (uint64_t)*cur_pos_src;
  memcpy(*cur_pos_dst, *cur_pos_src, copy_size);
  *cur_pos_src += copy_size;
  *cur_pos_dst += copy_size;

  // paste uuid atom to the end of the content of the trak atom
  unsigned char *uuid_atom = *cur_pos_dst;

  // set uuid atom size
  set_atom_size(uuid_atom, uuid_size);
  uuid_atom[4 + 0] = 'u';
  uuid_atom[4 + 1] = 'u';
  uuid_atom[4 + 2] = 'i';
  uuid_atom[4 + 3] = 'd';

  unsigned char *uuid_atom_content = uuid_atom + uuid_header_size;

  memcpy(uuid_atom_content, uuid_value, sizeof(uuid_value));
  memcpy(&uuid_atom_content[sizeof(uuid_value)], spherical_xml, sizeof(spherical_xml));

  // grow trak atom header by the content we added
  set_atom_size(*cur_pos_dst - ((uint64_t)*cur_pos_src - trak_atom_position), trak_atom_size + uuid_size);
  *cur_pos_dst += uuid_size;
}

bool insert_audio_spatial_metadata(unsigned char **cur_pos_src, unsigned char **cur_pos_dst,
                                   const uint64_t trak_atom_position, const uint32_t nb_channels) {
  // https://github.com/google/spatial-media/blob/master/docs/spatial-audio-rfc.md#metadata-format
  uint32_t sa3d_size = 20 + 4 * nb_channels;  // 20 + 4x 4channels
  uint64_t audio_atom_position[6];

  audio_atom_position[0] = trak_atom_position;

  // atom hierarchy : trak -> mdia -> minf -> stbl -> stsd -> mp4a -> SA3D
  CHECK_ATOM(audio_atom_position[1], audio_atom_position[0], 8, MDIA_ATOM);
  CHECK_ATOM(audio_atom_position[2], audio_atom_position[1], 8, MINF_ATOM);
  CHECK_ATOM(audio_atom_position[3], audio_atom_position[2], 8, STBL_ATOM);
  CHECK_ATOM(audio_atom_position[4], audio_atom_position[3], 8, STSD_ATOM);
  CHECK_ATOM(audio_atom_position[5], audio_atom_position[4], 16, MP4A_ATOM);

  unsigned char *sa3d_atom = (unsigned char *)malloc(sa3d_size);
  if (!sa3d_atom) {
    VideoStitch::Logger::warning("libavoutput") << "Could not allocate new sa3d atom" << std::endl;
    return false;
  }

  set_atom_size(sa3d_atom, sa3d_size);

  uint32_t j = 4;
  sa3d_atom[j++] = 'S';
  sa3d_atom[j++] = 'A';
  sa3d_atom[j++] = '3';
  sa3d_atom[j++] = 'D';
  sa3d_atom[j++] = 0;  // version
  sa3d_atom[j++] = 0;  // ambisonic_type = Periphonic
  uint32_t ambisonic_order = uint32_t(std::round(std::sqrt(nb_channels)) - 1);
  // ambisonic_order
  sa3d_atom[j++] = ambisonic_order >> 24 & 0xff;
  sa3d_atom[j++] = ambisonic_order >> 16 & 0xff;
  sa3d_atom[j++] = ambisonic_order >> 8 & 0xff;
  sa3d_atom[j++] = ambisonic_order & 0xff;
  sa3d_atom[j++] = 0;  // ambisonic_channel_ordering = ACN
  sa3d_atom[j++] = 0;  // ambisonic_normalization = SN3D
  // num_channels
  sa3d_atom[j++] = nb_channels >> 24 & 0xff;
  sa3d_atom[j++] = nb_channels >> 16 & 0xff;
  sa3d_atom[j++] = nb_channels >> 8 & 0xff;
  sa3d_atom[j++] = nb_channels & 0xff;
  for (uint32_t i = 0; i < nb_channels; i++) {
    // channel_map
    sa3d_atom[j++] = i >> 24 & 0xff;
    sa3d_atom[j++] = i >> 16 & 0xff;
    sa3d_atom[j++] = i >> 8 & 0xff;
    sa3d_atom[j++] = i & 0xff;
  }
  if (j != sa3d_size) {
    VideoStitch::Logger::warning("libavoutput") << "SA3D box size exceeded" << std::endl;
    free(sa3d_atom);
    return false;
  }

  uint64_t copy_size = audio_atom_position[5] + BE_32(audio_atom_position[5]) - (uint64_t)*cur_pos_src;
  // copy everything up to the current end of mp4a & update position after copy
  memcpy(*cur_pos_dst, *cur_pos_src, copy_size);
  *cur_pos_src += copy_size;
  *cur_pos_dst += copy_size;
  // update atom size in their new location
  for (int i = 0; i < 6; i++) {
    set_atom_size(*cur_pos_dst - ((uint64_t)*cur_pos_src - audio_atom_position[i]),
                  BE_32(*cur_pos_dst + audio_atom_position[i] - *cur_pos_src) + sa3d_size);
  }
  // copy SA3D atom & update position after copy
  memcpy(*cur_pos_dst, sa3d_atom, sa3d_size);
  *cur_pos_dst += sa3d_size;

  free(sa3d_atom);
  return true;
}

void splice_spatial_metadata(unsigned char **moov_atom_orig, uint64_t *moov_atom_size, const uint64_t target_size,
                             const uint32_t nb_channels) {
  /* Check there is enough space to insert the spatial metadata */
  uint32_t uuid_size = (uint32_t)sizeof(uuid_value) + (uint32_t)sizeof(spherical_xml) + 7;
  // 'sa3d' box (ambisonic only with 4 channels)
  uint32_t sa3d_size = (nb_channels == 4) ? 20 + 4 * nb_channels : 0;  // 20 + 4x 4channels

  uint64_t new_moov_size = *moov_atom_size + uuid_size + sa3d_size;
  unsigned char *new_moov_atom;
  if (target_size) {
    if ((new_moov_size) > (target_size - 8)) {
      VideoStitch::Logger::get(VideoStitch::Logger::Warning)
          << "[libavoutput] Not enough free space for atom size (" << target_size
          << " was reserved). Aborting metadata splicing." << std::endl;
      VideoStitch::Logger::get(VideoStitch::Logger::Warning)
          << "[libavoutput] Please, increase the minimum atom size using \"min_moov_size\" : " << new_moov_size + 8
          << std::endl;
      return;
    }
  }

  new_moov_atom = (unsigned char *)malloc((size_t)(target_size ? target_size : new_moov_size));
  if (!new_moov_atom) {
    VideoStitch::Logger::get(VideoStitch::Logger::Warning)
        << "[libavoutput] Could not allocate new moov atom" << std::endl;
    return;
  }

  const unsigned char *moov_atom = *moov_atom_orig;

  uint64_t video_trak_atom_position = 0;

  // current position of pointer for memcpy
  unsigned char *cur_pos_src = *moov_atom_orig;
  unsigned char *cur_pos_dst = new_moov_atom;

  uint64_t i = 8;
  while (i < *moov_atom_size - 4) {
    uint64_t trak_atom_position = find_atom_type((uint64_t)(moov_atom + i), *moov_atom_size - i, TRAK_ATOM);
    if (trak_atom_position == 0) {
      break;
    }
    uint64_t next_atom_position = find_atom_type(trak_atom_position + 8, BE_32(trak_atom_position) - 8, MDIA_ATOM);
    if (next_atom_position != 0) {
      next_atom_position = find_atom_type(next_atom_position + 8, BE_32(next_atom_position) - 8, HDLR_ATOM);
      // traverse tree, check mdia, hdlr = vide
      if (MDIA_VIDE == BE_32(next_atom_position + 16)) {
        if (video_trak_atom_position == 0) {
          video_trak_atom_position = trak_atom_position;
          insert_video_spatial_metadata(&cur_pos_src, &cur_pos_dst, trak_atom_position);
          if (sa3d_size == 0) {
            break;
          }
        } else {
          VideoStitch::Logger::warning("libavoutput")
              << "Multiple video tracks found. First track only will contain spatial metadata." << std::endl;
        }
        // traverse tree, check mdia, hdlr = soun
      } else if ((MDIA_SOUN == BE_32(next_atom_position + 16)) && (sa3d_size != 0)) {
        if (!insert_audio_spatial_metadata(&cur_pos_src, &cur_pos_dst, trak_atom_position, nb_channels)) {
          VideoStitch::Logger::warning("libavoutput") << "Aborting audio metadata splicing." << std::endl;
          break;
        }
        sa3d_size = 0;  // indicates atom has been updated
      }
    }
    i = trak_atom_position - (uint64_t)moov_atom + BE_32(trak_atom_position);
  }

  if ((video_trak_atom_position == 0) ||
      ((video_trak_atom_position + BE_32(video_trak_atom_position)) > ((uint64_t)moov_atom + *moov_atom_size))) {
    VideoStitch::Logger::get(VideoStitch::Logger::Warning)
        << "[libavoutput] Bad trak atom size. Aborting metadata splicing." << std::endl;
    free(new_moov_atom);
    return;
  }

  // copy moov trailing part
  memcpy(cur_pos_dst, cur_pos_src, (uint64_t)moov_atom + *moov_atom_size - (uint64_t)cur_pos_src);

  // grow moov atom header by the content we added
  set_atom_size(new_moov_atom, new_moov_size);

  if (target_size) {
    unsigned char *free_atom = &new_moov_atom[new_moov_size];
    set_atom_size(free_atom, target_size - new_moov_size);
    free_atom[4 + 0] = 'f';
    free_atom[4 + 1] = 'r';
    free_atom[4 + 2] = 'e';
    free_atom[4 + 3] = 'e';
    new_moov_size = (uint32_t)target_size;
  }

  free(*moov_atom_orig);
  *moov_atom_orig = new_moov_atom;
  *moov_atom_size = new_moov_size;
}

bool qt_faststart(const char *srcFile, const char *dstFile, const uint32_t nb_channels) {
  FILE *infile = NULL;
  FILE *outfile = NULL;
  unsigned char atom_bytes[ATOM_PREAMBLE_SIZE];
  uint32_t atom_type = 0;
  uint64_t atom_size = 0;
  uint64_t atom_offset = 0;
  uint64_t last_offset;
  unsigned char *moov_atom = NULL;
  unsigned char *ftyp_atom = NULL;
  uint64_t moov_atom_size = 0;
  uint64_t target_moov_size = 0;
  uint64_t ftyp_atom_size = 0;
  uint64_t i, j;
  uint32_t offset_count;
  uint64_t current_offset;
  uint64_t start_offset = 0;
  unsigned char *copy_buffer = NULL;
  int bytes_to_copy;
  bool in_place = false;

  if (!strcmp(srcFile, dstFile)) {
    in_place = true;
  }

  infile = VideoStitch::Io::openFile(srcFile, "rb");
  if (!infile) {
    perror(srcFile);
    goto error_out;
  }

  /* traverse through the atoms in the file to make sure that 'moov' is at the end */
  while (!feof(infile)) {
    if (fread(atom_bytes, ATOM_PREAMBLE_SIZE, 1, infile) != 1) {
      break;
    }
    atom_size = (uint32_t)BE_32(&atom_bytes[0]);
    atom_type = BE_32(&atom_bytes[4]);

    /* keep ftyp atom */
    if (atom_type == FTYP_ATOM) {
      ftyp_atom_size = atom_size;
      free(ftyp_atom);
      ftyp_atom = (unsigned char *)malloc((size_t)ftyp_atom_size);
      if (!ftyp_atom) {
        VideoStitch::Logger::get(VideoStitch::Logger::Warning)
            << "[libavoutput] Could not allocate " << atom_size << " bytes for ftyp atom" << std::endl;
        goto error_out;
      }
      fseeko(infile, -ATOM_PREAMBLE_SIZE, SEEK_CUR);
      if (fread(ftyp_atom, (size_t)atom_size, 1, infile) != 1) {
        perror(srcFile);
        goto error_out;
      }
      start_offset = ftello(infile);
    } else {
      if (in_place && (atom_type == MOOV_ATOM)) {
        fseeko(infile, -ATOM_PREAMBLE_SIZE, SEEK_CUR);
        break;
      }
      /* 64-bit special case */
      if (atom_size == 1) {
        if (fread(atom_bytes, ATOM_PREAMBLE_SIZE, 1, infile) != 1) {
          break;
        }
        atom_size = BE_64(&atom_bytes[0]);
        fseeko(infile, atom_size - ATOM_PREAMBLE_SIZE * 2, SEEK_CUR);
      } else {
        fseeko(infile, atom_size - ATOM_PREAMBLE_SIZE, SEEK_CUR);
      }
    }
    VideoStitch::Logger::get(VideoStitch::Logger::Debug)
        << "[libavoutput] " << char((atom_type >> 24) & 255) << char((atom_type >> 16) & 255)
        << char((atom_type >> 8) & 255) << char((atom_type >> 0) & 255) << " " << atom_offset << " " << atom_size
        << std::endl;
    if ((atom_type != FREE_ATOM) && (atom_type != JUNK_ATOM) && (atom_type != MDAT_ATOM) && (atom_type != MOOV_ATOM) &&
        (atom_type != PNOT_ATOM) && (atom_type != SKIP_ATOM) && (atom_type != WIDE_ATOM) && (atom_type != PICT_ATOM) &&
        (atom_type != UUID_ATOM) && (atom_type != FTYP_ATOM)) {
      VideoStitch::Logger::get(VideoStitch::Logger::Debug)
          << "[libavoutput] Encountered non-QT top-level atom (is this a QuickTime file?)" << std::endl;
      break;
    }
    atom_offset += atom_size;

    /* The atom header is 8 (or 16 bytes), if the atom size (which
     * includes these 8 or 16 bytes) is less than that, we won't be
     * able to continue scanning sensibly after this atom, so break. */
    if (atom_size < 8) break;
  }

  if (atom_type != MOOV_ATOM) {
    VideoStitch::Logger::get(VideoStitch::Logger::Warning)
        << "[libavoutput] Last atom in file was not a moov atom" << std::endl;
    free(ftyp_atom);
    fclose(infile);
    return false;
  }

  if (!in_place) {
    /* moov atom was, in fact, the last atom in the chunk; load the whole moov atom */
    fseeko(infile, -int64_t(atom_size), SEEK_END);
  }
  last_offset = ftello(infile);
  moov_atom_size = atom_size;
  moov_atom = (unsigned char *)malloc((size_t)moov_atom_size);
  if (!moov_atom) {
    VideoStitch::Logger::get(VideoStitch::Logger::Warning)
        << "[libavoutput] Could not allocate " << atom_size << " bytes for moov atom" << std::endl;
    goto error_out;
  }
  if (fread(moov_atom, (size_t)atom_size, 1, infile) != 1) {
    perror(srcFile);
    goto error_out;
  }

  if (in_place) {
    /* free atom should follow the moov atom to add the spatial metadata */
    if (fread(atom_bytes, ATOM_PREAMBLE_SIZE, 1, infile) != 1) {
      perror(srcFile);
      goto error_out;
    }
    atom_size = (uint32_t)BE_32(&atom_bytes[0]);
    atom_type = BE_32(&atom_bytes[4]);
    if (atom_type == FREE_ATOM) {
      target_moov_size = moov_atom_size + atom_size;
    } else {
      VideoStitch::Logger::get(VideoStitch::Logger::Warning)
          << "[libavoutput] There is no free atom (but " << char((atom_type >> 24) & 255)
          << char((atom_type >> 16) & 255) << char((atom_type >> 8) & 255) << char((atom_type >> 0) & 255)
          << ") right after the moov atom" << std::endl;
      VideoStitch::Logger::get(VideoStitch::Logger::Warning)
          << "[libavoutput] Cannot update moov atoms with extra spatial metadata" << std::endl;
      goto error_out;
    }
  }

  /* this utility does not support compressed atoms yet, so disqualify  files with compressed QT atoms */
  if (BE_32(&moov_atom[12]) == CMOV_ATOM) {
    VideoStitch::Logger::get(VideoStitch::Logger::Warning)
        << "[libavoutput] This utility does not support compressed moov atoms yet" << std::endl;
    goto error_out;
  }

  /* close; will be re-opened later */
  fclose(infile);
  infile = NULL;

  /* try to find the first video trak atom and splice in spatial metadata as a new uuid atom */
  /* try to find the first audio trak atom and splice in spatial metadata as a new S3AD atom for ambisonic */
  splice_spatial_metadata(&moov_atom, &moov_atom_size, target_moov_size, nb_channels);

  /* only need to update the moov+free atom */
  if (in_place) {
    if (moov_atom_size != target_moov_size) {
      goto error_out;
    }
    /* re-open the file for update */
    infile = VideoStitch::Io::openFile(srcFile, "r+b");
    if (!infile) {
      perror(srcFile);
      goto error_out;
    }

    /* dump the new moov atom */
    fseeko(infile, last_offset, SEEK_SET);
    VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "[libavoutput] Writing moov atom..." << std::endl;
    if (fwrite(moov_atom, (size_t)moov_atom_size, 1, infile) != 1) {
      perror(dstFile);
      goto error_out;
    }
    fclose(infile);
    free(moov_atom);
    free(ftyp_atom);
    return true;
  }

  /* crawl through the moov chunk in search of stco or co64 atoms */
  for (i = 4; i < moov_atom_size - 4; i++) {
    atom_type = BE_32(&moov_atom[i]);
    if (atom_type == STCO_ATOM) {
      VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "[libavoutput] Patching stco atom..." << std::endl;
      atom_size = BE_32(&moov_atom[i - 4]);
      if (i + atom_size - 4 > moov_atom_size) {
        VideoStitch::Logger::get(VideoStitch::Logger::Warning) << "[libavoutput] Bad atom size" << std::endl;
        goto error_out;
      }
      offset_count = BE_32(&moov_atom[i + 8]);
      for (j = 0; j < offset_count; j++) {
        current_offset = BE_32(&moov_atom[i + 12 + j * 4]);
        current_offset += moov_atom_size;
        moov_atom[i + 12 + j * 4 + 0] = (current_offset >> 24) & 0xFF;
        moov_atom[i + 12 + j * 4 + 1] = (current_offset >> 16) & 0xFF;
        moov_atom[i + 12 + j * 4 + 2] = (current_offset >> 8) & 0xFF;
        moov_atom[i + 12 + j * 4 + 3] = (current_offset >> 0) & 0xFF;
      }
      i += atom_size - 4;
    } else if (atom_type == CO64_ATOM) {
      VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "[libavoutput] Patching co64 atom..." << std::endl;
      atom_size = BE_32(&moov_atom[i - 4]);
      if (i + atom_size - 4 > moov_atom_size) {
        VideoStitch::Logger::get(VideoStitch::Logger::Warning) << "[libavoutput] Bad atom size" << std::endl;
        goto error_out;
      }
      offset_count = BE_32(&moov_atom[i + 8]);
      for (j = 0; j < offset_count; j++) {
        current_offset = BE_64(&moov_atom[i + 12 + j * 8]);
        current_offset += moov_atom_size;
        moov_atom[i + 12 + j * 8 + 0] = (unsigned char)((current_offset >> 56) & 0xFF);
        moov_atom[i + 12 + j * 8 + 1] = (unsigned char)((current_offset >> 48) & 0xFF);
        moov_atom[i + 12 + j * 8 + 2] = (unsigned char)((current_offset >> 40) & 0xFF);
        moov_atom[i + 12 + j * 8 + 3] = (unsigned char)((current_offset >> 32) & 0xFF);
        moov_atom[i + 12 + j * 8 + 4] = (unsigned char)((current_offset >> 24) & 0xFF);
        moov_atom[i + 12 + j * 8 + 5] = (unsigned char)((current_offset >> 16) & 0xFF);
        moov_atom[i + 12 + j * 8 + 6] = (unsigned char)((current_offset >> 8) & 0xFF);
        moov_atom[i + 12 + j * 8 + 7] = (unsigned char)((current_offset >> 0) & 0xFF);
      }
      i += atom_size - 4;
    }
  }

  /* re-open the input file and open the output file */
  infile = VideoStitch::Io::openFile(srcFile, "rb");
  if (!infile) {
    perror(srcFile);
    goto error_out;
  }

  if (start_offset > 0) { /* seek after ftyp atom */
    fseeko(infile, start_offset, SEEK_SET);
    last_offset -= start_offset;
  }

  outfile = VideoStitch::Io::openFile(dstFile, "wb");
  if (!outfile) {
    perror(dstFile);
    goto error_out;
  }

  /* dump the same ftyp atom */
  if (ftyp_atom_size > 0) {
    VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "[libavoutput] Writing ftyp atom..." << std::endl;
    if (fwrite(ftyp_atom, (size_t)ftyp_atom_size, 1, outfile) != 1) {
      perror(dstFile);
      goto error_out;
    }
  }

  /* dump the new moov atom */
  VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "[libavoutput] Writing moov atom..." << std::endl;
  if (fwrite(moov_atom, (size_t)moov_atom_size, 1, outfile) != 1) {
    perror(dstFile);
    goto error_out;
  }

  /* copy the remainder of the infile, from offset 0 -> last_offset - 1 */
  VideoStitch::Logger::get(VideoStitch::Logger::Debug) << "[libavoutput] Copying rest of file..." << std::endl;
  copy_buffer = (unsigned char *)malloc(COPY_BUFFER_SIZE);
  if (!copy_buffer) {
    VideoStitch::Logger::get(VideoStitch::Logger::Error)
        << "[libavoutput] Could not allocate " << COPY_BUFFER_SIZE << " bytes for copy_buffer" << std::endl;
    goto error_out;
  }
  while (last_offset) {
    if (last_offset > COPY_BUFFER_SIZE)
      bytes_to_copy = COPY_BUFFER_SIZE;
    else
      bytes_to_copy = (int)last_offset;

    if (fread(copy_buffer, bytes_to_copy, 1, infile) != 1) {
      perror(srcFile);
      goto error_out;
    }
    if (fwrite(copy_buffer, bytes_to_copy, 1, outfile) != 1) {
      perror(dstFile);
      goto error_out;
    }
    last_offset -= bytes_to_copy;
  }

  fclose(infile);
  fclose(outfile);
  free(moov_atom);
  free(ftyp_atom);
  free(copy_buffer);

  return true;

error_out:
  if (infile) fclose(infile);
  if (outfile) fclose(outfile);
  free(moov_atom);
  free(ftyp_atom);
  free(copy_buffer);
  return false;
}
