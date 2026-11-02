/* Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef XDRFILE_XTC_H
#define XDRFILE_XTC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "xdrfile.h"

/* All functions return exdrOK if succesfull.
 * (error codes defined in xdrfile.h).
 */

/* This function returns the number of atoms in the xtc file in *natoms */
int read_xtc_natoms(const char* fn, int* natoms);

/* This function returns the number of frames and the number of atoms
 * in the xtc file in *natoms and *nframes.
 * It also returns the starting position of each frame as bytes from the beginning of the file
 * in **offsets, which has to be freed manually.
 */
int read_xtc_header(const char* fn, int* natoms, unsigned long* nframes, int64_t** offsets);

/* Read one frame of an open xtc file */
int read_xtc(XDRFILE* xd, int natoms, int* step, float* time, matrix box, rvec* x, float* prec);

/* Write a frame to xtc file */
int write_xtc(XDRFILE* xd, int natoms, int step, float time, matrix box, rvec* x, float prec);

/* XTC small header size (natoms<=9).
 *  > int(4) magic
 *  > int(4) natoms
 *  > int(4) step
 *  > float(4) time
 *  > 9xfloat(4) box
 *  > int(4) natoms (again)
 */
#define XTC_SMALL_HEADER_SIZE 56

/* Size of uncompressed coordinates for one atom.
 * 3xfloat(4) x
 */
#define XTC_SMALL_COORDS_SIZE 12

/* XTC header size (natoms>=10).
 * Compressed trajectories contain some additional values:
 *  > float(4) preccission
 *  > 3xint(4) minint
 *  > 3xint(4) maxint
 *  > int(4) smallidx
 * See `xdrfile_compress_coord_double()`.
 */
#define XTC_HEADER_SIZE (XTC_SMALL_HEADER_SIZE + 32)

#ifdef __cplusplus
}
#endif

#endif
