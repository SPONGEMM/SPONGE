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

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "xdrfile.h"
#include "xdrfile_xtc.h"

#define XTC_MAGIC 1995

static int xtc_header(XDRFILE* xd, int* natoms, int* step, float* time, bool read) {
    int result, magic, n = 1;

    /* Note: read is same as write. He he he */
    magic = XTC_MAGIC;
    if ((result = xdrfile_write_int(&magic, n, xd)) != n) {
        if (read) {
            return exdrENDOFFILE;
        } else {
            return exdrINT;
        }
    }
    if (magic != XTC_MAGIC) {
        return exdrMAGIC;
    }
    if ((result = xdrfile_write_int(natoms, n, xd)) != n) {
        return exdrINT;
    }
    if ((result = xdrfile_write_int(step, n, xd)) != n) {
        return exdrINT;
    }
    if ((result = xdrfile_write_float(time, n, xd)) != n) {
        return exdrFLOAT;
    }

    return exdrOK;
}

static int xtc_coord(XDRFILE* xd, int* natoms, matrix box, rvec* x, float* prec, bool read) {
    int result;

    /* box */
    result = xdrfile_read_float(box[0], DIM * DIM, xd);
    if (DIM * DIM != result) {
        return exdrFLOAT;
    } else {
        if (read) {
            result = xdrfile_decompress_coord_float(x[0], natoms, prec, xd);
            if (result != *natoms) {
                return exdr3DX;
            }
        } else {
            result = xdrfile_compress_coord_float(x[0], *natoms, *prec, xd);
            if (result != *natoms) {
                return exdr3DX;
            }
        }
    }
    return exdrOK;
}

int read_xtc_natoms(const char* fn, int* natoms) {
    XDRFILE* xd;
    int step, result;
    float time;

    xd = xdrfile_open(fn, "r");
    if (NULL == xd) {
        return exdrFILENOTFOUND;
    }
    result = xtc_header(xd, natoms, &step, &time, true);
    xdrfile_close(xd);

    return result;
}

int read_xtc_header(const char* fn, int* natoms, unsigned long* nframes, int64_t** offsets) {
    XDRFILE* xd;
    int i, result, est_nframes, framebytes;
    int64_t filesize;
    *nframes = 0;

    read_xtc_natoms(fn, natoms);

    xd = xdrfile_open(fn, "r");
    if (NULL == xd) {
        return exdrFILENOTFOUND;
    }

    /* Go to file end */
    if (xdr_seek(xd, 0L, SEEK_END) != exdrOK) {
        xdrfile_close(xd);
        return exdrNR;
    }
    /* Cursor position is equivalent to file size */
    filesize = xdr_tell(xd);

    /* Dont bother with compression for nine atoms or less */
    if (*natoms <= 9) {
        xdrfile_close(xd);
        framebytes = XTC_SMALL_HEADER_SIZE + XTC_SMALL_COORDS_SIZE * (*natoms);
        *nframes =
            filesize / framebytes; /* Should we complain if framesize doesn't divide filesize? */
        *offsets = malloc(sizeof(int64_t) * (*nframes));
        if (*offsets == NULL) {
            /* failed to allocate memory for `offsets` */
            return exdrNOMEM;
        }
        for (i = 0; i < *nframes; i++) {
            (*offsets)[i] = i * framebytes;
        }
        return exdrOK;
    } else {
        /* Go back to the beginning of the file */
        if (xdr_seek(xd, (int64_t)XTC_HEADER_SIZE, SEEK_SET) != exdrOK) {
            xdrfile_close(xd);
            return exdrNR;
        }

        if (xdrfile_read_int(&framebytes, 1, xd) == 0) {
            xdrfile_close(xd);
            return exdrENDOFFILE;
        }
        framebytes = (framebytes + 3) & ~0x03; /* Rounding to the next 32-bit boundary */
        est_nframes = (int)(filesize / ((int64_t)(framebytes + XTC_HEADER_SIZE)) +
                            1); /* must be at least 1 for successful growth */
        /* First `framebytes` might be larger than average, so we would underestimate `est_nframes`
         */
        est_nframes += est_nframes / 5;

        /* Allocate memory for the frame index array */
        *offsets = malloc(sizeof(int64_t) * est_nframes);
        if (*offsets == NULL) {
            /* failed to allocate memory for `offsets` */
            xdrfile_close(xd);
            return exdrNOMEM;
        }
        (*offsets)[0] = 0;

        while (1) {
            /* Skip `framebytes` and next header */
            result = xdr_seek(xd, (int64_t)(framebytes + XTC_HEADER_SIZE), SEEK_CUR);
            if (result != exdrOK) {
                break;
            }

            (*nframes)++;

            if (*nframes == est_nframes) {
                /* grow the array exponentially */
                est_nframes += est_nframes * 2;
                *offsets = realloc(*offsets, sizeof(int64_t) * est_nframes);
                if (*offsets == NULL) {
                    /* failed to allocate memory for `offsets` */
                    result = exdrNOMEM;
                    break;
                }
            }

            /* Store position in `offsets`, adjust for header */
            (*offsets)[*nframes] = xdr_tell(xd) - (int64_t)(XTC_HEADER_SIZE);

            /* Read how much to skip next time */
            if (xdrfile_read_int(&framebytes, 1, xd) == 0) {
                result = exdrENDOFFILE;
                break;
            }
            framebytes = (framebytes + 3) & ~0x03; /* Rounding to the next 32-bit boundary */
        }

        xdrfile_close(xd);

        if (result != exdrENDOFFILE) {
            /* report error to caller */
            return result;
        }

        return exdrOK;
    }
}

/* Read subsequent frames */
int read_xtc(XDRFILE* xd, int natoms, int* step, float* time, matrix box, rvec* x, float* prec) {
    int result;

    if ((result = xtc_header(xd, &natoms, step, time, true)) != exdrOK) {
        return result;
    }

    if ((result = xtc_coord(xd, &natoms, box, x, prec, true)) != exdrOK) {
        return result;
    }

    return exdrOK;
}

/* Write a frame to xtc file */
int write_xtc(XDRFILE* xd, int natoms, int step, float time, matrix box, rvec* x, float prec) {
    int result;

    if ((result = xtc_header(xd, &natoms, &step, &time, false)) != exdrOK) {
        return result;
    }

    if ((result = xtc_coord(xd, &natoms, box, x, &prec, false)) != exdrOK) {
        return result;
    }

    return exdrOK;
}
