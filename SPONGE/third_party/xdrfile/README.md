# xdrfile (vendored)

Trimmed vendoring of libxdrfile, the reference implementation of the GROMACS
XTC/TRR trajectory container format. SPONGE only ships the subset required for
**writing** XTC trajectories.

## Provenance

| Field | Value |
|-------|-------|
| Upstream | libxdrfile (Erik Lindahl, David van der Spoel) |
| Version | 1.1.4 |
| Branch | `chemfiles` |
| Commit | `0d12c950bcc37bda591f7c5ce8256a281b74bd6d` |
| Source bundle | `cmake/third_party/xdrfile.bundle` |
| License | BSD 3-Clause, see `COPYING` |

Restore a full upstream checkout from the bundle with:

```sh
git clone cmake/third_party/xdrfile.bundle xdrfile-upstream
git -C xdrfile-upstream log --oneline -1
```

## Contents

```
include/xdrfile.h        XDR stream layer + portable primitive I/O
include/xdrfile_xtc.h    XTC read/write API
src/xdrfile.c
src/xdrfile_xtc.c
```

`xdrfile_trr.c` / `xdrfile_trr.h` are intentionally omitted: TRR output is not
implemented yet. Only the XTC path is compiled.

## Local modifications

Two SPONGE-specific extensions are applied on top of upstream (see
`patches/0001-sponge-runtime-extensions.patch`):

| API | Purpose |
|-----|---------|
| `xdrfile_flush(XDRFILE*)` | Flush pending stdio bytes without closing the file. Upstream has no flush entry point and `struct XDRFILE` is opaque, so a caller cannot reach the inner `FILE*`. |
| `xdrfile_setvbuf(XDRFILE*, size_t frame_bytes, int frames)` | Install a frame-aligned stdio buffer. Keeps one frame from straddling a buffer boundary so an interrupted write drops a whole frame instead of leaving a partial one, and cuts write syscalls by orders of magnitude. |

Supporting change: `struct XDRFILE` gained an owned `stdio_buffer` field that
`xdrfile_close()` releases.

Rationale for both: XTC carries no trailer and no frame index, so a file is
fully readable as long as it ends on a frame boundary. Flushing at that
boundary is therefore equivalent to closing as far as durable content goes
(`xdrfile_close()` itself only calls `fflush` + `fclose`). The buffer extension
supplies the frame atomicity that SPONGE's legacy `dat` trajectory gets from
`CONTROLLER::Set_File_Buffer()`.

### Replaying the patch on a new upstream revision

```sh
git clone cmake/third_party/xdrfile.bundle /tmp/xdrfile-upstream
git -C /tmp/xdrfile-upstream apply /path/to/patches/0001-sponge-runtime-extensions.patch
```

## Known upstream issues

`xdrstdio_ops` initialises `x_getpostn` / `x_setpostn` with 64-bit function
pointers while `struct xdr_ops` still declares them as `unsigned int`, which
GCC reports as `-Wincompatible-pointer-types`. This is harmless here: the
`xdr_getpos` / `xdr_setpos` macros that would use those slots are commented out
upstream (`src/xdrfile.c:116-117`), and `xdr_tell()` / `xdr_seek()` call
`xdrstdio_getpos` / `xdrstdio_setpos` directly, so frame offsets stay 64-bit and
remain correct past the 4 GiB mark.

## Re-vendoring checklist

1. Replace `include/` and `src/` with the upstream files.
2. Re-apply the patch above.
3. Confirm `xdrfile.c` still has the `_MSC_VER` branch in `xdrstdio_getpos` /
   `xdrstdio_setpos` (needed for the Windows `icx-cl` toolchain).
4. Do **not** clang-format this directory; `.clang-format` here disables it.
