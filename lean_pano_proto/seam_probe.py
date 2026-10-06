"""Plateau step and one-pixel valley on a stitched 16-bit TIFF.

The column score is the second difference of the column-mean luma,
normalized by the median absolute spread in a 2048-pixel window.
The plateau step is the mean of the 30 columns left of the valley
minus the mean of the 30 columns to its right.
"""

import struct
import sys
import numpy as np

HALF = 30
WIN = 2048


def _ifd(f, endian):
    n = struct.unpack(endian + "H", f.read(2))[0]
    tags = {}
    for _ in range(n):
        tag, typ, cnt, val = struct.unpack(endian + "HHII", f.read(12))
        tags[tag] = (typ, cnt, val)
    return tags


def _values(f, endian, typ, cnt, val):
    size = {3: 2, 4: 4}[typ]
    fmt = "H" if typ == 3 else "I"
    if cnt * size <= 4:
        raw = struct.pack(endian + "I", val)
        return list(struct.unpack(endian + fmt * cnt, raw[: cnt * size]))
    f.seek(val)
    return list(struct.unpack(endian + fmt * cnt, f.read(cnt * size)))


def column_luma(path):
    with open(path, "rb") as f:
        endian = "<" if f.read(2) == b"II" else ">"
        if struct.unpack(endian + "H", f.read(2))[0] != 42:
            raise SystemExit(f"{path} is not a TIFF")
        f.seek(struct.unpack(endian + "I", f.read(4))[0])
        tags = _ifd(f, endian)
        w = tags[256][2]
        h = tags[257][2]
        spp = tags[277][2]
        rps = tags[278][2]
        offs = _values(f, endian, *tags[273])
        acc = np.zeros(w, dtype=np.float64)
        y = 0
        for off in offs:
            rows = min(rps, h - y)
            f.seek(off)
            buf = np.frombuffer(f.read(rows * w * spp * 2), dtype=np.dtype(endian + "u2")).reshape(rows, w, spp)
            acc += (0.2126 * buf[:, :, 0] + 0.7152 * buf[:, :, 1] + 0.0722 * buf[:, :, 2]).sum(axis=0)
            y += rows
    return acc / h


def probe(path):
    prof = column_luma(path)
    w = prof.shape[0]
    d = np.zeros(w)
    d[1:-1] = prof[:-2] + prof[2:] - 2.0 * prof[1:-1]
    spread = np.ones(w)
    for x in range(0, w, 64):
        lo = max(1, x - WIN // 2)
        hi = min(w - 1, x + WIN // 2)
        spread[x : x + 64] = max(float(np.median(np.abs(d[lo:hi]))), 1e-9)
    z = d / spread
    x = int(np.argmax(z[HALF : w - HALF])) + HALF
    left = float(prof[x - HALF : x].mean())
    right = float(prof[x + 1 : x + 1 + HALF].mean())
    step = (left - right) / max(left, 1e-9) * 100.0
    depth = (float(prof[x]) - 0.5 * (float(prof[x - 1]) + float(prof[x + 1]))) / max(left, 1e-9) * 100.0
    print(
        f"{path} x={x} z={float(z[x]):.1f} step={step:+.2f}% depth={depth:.2f}% left={left:.1f} right={right:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    for path in sys.argv[1:]:
        probe(path)
