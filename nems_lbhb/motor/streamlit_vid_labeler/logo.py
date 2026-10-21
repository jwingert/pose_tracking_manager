"""The app logo: three real DLC poses of a moving ferret, as a rainbow skeleton.

The poses are baked in below as numbers, not shipped as an image -- the SVG is built at
runtime, so the repo carries no binary asset. :func:`poses_from_h5` is how the numbers
were made; re-run it to pick a different stride.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: app rename to Animal Tracking Tracker, with a logo drawn from real tracking data | date: 2026-09-25]

import numpy as np

# DLC's default labelling colour map is rainbow over the bodypart order; the logo keeps
# that order so each joint wears the colour it has in every labelled frame.
PARTS = ("front", "back", "leftear", "rightear", "shoulders", "midriff", "rear")
HEAD = (("front", "leftear"), ("leftear", "back"), ("back", "rightear"),
        ("rightear", "front"))
SPINE = (("back", "shoulders"), ("shoulders", "midriff"), ("midriff", "rear"))
EDGES = HEAD + SPINE

# made by poses_from_h5(**SOURCE): a ferret turning while it runs, 15 frames apart.
# heading 40 deg: at 30 the newest pose ran flat along the top edge and read as a crop
SOURCE = {'h5': '/auto/data/daq/StinkySquid/SQD042/sorted/SQD042a03_a_NFC.left_chimney-r0001.dlc.h5',
 'frames': [44772, 44787, 44802],
 'heading_deg': 40,
 'mirror': True}
POSES = [{'front': (59.7, 17.1),
  'back': (52.6, 23.7),
  'leftear': (66.4, 28.7),
  'rightear': (50.7, 14.7),
  'shoulders': (42.1, 33.4),
  'midriff': (20.5, 64.4),
  'rear': (6.0, 100.0)},
 {'front': (74.9, 4.3),
  'back': (64.9, 7.1),
  'leftear': (81.4, 16.1),
  'rightear': (66.9, 0.0),
  'shoulders': (52.2, 13.3),
  'midriff': (27.2, 38.0),
  'rear': (7.0, 61.3)},
 {'front': (94.0, 10.3),
  'back': (83.3, 6.9),
  'leftear': (91.0, 20.2),
  'rightear': (90.1, 1.3),
  'shoulders': (70.5, 3.3),
  'midriff': (32.0, 8.0),
  'rear': (8.4, 30.5)}]


def rainbow(n):
    """``n`` hex colours along matplotlib's 'rainbow' (DLC's default), no matplotlib."""
    out = []
    for x in np.linspace(0, 1, n):
        # matplotlib's rainbow: r = |2x - 0.5|, g = sin(pi x), b = cos(pi x / 2)
        r, g, b = abs(2 * x - 0.5), np.sin(np.pi * x), np.cos(np.pi * x / 2)
        out.append("#%02x%02x%02x" % tuple(int(round(255 * min(max(v, 0), 1)))
                                           for v in (r, g, b)))
    return out


COLOURS = dict(zip(PARTS, rainbow(len(PARTS))))


def poses_from_h5(path, frames, heading_deg=45.0, mirror=False):
    """Normalised ``[{part: (x, y)}]`` for ``frames`` of a DLC ``.h5``.

    Rotated so the middle pose's rear->front heading points at ``heading_deg`` (45 =
    up-right), optionally mirrored, scaled to fill a 100-unit box and centred in it.
    Rotation, mirroring and scale are the only liberties taken with the data.
    """
    import pandas as pd
    df = pd.read_hdf(path)
    scorer = df.columns.get_level_values(0)[0]
    xy = np.stack([df[scorer][p][["x", "y"]].to_numpy()[list(frames)] for p in PARTS], 1)
    xy = xy.astype(float)
    xy[..., 1] *= -1                                   # image rows grow downward
    if mirror:
        xy[..., 0] *= -1
    mid = xy[len(xy) // 2]
    head = mid[PARTS.index("front")] - mid[PARTS.index("rear")]
    angle = np.radians(heading_deg) - np.arctan2(head[1], head[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = xy @ rot.T
    lo, hi = xy.reshape(-1, 2).min(0), xy.reshape(-1, 2).max(0)
    scale = 100.0 / (hi - lo).max()
    xy = (xy - lo) * scale + (100 - (hi - lo) * scale) / 2    # centre both axes
    xy[..., 1] = 100 - xy[..., 1]                     # back to SVG's downward y
    return [{p: (round(float(x), 1), round(float(y), 1)) for p, (x, y) in zip(PARTS, pose)}
            for pose in xy]


def logo_svg(size=52, poses=None, stroke=7.0, joint=4.5):
    """The logo as an inline ``<svg>``: oldest pose faintest, newest solid.

    Each bone is a gradient between its two joints' colours, so the skeleton reads as
    one rainbow rather than seven segments.
    """
    poses = poses or POSES
    pad = stroke + joint
    parts, defs = [], []
    for k, pose in enumerate(poses):
        opacity = (0.38, 0.65, 1.0)[k] if len(poses) == 3 else 0.35 + 0.65 * k / max(len(poses) - 1, 1)
        group = [f'<g opacity="{opacity}">']
        for a, b in EDGES:
            gid = f"g{k}{a}{b}"
            (x1, y1), (x2, y2) = pose[a], pose[b]
            defs.append(f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
                        f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
                        f'<stop offset="0" stop-color="{COLOURS[a]}"/>'
                        f'<stop offset="1" stop-color="{COLOURS[b]}"/></linearGradient>')
            group.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                         f'stroke="url(#{gid})" stroke-width="{stroke}" '
                         f'stroke-linecap="round"/>')
        for p in PARTS:
            x, y = pose[p]
            group.append(f'<circle cx="{x}" cy="{y}" r="{joint}" fill="{COLOURS[p]}"/>')
        group.append("</g>")
        parts.append("".join(group))
    view = f"{-pad} {-pad} {100 + 2 * pad} {100 + 2 * pad}"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="{view}" style="vertical-align:middle">'
            f'<defs>{"".join(defs)}</defs>{"".join(parts)}</svg>')
# [AGENT EDIT END]
