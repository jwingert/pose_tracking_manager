"""Reusable schematic top-down ferret glyphs (head and whole body) for figures.

Glyphs are built in a unit frame (head centred at the origin, head radius = 1,
nose pointing +y; the body extends toward -y) and placed with an Affine2D
transform (scale -> rotate -> translate), so they can be dropped at any point,
size, and orientation, in data or axes coordinates.

    from nems_lbhb.ferret_glyph import draw_ferret_head, draw_ferret_body
    draw_ferret_head(ax, x=0.3, y=-0.2, size=0.15, angle=30)   # data coords
    draw_ferret_body(ax, 0.0, 0.0, size=0.1, angle=90)         # whole body
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: reusable rotatable ferret head + whole-body glyphs for neural-HRTF figure / arena | date: 2026-06-26]
import math

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms


def _affine(ax, x, y, size, angle, transform):
    base = {'data': ax.transData, 'axes': ax.transAxes}.get(transform, transform)
    return (mtransforms.Affine2D().scale(size).rotate_deg(angle).translate(x, y)
            + base)


# Head-radius display size (in points) at which the default line widths are
# calibrated. Line widths scale with the glyph's actual on-screen size so the
# outline/rounding stay a constant fraction of the glyph at any scale.
FERRET_REF_PPU = 22.0


# [AGENT EDIT START | agent: claude | user: wingertj | reason: clip glyphs to the arena box; several glyph patches already carry their own clip, and matplotlib allows only one per artist, so the two regions must be intersected | date: 2026-07-27]
def _clip_polygon(subject, clip):
    """Sutherland-Hodgman: clip polygon `subject` by CONVEX polygon `clip`.

    Both are (N, 2) vertex arrays in the same frame. `subject` may be concave.
    Returns the intersection, which may be empty.
    """
    poly = [tuple(p) for p in np.asarray(subject, float)]
    cp = np.asarray(clip, float)
    if len(cp) > 2 and np.allclose(cp[0], cp[-1]):
        cp = cp[:-1]
    # Normalise to counter-clockwise so "inside" is consistently to the left.
    if np.sum(cp[:, 0] * np.roll(cp[:, 1], -1)
              - np.roll(cp[:, 0], -1) * cp[:, 1]) < 0:
        cp = cp[::-1]

    def _isect(p1, p2, a, b):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, a, b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0.0:
            return p2
        s = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + s * (x2 - x1), y1 + s * (y2 - y1))

    for i in range(len(cp)):
        a, b = tuple(cp[i]), tuple(cp[(i + 1) % len(cp)])
        ex, ey = b[0] - a[0], b[1] - a[1]

        def _in(p):
            return ex * (p[1] - a[1]) - ey * (p[0] - a[0]) >= 0.0

        src, poly = poly, []
        if not src:
            break
        prev = src[-1]
        for cur in src:
            if _in(cur):
                if not _in(prev):
                    poly.append(_isect(prev, cur, a, b))
                poly.append(cur)
            elif _in(prev):
                poly.append(_isect(prev, cur, a, b))
            prev = cur
    return np.asarray(poly, float).reshape(-1, 2)
# [AGENT EDIT END]


def _lw_scale(ax, t):
    """Factor that keeps line widths proportional to the glyph's on-screen size."""
    p0 = t.transform((0.0, 0.0))
    p1 = t.transform((1.0, 0.0))
    px = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
    return (px * 72.0 / ax.figure.dpi) / FERRET_REF_PPU


def _head_factories(t):
    """Head silhouette shape factories in the unit frame (transform t)."""
    def neck(**kw):
        return mpatches.FancyBboxPatch(
            (-0.78, -1.5), 1.56, 1.7,
            boxstyle='round,pad=0,rounding_size=0.45', transform=t, **kw)

    def ear(sgn, **kw):
        return mpatches.Ellipse((sgn * 0.98, 0.0), 0.5, 0.9, angle=sgn * 20,
                                transform=t, **kw)

    def head_circle(**kw):
        return mpatches.Circle((0.0, 0.0), 1.0, transform=t, **kw)

    def snout(lw, **kw):
        # [AGENT EDIT | claude | user: wingertj | BLUNT tip: the nose defines the front of the face, so the muzzle ends in a flat edge sized to the nose rather than tapering to a point | 2026-07-28]
        # Gently bowed sides rising to a FLAT front edge (x -0.26..0.26 at
        # y=1.34). A pointed apex forced the nose to shrink or sit back to stay
        # inside the muzzle; blunting the tip lets the nose be the front of the
        # face at full width.
        # STRAIGHT sides: a triangle with a tiny flat tip, not bowed curves. The
        # thick stroke's round joins soften the corners on their own.
        verts = [(-0.74, 0.32),
                 (-0.04, 1.50),                 # left side -> tip corner
                 (0.04, 1.50),                  # flat front edge
                 (0.74, 0.32),                  # right side -> base
                 (-0.74, 0.32)]                 # CLOSEPOLY (straight base)
        codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
                 mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
        return mpatches.PathPatch(mpath.Path(verts, codes), lw=lw,
                                  joinstyle='round', transform=t, **kw)

    return neck, ear, head_circle, snout


def _body_factories(t):
    """Whole-body shape factories (long body, tail, shoulder/hip nubs).

    Proportions (head radius = 1): body top ~0.47 (under the head) to -7.72;
    tail from -6.9 to -11.1, so the part clear of the rump is ~1/4 of the
    nose(1.50)-to-tail-tip length.
    """
    def body(lw, **kw):
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: longer/narrower neck + widen toward the hind end | date: 2026-07-27]
        # Front-to-back profile (half-widths): a narrow NECK (0.68) that stays
        # narrow past the head circle (which ends at y=-1.0, so ~0.5 of neck is
        # visible), flaring to the shoulder (1.00) and then widening steadily
        # toward the hips (1.16, the widest point) before the rounded rump. The
        # animal is therefore heaviest at the back, as a ferret is from above.
        # The neck->shoulder flare is stepped through several points; a single
        # jump from the neck straight to full shoulder width leaves a boxy corner
        # that stacks with the shoulder nub into a double bump under the head.
        # Between the two limb pairs the midriff carries a full, gently convex
        # belly (1.12 -> 1.15) rather than a straight taper, and it is stretched
        # out so the fore and hind limbs sit further apart.
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: rump ran ~1 head-radius past the hind humps, reading as an over-long backside | date: 2026-07-30]
        # The rump is CROPPED: the hind humps end near y=-7.15, and the body now
        # closes at -7.72 (was -8.13), so only ~0.6 head-radii of butt sits
        # behind the legs. The taper is an OGIVE -- half-width eases off from the
        # widest point at -6.45 with steadily increasing slope (1.36 / 1.33 /
        # 1.25 / 1.13 / 0.97 / 0.77 / 0.53 / 0.26 / 0) rather than holding full
        # width and then dropping, which left a shoulder and a squared-off cap.
        # How early it can start is set by the HIP HUMP, not by taste: the body
        # must stay outside the hump's outer edge, or the hump breaks the
        # silhouette. Untilting and shortening the hump (see nub_specs) moved its
        # last crossing from y=-7.14 to -6.96, which is what buys this taper its
        # length -- 1.27 head-radii of run-out instead of 0.67.
        # The tail is shifted forward by the same amount so its root stays buried
        # in the rump and its widest point sits just behind the body edge.
        # [AGENT EDIT END]
        return mpatches.Polygon(
            [[-0.73, 0.47], [-0.73, -1.18], [-0.88, -1.75], [-1.03, -2.45],
             [-1.13, -3.20], [-1.20, -3.99], [-1.22, -4.83], [-1.28, -5.50],
             [-1.33, -6.00], [-1.36, -6.45], [-1.33, -6.70], [-1.25, -6.88],
             [-1.13, -7.05], [-0.97, -7.22], [-0.77, -7.38], [-0.53, -7.52],
             [-0.26, -7.63], [0.0, -7.72],
             [0.26, -7.63], [0.53, -7.52], [0.77, -7.38], [0.97, -7.22],
             [1.13, -7.05], [1.25, -6.88], [1.33, -6.70],
             [1.36, -6.45], [1.33, -6.00], [1.28, -5.50],
             [1.22, -4.83], [1.20, -3.99], [1.13, -3.20], [1.03, -2.45],
             [0.88, -1.75], [0.73, -1.18], [0.73, 0.47]],
            closed=True, lw=lw, joinstyle='round', transform=t, **kw)

    def tail(lw, **kw):
        # Tapers steadily to a narrow -- but not needle-sharp -- tip: the last
        # pair of points keeps half-width 0.13, enough to round off under the
        # outline halo without reading as blunt.
        return mpatches.Polygon(
            [[-0.34, -6.89], [-0.47, -7.99], [-0.28, -9.79], [-0.13, -10.84],
             [0.0, -11.09], [0.13, -10.84], [0.28, -9.79], [0.47, -7.99],
             [0.34, -6.89]],
            closed=True, lw=lw, joinstyle='round', transform=t, **kw)
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: splayed legs+claws read as a top-down lizard; a ferret's feet sit under the body, so only shoulder/hip nubs break the silhouette | date: 2026-07-27]
    def nub(lx, ly, ang, w, h, **kw):
        # Shoulder / hip bump: long axis runs ALONG the body, so the swell is
        # elongated front-to-back (a narrow bump reads as a stubby leg). Centred
        # near the body edge so it protrudes ~0.28 (shoulder) / ~0.32 (hip)
        # head-radii. `w`/`h` are the across- and along-body extents.
        return mpatches.Ellipse((lx, ly), w, h, angle=ang, transform=t, **kw)

    # (x, y, angle, width, length) for the front (shoulder) and hind (hip) nubs.
    # The HIND humps are deliberately bigger in every direction -- wider (0.80 vs
    # 0.62), longer (2.60 vs 1.58) and further out, since the body is widest at
    # the hips (~1.27) and narrower at the shoulders (~1.04).
    # Tilt on the SHOULDER nubs leans the front end inward (left -10 / right
    # +10), so the tip follows the neck taper instead of poking out through it as
    # a notch. The HIP humps use the opposite sign -- see below.
    # The hip hump runs the length of the broad hind quarters and its outer edge
    # merges TANGENTIALLY into the rump taper, so the posterior is nearly as wide
    # as the hump and there is no joint between them. Its rear sits behind
    # `rump_top` and is redrawn in the tail colour (see draw_ferret_body) so the
    # black hind end stays continuous across body and hump.
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: hip hump jutted past the body, so its rear plunged back to the outline as a visible corner | date: 2026-07-30]
    # The hip humps are pulled IN (centre 1.16 -> 1.08, width 0.86 -> 0.82) and
    # are now UNTILTED and shorter (2.30 -> 2.00). The corner at the back of the
    # hump was its rear TIP breaking the outline: the shoulders' +8 tilt leans a
    # nub's front end inward and its rear end OUTWARD, which on the long hip hump
    # swung the tip out to x=1.24 at y=-7.14 where the body is only ~1.15. An
    # ellipse's edge is vertical at its tip, so that 0.09 of overhang read as a
    # hard corner exactly where the rump starts to taper.
    # Zeroing the tilt and shortening puts the tip at (1.08, -7.00), buried 0.09
    # INSIDE the body, so the hump leaves the silhouette on its side instead --
    # tangentially, at y=-6.96, with the overhang already faded to 0.01. Its
    # protrusion decays smoothly from 0.16 at the hip to nothing.
    # Do NOT reverse the tilt to solve this: that swings the widest point forward
    # to y=-5.6, where the body is narrow, and the hump reads as an ear stuck on
    # at the waist with a hollow behind it.
    nub_specs = [(-0.98, -2.57, -10, 0.62, 1.58), (0.98, -2.57, 10, 0.62, 1.58),
                 (-1.08, -6.00, 0, 0.82, 2.00), (1.08, -6.00, 0, 0.82, 2.00)]
    # [AGENT EDIT END]
    # (x, y, angle, width, length) paws. Each sits immediately FORWARD of, and
    # partly overlapping, its hump. They are OVALS (a round dot reads as a bead)
    # tilted so the long axis points forward-outward, and they are big -- most of
    # each is hidden behind the body, so only the outer third or so shows.
    # [AGENT EDIT | claude | hind paws nudged out 1.47 -> 1.53: untilting the hip
    # hump widened its FRONT (the +8 tilt used to lean that end inward), so at
    # y=-5.45 the hump edge went 1.375 -> 1.42 and the paw all but vanished
    # behind it. 1.53 restores the ~0.11 of paw that shows. | 2026-07-30]
    foot_specs = [(-0.96, -1.80, 25, 0.62, 0.82), (0.96, -1.80, -25, 0.62, 0.82),
                  (-1.53, -5.45, 25, 0.66, 0.88), (1.53, -5.45, -25, 0.66, 0.88)]
    # (centre_y, height) of the brown bands covering each nub pair. The front
    # band reaches up over the neck to meet the head (a brown collar that matches
    # the head-only glyph's neck); the back band must be tall enough to cover the
    # elongated hip nubs, and everything BEHIND it is the black rump (see
    # draw_ferret_body) that runs continuously into the tail.
    band_specs = [(-1.76, 3.30), (-5.85, 2.00)]
    return body, tail, nub, nub_specs, band_specs, foot_specs
    # [AGENT EDIT END]


def _head_markings(ax, t, head_patch, body_color, mark_color, z, mask_lw,
                   snout=None, snout_lw=0.0, nose_color='#e79aa6'):
    """Sable markings on the head: brown back, bandit mask, eyes, nose.

    Returns the patches that carry their own clip (to the head), so a caller
    applying an outer clip knows not to overwrite them.
    """
    back = mpatches.Ellipse((0.0, -1.5), 2.9, 2.4, facecolor=mark_color,
                            edgecolor=mark_color, lw=0, transform=t, zorder=z)
    ax.add_patch(back)
    back.set_clip_path(head_patch)
    ax.add_patch(mpatches.Polygon(
        [[0.0, 1.05], [-0.62, 0.32], [0.62, 0.32]], closed=True, lw=mask_lw,
        joinstyle='round', facecolor=mark_color, edgecolor=mark_color,
        transform=t, zorder=z))
    band = mpatches.Ellipse((0.0, 0.52), 1.7, 0.95, facecolor=mark_color,
                            edgecolor=mark_color, lw=0, transform=t, zorder=z)
    ax.add_patch(band)
    band.set_clip_path(head_patch)
    eye = dict(facecolor='black', edgecolor='black', lw=0, transform=t,
               zorder=z + 1)
    # [AGENT EDIT | claude | user: wingertj | eyes forward (y 0.50 -> 0.76) up the longer muzzle; the brown band moved with them so they stay on dark fur | 2026-07-28]
    ax.add_patch(mpatches.Circle((-0.56, 0.76), 0.12, **eye))
    ax.add_patch(mpatches.Circle((0.56, 0.76), 0.12, **eye))
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: nose must reach the very front border of the face with NO rim of fur ahead of it | date: 2026-07-28]
    # Nose = the muzzle's OWN front cap. Drawing it as a copy of the snout path
    # with the same stroke width and clipping off everything behind a gentle arc
    # makes its outer edge coincide EXACTLY with the face border, so no cream can
    # show in front of it. (Any independently-drawn shape has to stop short of
    # the snout's stroked edge, which is what left a rim before.) The clip arc
    # gives the eye-facing side its shallow curve.
    nose_patches = []
    if snout is not None:
        nose = snout(snout_lw, facecolor=nose_color, edgecolor=nose_color,
                     zorder=z + 1)
        ax.add_patch(nose)
        nose.set_clip_path(mpath.Path(
            [(-0.34, 1.60), (0.0, 1.42), (0.34, 1.60),
             (0.34, 2.40), (-0.34, 2.40), (-0.34, 1.60)],
            [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3,
             mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]), t)
        nose_patches.append(nose)
    # [AGENT EDIT END]
    # Ear inner-fold curve (a small arc at the base of each ear). Thin, and
    # scales with the glyph (no fixed floor, which made it too thick when small).
    for sgn in (-1, 1):
        ax.add_patch(mpatches.Arc((sgn * 0.98, 0.0), 0.34, 0.62,
                                  angle=sgn * 20, theta1=205, theta2=335,
                                  edgecolor='black', lw=mask_lw * 0.28,
                                  transform=t, zorder=z + 2))
    return [back, band] + nose_patches


def draw_ferret_head(ax, x, y, size, angle=0.0, transform='data',
                     body_color='#f0e9d8', mark_color='#7a5230',
                     nose_color='#e79aa6',
                     outline_color='black', outline_lw=3.0, snout_lw=7.0,
                     mask_lw=5.0, zorder=20):
    """Draw a schematic sable-ferret head at (x, y).

    angle : degrees CCW, 0 = nose up. transform : 'data', 'axes', or a Transform.
    size : head radius in the (x, y) units. Returns the composite transform.
    """
    t = _affine(ax, x, y, size, angle, transform)
    s_lw = _lw_scale(ax, t)
    outline_lw *= s_lw
    snout_lw *= s_lw
    mask_lw *= s_lw
    z = zorder
    neck, ear, head_circle, snout = _head_factories(t)

    # Outer outline = enlarged black silhouette behind the fills.
    blk = dict(facecolor=outline_color, edgecolor=outline_color, zorder=z)
    ax.add_patch(neck(lw=outline_lw, **blk))
    for sgn in (-1, 1):
        ax.add_patch(ear(sgn, lw=outline_lw, **blk))
    ax.add_patch(head_circle(lw=outline_lw, **blk))
    ax.add_patch(snout(snout_lw + outline_lw, **blk))

    # Fills: neck brown, ears + head + snout cream.
    ax.add_patch(neck(lw=0, facecolor=mark_color, edgecolor=mark_color,
                      zorder=z + 1))
    for sgn in (-1, 1):
        ax.add_patch(ear(sgn, lw=0, facecolor=body_color, edgecolor=body_color,
                         zorder=z + 2))
    head_patch = head_circle(lw=0, facecolor=body_color, edgecolor=body_color,
                             zorder=z + 3)
    ax.add_patch(head_patch)
    ax.add_patch(snout(snout_lw, facecolor=body_color, edgecolor=body_color,
                       zorder=z + 3))

    _head_markings(ax, t, head_patch, body_color, mark_color, z + 4, mask_lw,
                   snout=snout, snout_lw=snout_lw, nose_color=nose_color)
    return t


def draw_ferret_body(ax, x, y, size, angle=0.0, transform='data',
                     body_color='#f0e9d8', mark_color='#7a5230', nub_color=None,
                     tail_color='#30261f', nose_color='#e79aa6', leg_color=None,
                     outline_color='black', outline_lw=3.0, snout_lw=7.0,
                     mask_lw=5.0, zorder=20, clip_box=None):
    """Draw a top-down whole-body sable ferret (head + body + tail + limb nubs).

    Same placement API as draw_ferret_head: `angle` (0 = nose up), `x, y`
    (the head centre), `size` (head radius in the (x, y) units), `transform`.
    The dorsal view is brown (sable back) with the head's cream face + mask.
    `size` is the head radius; the body/tail extend several head-radii toward -y.

    Seen from above a ferret's feet sit mostly under the body, so the limbs are
    drawn as small shoulder/hip nubs rather than splayed legs with claws.
    `tail_color` defaults to near-black, as on a sable ferret. `leg_color` is a
    back-compatible alias for `nub_color`.

    `clip_box` is an optional convex polygon of (x, y) vertices in DATA
    coordinates (e.g. an arena rectangle); the glyph is cropped to it. Patches
    that already carry an internal clip get the intersection of the two regions.
    Only meaningful for ``transform='data'``; ignored otherwise. The head
    markings stay clipped to the head, so a glyph whose HEAD straddles the
    boundary would still show its markings -- fine when heads sit inside the box.
    Returns the composite transform.
    """
    # [AGENT EDIT | claude | limb nubs replace splayed legs; sable tail is near-black | 2026-07-27]
    nub_color = nub_color or leg_color or mark_color
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: crop glyphs to the arena border | date: 2026-07-27]
    if transform != 'data':
        clip_box = None
    clip_box = None if clip_box is None else np.asarray(clip_box, float)
    n_patch0, n_line0 = len(ax.patches), len(ax.lines)
    glyph_affine = (mtransforms.Affine2D().scale(size).rotate_deg(angle)
                    .translate(x, y))
    self_clipped = set()

    def _set_clip(patch, local_verts):
        """Clip `patch` to a glyph-frame polygon, intersected with clip_box."""
        self_clipped.add(id(patch))
        if clip_box is None:
            patch.set_clip_path(mpath.Path(local_verts), t)
            return
        poly = _clip_polygon(
            glyph_affine.transform(np.asarray(local_verts, float)), clip_box)
        # An empty intersection means the piece lies wholly outside the box.
        patch.set_clip_path(mpath.Path(poly if len(poly) >= 3
                                       else np.zeros((3, 2))), ax.transData)
    # [AGENT EDIT END]
    tail_lw = 1.5   # small round-join so the tail tip stays fairly pointy
    body_lw = 3.0   # smooths the body-outline polygon corners
    t = _affine(ax, x, y, size, angle, transform)
    s_lw = _lw_scale(ax, t)
    outline_lw *= s_lw
    snout_lw *= s_lw
    mask_lw *= s_lw
    tail_lw *= s_lw
    body_lw *= s_lw
    z = zorder
    _neck, ear, head_circle, snout = _head_factories(t)
    body, tail, nub, nub_specs, band_specs, foot_specs = _body_factories(t)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: feet must sit BEHIND the body so its outline crosses them and they read as layered, not merged | date: 2026-07-28]
    # Paws go down FIRST, below every body layer. The body's black halo (drawn at
    # z, filled) then covers the inner part of each paw, so the body outline runs
    # across it and only the outer sliver shows -- the paw reads as tucked under
    # the animal rather than fused into its silhouette.
    for fx, fy, fang, fw, fh in foot_specs:
        ax.add_patch(mpatches.Ellipse((fx, fy), fw, fh, angle=fang,
                                      lw=outline_lw, facecolor=outline_color,
                                      edgecolor=outline_color, transform=t,
                                      zorder=z - 2))
        ax.add_patch(mpatches.Ellipse((fx, fy), fw, fh, angle=fang, lw=0,
                                      facecolor=nub_color, edgecolor=nub_color,
                                      transform=t, zorder=z - 1))
    # [AGENT EDIT END]

    # Outline pass (black halo) over every silhouette shape -> outer perimeter.
    blk = dict(facecolor=outline_color, edgecolor=outline_color, zorder=z)
    ax.add_patch(tail(tail_lw + outline_lw, **blk))
    ax.add_patch(body(body_lw + outline_lw, **blk))
    for lx, ly, ang, nw, nh in nub_specs:
        ax.add_patch(nub(lx, ly, ang, nw, nh, lw=outline_lw, **blk))
    for sgn in (-1, 1):
        ax.add_patch(ear(sgn, lw=outline_lw, **blk))
    ax.add_patch(head_circle(lw=outline_lw, **blk))
    ax.add_patch(snout(snout_lw + outline_lw, **blk))

    # Fills: cream body, near-black tail, brown nubs; body extends under the head.
    rump_top = band_specs[-1][0] - band_specs[-1][1] / 2
    ax.add_patch(tail(tail_lw, facecolor=tail_color, edgecolor=tail_color,
                      zorder=z + 1))
    body_patch = body(body_lw, facecolor=body_color, edgecolor=body_color,
                      zorder=z + 1)
    ax.add_patch(body_patch)
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: kill the light hairline tracing the rump outline where the black rump is painted over the cream body | date: 2026-07-27]
    # Keep the cream fill OUT of the rump. Painting an antialiased dark patch
    # over an antialiased cream patch that share the SAME path does not hide the
    # cream: at an edge pixel of coverage a the result retains a*(1-a) of the
    # lower colour, which traces the rump outline as a pale seam. Clipping the
    # cream away (overlapping slightly so no gap can open) removes the source.
    _set_clip(body_patch,
              [[-3.0, rump_top - 0.12], [3.0, rump_top - 0.12], [3.0, 3.0],
               [-3.0, 3.0]])
    # [AGENT EDIT END]
    for lx, ly, ang, nw, nh in nub_specs:
        ax.add_patch(nub(lx, ly, ang, nw, nh, lw=0, facecolor=nub_color,
                         edgecolor=nub_color, zorder=z + 1))

    # Brown bands across the body over each nub pair. Each band is a brown copy
    # of the body shape (so its sides line up exactly with the body edge -- no
    # gap), clipped to a horizontal strip spanning that leg pair.
    for cy, h in band_specs:
        band = body(body_lw, facecolor=mark_color, edgecolor=mark_color,
                    zorder=z + 2)
        ax.add_patch(band)
        # Clip to a horizontal strip via an explicit Path + transform (reliable
        # at any scale, unlike clipping to an unadded patch).
        _set_clip(band, [[-2.5, cy - h / 2], [2.5, cy - h / 2],
                         [2.5, cy + h / 2], [-2.5, cy + h / 2]])

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: sable ferrets are black from the hind quarters back, so carry the tail colour onto the rump up to the hind band | date: 2026-07-27]
    # Black rump: everything BEHIND the hind brown band, so the dark hind end
    # runs continuously from the body into the tail. Same clipped-body-copy trick
    # as the bands, so its edges line up exactly with the body outline.
    rump = body(body_lw, facecolor=tail_color, edgecolor=tail_color,
                zorder=z + 2)
    ax.add_patch(rump)
    _set_clip(rump, [[-2.5, -14.0], [2.5, -14.0], [2.5, rump_top],
                     [-2.5, rump_top]])
    # The hip humps run past `rump_top`, and the rump patch is body-shaped so it
    # cannot reach the part of a hump that protrudes beyond the body outline.
    # Repeat each hump in the tail colour over the same strip, so the black hind
    # end stays continuous across body AND hump instead of leaving a brown
    # crescent outside it.
    for lx, ly, ang, nw, nh in nub_specs:
        if ly - nh / 2 >= rump_top:
            continue
        dark = nub(lx, ly, ang, nw, nh, lw=0, facecolor=tail_color,
                   edgecolor=tail_color, zorder=z + 2)
        ax.add_patch(dark)
        _set_clip(dark, [[-3.0, -14.0], [3.0, -14.0], [3.0, rump_top],
                         [-3.0, rump_top]])
    # [AGENT EDIT END]

    for sgn in (-1, 1):
        ax.add_patch(ear(sgn, lw=0, facecolor=body_color, edgecolor=body_color,
                         zorder=z + 3))
    head_patch = head_circle(lw=0, facecolor=body_color, edgecolor=body_color,
                             zorder=z + 4)
    ax.add_patch(head_patch)
    ax.add_patch(snout(snout_lw, facecolor=body_color, edgecolor=body_color,
                       zorder=z + 4))

    for _p in _head_markings(ax, t, head_patch, body_color, mark_color,
                             z + 5, mask_lw, snout=snout, snout_lw=snout_lw,
                             nose_color=nose_color):
        self_clipped.add(id(_p))   # stay clipped to the head/nose, not the arena

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: hairline where the head's brown back-of-head marking (clipped to the head circle) met the body's brown collar | date: 2026-07-27]
    # Collar. The head's brown back-of-head marking is clipped to the HEAD
    # circle, and between y=-0.68 and -1.0 that circle runs INSIDE the body
    # outline -- so two patches of the SAME brown abut along the arc and leave an
    # antialiasing hairline that reads as a join between two pieces. Redraw that
    # same marking ellipse clipped to the BODY instead and lay it over the top:
    # the union covers the arc, and because it is the identical ellipse its own
    # visible edge falls exactly where the head marking's already does (so the
    # brown cap keeps its curved outline instead of gaining a straight edge).
    collar = mpatches.Ellipse((0.0, -1.5), 2.9, 2.4, facecolor=mark_color,
                              edgecolor=mark_color, lw=0, transform=t,
                              zorder=z + 6)
    ax.add_patch(collar)
    _set_clip(collar, body(0).get_xy())
    # [AGENT EDIT END]

    # (No claws: from above the paws are tucked under the body -- drawing splayed
    # toes made the glyph read as a lizard rather than a ferret.)

    # Whiskers: a few long, thin black lines from each side of the muzzle.
    whisk_len, whisk_lw = 1.15, 0.85 * s_lw
    # Origins sit just inside the muzzle EDGE at each height (the muzzle is
    # ~0.53/0.44/0.32 half-wide at y = 0.95/1.12/1.28), so the whiskers spring
    # from the side of the snout rather than from its midline.
    for ox, oy, wa in [(0.50, 0.95, 12), (0.42, 1.12, 24), (0.30, 1.28, 36)]:
        for side in (1, -1):
            a = math.radians(wa if side == 1 else 180 - wa)
            x0, y0 = side * ox, oy
            ax.plot([x0, x0 + whisk_len * math.cos(a)],
                    [y0, y0 + whisk_len * math.sin(a)],
                    color='black', lw=whisk_lw, solid_capstyle='round',
                    transform=t, zorder=z + 6)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: crop the glyph to the arena border | date: 2026-07-27]
    # Everything this call added that does not already carry its own (already
    # intersected) clip gets the box directly. Note add_patch assigns the axes
    # patch as a default clip, so "has no clip yet" has to be tracked explicitly
    # rather than tested with get_clip_path().
    if clip_box is not None:
        box_path = mpath.Path(clip_box)
        for art in (list(ax.patches)[n_patch0:] + list(ax.lines)[n_line0:]):
            if id(art) not in self_clipped:
                art.set_clip_path(box_path, ax.transData)
    # [AGENT EDIT END]
    return t


# Side-profile ferret silhouette traced from a real ferret photo (alert head-up
# pose), in the glyph unit frame: nose reared up at +x ≈ (4.8, 2.6), high arched
# back peak ≈ (-1.5, 2.9), tail trailing to ≈ (-8.4, -1.8), feet at y ≈ -1.95.
_FERRET_SIDE_OUTLINE = [
    (4.80, 2.57), (4.77, 2.18), (4.53, 1.73), (4.23, 1.26),
    (3.92, 0.79), (3.59, 0.33), (3.20, -0.10), (2.81, -0.49),
    (2.52, -0.86), (2.43, -1.26), (2.39, -1.62), (2.17, -1.84),
    (1.73, -1.85), (1.24, -1.68), (0.75, -1.41), (0.26, -1.16),
    (-0.27, -1.03), (-0.77, -1.05), (-1.10, -1.23), (-1.17, -1.49),
    (-1.17, -1.71), (-1.40, -1.85), (-1.88, -1.91), (-2.45, -1.93),
    (-3.05, -1.95), (-3.64, -1.95), (-4.20, -1.91), (-4.73, -1.83),
    (-5.24, -1.81), (-5.77, -1.86), (-6.34, -1.92), (-6.92, -1.94),
    (-7.52, -1.94), (-8.05, -1.91), (-8.40, -1.78), (-8.39, -1.60),
    (-8.05, -1.43), (-7.54, -1.31), (-6.98, -1.21), (-6.44, -1.08),
    (-5.94, -0.86), (-5.53, -0.52), (-5.22, -0.09), (-4.95, 0.40),
    (-4.68, 0.89), (-4.36, 1.36), (-3.99, 1.79), (-3.57, 2.17),
    (-3.10, 2.49), (-2.61, 2.74), (-2.08, 2.89), (-1.53, 2.91),
    (-0.99, 2.80), (-0.47, 2.62), (0.03, 2.38), (0.52, 2.11),
    (1.02, 1.92), (1.53, 1.90), (2.01, 2.08), (2.48, 2.37),
    (2.95, 2.65), (3.47, 2.84), (4.02, 2.89), (4.51, 2.81),
]


def draw_ferret_side(ax, x, y, size, angle=0.0, transform='data',
                     body_color='#e9e1cf', mark_color='#6f4a2e',
                     dark_color='#3b2a1f', outline_color='black',
                     outline_lw=1.5, zorder=20):
    """Modular side-profile sable ferret, alert head-up pose (nose toward +x).

    Built from grouped simple shapes -- body trunk, rump, head+neck, head,
    muzzle, two ears, four legs and a tail -- rather than one traced outline, so
    every part stays a distinct, recolourable/poseable piece while the combined
    silhouette still approximates a real ferret (cf. `_FERRET_SIDE_OUTLINE`, the
    photo trace used as the shape-fit target). Same placement API.
    """
    t = _affine(ax, x, y, size, angle, transform)
    olw = outline_lw * _lw_scale(ax, t)
    z = zorder
    rr = 'round,pad=0,rounding_size='

    # --- part geometry (unit frame; head reared up toward +x) ---------------
    # cream body masses: (cx, cy, w, h, angle_deg)
    masses = [
        (-1.5, 0.5, 6.8, 3.7, 0),      # trunk
        (-3.55, 0.15, 2.9, 2.9, 0),    # rump / haunch
        (1.0, -0.25, 3.1, 3.1, 0),     # chest / shoulder
        (2.4, 1.2, 2.9, 1.9, 45),      # head + neck (rises to the head)
        (3.7, 2.0, 1.8, 1.7, 12),      # head
        (4.4, 2.35, 1.1, 0.8, 30),     # muzzle
    ]
    EARS = [(3.45, 2.82, 0.56, 0.64, 10), (3.0, 2.66, 0.48, 0.56, 10)]
    LEGS = [(1.85, -0.7), (1.25, -0.7), (-2.1, -0.7), (-3.25, -0.7)]  # (cx, top_y)
    # Two-piece drooping tail. The hind (distal) piece lies flat along the floor
    # (dragging, y ~ -1.95); the connecting (proximal) piece slopes up from that
    # floor portion to the rear of the animal (the rump).
    # The two pieces share the exact junction edge JT->JB so they abut with no
    # gap or step; the halo scheme covers the internal seam, giving a smooth bend.
    _JT, _JB = (-6.1, -1.2), (-6.0, -1.9)                 # shared junction edge
    TAIL_PIECES = [
        # proximal: thick bar sloping from the rump down to the junction
        [(-3.9, 0.4), _JT, _JB, (-3.5, -0.15)],
        # distal: lies flat along the floor, tapering to a pointed tip
        [_JT, (-8.1, -1.48), (-9.3, -1.88), (-8.4, -1.97), _JB],
    ]

    def ell(spec, lw, **k):
        cx, cy, w, h, a = spec
        return mpatches.Ellipse((cx, cy), w, h, angle=a, lw=lw, transform=t, **k)

    def leg(cx, top, lw, **k):
        return mpatches.FancyBboxPatch((cx - 0.3, -1.95), 0.6, top + 1.95,
                                       boxstyle=rr + '0.3', lw=lw, transform=t, **k)

    def tail(verts, lw, **k):
        return mpatches.Polygon(verts, closed=True, lw=lw, joinstyle='round',
                                transform=t, **k)

    blk = dict(facecolor=outline_color, edgecolor=outline_color)
    # --- outline halos: ears + tail + legs behind, then the body masses ------
    for spec in EARS:
        ax.add_patch(ell(spec, olw, zorder=z, **blk))
    for tv in TAIL_PIECES:
        ax.add_patch(tail(tv, olw, zorder=z, **blk))
    for cx, top in LEGS:
        ax.add_patch(leg(cx, top, olw, zorder=z, **blk))
    for spec in masses:
        ax.add_patch(ell(spec, olw, zorder=z, **blk))
    # --- fills: cream ears + dark legs/tail behind, cream body masses on top --
    for spec in EARS:
        ax.add_patch(ell(spec, 0, facecolor=body_color, edgecolor=body_color,
                         zorder=z + 1))
    for tv in TAIL_PIECES:
        ax.add_patch(tail(tv, 0, facecolor=dark_color, edgecolor=dark_color,
                          zorder=z + 1))
    for cx, top in LEGS:
        ax.add_patch(leg(cx, top, 0, facecolor=dark_color, edgecolor=dark_color,
                         zorder=z + 1))
    mass_patches = []
    for spec in masses:
        p = ell(spec, 0, facecolor=body_color, edgecolor=body_color, zorder=z + 2)
        ax.add_patch(p)
        mass_patches.append(p)
    headneck_patch, head_patch = mass_patches[3], mass_patches[4]

    # --- sable markings (each clipped to the relevant part) -----------------
    throat = ell((2.7, 0.35, 2.3, 1.9, 45), 0, facecolor=mark_color,
                 edgecolor=mark_color, zorder=z + 3)
    ax.add_patch(throat)
    throat.set_clip_path(headneck_patch)
    mask = ell((3.82, 1.85, 1.1, 1.3, 50), 0, facecolor=dark_color,
               edgecolor=dark_color, zorder=z + 4)
    ax.add_patch(mask)
    mask.set_clip_path(head_patch)

    # --- eye, nose (pink), whiskers -----------------------------------------
    ax.add_patch(mpatches.Circle((3.9, 2.0), 0.12, facecolor='black',
                                 edgecolor='black', lw=0, transform=t,
                                 zorder=z + 5))                          # eye
    ax.add_patch(mpatches.Circle((4.62, 2.42), 0.12, facecolor='#b07a82',
                                 edgecolor=dark_color, lw=olw * 0.4,
                                 transform=t, zorder=z + 5))             # nose
    wlw = 0.6 * _lw_scale(ax, t)
    for ox, oy, wa in [(4.52, 2.34, 34), (4.6, 2.46, 20), (4.44, 2.22, 48)]:
        a = math.radians(wa)
        ax.plot([ox, ox + 1.7 * math.cos(a)], [oy, oy + 1.7 * math.sin(a)],
                color='black', lw=wlw, solid_capstyle='round', transform=t,
                zorder=z + 5)
    return t


def draw_implant(ax, x, y, size, angle=0.0, transform='data',
                 body_color='#bcbcbc', dark='#8f8f8f', light='#d6d6d6',
                 outline_color='0.25', outline_lw=2.0, zorder=20):
    """Schematic chronic neural implant / headstage: cap + body + chamber collar
    + base plate + probe shank, with screw and side-port details.

    Unit frame: cap on top (+y), probe out the bottom (−y). Same placement API.
    """
    t = _affine(ax, x, y, size, angle, transform)
    olw = outline_lw * _lw_scale(ax, t)
    z = zorder
    rr = 'round,pad=0,rounding_size='
    edge = dict(edgecolor=outline_color, lw=olw, transform=t)

    ax.add_patch(mpatches.Ellipse((0.1, -2.15), 2.1, 0.4, facecolor='0.85',
                                  edgecolor='none', lw=0, transform=t,
                                  zorder=z - 1, alpha=0.6))                 # shadow
    ax.add_patch(mpatches.Rectangle((-0.04, -2.55), 0.08, 1.25,
                                    facecolor='0.3', edgecolor='0.3', lw=0,
                                    transform=t, zorder=z))                 # probe shank
    ax.add_patch(mpatches.FancyBboxPatch((-1.0, -1.6), 2.0, 0.6,
                                         boxstyle=rr + '0.14', facecolor=light,
                                         zorder=z + 1, **edge))             # base plate
    ax.add_patch(mpatches.Ellipse((0.0, -0.98), 2.1, 0.62, facecolor=dark,
                                  zorder=z + 2, **edge))                    # chamber collar
    ax.add_patch(mpatches.FancyBboxPatch((-0.88, -1.05), 1.76, 2.35,
                                         boxstyle=rr + '0.2', facecolor=body_color,
                                         zorder=z + 3, **edge))             # main body
    ax.add_patch(mpatches.FancyBboxPatch((0.34, -1.0), 0.5, 2.25,
                                         boxstyle=rr + '0.2', facecolor=dark,
                                         edgecolor='none', lw=0, transform=t,
                                         zorder=z + 4))                     # side facet (3D hint)
    ax.add_patch(mpatches.FancyBboxPatch((-1.02, 1.12), 2.04, 0.72,
                                         boxstyle=rr + '0.2', facecolor=light,
                                         zorder=z + 5, **edge))             # cap / lid
    screw = dict(facecolor=dark, edgecolor=outline_color, lw=olw * 0.5,
                 transform=t, zorder=z + 6)
    for sx in (-0.45, 0.45):
        ax.add_patch(mpatches.Circle((sx, 1.48), 0.14, **screw))           # cap screws
    for sx in (-0.5, 0.5):
        ax.add_patch(mpatches.Circle((sx, -1.32), 0.12, **screw))          # base screws
    ax.add_patch(mpatches.Circle((-0.55, 0.2), 0.2, facecolor=dark,
                                 edgecolor=outline_color, lw=olw * 0.6,
                                 transform=t, zorder=z + 6))               # side port
    return t
# [AGENT EDIT END]
