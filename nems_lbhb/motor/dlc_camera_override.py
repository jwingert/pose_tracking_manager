# [AGENT EDIT START | agent: claude | user: wingertj | reason: the camera-override logic
# lived in a CLI script, but the GUI is where a user actually notices a mix-up (they are
# looking at the frames); both need the same validated write path | date: 2026-08-12]
"""Assert which psi camera slot holds the ceiling view for one session.

psi names video files after the camera SLOT, not the content.  When a USB port
re-enumerates between sessions the slots swap, and the top-down view lands in
``video_side``/``video_side2`` while ``recording.avi`` holds the face or a side view.
Nothing in the filenames, the trial log or celldb records that, so it has to be asserted
by a human who has looked at the frames -- which is what this module supports, for both
``scripts/dlc_set_camera_override.py`` and the labeler GUI's camera panel.

The assertion is stored as ``[camera_override.<session stem>]`` in
``nems_lbhb/configs/dlc_registry.toml``.  ``cam_tag`` is derived from the slot rather
than accepted from the caller: getting it wrong would mis-align every trial while looking
deliberate in the file.

Nothing here imports DeepLabCut, and OpenCV is imported lazily inside the two functions
that decode video, so the registry half of the module loads in every environment.
"""
import os
from datetime import date

from nems_lbhb.motor import dlc_registry

SLOTS = ('recording.avi', 'video_side.avi', 'video_side2.avi')

# fractions of the shortest slot's duration, so every slot is sampled at the same
# wall-clock moments and the views can actually be compared
SAMPLE_FRACTIONS = (0.15, 0.35, 0.55, 0.75)


def session_dir(path):
    """The session directory for a video path, however it was given.

    Accepts ``<sess>/recording.avi``, the parent-level ``<sess>.avi`` symlink that
    ``dlc_fit_script`` creates, or the session directory itself.
    """
    p = str(path).rstrip('/')
    if os.path.isdir(p):
        return os.path.abspath(p)
    base = os.path.basename(p)
    if base in SLOTS or base.startswith(('recording', 'video_')):
        return os.path.abspath(os.path.dirname(p))
    return os.path.abspath(os.path.splitext(p)[0])        # <sess>.avi -> <sess>/


def session_stem(path):
    """The registry key for a session: its directory name (e.g. 'REI084a10_a_NFB')."""
    return os.path.basename(session_dir(path))


def cam_tag_for(video):
    """'camB'/'camC' for a side slot, ``None`` for the base slot (``recording.avi``)."""
    return dlc_registry._CAM_TAG_FOR_VIDEO.get(video)


def slot_info(path):
    """Per-slot frame count, fps, duration and frame size for every slot present.

    Returns an ordered dict keyed by slot filename.  Frame count and fps do NOT identify
    a view -- two cameras often match on both -- so this is context for looking at the
    pictures, never a substitute for it.
    """
    import cv2
    cv2.setLogLevel(0)
    sess = session_dir(path)
    info = {}
    for slot in SLOTS:
        p = os.path.join(sess, slot)
        if not os.path.exists(p):
            continue
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        info[slot] = dict(path=p, n_frames=n, fps=fps, duration=n / fps if fps else 0.0,
                          width=int(cap.get(3)), height=int(cap.get(4)),
                          cam_tag=cam_tag_for(slot))
        cap.release()
    return info


def sample_frames(path, info=None, fractions=SAMPLE_FRACTIONS, size=(320, 240)):
    """``{slot: [(seconds, grayscale frame), ...]}`` at matched wall-clock times.

    Contrast-stretched per frame: a slot recorded at 100 fps has a short exposure and is
    nearly unreadable raw, which is exactly the slot a mix-up tends to involve.
    """
    import cv2
    info = info or slot_info(path)
    if not info:
        return {}
    shortest = min(v['duration'] for v in info.values())
    times = [shortest * f for f in fractions]
    out = {}
    for slot, meta in info.items():
        cap = cv2.VideoCapture(meta['path'])
        tiles = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * meta['fps']))
            ok, frame = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
            tiles.append((t, cv2.resize(g, size)))
        cap.release()
        out[slot] = tiles
    return out


def current(path, registry_path=None):
    """The override recorded for this session, or ``None``."""
    reg = dlc_registry.load_registry(registry_path)
    return (reg.get('camera_override') or {}).get(session_stem(path))


def existing_outputs(path):
    """Any ``.dlc.h5`` already sitting in this session's ``sorted/``.

    An override means whatever was analysed before came from the WRONG camera, so these
    have to be quarantined and re-run.  Surfacing them is what stops that being silent.
    """
    sess = session_dir(path)
    sorted_dir = os.path.join(os.path.dirname(sess), 'sorted')
    stem = os.path.basename(sess)
    if not os.path.isdir(sorted_dir):
        return []
    return sorted([os.path.join(sorted_dir, n) for n in os.listdir(sorted_dir)
                   if n.startswith(stem) and n.endswith('.dlc.h5')])


def render_block(stem, video, by, note='', when=None):
    """The TOML block that would be written.  Pure -- safe to show before committing."""
    tag = cam_tag_for(video)
    note = (note or 'camera slot mix-up; ceiling view is in %s' % video).replace('"', "'")
    return ('\n[camera_override.%s]\n'
            'video = "%s"\n'
            'cam_tag = "%s"\n'
            'verified_by = "%s"\n'
            'verified_date = "%s"\n'
            'note = "%s"\n'
            % (stem, video, tag, by, (when or date.today()).isoformat(), note))


def add(path, video, by, note='', registry_path=None, dry_run=False):
    """Record the assertion.  Returns ``(block_text, commit_command)``.

    Refuses rather than guessing when:

    * ``video`` is not a psi slot, or is ``recording.avi`` -- that is already the default,
      so an override would assert nothing;
    * the named file is not there;
    * an entry for this session already exists.  Overwriting an assertion someone made
      after looking at the frames should be a visible, deliberate edit, not a click.

    Never commits.  The registry is git-tracked and shared, and this runs from a Streamlit
    app that may be running as a service account in a dirty checkout.
    """
    if video not in dlc_registry._CAM_TAG_FOR_VIDEO:
        raise ValueError('%r is not a psi camera slot (expected one of %s)'
                         % (video, sorted(dlc_registry._CAM_TAG_FOR_VIDEO)))
    if cam_tag_for(video) is None:
        raise ValueError('recording.avi is already the default camera; an override is only '
                         'needed when the ceiling view is on a side slot.')
    sess = session_dir(path)
    stem = os.path.basename(sess)
    target = os.path.join(sess, video)
    if not os.path.exists(target):
        raise FileNotFoundError('%s does not exist' % target)

    reg_path = registry_path or dlc_registry.DEFAULT_REGISTRY_PATH
    text = open(reg_path).read()
    if ('[camera_override.%s]' % stem) in text:
        raise ValueError('the registry already has an entry for %s; edit it by hand, so '
                         'that overwriting a human assertion stays visible' % stem)

    block = render_block(stem, video, by, note=note)
    anchor = '\n[training_defaults]'
    if anchor not in text:
        raise ValueError('cannot find the [training_defaults] anchor in %s' % reg_path)

    commit = ('git add %s && git commit -m "dlc: camera override for %s (%s)"'
              % (os.path.relpath(reg_path, _repo_root(reg_path)), stem, video))
    if dry_run:
        return block, commit

    tmp = reg_path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(text.replace(anchor, block + anchor, 1))
    os.replace(tmp, reg_path)

    dlc_registry.clear_cache()
    over = dlc_registry.load_registry(reg_path).get('camera_override', {})   # validates
    if stem not in over:
        raise RuntimeError('the entry for %s did not round-trip through the registry' % stem)
    return block, commit


def _repo_root(reg_path):
    """Two levels above nems_lbhb/configs/, so the printed git command is repo-relative."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(reg_path))))
# [AGENT EDIT END]
