"""
Path settings for where to store training data and model results
"""

try:
    import pupil_settings_local as psl
except ImportError:
    psl = None

if hasattr(psl, 'DEFAULT_DLC_MODEL'):
    DEFAULT_DLC_MODEL = psl.DEFAULT_DLC_MODEL
else:
    #DEFAULT_DLC_MODEL = '/auto/data/dlc/multivid-CLL-2022-01-14/config.yaml'
    # Prince LH
    #DEFAULT_DLC_MODEL = '/auto/data/dlc/free_top_2022_11/config.yaml'
    # Prince RH
    DEFAULT_DLC_MODEL = '/auto/data/dlc/free_top_RH-jereme-2023-02-17/config.yaml'

# [AGENT EDIT START | agent: claude | user: wingertj | reason: routing moved to nems_lbhb/configs/dlc_registry.toml | date: 2026-08-06]
def get_dlc_model(avipath):
    """Resolve a video path to a DLC config.yaml. Routing rules live in the registry TOML."""
    from nems_lbhb.motor.dlc_registry import config_for_video
    return config_for_video(avipath)
# [AGENT EDIT END]
