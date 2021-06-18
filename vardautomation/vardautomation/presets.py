"""Presets"""

__all__ = [
    'Preset', 'NoPreset',
    'PresetBD', 'PresetWEB',
    'PresetAAC', 'PresetOpus', 'PresetEAC3', 'PresetFLAC',
    'PresetChapOGM',
    'PresetChapXML'
]

import os
from typing import Callable, Optional

import lvsfunc
import vapoursynth as vs

core = vs.core


# TODO: Rewrite this logic
class Preset():  # noqa
    def __init__(self, idx: Optional[Callable[[str], vs.VideoNode]],
                 a_src: Optional[str], a_src_cut: Optional[str],
                 a_enc_cut: Optional[str], chapter: Optional[str]) -> None:
        self.path = None
        self.path_without_ext = None
        self.idx = idx
        self.name = None
        self.a_src = a_src
        self.a_src_cut = a_src_cut
        self.a_enc_cut = a_enc_cut
        self.chapter = chapter
        super().__init__()


PresetBD = Preset(
    idx=core.lsmas.LWLibavSource,
    a_src='{path:s}_track_{num:s}.wav',
    a_src_cut='{path:s}_cut_track_{num:s}.wav',
    a_enc_cut=None,
    chapter=None
)
PresetWEB = Preset(
    idx=core.ffms2.Source,
    a_src=None,
    a_src_cut=None,
    a_enc_cut='',
    chapter=''
)
PresetAAC = Preset(
    idx=None,
    a_src='{path:s}_track_{num:s}.aac',
    a_src_cut='{path:s}_cut_track_{num:s}.aac',
    a_enc_cut='{path:s}_cut_enc_track_{num:s}.m4a',
    chapter=None
)
PresetOpus = Preset(
    idx=None,
    a_src='{path:s}_track_{num:s}.opus',
    a_src_cut='{path:s}_cut_track_{num:s}.opus',
    a_enc_cut='{path:s}_cut_enc_track_{num:s}.opus',
    chapter=None
)
PresetEAC3 = Preset(
    idx=None,
    a_src='{path:s}_track_{num:s}.eac3',
    a_src_cut='{path:s}_cut_track_{num:s}.eac3',
    a_enc_cut='{path:s}_cut_enc_track_{num:s}.eac3',
    chapter=None
)
PresetFLAC = Preset(
    idx=None,
    a_src='{path:s}_track_{num:s}.flac',
    a_src_cut='{path:s}_cut_track_{num:s}.flac',
    a_enc_cut='{path:s}_cut_enc_track_{num:s}.flac',
    chapter=None
)
NoPreset = Preset(
    idx=lvsfunc.misc.source,
    a_src='',
    a_src_cut='',
    a_enc_cut='',
    chapter=''
)
PresetChapOGM = Preset(
    idx=None,
    a_src=None,
    a_src_cut=None,
    a_enc_cut=None,
    chapter=os.path.abspath('chapters/{name:s}.txt')
)
PresetChapXML = Preset(
    idx=None,
    a_src=None,
    a_src_cut=None,
    a_enc_cut=None,
    chapter=os.path.abspath('chapters/{name:s}.xml')
)
