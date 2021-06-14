"""Presets"""

__all__ = ['Preset', 'PresetBD', 'PresetWEB', 'PresetAAC', 'PresetOpus', 'PresetEAC3', 'PresetFLAC', 'NoPreset']

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
        self.src = None
        self.idx = idx
        self.name = None
        self.a_src = a_src
        self.a_src_cut = a_src_cut
        self.a_enc_cut = a_enc_cut
        self.chapter = chapter
        super().__init__()


PresetBD = Preset(core.lsmas.LWLibavSource, '{path:s}_track_{num:s}.wav', '{path:s}_cut_track_{num:s}.wav', None, os.path.abspath('chapters/{name:s}.txt'))
PresetWEB = Preset(core.ffms2.Source, None, None, '', '')
PresetAAC = Preset(None, '{path:s}_track_{num:s}.aac', '{path:s}_cut_track_{num:s}.aac', '{path:s}_cut_enc_track_{num:s}.m4a', None)
PresetOpus = Preset(None, '{path:s}_track_{num:s}.opus', '{path:s}_cut_track_{num:s}.opus', '{path:s}_cut_enc_track_{num:s}.opus', None)
PresetEAC3 = Preset(None, '{path:s}_track_{num:s}.eac3', '{path:s}_cut_track_{num:s}.eac3', '{path:s}_cut_enc_track_{num:s}.eac3', None)
PresetFLAC = Preset(None, '{path:s}_track_{num:s}.flac', '{path:s}_cut_track_{num:s}.flac', '{path:s}_cut_enc_track_{num:s}.flac', None)

NoPreset = Preset(lvsfunc.misc.source, '', '', '', '')
