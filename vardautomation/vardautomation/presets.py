"""Presets"""

__all__ = [
    'Preset', 'NoPreset',
    'PresetBD', 'PresetWEB',
    'PresetAAC', 'PresetOpus', 'PresetEAC3', 'PresetFLAC',
    'PresetChapOGM',
    'PresetChapXML'
]

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional

import vapoursynth as vs

from .vpathlib import VPath

core = vs.core


class PresetType(IntEnum):
    NO_PRESET = 0
    VIDEO = 10
    AUDIO = 20
    CHAPTER = 30


@dataclass
class Preset:
    idx: Optional[Callable[[str], vs.VideoNode]]
    a_src: Optional[VPath]
    a_src_cut: Optional[VPath]
    a_enc_cut: Optional[VPath]
    chapter: Optional[VPath]
    preset_type: PresetType



NoPreset = Preset(
    idx=None,
    a_src=VPath(''),
    a_src_cut=VPath(''),
    a_enc_cut=VPath(''),
    chapter=VPath(''),
    preset_type=PresetType.NO_PRESET
)
PresetBD = Preset(
    idx=core.lsmas.LWLibavSource,
    a_src=VPath('{work_filename:s}_track_{num:s}.wav'),
    a_src_cut=VPath('{work_filename:s}_cut_track_{num:s}.wav'),
    a_enc_cut=None,
    chapter=None,
    preset_type=PresetType.VIDEO
)
PresetWEB = Preset(
    idx=core.ffms2.Source,
    a_src=None,
    a_src_cut=None,
    a_enc_cut=VPath(''),
    chapter=None,
    preset_type=PresetType.VIDEO
)
PresetAAC = Preset(
    idx=None,
    a_src=VPath('{work_filename:s}_track_{num:s}.aac'),
    a_src_cut=VPath('{work_filename:s}_cut_track_{num:s}.aac'),
    a_enc_cut=VPath('{work_filename:s}_cut_enc_track_{num:s}.m4a'),
    chapter=None,
    preset_type=PresetType.AUDIO
)
PresetOpus = Preset(
    idx=None,
    a_src=VPath('{work_filename:s}_track_{num:s}.opus'),
    a_src_cut=VPath('{work_filename:s}_cut_track_{num:s}.opus'),
    a_enc_cut=VPath('{work_filename:s}_cut_enc_track_{num:s}.opus'),
    chapter=None,
    preset_type=PresetType.AUDIO
)
PresetEAC3 = Preset(
    idx=None,
    a_src=VPath('{work_filename:s}_track_{num:s}.eac3'),
    a_src_cut=VPath('{work_filename:s}_cut_track_{num:s}.eac3'),
    a_enc_cut=VPath('{work_filename:s}_cut_enc_track_{num:s}.eac3'),
    chapter=None,
    preset_type=PresetType.AUDIO
)
PresetFLAC = Preset(
    idx=None,
    a_src=VPath('{work_filename:s}_track_{num:s}.flac'),
    a_src_cut=VPath('{work_filename:s}_cut_track_{num:s}.flac'),
    a_enc_cut=VPath('{work_filename:s}_cut_enc_track_{num:s}.flac'),
    chapter=None,
    preset_type=PresetType.AUDIO
)
PresetChapOGM = Preset(
    idx=None,
    a_src=None,
    a_src_cut=None,
    a_enc_cut=None,
    chapter=VPath('chapters/{name:s}.txt'),
    preset_type=PresetType.CHAPTER
)
PresetChapXML = Preset(
    idx=None,
    a_src=None,
    a_src_cut=None,
    a_enc_cut=None,
    chapter=VPath('chapters/{name:s}.xml'),
    preset_type=PresetType.CHAPTER
)
