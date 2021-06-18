"""Automation module"""

__all__ = ['FileInfo']

import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Union

import vapoursynth as vs
from prettyprinter import pretty_call, pretty_repr, register_pretty
from prettyprinter.doc import Doc
from prettyprinter.prettyprinter import PrettyContext

from .presets import NoPreset, Preset

core = vs.core

# TODO: Better english because I’m fucking bad


class FileInfo():  # noqa: PLR0902
    """File info object"""
    path: Path
    path_without_ext: Path

    idx: Optional[Callable[[str], vs.VideoNode]]
    preset: List[Preset]

    name: str

    a_src: Optional[str]
    a_src_cut: Optional[str]
    a_enc_cut: Optional[str]
    chapter: Optional[str]

    clip: vs.VideoNode
    frame_start: Optional[int]
    frame_end: Optional[int]
    clip_cut: vs.VideoNode

    name_clip_output: Path
    name_file_final: Path

    name_clip_output_lossless: Path
    do_lossless: bool

    qpfile: Path
    do_qpfile: bool


    def __init__(self, path: Path, /,
                 frame_start: Optional[int] = None, frame_end: Optional[int] = None, *,
                 idx: Optional[Callable[[str], vs.VideoNode]] = None,
                 preset: Union[List[Preset], Preset] = NoPreset) -> None:
        """Helper which allows to store the data related to your file to be encoded

        Args:
            path (Path):
                Path to your source file.

            frame_start (Optional[int], optional):
                Number of frames to trim at the beginning of the clip. Python slicing.
                Defaults to None.

            frame_end (Optional[int], optional):
                Number of frames to trim at the end of the clip. Python slicing.
                Defaults to None.

            idx (Optional[Callable[[str], vs.VideoNode]], optional):
                Indexer used to index the video track.
                Defaults to lvsfunc.misc.source.

            preset (Union[List[Preset], Preset], optional):
                Preset used to fill idx, a_src, a_src_cut, a_enc_cut and chapter attributes.
                Defaults to NoPreset.
        """
        self.path = path
        self.path_without_ext = self.path.with_suffix('')
        self.idx = idx

        self.name = Path(sys.argv[0]).stem

        self.a_src, self.a_src_cut, self.a_enc_cut, self.chapter = (None, ) * 4

        # TODO: Rewrite this logic
        self.preset = [preset] if isinstance(preset, Preset) else preset
        self._params_fill_preset()

        if self.idx:
            self.clip = self.idx(str(path))
            self.frame_start = frame_start
            self.frame_end = frame_end
            self.clip_cut = self.clip[self.frame_start:self.frame_end] if (self.frame_start or self.frame_end) else self.clip

        self.name_clip_output = Path(self.name + '.265')
        self.name_file_final = Path(self.name + '.mkv')

        self.name_clip_output_lossless = Path(self.name + '_lossless.mkv')
        self.do_lossless = False

        self.qpfile = Path(self.name + '_qpfile.log')
        self.do_qpfile = False

        super().__init__()

    def __repr__(self) -> str:
        @register_pretty(FileInfo)
        def _repr(value: object, ctx: PrettyContext) -> Doc:
            return pretty_call(ctx, FileInfo, vars(value))

        return pretty_repr(self)

    def _params_fill_preset(self) -> None:
        for pre in self.preset:
            for d1, d2 in zip(self.__dict__.items(), pre.__dict__.items()):  # noqa: PLC0103
                _, v = d1  # noqa: PLC0103
                kp, vp = d2  # noqa: PLC0103
                if isinstance(vp, str):
                    vp = vp.format(path=str(self.path_without_ext), name=self.name, num='{}')  # noqa: PLC0103
                setattr(self, kp, vp if not v else v)

    def cleanup(self, *,  # noqa
                a_src: bool = True, a_src_cut: bool = True, a_enc_cut: bool = True,
                chapter: bool = False) -> None:
        files = (self.a_src, self.a_src_cut, self.a_enc_cut, self.chapter)
        booleans = (a_src, a_src_cut, a_enc_cut, chapter)

        for file, boolean in zip(files, booleans):
            if boolean and file and Path(file).exists():
                os.remove(file)
            for i in range(10):
                if boolean and file and file.format(i) and Path(file.format(i)).exists():
                    os.remove(file.format(i))
