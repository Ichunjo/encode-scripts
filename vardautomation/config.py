"""Automation module"""
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Union

import vapoursynth as vs

from .presets import NoPreset, Preset

core = vs.core

# TODO: Better english because I’m fucking bad


class FileInfo():  # noqa: PLR0902
    """File info object"""
    path: str
    src: str
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

    name_clip_output: str
    name_file_final: str

    name_clip_output_lossless: str
    do_lossless: bool


    def __init__(self, src: str, /,
                 frame_start: Optional[int] = None, frame_end: Optional[int] = None, *,
                 idx: Optional[Callable[[str], vs.VideoNode]] = None, preset: Union[List[Preset], Preset] = NoPreset) -> None:
        """Helper which allows to store the data related to your file to be encoded

        Args:
            src (str):
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
        self.path = str(Path(src).parent.joinpath(Path(src).stem))
        self.src = src
        self.idx = idx

        self.name = Path(sys.argv[0]).stem

        self.a_src = None
        self.a_src_cut = None
        self.a_enc_cut = None
        self.chapter = None

        self.preset = [preset] if isinstance(preset, Preset) else preset
        self._params_fill_preset()

        if self.idx:
            self.clip = self.idx(src)
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.clip_cut = self.clip[self.frame_start:self.frame_end] if (self.frame_start or self.frame_end) else self.clip

        self.name_clip_output = self.name + '.265'
        self.name_file_final = self.name + '.mkv'

        self.name_clip_output_lossless = self.name + '_lossless.mkv'
        self.do_lossless = False

        super().__init__()

    # TODO: Make a better __str__
    def __str__(self) -> str:
        txt = 'File infos:\n'
        txt += f'Full path: {self.src}\n'
        txt += f'Clip format: \n{self.clip_cut.format}\n'
        txt += f'Name: {self.name}\n'
        return txt

    def _params_fill_preset(self) -> None:
        for pre in self.preset:
            for d1, d2 in zip(self.__dict__.items(), pre.__dict__.items()):  # noqa: PLC0103
                _, v = d1  # noqa: PLC0103
                kp, vp = d2  # noqa: PLC0103
                if isinstance(vp, str):
                    vp = vp.format(path=self.path, name=self.name, num='{}')  # noqa: PLC0103
                setattr(self, kp, vp if not v else v)

    def cleanup(self, *,  # noqa
                a_src: bool = True, a_src_cut: bool = True, a_enc_cut: bool = True,
                chapter: bool = False, name_clip_output: bool = False) -> None:
        files = (self.a_src, self.a_src_cut, self.a_enc_cut, self.chapter, self.name_clip_output)
        booleans = (a_src, a_src_cut, a_enc_cut, chapter, name_clip_output)

        for file, boolean in zip(files, booleans):
            if boolean and file and Path(file).exists():
                os.remove(file)
            for i in range(10):
                if boolean and file.format(i) and Path(file.format(i)).exists():
                    os.remove(file.format(i))
