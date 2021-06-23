"""Path module"""

__all__ = ['Patch']

import platform
import shutil
from subprocess import call
from typing import NoReturn, Optional, Tuple

import vapoursynth as vs

from .automation import BasicTool, VideoEncoder
from .config import FileInfo
from .vpathlib import AnyPath, VPath


class Patch():  # noqa
    """Allow easy video patching"""
    ffmsindex: VPath = VPath('ffmsindex')
    mkvmerge: VPath = VPath('mkvmerge')

    workdir: VPath
    fix_raw: VPath
    fix_mkv: VPath

    file_to_fix: VPath
    filtered_clip: vs.VideoNode
    frame_start: int
    frame_end: int
    encoder: VideoEncoder
    file: FileInfo
    output_filename: Optional[str]

    def __init__(self,
                 file_to_fix: AnyPath, filtered_clip: vs.VideoNode,
                 frame_start: int, frame_end: int,
                 encoder: VideoEncoder, file: FileInfo, *,
                 output_filename: Optional[str] = None) -> None:
        """TODO: Make a proper docstring

        Args:
            file_to_fix (AnyPath): [description]
            filtered_clip (vs.VideoNode): [description]
            frame_start (int): [description]
            frame_end (int): [description]
            encoder (VideoEncoder): [description]
            file (FileInfo): [description]
            output_filename (Optional[str], optional): [description]. Defaults to None.
        """
        self.file_to_fix = VPath(file_to_fix).resolve()
        self.filtered_clip = filtered_clip
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.encoder = encoder
        self.file = file
        self.output_filename = output_filename

        whech = self._where_which()
        if call([whech, self.ffmsindex]) != 0:
            self._throw_error(self.ffmsindex.to_str())
        if call([whech, self.mkvmerge]) != 0:
            self._throw_error(self.mkvmerge.to_str())

    def run(self) -> None:
        """Launch patch"""
        # Local folder
        self.workdir = self.file_to_fix.parent / (self.file.name + '_temp')
        self.workdir.mkdir()
        self.fix_raw = self.workdir / 'fix'
        self.fix_mkv = self.workdir / 'fix.mkv'

        start, end = self._generate_keyframes()
        self._encode(self.filtered_clip[start:end])
        self._cut_and_merge(start, end)

    def do_cleanup(self) -> None:
        """Delete workdir folder"""
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _generate_keyframes(self) -> Tuple[int, int]:
        idx_file = self.workdir / 'index.ffindex'
        kf_file = idx_file.with_suffix(idx_file.suffix + '_track00.kf.txt')

        idxing = BasicTool(
            self.ffmsindex,
            ['-k', '-f', self.file_to_fix.to_str(), idx_file.to_str()]
        )
        idxing.run()

        with kf_file.open('r', encoding='utf-8') as f:
            kfsstr = f.read().splitlines()

        # Convert to int and add the last frame
        kfsint = [int(x) for x in kfsstr[2:]] + [self.filtered_clip.num_frames]

        fra_s, fra_e = None, None

        for i, kf in enumerate(kfsint):
            if kf > self.frame_start:
                fra_s = kfsint[i-1]
                break
            if kf == self.frame_start:
                fra_s = kf
                break

        for i, kf in enumerate(kfsint):
            if kf >= self.frame_end:
                fra_e = kf
                break

        if fra_s is None or fra_e is None:
            raise ValueError('_generate_keyframes: Something is wrong in frame_start or frame_end')

        return fra_s, fra_e

    def _encode(self, clip: vs.VideoNode) -> None:
        self.file.name_clip_output = self.fix_raw

        self.encoder.run_enc(clip, self.file)

        merge = BasicTool(self.mkvmerge, ['-o', self.fix_mkv.to_str(), self.fix_raw.to_str()])
        merge.run()

    def _cut_and_merge(self, start: int, end: int) -> None:
        name = self.file_to_fix.stem
        tmp = self.workdir / f'{name}_tmp.mkv'
        tmpnoaudio = self.workdir / f'{name}_tmp_noaudio.mkv'

        final = self.file_to_fix.parent
        if self.output_filename is not None:
            final /= self.output_filename
        else:
            final /= f'{name}_new.mkv'


        if start == 0:
            split_args = ['--split', f'frames:{end}']
        else:
            split_args = ['--split', f'frames:{start},{end}']
        merge = BasicTool(
            self.mkvmerge,
            ['-o', tmp.to_str(), '--no-audio', '--no-track-tags', '--no-chapters',
             self.file_to_fix.to_str(), *split_args]
        )
        merge.run()


        tmp001 = self.workdir / f'{tmp.stem}-001.mkv'
        tmp002 = self.workdir / f'{tmp.stem}-002.mkv'
        tmp003 = self.workdir / f'{tmp.stem}-003.mkv'

        if start == 0:
            merge_args = [self.fix_mkv.to_str(), '+', tmp002.to_str()]
        elif end == self.filtered_clip.num_frames:
            merge_args = [tmp001.to_str(), '+', self.fix_mkv.to_str()]
        else:
            merge_args = [tmp001.to_str(), '+', self.fix_mkv.to_str(), '+', tmp003.to_str()]

        merge = BasicTool(
            self.mkvmerge,
            ['-o', tmpnoaudio.to_str(), '--no-audio', '--no-track-tags', '--no-chapters', *merge_args]
        )
        merge.run()
        merge = BasicTool(
            self.mkvmerge,
            ['-o', final.to_str(), tmpnoaudio.to_str(), '--no-video', self.file_to_fix.to_str()]
        )
        merge.run()

    @staticmethod
    def _where_which() -> str:
        return 'where' if platform.system() == 'Windows' else 'which'

    @staticmethod
    def _throw_error(file_not_found: str) -> NoReturn:
        raise FileNotFoundError(f'{file_not_found} not found!')
