
import platform
from pathlib import Path
from subprocess import call
from typing import NoReturn, Tuple

import vapoursynth as vs

from .automation import BasicTool, VideoEncoder
from .config import FileInfo


class Patch():
    workdir: str = '_temp_workdir'
    fix_raw: str = f'{workdir}/fix'
    fix_mkv: str = f'{workdir}/fix.mkv'
    ffmsindex: str = 'ffmsindex'
    mkvmerge: str = 'mkvmerge'

    file_to_fix: Path
    filtered_clip: vs.VideoNode
    frame_start: int
    frame_end: int
    encoder: VideoEncoder
    file: FileInfo

    def __init__(self, file_to_fix: Path, filtered_clip: vs.VideoNode,
                 frame_start: int, frame_end: int,
                 encoder: VideoEncoder, file: FileInfo) -> None:
        self.file_to_fix = file_to_fix
        self.filtered_clip = filtered_clip
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.encoder = encoder
        self.file = file


        whech = self._where_which()
        if call([whech, self.ffmsindex]) != 0:
            self._throw_error(self.ffmsindex)
        if call([whech, self.mkvmerge]) != 0:
            self._throw_error(self.mkvmerge)

        start, end = self._generate_keyframes()
        self._encode(self.filtered_clip[start:end])
        self._cut_and_merge(start, end)


    def _generate_keyframes(self) -> Tuple[int, int]:
        idx_file = f'{self.workdir}/index.ffindex'
        kf_file = f'{idx_file}_track00.kf.txt'

        idxing = BasicTool(self.ffmsindex, ['-k', '-f', str(self.file_to_fix), idx_file])
        idxing.run()

        with open(kf_file, 'r', encoding='utf-8') as f:
            kfs = f.read().splitlines()

        kfs = list(map(int, kfs[2:]))
        # Add the last frame
        kfs.append(self.filtered_clip.num_frames)


        fra_s, fra_e = None, None

        for i, kf in enumerate(kfs):
            if kf > self.frame_start:
                fra_s = kfs[i-1]
                break
            if kf == self.frame_start:
                fra_s = kf
                break

        for i, kf in enumerate(kfs):
            if kf >= self.frame_end:
                fra_e = kf
                break

        if fra_s is None or fra_e is None:
            raise ValueError('Something is wrong in frame_start or frame_end')

        return fra_s, fra_e

    def _encode(self, clip: vs.VideoNode) -> None:
        self.file.name_clip_output = self.fix_raw

        self.encoder.run_enc(clip, self.file)

        merge = BasicTool('mkvmerge', ['-o', self.fix_mkv, self.fix_raw])
        merge.run()

    def _cut_and_merge(self, start: int, end: int) -> None:
        name = str(self.file_to_fix.stem)
        tmp = f'{self.workdir}/{name}_tmp.mkv'
        tmpnoaudio = f'{self.workdir}/{name}_tmp_noaudio.mkv'
        final = f'{name}_new.mkv'

        if start == 0:
            split_args = ['--split', f'frames:{end}']
        else:
            split_args = ['--split', f'frames:{start},{end}']
        merge = BasicTool(self.mkvmerge, ['-o', tmp, '--no-audio', '--no-track-tags', '--no-chapters', str(self.file_to_fix), *split_args])
        merge.run()

        if start == 0:
            merge_args = [self.fix_mkv, '+', f'{self.workdir}/{tmp[:-4]}-002.mkv']
        elif end == self.filtered_clip.num_frames:
            merge_args = [f'{self.workdir}/{tmp[:-4]}-001.mkv', '+', self.fix_mkv]
        else:
            merge_args = [f'{self.workdir}/{tmp[:-4]}-001.mkv', '+', self.fix_mkv, '+', f'{self.workdir}/{tmp[:-4]}-003.mkv']

        merge = BasicTool(self.mkvmerge, ['-o', tmpnoaudio, '--no-audio', '--no-track-tags', '--no-chapters', *merge_args])
        merge.run()
        merge = BasicTool(self.mkvmerge, ['-o', final, tmpnoaudio, '--no-video', str(self.file_to_fix)])
        merge.run()

    @staticmethod
    def _where_which() -> str:
        return 'where' if platform.system() == 'Windows' else 'which'

    @staticmethod
    def _throw_error(file_not_found: str) -> NoReturn:
        raise FileNotFoundError(f'{file_not_found} not found!')
