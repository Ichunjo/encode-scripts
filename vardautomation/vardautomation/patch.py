"""Path module"""

__all__ = ['Patch']

import shutil
from itertools import chain
from typing import List, Optional, Set, Tuple, Union

import vapoursynth as vs
from lvsfunc.types import Range
from lvsfunc.util import normalize_ranges

from .automation import BasicTool, VideoEncoder
from .config import FileInfo
from .vpathlib import VPath


class Patch:
    """Allow easy video patching"""
    ffmsindex: VPath = VPath('ffmsindex')
    mkvmerge: VPath = VPath('mkvmerge')

    encoder: VideoEncoder
    clip: vs.VideoNode
    file: FileInfo
    ranges: List[Tuple[int, int]]

    workdir: VPath
    output_filename: VPath

    _file_to_fix: VPath

    def __init__(self, encoder: VideoEncoder, clip: vs.VideoNode, file: FileInfo,
                 ranges: Union[Range, List[Range]],
                 output_filename: Optional[str] = None) -> None:
        """Patching your videos has never been so easy!"""
        self.encoder = encoder
        self.clip = clip
        self.file = file
        self.file.do_qpfile = False

        self.ranges = normalize_ranges(self.clip, ranges)

        self._file_to_fix = self.file.name_file_final

        final = self._file_to_fix.parent

        self.workdir = final / (self.file.name + '_temp')
        if output_filename is not None:
            self.output_filename = VPath(output_filename)
        else:
            self.output_filename = final / f'{self._file_to_fix.stem}_new.mkv'

        if self.workdir.exists():
            raise FileExistsError(f'Patch: {self.workdir.resolve().to_str()} already exists!')

    def run(self) -> None:
        """Launch patch"""
        # Local folder
        self.workdir.mkdir()

        self._resolve_range()
        self._encode()
        self._cut_and_merge()

    def do_cleanup(self) -> None:
        """Delete workdir folder"""
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _resolve_range(self) -> None:
        idx_file = self.workdir / 'index.ffindex'
        kf_file = idx_file.with_suffix(idx_file.suffix + '_track00.kf.txt')

        BasicTool(
            self.ffmsindex,
            ['-k', '-f', self._file_to_fix.to_str(), idx_file.to_str()]
        ).run()

        with kf_file.open('r', encoding='utf-8') as f:
            kfsstr = f.read().splitlines()

        # Convert to int and add the last frame
        kfsint = [int(x) for x in kfsstr[2:]] + [self.clip.num_frames]

        rng = self.__resolve_range_with_kfs(kfsint)
        rng = self.__resolve_range_from_kfs(rng)

        if len(rng) == 1:
            if rng[0][0] == 0 and rng[0][1] == self.clip.num_frames:
                raise ValueError('Don\'t use Patch, just redo your encode')

        self.ranges = rng

    def _encode(self) -> None:
        for i, rng in enumerate(self.ranges, start=1):
            fix = self.workdir / f'fix-{i:03.0f}'
            self.file.name_clip_output = fix
            self.encoder.run_enc(self.clip[rng[0]:rng[1]], self.file)

            BasicTool(self.mkvmerge, ['-o', fix.with_suffix('.mkv').to_str(), fix.to_str()]).run()

    def _cut_and_merge(self) -> None:
        tmp = self.workdir / 'tmp.mkv'
        tmpnoaudio = self.workdir / 'tmp_noaudio.mkv'

        if (start := (rng := list(chain.from_iterable(self.ranges)))[0]) == 0:
            rng = rng[1:]
        if rng[::-1][0] == self.clip.num_frames:
            rng = rng[:-1]
        split_args = ['--split', 'frames:' + ','.join(map(str, rng))]

        BasicTool(
            self.mkvmerge,
            ['-o', tmp.to_str(), '--no-audio', '--no-track-tags', '--no-chapters',
             self._file_to_fix.to_str(), *split_args]
        ).run()

        tmp_files = sorted(self.workdir.glob('tmp-???.mkv'))
        fix_files = sorted(self.workdir.glob('fix-???.mkv'))

        # merge_args: List[str] = []
        # for i, tmp in enumerate(tmp_files):
        #     merge_args += [
        #         fix_files[int(i/2)].to_str()
        #         if i % 2 == (0 if start == 0 else 1) else tmp.to_str()
        #     ] + ['+']
        merge_args = [
            fix_files[int(i/2)].to_str() if i % 2 == (0 if start == 0 else 1) else tmp.to_str()
            for i, tmp in enumerate(tmp_files)
        ]

        BasicTool(
            self.mkvmerge,
            ['-o', tmpnoaudio.to_str(),
             '--no-audio', '--no-track-tags', '--no-chapters',
             '[', *merge_args, ']',
             '--append-to', ','.join([f'{i+1}:0:{i}:0' for i in range(len(merge_args) - 1)])]
        ).run()
        BasicTool(
            self.mkvmerge,
            ['-o', self.output_filename.to_str(), tmpnoaudio.to_str(), '--no-video', self._file_to_fix.to_str()]
        ).run()

    def __resolve_range_with_kfs(self, kfs: List[int]) -> List[Tuple[int, int]]:
        rng_set: Set[Tuple[int, int]] = set()
        for start, end in self.ranges:
            s, e = (None, ) * 2
            for i, kf in enumerate(kfs):
                if kf > start:
                    s = kfs[i-1]
                    break
                if kf == start:
                    s = kf
                    break

            for i, kf in enumerate(kfs):
                if kf >= end:
                    e = kf
                    break

            if s is None or e is None:
                raise ValueError('_resolve_range: Something is wrong in `s` or `e`')

            rng_set.add((s, e))

        return sorted(rng_set)

    @staticmethod
    def __resolve_range_from_kfs(rng: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Changes something like that:
        # [(0, 67), (0, 115), (67, 204), (67, 377), (115, 204), (962, 2006), (3960, 5053), (4883, 5053)]
        # to:
        # [(0, 377), (962, 2006), (3960, 5053)]
        rng_res: List[Tuple[int, int]] = []
        i = 0
        while i < len(rng):
            if i < len(rng) - 1 and rng[i][1] > rng[i+1][0]:
                j = 0
                frames: Tuple[Set[int], Set[int]] = (set(), set())
                while i + j < len(rng) - 1 and rng[j+i][1] > rng[j+i+1][0]:
                    for k in range(2):
                        frames[k].add(rng[j+i][k])
                        frames[k].add(rng[j+i+1][k])
                    j += 1
                rng_res.append((min(frames[0]), max(frames[1])))
                i += j + 1
            else:
                rng_res.append(rng[i])
                i += 1
        return rng_res
