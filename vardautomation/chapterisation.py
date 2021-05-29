# noqa
import os
from typing import Any, List, Sequence

import vapoursynth as vs

from .colors import Colors


class Chapter():
    """Chapter object"""
    chapter_file: str

    def __init__(self, chapter_file: str) -> None:
        """Take simple txt chapter file path as paramater"""
        self.chapter_file = chapter_file
        super().__init__()

    def create(self, num_entries: int, names: List[str], frames: List[int], src_clip: vs.VideoNode) -> None:
        """Create a txt chapter file with given parameters

        Args:
            num_entries (int):
                Number of chapters.

            names (List[str]):
                Name of chapters.

            frames (List[int]):
                Frames where chapters begin.

            src_clip (vs.VideoNode):
                Source clip.
        """
        datas: List[Any] = [names, frames]
        if all(len(lst) != num_entries for lst in datas):
            raise ValueError('create: "names" and "frames" must have the same length as the given "num_entries"')

        with open(self.chapter_file, 'w') as file:
            for i, name, frame in zip(range(1, num_entries + 1), names, frames):
                file.writelines([f'CHAPTER{i:02.0f}={self._f2ts(frame, src_clip)}\n',
                                 f'CHAPTER{i:02.0f}NAME={name}\n'])
            print(Colors.INFO)
            print(f'Chapter file sucessfuly created at: {self.chapter_file}')
            print(f'{Colors.RESET}\n')

    def copy(self, destination: str) -> None:
        """Copy source chapter to destination.

        Args:
            destination (str): Destination path.
        """
        os.system(f'copy "{self.chapter_file}" "{destination}"')

    def set_names(self, names: Sequence[str]) -> None:
        """Change chapter names.

        Args:
            names (List[str]): List of names.
        """
        data = self._get_data()

        newchapnames: List[str] = []
        for i, chapname in enumerate(data[1::2]):
            if names[i] != '':
                newchapnames += [f'CHAPTER{i+1:02.0f}NAME={names[i]}\n']
            else:
                newchapnames += chapname

        data = [val for tup in zip(data[::2], newchapnames) for val in tup]

        with open(self.chapter_file, 'w') as file:
            file.writelines(data)

        print(Colors.INFO)
        print(f'Chapter names sucessfuly updated at: {self.chapter_file}')
        print(f'{Colors.RESET}\n')

    def shift_times(self, frames: int, src_clip: vs.VideoNode) -> None:
        """Shift times by given number of frames.

        Args:
            frames (int): Number of frames to shift
            src_clip (vs.VideoNode): Source clip.
        """
        data = self._get_data()

        shifttime = self._f2seconds(frames, src_clip)

        chaptimes = data[::2]
        newchaptimes: List[str] = []
        for chaptime in chaptimes:
            chap, time = chaptime.split('=')
            time = time[:-1]
            seconds = max(0, self._ts2seconds(time) + shifttime)
            time = self._seconds2ts(seconds)
            newchaptimes += [f'{chap}={time}\n']

        data = [val for tup in zip(newchaptimes, data[1::2]) for val in tup]

        with open(self.chapter_file, 'w') as file:
            file.writelines(data)

        print(Colors.INFO)
        print(f'Chapter names sucessfuly shifted at: {self.chapter_file}')
        print(f'{Colors.RESET}\n')

    def _f2seconds(self, f: int, src_clip: vs.VideoNode, /) -> float:  # noqa
        if f == 0:
            s = 0.0  # noqa

        t = round(float(10 ** 9 * f * src_clip.fps ** -1))  # noqa
        s = t / 10 ** 9  # noqa
        return s

    def _f2ts(self, f: int, src_clip: vs.VideoNode, /, *, precision: int = 3) -> str:  # noqa
        s = self._f2seconds(f, src_clip)  # noqa
        ts = self._seconds2ts(s, precision=precision)  # noqa
        return ts

    def _ts2seconds(self, ts: str, /) -> float:  # noqa
        h, m, s = map(float, ts.split(':'))  # noqa
        return h * 3600 + m * 60 + s

    def _seconds2ts(self, s: float, /, *, precision: int = 3) -> str:  # noqa
        m = s // 60  # noqa
        s %= 60  # noqa
        h = m // 60  # noqa
        m %= 60  # noqa

        return self._compose_ts(h, m, s, precision=precision)

    def _get_data(self) -> List[str]:
        with open(self.chapter_file, 'r') as file:
            data = file.readlines()
        return data

    @staticmethod
    def _compose_ts(h: float, m: float, s: float, /, *, precision: int = 3) -> str:
        if precision == 0:  # noqa
            return f"{h:02.0f}:{m:02.0f}:{round(s):02}"
        elif precision == 3:
            return f"{h:02.0f}:{m:02.0f}:{s:06.3f}"
        elif precision == 6:
            return f"{h:02.0f}:{m:02.0f}:{s:09.6f}"
        elif precision == 9:
            return f"{h:02.0f}:{m:02.0f}:{s:012.9f}"
        else:
            raise ValueError('precision must be <= 9 and >= 0')
