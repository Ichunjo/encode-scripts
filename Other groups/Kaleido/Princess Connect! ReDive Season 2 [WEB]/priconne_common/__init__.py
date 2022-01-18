from typing import List, Sequence

import vapoursynth as vs
from vardautomation import JAPANESE, X265, AudioStream, FileInfo, Mux, Patch, RunnerConfig, SelfRunner, VideoStream
from vardefunc import AddGrain
from vardefunc.types import Range
from vsmask.better_vsutil import join, split
from vsmask.edge import TheToof
from vsutil import Range as CRange
from vsutil import scale_value

from .edgecleaner import edge_cleaner
from .finedehalo import fine_dehalo


class Encode:
    def __init__(self, file: FileInfo, clip: vs.VideoNode) -> None:
        self.file = file
        self.clip = clip

    def run(self, num: str, audio: str) -> None:
        enc = X265('priconne_common/x265_settings_web')
        enc.resumable = True

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, '', JAPANESE),
                AudioStream(audio, 'AAC 2.0', JAPANESE),
                None
            ), merge_args={'--ui-language': 'en'}
        )

        config = RunnerConfig(enc, muxer=muxer)

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()
        runner.upload_ftp('Stalleido', f'/kaleido/.projects/Priconne S2/{num}')

    def do_patch(self, ranges: Range | List[Range]) -> None:
        p = Patch(X265('priconne_common/x265_settings_web'), self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()


core = vs.core
graigasm_args = dict(
    thrs=[x << 8 for x in (32, 80, 128, 176)],
    strengths=[(0.25, 0.1), (0.15, 0.05), (0.10, 0.0), (0.0, 0.0)],
    sizes=(1.2, 1.1, 1, 1),
    sharps=(70, 60, 50, 50),
    grainers=[
        AddGrain(seed=333, constant=False),
        AddGrain(seed=333, constant=False),
        AddGrain(seed=333, constant=True)
    ]
)


class TheToof2(TheToof):
    def _merge(self, clips: Sequence[vs.VideoNode]) -> vs.VideoNode:
        c = clips[0]
        assert c.format
        neutral = 1 << (c.format.bits_per_sample-1)
        peak = (1 << c.format.bits_per_sample)-1
        return core.std.Expr(clips, f'x y z a max max max {neutral} / 0.86 pow {peak} *')


def toooon(clip: vs.VideoNode, strength: float = 1.0, l_thr: int = 2, h_thr: int = 12,
           blur: int = 2, depth: int = 32) -> vs.VideoNode:
    """
    str (float) - Strength of the line darken. Default is 1.0

    l_thr (int) - Lower threshold for the linemask. Default is 2

    h_thr (int) - Upper threshold for the linemask. Default is 12

    blur (int)  - "blur" parameter of AWarpSharp2. Default is 2

    depth (int) - "depth" parameter of AWarpSharp2. Default is 32
    """
    assert clip.format
    neutral = 1 << (clip.format.bits_per_sample - 1)
    peak = (1 << clip.format.bits_per_sample) - 1
    multiple = peak / 255

    lthr = neutral + scale_value(l_thr, 8, clip.format.bits_per_sample, CRange.FULL, CRange.FULL)
    lthr8 = lthr / multiple
    hthr = neutral + scale_value(h_thr, 8, clip.format.bits_per_sample, CRange.FULL, CRange.FULL)
    hthr8 = hthr / multiple
    ludiff = h_thr - l_thr

    luma, *chroma = split(clip)

    diff = core.std.MakeDiff(luma.std.Maximum().std.Minimum(), luma)
    warped = padding(diff, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=depth).std.Crop(6, 6, 6, 6)
    limit = core.std.Expr([diff, warped], 'x y min')
    limit = core.std.Expr(
        [limit, limit.std.Maximum()],
        (f'y {lthr} <= {neutral} y {hthr} >= x {hthr8} y {multiple} / - 128 * x {multiple} / y {multiple} '
         f'/ {lthr8} - * + {ludiff} / {multiple} * ? {neutral} - {strength} * {neutral} + ?')
    )
    diff = core.std.MakeDiff(luma, limit)

    return join([diff] + chroma, clip.format.color_family)


def padding(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    return clip.resize.Point(
        clip.width + left + right, clip.height + top + bottom,
        src_left=-left, src_top=-top,
        src_width=clip.width + left + right, src_height=clip.height + top + bottom
    )
