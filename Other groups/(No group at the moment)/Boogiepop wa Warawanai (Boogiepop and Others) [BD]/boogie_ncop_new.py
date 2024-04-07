import os

from pathlib import Path

import vapoursynth as vs

from betterdescaletarget import RescaleFrac, cambi_mask, denoise
from lvsfunc import get_match_centers_scaling
from vodesfunc import Clamped_Doubler, DescaleTarget, Waifu2x_Doubler
from vsdeband import AddGrain, Placebo
from vsdenoise import nl_means
from vsexprtools import ExprOp, combine, norm_expr
from vskernels import Bicubic, Bilinear, FFmpegBicubic, Gaussian, Hermite, Kernel
from vsmasktools import Morpho, XxpandMode, adg_mask, dre_edgemask
from vsmuxtools import Chapters, Opus, Setup, VideoFile, do_audio, mux, src_file, x265
from vspreview.api import is_preview
from vsrgtools import box_blur
from vstools import (
    DitherType, core, depth, finalize_clip, get_neutral_value, get_y, join, replace_ranges,
    set_output, split
)

core.set_affinity(24, 2 << 13)
# core.set_affinity(range(0, 24, 2), 2 << 13)
os.environ["CUDA_MODULE_LOADING"] = "LAZY"


EPISODE = 'NCOP'
(OPSTART, OPEND) = (0, 2160)


NCOP = src_file(
    r'[BDMV][アニメ] ブギーポップは笑わない (2019) [BD Vol.1-5][FIN]\BOOGIEPOP_AND_OTHERS_1\BDMV\STREAM\00003.m2ts',
    force_lsmas=True, trim=(24, -24)
)

STATIC_NOISE_RANGES= [
    (OPSTART + 876, OPSTART + 970),
    (OPSTART + 1030, OPSTART + 1064),
    (OPSTART + 1378, OPSTART + 1824),
]



def main() -> vs.VideoNode:
    src = NCOP.init_cut()
    src_16 = depth(src, 16)
    src_16 += src_16[-1] * 2

    # Fix bad edges, probably wrong chromatic aberration lul
    y, u, v = split(src_16)
    u = core.fb.FillBorders(u, 1, 0, 1, 1)
    v = core.fb.FillBorders(v, 1, 0, 1, 1)
    border = join(depth(y, 16), u, v)
    border = replace_ranges(src_16, border, (OPSTART + 399, OPSTART + 1834), exclusive=True)
    out = border
    # set_output(border, 'border')
    out = depth(out, 32)

    # Extract dynamic noise and static noise
    den = denoise(out, sigma=2.0, strength=0.5, thSAD=200, tr=2)
    noise = norm_expr([out, den], ('x y -', f'{get_neutral_value(out, True)}'))
    bruteforce = nl_means(den, 8.0, tr=0, sr=3, simr=4, planes=0)
    out = den

    native_res = get_match_centers_scaling(out, target_height=720)
    dt1 = DescaleTarget(
        **native_res,
        kernel=FFmpegBicubic,
        upscaler=Waifu2x_Doubler('trt', tiles=(2, 1)),
        downscaler=Hermite(linear=True),
        line_mask=False, credit_mask=False
    )
    dt2 = DescaleTarget(
        **native_res,
        kernel=Bilinear,
        upscaler=Clamped_Doubler(sharp_doubler=Waifu2x_Doubler('trt', tiles=(2, 1)), ratio=75),
        downscaler=Gaussian(linear=True, sigma=0.375),
        line_mask=False, credit_mask=False,
    )
    upscaled1 = dt1.get_upscaled(out)
    upscaled2 = dt2.get_upscaled(out).bore.FixBrightness(0, 1, 2, 0)
    upscaled = replace_ranges(upscaled1, upscaled2, (OPSTART + 399, OPSTART + 1834), exclusive=True)
    out = upscaled

    # Make a mask based on the static noise extracted
    staticnoise = norm_expr([den, bruteforce, box_blur(bruteforce, planes=0)], ('z 0.8 < x y - 0.0 ?', '0.0'))
    staticnoise_mask = Morpho.maximum(norm_expr(staticnoise, 'x abs 160 *', planes=0), 3, 3, planes=0).std.Limiter()
    staticnoise_mask = box_blur(staticnoise_mask, 2, 2, planes=0)

    # Add lineart and areas where there are a lot of details in the staticnoise_mask
    edgemask = dre_edgemask(depth(den, 16), brz=10 / 255).std.Maximum().std.Minimum()
    edgemask = box_blur(edgemask, 2, 2)
    blank = den.std.BlankClip(keep=True)
    staticnoise_mask = norm_expr(
        (staticnoise_mask, replace_ranges(blank, join(depth(edgemask, 32), blank), STATIC_NOISE_RANGES, exclusive=True)),
        'x y + 0.0 1.0 clamp'
    )

    def sharp_grain_kernel(sharp: float) -> Kernel:
        b = sharp / -50 + 1
        c = (1 - b) / 2
        return Bicubic(b, c)

    neutral_chroma = join(out, out.std.BlankClip())
    staticgrains = [
        AddGrain((0.2, 0.0), 3, sharp_grain_kernel(80), dynamic=False, neutral_out=True, seed=111).grain(neutral_chroma),
        AddGrain((0.1, 0.0), 4.2, sharp_grain_kernel(110), dynamic=False, neutral_out=True, seed=333).grain(neutral_chroma),
        AddGrain((0.1, 0.0), (4.2 * 5/3, 4.2), sharp_grain_kernel(90), dynamic=False, neutral_out=True, seed=666).grain(neutral_chroma),
        AddGrain((0.1, 0.0), (4.2, 4.2 * 5/3), sharp_grain_kernel(90), dynamic=False, neutral_out=True, seed=999).grain(neutral_chroma)
    ]
    staticgrain = norm_expr(staticgrains, ('x 0.5 - y 0.5 - z 0.5 - a 0.5 - + + +', ''), 0)
    newstatic = combine([out, staticgrain], ExprOp.ADD)
    limit = norm_expr(den, ('x 0.035 < 0 x 0.035 >= x 0.08 < and x 0.035 - 0.08 0.035 - / 1.0 ? ? 0.75 *', '0'), 0)
    set_output(get_y(limit))
    newstatic = core.std.MaskedMerge(out, newstatic, limit, 0)
    newstatic = core.std.MaskedMerge(newstatic, out, staticnoise_mask)
    newstatic = replace_ranges(out, newstatic, STATIC_NOISE_RANGES, exclusive=True)
    out = newstatic
    out = depth(out, 16)


    deband_mask = cambi_mask(out, 2, merge_previous=True).resize.Point(format=vs.GRAY16).std.Expr('x 1.2 *')
    db = core.std.MaskedMerge(
        Placebo.deband(out, radius=16, thr=1.3, iterations=12),
        Placebo.deband(out, radius=16, thr=2.5, iterations=16),
        deband_mask
    )
    db_2 = Placebo.deband(out, radius=25, thr=4.5, iterations=20)
    db_2 = core.std.MaskedMerge(db_2, out, Morpho.expand(edgemask, 10, 10, XxpandMode.ELLIPSE))
    db = replace_ranges(db, db_2, [(OPSTART + 1064, OPSTART + 1146), (OPSTART + 1172, OPSTART + 1378)], exclusive=True)
    deband = core.std.MaskedMerge(db, out, edgemask)
    out = deband


    adgmask = adg_mask(out, 14)
    noiseb = combine((depth(out, 32), noise), ExprOp.ADD)
    noiseb = core.std.MaskedMerge(out, depth(noiseb, 16), adgmask)
    out = depth(noiseb, 16)

    grain_mask1 = norm_expr(deband_mask, 'x 6000 < 0 x ?')
    grain_mask1 = box_blur(grain_mask1, 2, 4)
    grain_mask2 = cambi_mask(depth(src_16, 8, dither_type=DitherType.NONE), 2, window_size=40)
    grain_mask2 = norm_expr(grain_mask2, 'x 0.004 > x 20 * 1.0 min 0 ?')
    grain_mask2 = box_blur(Morpho.expand(grain_mask2, 17, mode=XxpandMode.ELLIPSE), 4, 4)
    grain_mask = norm_expr((grain_mask1, grain_mask2), 'x y 65535 * + 0 65535 clamp')
    grain = AddGrain((0.5, 0.0), 1.2, sharp_grain_kernel(35), dynamic=True, neutral_out=False, seed=333).grain(out)
    grain = core.std.MaskedMerge(out, grain, grain_mask)
    grain = AddGrain((0.0, 0.1), 1.2, dynamic=True, neutral_out=False, seed=333).grain(grain)
    grain = core.std.MaskedMerge(out, grain, adgmask)
    out = grain

    return finalize_clip(out)


if is_preview():
    main()
else:
    final = main()

    setup = Setup(
        EPISODE,
        bdmv_dir="[BDMV][アニメ] ブギーポップは笑わない (2019) [BD Vol.1-5][FIN]/BOOGIEPOP_AND_OTHERS_1"
    )

    enc = x265('boogiepop_common/x265_settings', qp_clip=NCOP.src_cut, quiet_merging=False)
    enc.executable = 'x265-x64-v3.5+131-aMod-gcc13.1.0-opt-znver2.exe'
    enc._update_progress = lambda current_frame, total_frames: None
    assert setup.work_dir
    encoded = Path(setup.work_dir).joinpath("encoded.265").resolve()
    video = VideoFile(encoded) if encoded.exists() else enc.encode(final)

    audio = do_audio(NCOP, encoder=Opus())

    chaps = Chapters(NCOP)

    mux(
        video.to_track('JPBD encode', 'jpn'),
        audio.to_track('Opus 2.0', 'jpn'),
        chaps,
        quiet=False, print_cli=True
    )
