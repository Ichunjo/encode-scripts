import os

from pathlib import Path

import vapoursynth as vs

from betterdescaletarget import RescaleFrac, cambi_mask, denoise
from lvsfunc import get_match_centers_scaling
from vodesfunc import Waifu2x_Doubler
from vsaa import based_aa
from vsdeband import AddGrain, Placebo
from vsdenoise import nl_means
from vsexprtools import ExprOp, combine, norm_expr
from vskernels import Bicubic, FFmpegBicubic, Hermite, Kernel
from vsmasktools import Morpho, XxpandMode, adg_mask, diff_creditless_oped, dre_edgemask, region_rel_mask
from vsmuxtools import Chapters, Opus, Setup, VideoFile, do_audio, mux, src_file, x265
from vspreview.api import is_preview
from vsrgtools import box_blur
from vsscale import FSRCNNXShader
from vstools import (
    DitherType, core, depth, finalize_clip, get_neutral_value, insert_clip, join, replace_ranges,
    set_output
)

core.set_affinity(24, 2 << 13)
os.environ["CUDA_MODULE_LOADING"] = "LAZY"


JPBD = src_file(
    r'[BDMV][アニメ] ブギーポップは笑わない (2019) [BD Vol.1-5][FIN]\BOOGIEPOP_AND_OTHERS_2\BDMV\STREAM\00001.m2ts',
    force_lsmas=True, trim=(24, -24)
)

EPISODE = '05'
(OPSTART, OPEND) = (0, 2158)
(EDSTART, EDEND) = (31888, JPBD.src_cut.num_frames)

# Chapters(JPBD)

NCED = src_file(
    r'[BDMV][アニメ] ブギーポップは笑わない (2019) [BD Vol.1-5][FIN]\BOOGIEPOP_AND_OTHERS_1\BDMV\STREAM\00004.m2ts',
    force_lsmas=True, trim=(24, -24 + EDEND - EDSTART - 2160)
)

OP = src_file(r'_workdir\OP\encoded_ffv1.mkv', force_lsmas=True, trim=(0, OPEND - OPSTART))


STATIC_NOISE_RANGES= [
    (EDSTART, EDEND)
]


def main() -> vs.VideoNode:
    src = JPBD.init_cut()
    src_16 = depth(src, 16)
    set_output(src_16)

    out = depth(src_16, 32)

    # Extract dynamic noise and static noise
    den = denoise(out, sigma=2.0, strength=0.5, thSAD=200, tr=2)
    noise = norm_expr([out, den], ('x y -', f'{get_neutral_value(out, True)}'))
    bruteforce = nl_means(den, 8.0, tr=0, sr=3, simr=4, planes=0)
    out = den

    # Descale process
    oped_credit_mask = diff_creditless_oped(
        src,
        ncop=core.std.BlankClip(),
        nced=NCED.init_cut(),
        thr=0.81,
        opstart=None,
        opend=None,
        edstart=EDSTART,
        edend=EDEND - 1,
        prefilter=True,
    )
    oped_credit_mask = region_rel_mask(oped_credit_mask, 20, 20, 20, 20)

    native_res = get_match_centers_scaling(out, target_height=720)
    rs = RescaleFrac(
        out, **native_res,
        kernel=FFmpegBicubic,
        upscaler=Waifu2x_Doubler('trt', tiles=(2, 1)),
        downscaler=Hermite(linear=True))
    rs.credit_mask = oped_credit_mask
    aa = based_aa(rs.doubled, 2, supersampler=FSRCNNXShader.x56)
    rs.doubled = replace_ranges(rs.doubled, aa, [(7882, 8182), (11506, 11578), (21322, 21472)], exclusive=True)
    out = rs.upscale.with_chroma()

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
    limit = norm_expr(den, ('x 0.035 < 0 x 0.035 >= x 0.08 < and x 0.035 - 0.08 0.035 - / 1.0 ? ? 0.75 *', '0'), 0)
    newstatic = combine([out, staticgrain], ExprOp.ADD)
    newstatic = core.std.MaskedMerge(out, newstatic, limit, 0)
    newstatic = core.std.MaskedMerge(newstatic, out, staticnoise_mask)
    newstatic = replace_ranges(out, newstatic, STATIC_NOISE_RANGES, exclusive=True)
    out = newstatic
    out = depth(out, 16)


    db_weak = Placebo.deband(out, radius=16, thr=1.3, iterations=12)
    db_strg = Placebo.deband(out, radius=16, thr=2.5, iterations=16)
    deband_mask = cambi_mask(out, 2, merge_previous=True).resize.Point(format=vs.GRAY16).std.Expr('x 1.2 *')
    db = core.std.MaskedMerge(db_weak, db_strg, deband_mask)
    # Scenefiltering
    db = replace_ranges(db, db_weak, [(5447, 5803), (7495, 7549), (18978, 19140)], exclusive=True)
    db = replace_ranges(db, db_strg, [(16675, 16872)], exclusive=True)
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
    grain = AddGrain((0.4, 0.0), 1.2, sharp_grain_kernel(35), dynamic=True, neutral_out=False, seed=333).grain(out)
    grain = core.std.MaskedMerge(out, grain, grain_mask)
    grain = AddGrain((0.0, 0.1), 1.2, dynamic=True, neutral_out=False, seed=333).grain(grain)
    grain = core.std.MaskedMerge(out, grain, adgmask)
    out = grain

    finalized = finalize_clip(out)
    finalized = insert_clip(finalized, OP.src_cut, OPSTART)
    set_output(finalized)

    return finalized


if is_preview():
    main()
else:
    final = main()

    setup = Setup(
        EPISODE,
        bdmv_dir="[BDMV][アニメ] ブギーポップは笑わない (2019) [BD Vol.1-5][FIN]/BOOGIEPOP_AND_OTHERS_1"
    )

    enc = x265('boogiepop_common/x265_settings', qp_clip=JPBD.init_cut(), quiet_merging=False)
    enc.executable = 'x265-x64-v3.5+131-aMod-gcc13.1.0-opt-znver2.exe'
    enc._update_progress = lambda current_frame, total_frames: None
    assert setup.work_dir
    encoded = Path(setup.work_dir).joinpath("encoded.265").resolve()
    video = VideoFile(encoded) if encoded.exists() else enc.encode(final)

    audio = do_audio(JPBD, encoder=Opus())

    chaps = Chapters(JPBD)

    mux(
        video.to_track('JPBD encode', 'jpn'),
        audio.to_track('Opus 2.0', 'jpn'),
        chaps,
        quiet=False, print_cli=True
    )
