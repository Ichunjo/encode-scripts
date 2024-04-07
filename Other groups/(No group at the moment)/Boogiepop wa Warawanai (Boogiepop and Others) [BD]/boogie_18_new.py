import os

from pathlib import Path

import vapoursynth as vs

from betterdescaletarget import RescaleFrac, cambi_mask, denoise
from lvsfunc import get_match_centers_scaling
from vodesfunc import Waifu2x_Doubler
from vsdeband import AddGrain, Placebo
from vsexprtools import ExprOp, combine, norm_expr
from vskernels import Bicubic, FFmpegBicubic, Hermite, Kernel
from vsmasktools import Morpho, XxpandMode, adg_mask, diff_creditless, dre_edgemask
from vsmuxtools import Chapters, Opus, Setup, VideoFile, do_audio, mux, src_file, x265
from vspreview.api import is_preview
from vsrgtools import box_blur
from vstools import (
    DitherType, KwargsT, core, depth, finalize_clip, get_neutral_value, get_u, get_v, get_y, join,
    replace_ranges, set_output
)

core.set_affinity(24, 2 << 13)
os.environ["CUDA_MODULE_LOADING"] = "LAZY"


JPBD = src_file(
    r'[BDMV][アニメ] ブギーポップは笑わない (2019) [BD Vol.1-5][FIN]\BOOGIEPOP_AND_OTHERS_5\BDMV\STREAM\00004.m2ts',
    force_lsmas=True, trim=(24, -24)
)

# Chapters(JPBD)

EPISODE = '18'
(EDSTART, EDEND) = (31836, 33999)


def main() -> vs.VideoNode:
    src = JPBD.init_cut()
    src_16 = depth(src, 16)
    set_output(src_16)
    out = depth(src_16, 32)
    
    fix_row = core.akarin.Expr(out, 'Y 944 = Y 945 = Y 134 = or or x[0,-2] x[0,2] + 2 / x ?')
    fix_row = replace_ranges(out, fix_row, (33999, None), exclusive=True)
    out = fix_row

    # Extract dynamic noise and static noise
    den = denoise(out, 200, 2, 0.5, nl_args=KwargsT(num_streams=4))
    noise = norm_expr([out, den], ('x y -', f'{get_neutral_value(out, True)}'))
    out = den


    # Descale process
    native_res = get_match_centers_scaling(out, target_height=720)
    rs = RescaleFrac(
        out, **native_res,
        kernel=FFmpegBicubic,
        upscaler=Waifu2x_Doubler('trt', tiles=(2, 1)),
        downscaler=Hermite(linear=True)
    )

    src_cropped = get_y(depth(out, 16)).std.Crop(0, 0, 135, 135).edgefixer.Continuity(0, 1, 0, 1)
    src_cropped = depth(src_cropped, 32)
    rs_ed = RescaleFrac(
        src_cropped, **get_match_centers_scaling(src_cropped, 1280, None),
        kernel=FFmpegBicubic,
        upscaler=Waifu2x_Doubler('trt', tiles=(2, 1)),
        downscaler=Hermite(linear=True),
    )
    creditmask = diff_creditless(rs_ed.rescale, rs_ed.clipy, 0.35, prefilter=1).std.Maximum()
    creditmask = box_blur(creditmask)
    creditmask = replace_ranges(creditmask.std.BlankClip(keep=True), creditmask, [(EDSTART, EDEND)], exclusive=True)
    rs_ed.credit_mask = creditmask

    out = replace_ranges(
        rs.upscale.with_chroma(),
        join(
            rs_ed.upscale.std.AddBorders(0, 0, 135, 135, 0.0186),
            get_u(out).std.Crop(0, 0, 68, 68).std.AddBorders(0, 0, 68, 68),
            get_v(out).std.Crop(0, 0, 68, 68).std.AddBorders(0, 0, 68, 68)
        ),
        (EDSTART, EDEND), exclusive=True
    ).std.CopyFrameProps(out)

    edgemask = dre_edgemask(depth(den, 16), brz=10 / 255).std.Maximum().std.Minimum()
    edgemask = box_blur(edgemask, 2, 2)

    def sharp_grain_kernel(sharp: float) -> Kernel:
        b = sharp / -50 + 1
        c = (1 - b) / 2
        return Bicubic(b, c)

    out = depth(out, 16)

    db_weak = Placebo.deband(out, radius=16, thr=1.3, iterations=12)
    db_strg = Placebo.deband(out, radius=16, thr=2.5, iterations=16)
    db_strg2 = Placebo.deband(out, radius=20, thr=3.25, iterations=18)
    deband_mask = cambi_mask(out, 2, merge_previous=True).resize.Point(format=vs.GRAY16).std.Expr('x 1.2 *')
    db = core.std.MaskedMerge(db_weak, db_strg, deband_mask)
    # Scenefiltering
    db = replace_ranges(db, db_strg, [(1329, 1412), (4600, 4776), (4923, 4995), (5568, 5709), (6234, 7512),
                                      (7783, 8192), (13168, 13240), (18115, 18551), (18611, 18683)], exclusive=True)
    db = replace_ranges(db, db_strg2, [(387, 503),], exclusive=True)

    deband = core.std.MaskedMerge(db, out, edgemask)
    #
    # Scenefiltering
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
    set_output(finalized)

    return finalized


if is_preview():
    main()
    # set_output(JPBD.init_cut())
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
