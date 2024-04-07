import os

from pathlib import Path

import vapoursynth as vs

from betterdescaletarget import RescaleFrac, cambi_mask, denoise
from lvsfunc import get_match_centers_scaling
from vodesfunc import Waifu2x_Doubler
from vsaa import based_aa
from vsdeband import AddGrain, Placebo
from vsexprtools import ExprOp, combine, norm_expr
from vskernels import Bicubic, FFmpegBicubic, Hermite, Kernel
from vsmasktools import Morpho, XxpandMode, adg_mask, diff_creditless, dre_edgemask
from vsmuxtools import Chapters, Opus, Setup, VideoFile, do_audio, mux, src_file, x265
from vspreview.api import is_preview
from vsrgtools import box_blur
from vsscale import FSRCNNXShader
from vstools import DitherType, core, depth, finalize_clip, get_neutral_value, replace_ranges, set_output

core.set_affinity(24, 2 << 13)
os.environ["CUDA_MODULE_LOADING"] = "LAZY"


JPBD = src_file(
    r'[BDMV][アニメ] ブギーポップは笑わない (2019) [BD Vol.1-5][FIN]\BOOGIEPOP_AND_OTHERS_3\BDMV\STREAM\00002.m2ts',
    force_lsmas=True, trim=(24, -24)
)

# Chapters(JPBD)

EPISODE = '09'
(EDSTART, EDEND) = (31904, JPBD.src_cut.num_frames)


def main() -> vs.VideoNode:
    src = JPBD.init_cut()
    src_16 = depth(src, 16)
    set_output(src_16)

    out = depth(src_16, 32)

    # Extract dynamic noise and static noise
    den = denoise(out, sigma=2.0, strength=0.5, thSAD=200, tr=2)
    noise = norm_expr([out, den], ('x y -', f'{get_neutral_value(out, True)}'))
    out = den

    native_res = get_match_centers_scaling(out, target_height=720)
    rs = RescaleFrac(
        out, **native_res,
        kernel=FFmpegBicubic,
        upscaler=Waifu2x_Doubler('trt', tiles=(2, 1)),
        downscaler=Hermite(linear=True)
    )
    creditmask = diff_creditless(rs.rescale, rs.clipy, 0.35, prefilter=1).std.Maximum()
    creditmask = box_blur(creditmask)
    creditmask = replace_ranges(creditmask.std.BlankClip(keep=True), creditmask, [(EDSTART, None)], exclusive=True)
    rs.credit_mask = creditmask

    doubled = rs.doubled
    aa = based_aa(doubled, 2, supersampler=FSRCNNXShader.x56)
    rs.doubled = replace_ranges(doubled, aa, [(7183, 7553), ], exclusive=True)
    out = rs.upscale.with_chroma()

    # Add lineart and areas where there are a lot of details in the staticnoise_mask
    edgemask = dre_edgemask(depth(den, 16), brz=10 / 255).std.Maximum().std.Minimum()
    edgemask = box_blur(edgemask, 2, 2)


    def sharp_grain_kernel(sharp: float) -> Kernel:
        b = sharp / -50 + 1
        c = (1 - b) / 2
        return Bicubic(b, c)

    out = depth(out, 16)

    db_weak = Placebo.deband(out, radius=16, thr=1.3, iterations=12)
    db_strg = Placebo.deband(out, radius=16, thr=2.5, iterations=16)
    db_strg2 = Placebo.deband(out, radius=20, thr=3.25, iterations=16)
    deband_mask = cambi_mask(out, 2, merge_previous=True).resize.Point(format=vs.GRAY16).std.Expr('x 1.2 *')
    db = core.std.MaskedMerge(db_weak, db_strg, deband_mask)
    # Scenefiltering
    db = replace_ranges(db, db_strg, [(25804, 26401),], exclusive=True)
    db = replace_ranges(db, db_strg2, [(20483, 20531), (22643, 22679), (25119, 25220), (25364, 25536), (30378, 30497)], exclusive=True)
    # 
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
