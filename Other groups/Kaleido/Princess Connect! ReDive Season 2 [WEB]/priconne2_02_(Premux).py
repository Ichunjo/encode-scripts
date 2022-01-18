"""PriConne S2 Script"""

import vapoursynth as vs
from debandshit import dumb3kdb
from lvsfunc.kernels import Catrom, Lanczos
from vardautomation import FileInfo, VPath, get_vs_core
from vardefunc import (
    DebugOutput, Eedi3SR, Graigasm, Nnedi3SS, Thresholds, YUVPlanes, diff_rescale_mask,
    finalise_output, initialise_clip, initialise_input, merge_chroma, remap_rfs, select_frames,
    thresholding
)
from vsdenoise import knl_means_cl
from vsmask.edge import FDOGTCanny
from vsmask.util import XxpandMode, expand, inpand
from vsutil import depth, get_w, get_y

from priconne_common import Encode, TheToof2, edge_cleaner, fine_dehalo, graigasm_args, toooon

NUM = __file__[-14:-12]

core = get_vs_core(range(0, 24, 2))

FILE = FileInfo(f'raws/[FeelsBadSubs] Princess Connect! Re - Dive Season 2 - {NUM} [1080p].mkv')
NCOP_YT_AVC = core.lsmas.LWLibavSource('アニメ「プリンセスコネクト！Re -Dive Season 2」オープニング・テーマ「Lost Princess」ノンテロップ映像 [jMRKYvE5h6A]avc.mkv')

DEBUG = DebugOutput(
    FILE.clip_cut,
    # NCOP_YT_AVC.std.BlankClip(length=408).std.AssumeFPS(FILE.clip_cut) + NCOP_YT_AVC.std.SelectEvery(5, [1, 2, 3, 4]),
    props=0
)

dims = (get_w(882), 882)
OPSTART, OPEND = 408, 2566
EDSTART, EDEND = 31768, 33926


@DEBUG.catch(op='@=')
@finalise_output
@initialise_input(bits=8)
def filtering(src: vs.VideoNode = FILE.clip_cut) -> vs.VideoNode:
    global DEBUG
    out = src

    rescale_mask = diff_rescale_mask(out, 882, Lanczos(5), thr=80, expand=2)
    # DEBUG[10] = rescale_mask

    out = depth(out, 32)

    src_op = depth(out, 32)[OPSTART:OPEND]
    yt_avc = initialise_clip(NCOP_YT_AVC.std.SelectEvery(5, [1, 2, 3, 4]), 32)
    yt_dblk = vsort_dpir(get_y(yt_avc), 100)
    yt_dblk = merge_chroma(yt_dblk, yt_avc)

    src_blurred = src_op.bilateralgpu.Bilateral(5, 1, radius=7)
    ref_blurred = yt_avc.bilateralgpu.Bilateral(5, 1, radius=7)
    diff_blur = core.std.Expr([src_op, src_blurred], 'x y -')
    undim = core.std.Expr([ref_blurred, diff_blur], 'x y +')
    # DEBUG <<= dict(CR=FILE.clip_cut[OPSTART:OPEND], AVC=yt_avc)
    # DEBUG <<= undim
    indices = list(zip([0] * out.num_frames, range(out.num_frames)))
    yt_avc_i = list(zip([1] * undim.num_frames, range(undim.num_frames)))
    undim_i = list(zip([2] * undim.num_frames, range(undim.num_frames)))
    indices[OPSTART + 1388:OPSTART + 1573] = undim_i[1388:1573]
    indices[OPSTART + 1642:OPSTART + 1705] = yt_avc_i[1642:1705]
    out = select_frames([out, yt_dblk, undim], indices)
    DEBUG @= out

    # Get ride of the weird dithering pattern
    denoise = knl_means_cl(out, [0.4, 0.4], tr=0, sr=2, simr=8)
    limit = core.akarin.Expr([denoise, out], 'x y 1.5 255 / - y 1 255 / + clamp')
    out = remap_rfs(limit, out, [(OPSTART + 1388, OPSTART + 1573), (OPSTART + 1642, OPSTART + 1705)])
    # DEBUG <<= out


    with YUVPlanes(out) as c:
        y = c.Y
        # TCunny masks ouputs values out of range
        line_mask = FDOGTCanny().get_mask(y).akarin.Expr('x 0 1 clamp')
        line_mask = expand(line_mask, 3, 3, mode=XxpandMode.ELLIPSE)
        line_mask = inpand(line_mask, 2, 2, mode=XxpandMode.ELLIPSE)
        # DEBUG <<= line_mask

        # Pretty smooth descale
        tdescale = Lanczos(5).descale(c.Y, *dims)
        downscale = Catrom().scale(c.Y, *dims)
        descale = core.std.Expr([tdescale, downscale], 'x y min')

        upscale = Nnedi3SS(shifter=Catrom()).scale(descale, *[d * 2 for d in dims])
        aaa = Eedi3SR(True, True, 0.2, 0.55, gamma=400, mdis=15).aa(upscale)

        rescale = Catrom().scale(aaa, 1920, 1080)
        c.Y = core.std.MaskedMerge(y, rescale, line_mask)
        # DEBUG <<= c.Y
    out = depth(c.clip, 16)

    with YUVPlanes(out) as c:
        y = c.Y
        # Restore lineart's derkiness
        dline_mask = TheToof2().get_mask(y)
        dark = core.std.MaskedMerge(y, toooon(y, 0.35), dline_mask)
        c.Y = dark
    out = c.clip

    # Remove halos left
    dehalo = fine_dehalo(out, rx=1.8, darkstr=0)
    out = dehalo
    clean = edge_cleaner(dehalo, 15, hot=True, smode=1)
    out = clean

    base = depth(src, 16)
    rescale_mask = depth(rescale_mask, 16)
    credit = out
    credit = remap_rfs(credit, core.std.MaskedMerge(out, base, rescale_mask),
                       [(OPSTART, OPEND), (2579, 2723)])
    credit = remap_rfs(credit, out, [(OPSTART + 1388, OPSTART + 1573), (OPSTART + 1642, OPSTART + 1705)])
    credit = remap_rfs(credit, depth(limit, 16), (EDSTART, EDEND))
    out = credit

    # Use this mask again
    thrs = [
        Thresholds(line_mask.std.BlankClip(), 0, 0, 0.04, 0.06),
        Thresholds(line_mask, 0.04, 0.6, 1, 1),
    ]
    line_mask = thresholding(*thrs, base=line_mask, guidance=line_mask).std.Expr('x 5.5 * 1 min')
    line_mask = core.std.BoxBlur(line_mask, 0, 2, 1, 2, 1)
    # DEBUG <<= line_mask
    # Source is pretty clean and doesn't really have banding but well
    deband = dumb3kdb(out, 31, 33, 24)
    out = core.std.MaskedMerge(deband, out, line_mask.resize.Point(format=vs.GRAY16))

    # Good old graigasm
    grain = Graigasm(**graigasm_args).graining(out)  # type: ignore
    out = grain

    eyecatch = remap_rfs(out, base, [(9504, 9540), (11114, 11147), (12120, 12156), (12540, 12612)])
    out = eyecatch

    return out


def vsort_dpir(clip: vs.VideoNode, strength: float) -> vs.VideoNode:
    return core.ort.Model(
        [clip, clip.std.BlankClip(format=vs.GRAYS, color=strength / 255)],
        r'C:\Users\Varde\AppData\Roaming\VapourSynth\vs-mlrt shit\models.v5\models\dpir\drunet_deblocking_grayscale.onnx',
        provider='CUDA', fp16=True
    )


if __name__ == '__main__':
    Encode(FILE, filtering()).run(NUM, VPath('Princess Connect! Re Dive S2 - 02 (BS11)_track_cut_{track_number}.m4a').to_str())
else:
    pass
    # filtering()
