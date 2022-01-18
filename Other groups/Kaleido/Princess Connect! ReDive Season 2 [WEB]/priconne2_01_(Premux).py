"""PriConne S2 Script"""

import vapoursynth as vs
from debandshit import dumb3kdb
from lvsfunc.kernels import Catrom, Lanczos
from vardautomation import X265, FileInfo, get_vs_core
from vardefunc import (
    DebugOutput, DuplicateFrame, Eedi3SR, Graigasm, Nnedi3SS, Thresholds, YUVPlanes,
    diff_rescale_mask, finalise_output, initialise_input, remap_rfs, thresholding
)
from vsdenoise import knl_means_cl
from vsmask.edge import FDOGTCanny
from vsmask.util import XxpandMode, expand, inpand
from vsutil import depth, get_w

from priconne_common import TheToof2, edge_cleaner, fine_dehalo, graigasm_args, toooon

NUM = __file__[-14:-12]

core = get_vs_core(range(0, 24, 2))

FILE = FileInfo(f'raws/[FeelsBadSubs] Princess Connect! Re - Dive Season 2 - {NUM} [1080p].mkv')
FILE.trims_or_dfs = [(0, None), DuplicateFrame(FILE.clip_cut.num_frames - 1)]

DEBUG = DebugOutput(FILE.clip_cut, props=0)

dims = (get_w(882), 882)


@DEBUG.catch
@finalise_output
@initialise_input(bits=8)
def filtering(src: vs.VideoNode = FILE.clip_cut) -> vs.VideoNode:
    global DEBUG
    out = src

    rescale_mask = diff_rescale_mask(out, 882, Lanczos(5), thr=105)
    out = depth(out, 32)

    # Get ride of the weird dithering pattern
    denoise = knl_means_cl(out, [0.4, 0.4], tr=0, sr=2, simr=8)
    limit = core.akarin.Expr([denoise, out], 'x y 1.5 255 / - y 1 255 / + clamp')
    out = limit
    DEBUG <<= out


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

        upscale = Nnedi3SS(opencl=True, shifter=Catrom()).scale(descale, *[d * 2 for d in dims])
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
        # DEBUG <<= (y, toooon(y, 0.25))
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
                       [(0, 2118), (27763, 31107), (31438, 33925)])
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

    eyecatch = remap_rfs(out, base, [(12059, 12095), (13825, 13861), (33925, None)])
    out = eyecatch

    return out


if __name__ == '__main__':
    enc = X265('priconne_common/x265_settings_web')
    enc.resumable = True
    enc.run_enc(filtering(), FILE)
else:
    pass
    # filtering()
