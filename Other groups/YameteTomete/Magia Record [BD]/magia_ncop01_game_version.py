"""Magia Record NCOP Game version script"""
__author__ = 'Vardë'

import sys
from pathlib import Path

import vardefunc as vdf
import havsfunc as hvf
import G41Fun as gf
import placebo
import xvs

from vsutil import depth, split, join
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core

X264 = 'x264_tmod'

NAME = Path(sys.argv[0]).stem
OUTPUT = NAME + '.264'

VIDEO_FILE = 'magia_op_2.mkv'
CLIP_SRC = lvf.src(VIDEO_FILE, force_lsmas=True)
# vdf.set_ffms2_log_level('warning')
# AUDIO_SRC = Path(VIDEO_FILE).stem + '.eac3'

X264_ARGS = dict(
    threads=12, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-1:-1', me='tesa', subme=10, psy_rd='1.0:0.00', merange=48,
    keyint=240, min_keyint=12, rc_lookahead=84, crf=14.75, qcomp=0.72, aq_mode=3, aq_strength=0.85,
    qpstep=7, scenecut=45
)


def do_filter():
    """Vapoursynth filtering"""
    src = CLIP_SRC
    out = depth(src, 16)


    clip = out
    clip = core.std.FreezeFrames(clip, 1432, 1433, 1434)
    clip = core.std.FreezeFrames(clip, 1503, 1514, 1515)
    out = clip



    planes = split(out)
    planes[1], planes[2] = [core.resize.Spline16(plane, src_left=-0.25) for plane in planes[1:]]
    out = join(planes)



    # qual=2 produces weird artefacts and a stronger alpha/beta/gamma smudges details.
    new_fields = core.eedi3m.EEDI3(
        out, 1, alpha=0.4, beta=0.5, gamma=300, nrad=3, mdis=20, vcheck=3,
        sclip=core.nnedi3.nnedi3(out, 1, nsize=3, nns=3, qual=1, pscrn=4)
    )
    out = new_fields



    denoise = gf.MCDegrainSharp(out, tr=3, bblur=0.65, csharp=lambda x: hvf.LSFmod(x, strength=85, Smode=3, Lmode=1, edgemode=0),
                                thSAD=285, rec=True)
    out = denoise



    edge_cleaner = hvf.EdgeCleaner(out, 35, rmode=13, smode=1, hot=True)
    out = edge_cleaner



    antialias = lvf.sraa(out, 2, 13, downscaler=core.resize.Bicubic, alpha=0.25, beta=0.35, gamma=400)
    antialias_a = lvf.sraa(out, 1.4, 9, downscaler=core.resize.Bicubic)
    antialias = lvf.rfs(antialias, antialias_a, [(1223, 1229)])
    out = antialias



    chromableed = xvs.WarpFixChromaBlend(out, 72, depth=8)
    out = chromableed




    detail_mask = lvf.denoise.detail_mask(out, brz_a=2300, brz_b=1000)
    deband = placebo.deband(out, 17, 4.25, grain=6)
    deband_a = placebo.deband(out, 17, 8, 2, grain=6)
    deband_b = placebo.deband(out, 17, 6, grain=6)
    deband = lvf.rfs(deband, deband_a, [(1230, 1330)])
    deband = lvf.rfs(deband, deband_b, [(1678, 1889)])


    deband = core.std.MaskedMerge(deband, out, detail_mask)
    out = deband


    return depth(out, 10)



def do_encode(clip: vs.VideoNode):
    """Compression with x264"""
    vdf.encode(clip, X264, OUTPUT, **X264_ARGS, frames=clip.num_frames)




if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
