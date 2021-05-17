"""Midnight Crazy Trail script"""
__author__ = 'Vardë'

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import vardefunc as vdf
import havsfunc as hvf
import awsmfunc as awf
import placebo
import xvs

from vsutil import depth, get_y, split, join
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core

X264 = 'x264_tmod'

NAME = Path(sys.argv[0]).stem
OUTPUT = NAME + '.264'

VIDEO_FILE = 'mct.mkv'
CLIP_SRC = lvf.src(VIDEO_FILE)
vdf.set_ffms2_log_level('warning')
AUDIO_SRC = Path(VIDEO_FILE).stem + '.eac3'

X264_ARGS = dict(
    threads=12, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='0:0', me='umh', subme=10, psy_rd='0.90:0.00', merange=32,
    keyint=240, min_keyint=23, rc_lookahead=84, crf=15, qcomp=0.72, aq_mode=3, aq_strength=0.85,
    qpstep=7
)



def knlm_denoise(clip: vs.VideoNode, h: Tuple[float] = (1.2, 0.8), knlm_args: Optional[Dict[str, Any]] = None):
    knargs = dict(a=2, d=3, device_type='gpu', device_id=0)
    if knlm_args is not None:
        knargs.update(knlm_args)

    clip = core.knlm.KNLMeansCL(clip, h=h[0], channels='Y', **knargs)
    clip = core.knlm.KNLMeansCL(clip, h=h[1], channels='UV', **knargs)

    return clip

def do_filter():
    """Vapoursynth filtering"""
    scene_change = []
    with open('k.log') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            scene_change.append(int(line[0]))


    src = CLIP_SRC

    # Lol?
    border = awf.bbmod(src, 4, thresh=128, blur=15, y=True, u=True, v=True)
    border = lvf.ef(border, [6, 3, 3], radius=[12, 6, 6])
    border = lvf.rfs(src, border, scene_change)
    out = depth(border, 16)



    # joletb has stronk eyes
    planes = split(out)
    planes[1], planes[2] = [core.resize.Spline16(plane, src_top=-0.25) for plane in planes[1:]]
    out = join(planes)


    # qual=2 produces weird artefacts and a stronger alpha/beta/gamma smudges details.
    new_fields = core.eedi3m.EEDI3(
        out, 1, alpha=0.25, beta=0.3, gamma=400, nrad=3, mdis=20, vcheck=3,
        sclip=core.nnedi3.nnedi3(out, 1, nsize=3, nns=3, qual=1, pscrn=4)
    )
    out = new_fields



    # omegartifacted frames
    def freeze_frame_after(clip, frame):
        return core.std.FreezeFrames(clip, frame, frame, frame+1)
    def freeze_frame_before(clip, frame):
        return core.std.FreezeFrames(clip, frame, frame, frame-1)

    cursed_frames = [2326, 8907, 12211, 12551, 13990, 14403, 15462, 17673, 19382, 23099,
                     23738, 24031, 24802, 25083]
    for cursed_frame in cursed_frames:
        out = freeze_frame_after(out, cursed_frame)
    cursed_frames = [5695, 9115, 9116, 17671, 18432]
    for cursed_frame in cursed_frames:
        out = freeze_frame_before(out, cursed_frame)



    # omegartifacted frames ²
    denoise_li = hvf.SMDegrain(out, thSAD=200)
    denoise_hi = hvf.SMDegrain(out, thSAD=350)
    denoise = lvf.rfs(denoise_li, denoise_hi, scene_change)


    def hard_denoise(clip):
        clip = hvf.SMDegrain(clip, thSAD=500)
        clip = knlm_denoise(clip, (2, 2), dict(a=8))
        return clip

    cursed_frames = [595, 1191, 2643, 2663, 2664, 2665, 2666, 2667, 2671, 2672, 2674, 2675,
                     2679, 3999, 4419, 6351, 6355, 6547, 8906, 11731, 14176, 14177, 14178, 14179,
                     18430, 18435, 18437, 18438, 18439, 27766]
    cursed_frames += range(10767, 10776)
    cursed_frames += range(25013, 25018)
    cursed_frames += range(27663, 27668)
    cursed_frames += range(29642, 29646)
    cursed_frames += range(31384, 31388)

    uncursed = hard_denoise(out)
    denoise = lvf.rfs(denoise, uncursed, cursed_frames)
    out = denoise


    # It helps to fix the the aliasing left
    antialias = lvf.sraa(out, rep=13)
    out = antialias


    # Compensative line darkening and sharpening
    luma = get_y(out)
    darken = hvf.Toon(luma, 0.20)
    darken_mask = core.std.Expr(
        [core.std.Convolution(luma, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False),
         core.std.Convolution(luma, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False)],
        ['x y max {neutral} / 0.86 pow {peak} *'.format(neutral=1 << (luma.format.bits_per_sample-1),
                                                        peak=(1 << luma.format.bits_per_sample)-1)])
    darken = core.std.MaskedMerge(luma, darken, darken_mask)


    # Slight sharp through CAS
    sharp = hvf.LSFmod(darken, strength=65, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True)
    out = vdf.merge_chroma(sharp, out)




    # Chroma planes are pure crap
    chromableed = xvs.WarpFixChromaBlend(out, 96, depth=8)
    out = chromableed



    detail_mask = lvf.denoise.detail_mask(out, brz_a=2700, brz_b=1500)
    deband = placebo.deband(out, 17, 5.75, grain=4)
    deband_b = placebo.deband(out, 24, 8, 2, grain=4)
    deband_c = placebo.deband(out, 17, 8, grain=4)
    deband_d = placebo.deband(out, 20, 12, 3, grain=4)

    deband = lvf.rfs(deband, deband_b, [(4596, 4669), (23036, 23098)])
    deband = lvf.rfs(deband, deband_c, [(1646, 1711), (29768, 29840), (29932, 30037), (30163, 30243)])
    deband = lvf.rfs(deband, deband_d, [(1712, 1830)])

    deband = core.std.MaskedMerge(deband, out, detail_mask)
    out = deband


    return depth(out, 10)



def do_encode(clip: vs.VideoNode):
    """Compression with x264"""
    vdf.encode(clip, X264, OUTPUT, **X264_ARGS, frames=clip.num_frames)
    # ffv1_args = [
    #     'ffmpeg', '-i', '-', '-vcodec', 'ffv1', '-coder', '1', '-context', '0',
    #     '-g', '1', '-level', '3', '-threads', '8',
    #     '-slices', '24', '-slicecrc', '1', NAME + "_lossless.mkv"
    # ]
    # print("Encoder command: ", " ".join(ffv1_args), "\n")
    # import subprocess
    # process = subprocess.Popen(ffv1_args, stdin=subprocess.PIPE)
    # clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
    #             print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
    # process.communicate()




if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
