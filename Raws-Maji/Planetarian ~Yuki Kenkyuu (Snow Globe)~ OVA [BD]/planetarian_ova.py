"""Planetarian script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path
from acsuite import eztrim

from adptvgrnMod import sizedgrn
import debandshit as dbs
import vardefunc as vdf
import havsfunc as hvf
import mvsfunc as mvf

from vsutil import depth, get_y, get_w, insert_clip
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core

class InfosBD(NamedTuple):
    path: str
    src: str
    src_clip: vs.VideoNode
    frame_start: int
    frame_end: int
    src_cut: vs.VideoNode
    a_src: str
    a_src_cut: str
    a_enc_cut: str
    name: str
    output: str
    chapter: str
    output_final: str


def infos_bd(path, frame_start, frame_end) -> InfosBD:
    src = path + '.m2ts'
    # src_clip = lvf.src(src, stream_index=0, ff_loglevel=4)
    src_clip = lvf.src(src, stream_index=0)
    src_cut = src_clip[frame_start:frame_end] if (frame_start or frame_end) else src_clip
    a_src = path + '_track_{}.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'[BDMV][210120] Planetarian ~Yuki Kenkyuu (Snow Globe)~ OVA Vol.1 Fin\PLANETARIAN_SNOWGLOVE\BDMV\STREAM\00000', 24, -24)


def eedi3_upscale(clip: vs.VideoNode, correct_shift: bool = True)-> vs.VideoNode:
    nnargs: Dict[str, Any] = dict(nsize=4, nns=4, qual=2, etype=1, pscrn=1)
    eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.8, gamma=1000, nrad=1, mdis=15)

    clip = clip.std.Transpose()
    clip = clip.eedi3m.EEDI3(0, True, sclip=clip.nnedi3.nnedi3(0, True, **nnargs), **eeargs)
    clip = clip.std.Transpose()
    clip = clip.eedi3m.EEDI3(0, True, sclip=clip.nnedi3.nnedi3(0, True, **nnargs), **eeargs)

    return core.resize.Bicubic(clip, src_top=.5, src_left=.5) if correct_shift else clip


def dumb3kdb(clip: vs.VideoNode, radius=16, strength=41):
    div = (strength - 1) % 16
    if strength < 17:
        return clip
    if div == 0:
        return clip.f3kdb.Deband(radius, strength, strength, strength, 0, 0, output_depth=16)
    lo_str = strength - div
    hi_str = strength - div + 16
    lo_clip = clip.f3kdb.Deband(radius, lo_str, lo_str, lo_str, 0, 0, output_depth=16)
    hi_clip = clip.f3kdb.Deband(radius, hi_str, hi_str, hi_str, 0, 0, output_depth=16)
    return core.std.Merge(lo_clip, hi_clip, (strength - lo_str) / 16)


def single_rate_antialiasing(clip: vs.VideoNode, rep: Optional[int] = None,
                             **eedi3_args: Any)-> vs.VideoNode:
    """Drop half the field with eedi3+nnedi3 and interpolate them.

    Args:
        clip (vs.VideoNode): Source clip.
        rep (Optional[int], optional): Repair mode. Defaults to None.

    Returns:
        vs.VideoNode: [description]
    """
    nnargs: Dict[str, Any] = dict(nsize=0, nns=3, qual=1)
    eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)
    eeargs.update(eedi3_args)

    eedi3_fun, nnedi3_fun = core.eedi3m.EEDI3, core.nnedi3.nnedi3

    flt = core.std.Transpose(clip)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)
    flt = core.std.Transpose(flt)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)

    if rep:
        flt = core.rgsf.Repair(flt, clip, rep)

    return flt


def smhdegrain(clip: vs.VideoNode, tr, th)-> vs.VideoNode:
    return hvf.SMDegrain(clip, tr=tr, thSAD=th, plane=4)

def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 16)
    out = src


    h = 900
    w = get_w(h)


    # Remove the grain
    ref = hvf.SMDegrain(out, tr=1, thSAD=300, plane=4)
    degrain = mvf.BM3D(out, sigma=[1.5, 1], radius1=1, ref=ref)
    degrain = insert_clip(degrain, smhdegrain(out[5539:5670], 2, 280), 5539)
    degrain = insert_clip(degrain, smhdegrain(out[5933:5992], 2, 200), 5933)
    degrain = insert_clip(degrain, smhdegrain(out[6115:6180], 2, 200), 6115)
    degrain = insert_clip(degrain, smhdegrain(out[6180:6281], 2, 200), 6180)
    degrain = insert_clip(degrain, smhdegrain(out[39303:39482], 2, 280), 39303)
    degrain = insert_clip(degrain, smhdegrain(out[40391:40837], 2, 200), 40391)
    degrain = insert_clip(degrain, smhdegrain(out[40908:41087], 2, 280), 40908)
    degrain = insert_clip(degrain, smhdegrain(out[41671:41791], 2, 280), 41671)
    degrain = insert_clip(degrain, smhdegrain(out[41791:41977], 2, 280), 41791)
    degrain = insert_clip(degrain, smhdegrain(out[41977:42073], 2, 280), 41977)

    degrain = insert_clip(degrain, smhdegrain(out[43083:44462], 2, 350), 43083)
    degrain = lvf.rfs(degrain, out, [(51749, 52387)])
    out = depth(degrain, 32)



    luma = get_y(out)
    line_mask = vdf.edge_detect(luma, 'kirsch', 0.075, (1, 1)).std.Median().std.Inflate()


    descale = core.descale.Debilinear(luma, w, h)
    upscale = eedi3_upscale(descale)
    antialias = single_rate_antialiasing(upscale, 13, alpha=0.2, beta=0.6, gamma=300, mdis=15).resize.Bicubic(src.width, src.height)

    rescale = core.std.MaskedMerge(luma, antialias, line_mask)
    merged = vdf.merge_chroma(rescale, out)
    out = depth(merged, 16)


    y = get_y(out)
    detail_light_mask = lvf.denoise.detail_mask(y.std.Median(), brz_a=2500, brz_b=1200)


    pf = out.std.Convolution([1] * 9).std.Merge(out, 0.45)

    diffdb = core.std.MakeDiff(out, pf)

    deband = dumb3kdb(pf, 16, 30)
    deband_b = dbs.f3kbilateral(pf, 20, 100)
    deband = lvf.rfs(deband, deband_b, [(43083, 44461)])

    deband = core.std.MergeDiff(deband, diffdb)
    deband = core.std.MaskedMerge(deband, out, detail_light_mask)

    deband = lvf.rfs(deband, out, [(51749, 52387)])
    out = deband


    sharp = hvf.LSFmod(out, strength=65, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True)
    out = sharp




    ref = get_y(out).std.PlaneStats()
    adgmask_a = core.adg.Mask(ref, 30)
    adgmask_b = core.adg.Mask(ref, 12)


    stgrain = sizedgrn(out, 0.1, 0.05, 1.00)
    stgrain = core.std.MaskedMerge(out, stgrain, adgmask_b)
    stgrain = core.std.MaskedMerge(out, stgrain, adgmask_a.std.Invert())

    dygrain = sizedgrn(out, 0.2, 0.05, 1.05, sharp=60, static=False)
    dygrain = core.std.MaskedMerge(out, dygrain, adgmask_a)
    grain = core.std.MergeDiff(dygrain, out.std.MakeDiff(stgrain))
    out = grain



    ref = src
    rescale_mask = vdf.drm(ref, h, mthr=65, sw=4, sh=4)
    credit = out
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, rescale_mask, 0),
                     [(0, 2956), (43104, 45749)])
    credit = lvf.rfs(credit, ref, [(45824, 50401), (52388, src.num_frames-1)])
    out = credit



    return depth(out, 10)


def do_encode(clip):
    """Compression with x26X"""
    if not os.path.isfile(JPBD.output):
        print('\n\n\nVideo encoding')
        bits = clip.format.bits_per_sample
        x265_cmd = f'x265 -o {JPBD.output} - --y4m' + ' '
        x265_cmd += f'--csv {JPBD.name}_log_x265.csv --csv-log-level 2' + ' '
        x265_cmd += '--preset slower' + ' '
        x265_cmd += f'--frames {clip.num_frames} --fps {clip.fps_num}/{clip.fps_den} --output-depth {bits}' + ' '
        x265_cmd += '--rd 3 --no-rect --no-amp --rskip 1 --tu-intra-depth 2 --tu-inter-depth 2 --tskip' + ' '
        x265_cmd += '--merange 48 --weightb' + ' '
        x265_cmd += '--no-strong-intra-smoothing' + ' '
        x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.25 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 60 --rc-lookahead 84 --bframes 16' + ' '
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 0.85 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=-1:-1 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'# + ' '

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()


    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '3:', JPBD.a_src.format(1), '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(1), JPBD.a_src_cut.format(1))

    print('\n\n\nAudio encoding')
    qaac_args = ['qaac', JPBD.a_src_cut.format(1), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(1)]
    subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder', '-print_format', 'default=nokey=1:noprint_wrappers=1', JPBD.a_enc_cut.format(1)]
    encoder_name = subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')
    f = open("tags_aac.xml", 'w')
    f.writelines(['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
                  '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}</String>', '</Simple>',
                  '</Tag>', '</Tags>'])
    f.close()

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', JPBD.output_final,
                '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', JPBD.output,
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1),
                '--chapter-language', 'jpn', '--chapters', JPBD.chapter]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    # Clean up
    files = [JPBD.a_src, JPBD.a_src_cut.format(1),
             JPBD.a_enc_cut.format(1), 'tags_aac.xml']
    for file in files:
        if os.path.exists(file):
            os.remove(file)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
