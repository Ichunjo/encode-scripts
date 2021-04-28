"""Assault Lily Bouquet script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from typing import NamedTuple
from pathlib import Path
from functools import partial
from acsuite import eztrim

from adptvgrnMod import sizedgrn
import debandshit as dbs
import vardefunc as vdf
import havsfunc as hvf
import mvsfunc as mvf
import zzfunc as zzf

from vsutil import depth, get_y, fallback, split, join
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

JPBD = infos_bd(r'[BDMV][210127][BRMM-10341][アサルトリリィ BOUQUET 1]\BRMM_10341\BDMV\STREAM\00000', 0, 34645)
JPBD_NCOP = infos_bd(r'[BDMV][210127][BRMM-10341][アサルトリリィ BOUQUET 1]\BRMM_10341\BDMV\STREAM\00003', 0, -26)
JPBD_NCED = infos_bd(r'[BDMV][210127][BRMM-10341][アサルトリリィ BOUQUET 1]\BRMM_10341\BDMV\STREAM\00004', 0, -39)


# Code stolen from Zastin
def mt_xxpand_multi(clip, sw=1, sh=None, mode='square', planes=None, start=0, M__imum=core.std.Maximum, **params):
    sh = fallback(sh, sw)
    planes = list(range(clip.format.num_planes)) if planes is None else [planes] if isinstance(planes, int) else planes

    if mode == 'ellipse':
        coordinates = [[1]*8, [0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0]]
    elif mode == 'losange':
        coordinates = [[0, 1, 0, 1, 1, 0, 1, 0]] * 3
    else:
        coordinates = [[1]*8] * 3

    clips = [clip]

    end = min(sw, sh) + start

    for x in range(start, end):
        clips += [M__imum(clips[-1], coordinates=coordinates[x % 3], planes=planes, **params)]

    for x in range(end, end + sw - sh):
        clips += [M__imum(clips[-1], coordinates=[0, 0, 0, 1, 1, 0, 0, 0], planes=planes, **params)]

    for x in range(end, end + sh - sw):
        clips += [M__imum(clips[-1], coordinates=[0, 1, 0, 0, 0, 0, 1, 0], planes=planes, **params)]

    return clips

maxm = partial(mt_xxpand_multi, M__imum=core.std.Maximum)
minm = partial(mt_xxpand_multi, M__imum=core.std.Minimum)


# Code stolen from Zastin
def morpho_mask(clip: vs.VideoNode):
    # y, u, v = split(clip)
    y = get_y(clip)
    # stats = y.std.PlaneStats()
    # agm3 = core.adg.Mask(stats, 3)
    # agm6 = core.adg.Mask(stats, 6)

    ymax = maxm(y, sw=40, mode='ellipse')
    ymin = minm(y, sw=40, mode='ellipse')
    # umax = maxm(u, sw=16, mode='ellipse')
    # umin = minm(u, sw=16, mode='ellipse')
    # vmax = maxm(v, sw=12, mode='ellipse')
    # vmin = minm(v, sw=12, mode='ellipse')


    thr = 3.2 * 256
    ypw0 = y.std.Prewitt()
    ypw = ypw0.std.Binarize(thr).rgvs.RemoveGrain(11)


    rad = 3
    thr = 2.5 * 256
    yrangesml = core.std.Expr([ymax[3], ymin[3]], 'x y - abs')
    yrangesml = yrangesml.std.Binarize(thr).std.BoxBlur(0, 2, 1, 2, 1)


    rad = 16
    thr = 4 * 256
    yrangebig0 = core.std.Expr([ymax[rad], ymin[rad]], 'x y - abs')
    yrangebig = yrangebig0.std.Binarize(thr)
    yrangebig = minm(yrangebig, sw=rad * 3 // 4, threshold=65536//((rad * 3 // 4) + 1), mode='ellipse')[-1]
    yrangebig = yrangebig.std.BoxBlur(0, rad//4, 1, rad//4, 1)


    rad = 30
    thr = 1 * 256
    ymph = core.std.Expr([y, maxm(ymin[rad], sw=rad, mode='ellipse')[rad],
                          minm(ymax[rad], sw=rad, mode='ellipse')[rad]], 'x y - z x - max')
    ymph = ymph.std.Binarize(384)
    ymph = ymph.std.Minimum().std.Maximum()
    ymph = ymph.std.BoxBlur(0, 4, 1, 4, 1)


    grad_mask = zzf.combine([ymph, yrangesml, ypw])
    # grain_mask = core.std.Expr([yrangebig, grad_mask, ypw0.std.Binarize(2000).std.Maximum().std.Maximum()], '65535 y - x min z -').std.BoxBlur(0,16,1,16,1)

    return grad_mask, yrangebig



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



def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src += src[-1]
    src = depth(src, 16)
    out = src

    opstart, opend = 792, 2948
    edstart, edend = 32248, 34404


    # Remove the noise
    ref = hvf.SMDegrain(out, tr=1, thSAD=300, plane=4)
    denoise = mvf.BM3D(out, sigma=[2, 1], radius1=1, ref=ref)
    out = denoise




    # Limited antialiasing
    y = get_y(out)
    lineart = core.std.Sobel(y).std.Binarize(75<<8).std.Maximum().std.Inflate()
    aa = lvf.sraa(y, 2, 13, downscaler=partial(core.resize.Bicubic, filter_param_a=-0.5, filter_param_b=0.25))
    sharp = hvf.LSFmod(aa, strength=70, Smode=3, edgemode=0, source=y)


    y = core.std.MaskedMerge(y, sharp, lineart)
    out = vdf.merge_chroma(y, out)





    # Deband masks
    planes = split(out)

    preden = core.dfttest.DFTTest(out, sigma=14.0, sigma2=10.0, sbsize=1, sosize=0).rgvs.RemoveGrain(3)
    grad_mask, yrangebig = morpho_mask(preden)


    rgb = core.resize.Bicubic(preden, 960, 540, format=vs.RGB48).std.SetFrameProp('_Matrix', delete=True)
    chroma_masks = [core.std.Prewitt(p).std.Binarize(5500).rgvs.RemoveGrain(11) for p in split(rgb) + split(preden)[1:]]
    chroma_mask = core.std.Expr(chroma_masks, 'x y + z + a + b +').std.Maximum()
    chroma_mask = core.std.Merge(chroma_mask, chroma_mask.std.Minimum()).std.BoxBlur(0, 1, 1, 1, 1)

    fdogmask = vdf.edge_detect(preden.std.Median(), 'FDOG', 8250, (4, 1)).std.BoxBlur(0, 2, 1, 2, 1)
    detail_mask = detail_mask_func(planes[0].std.BoxBlur(0, 1, 1, 1, 1), brz_a=3300, brz_b=2000)


    # Debanding stages
    debands = [None] * 3
    deband_y_strong = dbs.f3kbilateral(planes[0], 16, 65)
    deband_y_normal = dumb3kdb(planes[0], 16, 36)
    deband_y_light = core.std.MaskedMerge(dumb3kdb(planes[0], 14, 32), planes[0], detail_mask)
    debands[1] = dbs.f3kbilateral(planes[1], 12, 50)
    debands[2] = dbs.f3kbilateral(planes[2], 12, 50)


    debands[0] = core.std.MaskedMerge(deband_y_strong, deband_y_normal, grad_mask)
    debands[0] = core.std.MaskedMerge(debands[0], deband_y_light, yrangebig)
    debands[1], debands[2] = [core.std.MaskedMerge(debands[i], planes[i], chroma_mask) for i in range(1, 3)]
    deband = join(debands)
    deband = lvf.rfs(deband, core.std.MaskedMerge(dbs.f3kbilateral(out, 20, 97), out, fdogmask), [(opstart+2019, opstart+2036), (opstart+1504, opstart+1574)])
    out = deband




    # Regraining
    ref = get_y(out).std.PlaneStats()
    adgmask_a = core.adg.Mask(ref, 25)
    adgmask_b = core.adg.Mask(ref, 10)

    stgrain_a = core.grain.Add(out, 0.1, 0, seed=333)
    stgrain_a = core.std.MaskedMerge(out, stgrain_a, adgmask_b.std.Invert())

    stgrain_b = sizedgrn(out, 0.2, 0.1, 1.15, sharp=80, seed=333)
    stgrain_b = core.std.MaskedMerge(out, stgrain_b, adgmask_b)
    stgrain_b = core.std.MaskedMerge(out, stgrain_b, adgmask_a.std.Invert())
    stgrain = core.std.MergeDiff(stgrain_b, out.std.MakeDiff(stgrain_a))

    dygrain = sizedgrn(out, 0.3, 0.1, 1.25, sharp=80, static=False, seed=333)
    dygrain = core.std.MaskedMerge(out, dygrain, adgmask_a)
    grain = core.std.MergeDiff(dygrain, out.std.MakeDiff(stgrain))
    out = grain




    # Credits OP and ED
    ref = src
    src_ncop, src_nced = [depth(c, 16) for c in [JPBD_NCOP.src_cut, JPBD_NCED.src_cut]]
    src_c, src_ncop, src_nced = [c.std.BoxBlur(0, 2, 1, 2, 1) for c in [src, src_ncop, src_nced]]

    opening_mask = vdf.dcm(out, src_c[opstart:opend+1], src_ncop[:opend-opstart+1], opstart, opend, 2, 2)
    ending_mask = vdf.dcm(out, src_c[edstart:edend+1], src_nced[:edend-edstart+1], edstart, edend, 2, 2)
    credit_mask = core.std.Expr([opening_mask, ending_mask], 'x y +').std.Inflate()

    credit = lvf.rfs(out, core.std.MaskedMerge(out, ref, credit_mask), [(opstart, opend), (edstart, edend)])
    out = credit





    # Fix letterboxes ending
    crop = core.std.Crop(out, 0, 0, 132, 132).edgefixer.ContinuityFixer(0, 1, 0, 0)
    crop = core.std.AddBorders(crop, 0, 0, 132, 132)
    out = lvf.rfs(out, crop, [(edstart, edend)])




    return depth(out, 10).std.Limiter(16<<2, [235<<2, 240<<2], [0, 1, 2])



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
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 1.0 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=-1:-1 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'# + ' '

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()


    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src.format(1), '-log=NUL']
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
