"""ReZero S2 script"""
__author__ = 'Vardë'

import os
import sys
import shlex
import subprocess
from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path
from acsuite import eztrim

from adptvgrnMod import adptvgrnMod
import debandshit as dbs
import vardefunc as vdf
import awsmfunc as awf
import mvsfunc as mvf
import G41Fun as gf
import placebo

from vsutil import depth, get_y, get_w, iterate, insert_clip
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
    src_clip = lvf.src(src, stream_index=0, ff_loglevel=4)
    src_cut = src_clip[frame_start:frame_end] if (frame_start or frame_end) else src_clip
    a_src = path + '_track_{}.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = '_chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'REZERO_2ND_DISC3\BDMV\STREAM\00001', 0, 34766)
JPBD_NCOP = infos_bd(r'REZERO_2ND_DISC1\BDMV\STREAM\00018', 0, -24)


def hybrid_denoise(clip: vs.VideoNode, knlm_h: float = 0.5, sigma: float = 2,
                   knlm_args: Optional[Dict[str, Any]] = None,
                   bm3d_args: Optional[Dict[str, Any]] = None)-> vs.VideoNode:
    """Denoise luma with BM3D and chroma with knlmeansCL

    Args:
        clip (vs.VideoNode): Source clip.
        knlm_h (float, optional): h parameter in knlm.KNLMeansCL. Defaults to 0.5.
        sigma (float, optional): Sigma parameter in mvf.BM3D. Defaults to 2.
        knlm_args (Optional[Dict[str, Any]], optional): Optional extra arguments for knlm.KNLMeansCL. Defaults to None.
        bm3d_args (Optional[Dict[str, Any]], optional): Optional extra arguments for mvf.BM3D. Defaults to None.

    Returns:
        vs.VideoNode: [description]
    """
    knargs = dict(a=2, d=3, device_type='gpu', device_id=0, channels='UV')
    if knlm_args is not None:
        knargs.update(knlm_args)

    b3args = dict(radius1=1, profile1='fast')
    if bm3d_args is not None:
        b3args.update(bm3d_args)

    luma = get_y(clip)
    luma = mvf.BM3D(luma, sigma, **b3args)
    chroma = core.knlm.KNLMeansCL(clip, h=knlm_h, **knargs)

    return vdf.merge_chroma(luma, chroma)



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

    eedi3_fun, nnedi3_fun = core.eedi3m.EEDI3, core.nnedi3cl.NNEDI3CL

    flt = core.std.Transpose(clip)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)
    flt = core.std.Transpose(flt)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)

    if rep:
        flt = core.rgsf.Repair(flt, clip, rep)

    return flt

def line_mask_func(clip: vs.VideoNode)-> vs.VideoNode:
    coord = [1, 2, 1, 2, 2, 1, 2, 1]
    mask = core.retinex.MSRCP(depth(clip, 16), sigma=[40, 150, 275], upper_thr=0.0025)
    mask = gf.EdgeDetect(mask, 'FDOG', multi=1.5)
    return mask.std.Maximum(coordinates=coord).std.Minimum(coordinates=coord)

def detail_dark_mask_func(clip: vs.VideoNode, brz_a: int, brz_b: int)-> vs.VideoNode:
    ret = core.retinex.MSRCP(clip, sigma=[100, 250, 800], upper_thr=0.005)
    return lvf.denoise.detail_mask(ret, brz_a=brz_a, brz_b=brz_b)


def mask_opening(ref: vs.VideoNode)-> vs.VideoNode:
    mask = core.imwri.Read('rezeros2_ncop_0-16.png')*17 + core.imwri.Read('rezeros2_ncop_17-18.png')*2
    mask += core.imwri.Read('rezeros2_ncop_19.png')
    mask += core.imwri.Read('rezeros2_ncop_20.png')
    mask += core.imwri.Read('rezeros2_ncop_21.png')
    mask += core.imwri.Read('rezeros2_ncop_22.png')
    mask += core.imwri.Read('rezeros2_ncop_23.png')
    mask += core.imwri.Read('rezeros2_ncop_24.png')
    mask += core.imwri.Read('rezeros2_ncop_25.png')
    mask += core.imwri.Read('rezeros2_ncop_26.png')
    mask += core.imwri.Read('rezeros2_ncop_27.png')
    mask += core.imwri.Read('rezeros2_ncop_28.png')
    mask += core.imwri.Read('rezeros2_ncop_29.png')
    mask += core.imwri.Read('rezeros2_ncop_30.png')
    mask += core.imwri.Read('rezeros2_ncop_31.png')
    mask += core.imwri.Read('rezeros2_ncop_32.png')
    mask += core.imwri.Read('rezeros2_ncop_33.png')
    mask += core.imwri.Read('rezeros2_ncop_34.png')
    mask += core.imwri.Read('rezeros2_ncop_35.png')
    mask += core.imwri.Read('rezeros2_ncop_36.png')
    mask += core.imwri.Read('rezeros2_ncop_37.png')
    mask += core.imwri.Read('rezeros2_ncop_38.png')

    ref = ref.std.BlankClip(format=vs.GRAY16)
    mask = core.resize.Bicubic(mask, format=vs.GRAY16, matrix_s='709').std.AssumeFPS(ref)

    return insert_clip(ref, mask, OPSTART)



OPSTART, OPEND = 1630, 3788


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    h = 720
    w = get_w(h)


    edgesfix = awf.bbmod(src, 1, 1, 1, 1, 48, 500)
    out = edgesfix


    clip = depth(out, 32)
    denoise = hybrid_denoise(clip, 0.45, 1.5)
    out = denoise


    luma = get_y(out)
    line_mask = line_mask_func(luma)

    descale = core.descale.Debilinear(luma, w, h)
    upscale = vdf.nnedi3_upscale(descale, pscrn=1)
    antialias = single_rate_antialiasing(upscale, 13, alpha=0.4, beta=0.3, gamma=400, mdis=15).resize.Bilinear(src.width, src.height)

    rescale = core.std.MaskedMerge(luma, antialias, depth(line_mask, 32))
    merged = vdf.merge_chroma(rescale, out)
    out = depth(merged, 16)



    preden = core.knlm.KNLMeansCL(get_y(out), h=0.75, a=2, d=3, device_type='gpu', device_id=0)
    detail_dark_mask = detail_dark_mask_func(preden, brz_a=8000, brz_b=6000)
    detail_light_mask = lvf.denoise.detail_mask(preden, brz_a=2500, brz_b=1200)
    detail_mask = core.std.Expr([detail_dark_mask, detail_light_mask], 'x y +').std.Median()
    detail_mask_grow = iterate(detail_mask, core.std.Maximum, 2)
    detail_mask_grow = iterate(detail_mask_grow, core.std.Inflate, 2).std.Convolution([1, 1, 1, 1, 1, 1, 1, 1, 1])

    detail_mask = core.std.Expr([preden, detail_mask_grow, detail_mask], f'x {32<<8} < y z ?')


    op_mask = mask_opening(out)
    op_mask = iterate(op_mask, core.std.Deflate, 2)

    deband_a = dbs.f3kpf(out, 17, 36, 42, thr=0.5, elast=2, thrc=0.2)
    deband_b = placebo.deband(out, 18, 5.5, 2, 0)
    deband_c = placebo.deband(out, 22, 10, 3, 0)

    deband = core.std.MaskedMerge(deband_a, deband_b, preden)
    deband = core.std.MaskedMerge(deband, out, detail_mask)


    deband = lvf.rfs(deband, core.std.MaskedMerge(deband_c, deband, op_mask), [(OPSTART+0, OPSTART+38)])
    deband = lvf.rfs(deband, deband_b, [(OPSTART+236, OPSTART+284)])
    deband = lvf.rfs(deband, deband_c, [(OPSTART+1934, OPSTART+1944)])
    deband = core.neo_f3kdb.Deband(deband, preset='depth', grainy=24, grainc=24)
    out = deband


    ref, src, src_ncop = [depth(x, 16) for x in [denoise, src, JPBD_NCOP.src_cut]]
    credit = out
    credit = lvf.rfs(credit, ref, [(33725, 33797), (34719, src.num_frames-1)])
    out = credit


    src_c, src_ncop = [c.knlm.KNLMeansCL(a=7, h=35, d=0, device_type='gpu') for c in [src, src_ncop]]
    opening_mask = vdf.dcm(out, src_c[OPSTART:OPEND+1], src_ncop[:OPEND-OPSTART+1], OPSTART, OPEND, 4, 4).std.Inflate()
    credit = lvf.rfs(out, core.std.MaskedMerge(out, ref, opening_mask), [(OPSTART, OPEND)])
    out = credit



    grain = adptvgrnMod(out, 0.3, 0.15, 1.25, luma_scaling=8, sharp=80, static=False, lo=19)
    out = grain



    return depth(out, 10)


def do_encode(clip):
    """Compression with x26X"""
    vdf.generate_keyframes(JPBD.src_cut, JPBD.name + '_keyframes.txt')

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
        x265_cmd += '--psy-rd 2.0 --psy-rdoq 1.5 --no-open-gop --keyint 240 --min-keyint 24 --scenecut 60 --rc-lookahead 84 --bframes 16' + ' '
        x265_cmd += '--crf 15 --aq-mode 3 --aq-strength 1.15 --cbqpoffs -3 --crqpoffs -3 --qcomp 0.70' + ' '
        x265_cmd += f'--qpfile {JPBD.name}_keyframes.txt' + ' '
        x265_cmd += '--deblock=-1:-1 --no-sao --no-sao-non-deblock' + ' '
        # x265_cmd += f'--zones {OPSTART},{OPEND},b=1.45' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'# + ' '

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '3:', JPBD.a_src.format(1), '4:', JPBD.a_src.format(2), '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(1), JPBD.a_src_cut.format(1))
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src.format(2), JPBD.a_src_cut.format(2))

    print('\n\n\nAudio encoding')
    qaac_args = ['qaac', JPBD.a_src_cut.format(1), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(1)]
    subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')
    qaac_args = ['qaac', JPBD.a_src_cut.format(2), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(2)]
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
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0 Commentary', '--language', '0:jpn', JPBD.a_enc_cut.format(2),
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
