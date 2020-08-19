"""Symphogear G script"""
__author__ = 'Vardë'

import os
import sys
import subprocess
from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path
from acsuite import eztrim

from adptvgrnMod import adptvgrnMod
from vsTAAmbk import TAAmbk
import debandshit as dbs
import vardefunc as vdf
import muvsfunc as muvf
import havsfunc as hvf
import mvsfunc as mvf
import G41Fun as gf
import rekt as rkt
import xvs


from vsutil import depth, get_y, get_w, iterate
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
    src_clip = lvf.src(path + '.m2ts')
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'戦姫絶唱シンフォギアＧ\[BDMV][131002] 戦姫絶唱シンフォギアG 1\KIXA_90350\BDMV\STREAM\00003', 0, -24)
X265 = 'x265'

def hybrid_denoise(clip: vs.VideoNode, knlm_h: float = 0.5, sigma: float = 2,
                   knlm_args: Optional[Dict[str, Any]] = None,
                   bm3d_args: Optional[Dict[str, Any]] = None)-> vs.VideoNode:
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

def _rescale_mask(original: vs.VideoNode, upscaled: vs.VideoNode, thr: float)-> vs.VideoNode:
    mask = core.std.Expr([original, upscaled], 'x y - abs').std.Binarize(thr)
    mask = iterate(mask, core.std.Maximum, 3)
    return iterate(mask, core.std.Inflate, 2)

def single_rate_antialiasing(clip: vs.VideoNode, rep: Optional[int] = None,
                             **eedi3_args: Any)-> vs.VideoNode:
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

def _aa_extra(clip: vs.VideoNode)-> vs.VideoNode:
    downscaler_fun = lambda x, wid, hei: \
        core.fmtc.resample(x, wid, hei, kernel='gauss', invks=True, invkstaps=2, taps=1, a1=32)
    luma = get_y(clip)
    antiaa = lvf.sraa(luma, 2, height=720)
    antiaa = TAAmbk(antiaa, 'Eedi3SangNom', mtype=2, alpha=0.75, beta=0.2, gamma=40, aa=96)
    antiaa = lvf.sraa(antiaa, 2, height=clip.height, downscaler=downscaler_fun)
    return vdf.merge_chroma(antiaa, clip)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 32)
    src += src[-1]

    h = 720
    w = get_w(h)
    b, c = vdf.get_bicubic_params('robidoux')
    full_stuff = [(313, 1993), (15602, 15757), (19240, 19295)]




    denoise = hybrid_denoise(src, 0.5, 2)
    denoise_b = hybrid_denoise(src, 0.6, 7.5, bm3d_args=dict(profile1='high'))
    deband = lvf.rfs(denoise, denoise_b, [(4002, 4061)])
    out = denoise


    luma = get_y(out)
    line_mask = vdf.edge_detect(luma, 'FDOG', 0.05, (1, 1))


    descale = core.descale.Debicubic(luma, w, h, b, c)
    upscale = vdf.fsrcnnx_upscale(descale, None, descale.height*2, 'shaders/FSRCNNX_x2_56-16-4-1.glsl', core.resize.Point)

    antialias = single_rate_antialiasing(upscale, 13, alpha=0.3, beta=0.45, gamma=320, mdis=18)

    scaled = muvf.SSIM_downsample(antialias, src.width, src.height, kernel='Bicubic')
    rescale = core.std.MaskedMerge(luma, scaled, line_mask)
    merged = vdf.merge_chroma(rescale, out)
    out = depth(merged, 16)



    antialias = lvf.rfs(out, _aa_extra(depth(denoise, 16)), [(2794, 2949)])
    out = antialias


    # Slight sharp though CAS
    sharp = hvf.LSFmod(out, strength=75, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True)
    out = sharp


    dering = gf.HQDeringmod(out, thr=16, darkthr=0.1)
    out = dering



    warp = xvs.WarpFixChromaBlend(out, thresh=48, depth=8)
    out = warp


    preden = core.knlm.KNLMeansCL(out, d=0, a=3, h=0.6, device_type='GPU', channels='Y')
    deband_mask = lvf.denoise.detail_mask(preden, brz_a=2000, brz_b=800, rad=4)
    deband = dbs.f3kpf(out, 17, 42, 42)
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    deband_b = rkt.rekt_fast(deband, lambda x: core.neo_f3kdb.Deband(x, 8, 96, 96, 96, sample_mode=1, keep_tv_range=True), 1112, 264, 184, 0)
    deband = lvf.rfs(deband, deband_b, [(4002, 4061)])
    out = deband


    adg_mask = core.adg.Mask(out.std.PlaneStats(), 20).std.Expr(f'x x {128<<8} - 0.25 * +')
    grain = core.grain.Add(out, 0.2, constant=True)
    grain = core.std.MaskedMerge(out, grain, adg_mask, 0)
    grain_b = adptvgrnMod(grain, 0.6, size=1.85, static=False, luma_scaling=4, grain_chroma=False)
    grain = lvf.rfs(grain, grain_b, [(204, 2049)])
    out = grain



    rescale_mask = vdf.drm(luma, b=b, c=c, sw=4, sh=4)
    ref, rescale_mask, src = [depth(x, 16) for x in [denoise, rescale_mask, src]]

    credit = lvf.rfs(out, core.std.MaskedMerge(out, ref, rescale_mask), full_stuff)
    credit = lvf.rfs(credit, core.std.MaskedMerge(credit, ref, vdf.region_mask(rescale_mask, 1450, 0, 947, 0)),
                     [(33860, 33925)])
    credit = lvf.rfs(credit, src, [(0, 203), (2049, 2229), (31768, 33839)])
    out = credit



    smooth = gf.JohnFPS(out[31888:33840], 60000, 1001)
    clips = [
        out[:31888],
        smooth,
        out[33840:]
    ]
    vfr = muvf.VFRSplice(clips, 'symphogearg_01_timecode.txt')
    cfr = core.std.AssumeFPS(vfr, src)
    out = cfr



    return depth(out, 10)



def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x265"""
    print('\n\n\nVideo encoding')
    if not os.path.exists(JPBD.output):
        x265_args = [
            X265, "--y4m", "--frames", f"{clip.num_frames}", "--sar", "1", "--output-depth", "10",
            "--colormatrix", "bt709", "--colorprim", "bt709", "--transfer", "bt709", "--range", "limited",
            "--min-luma", str(16<<2), "--max-luma", str(235<<2),
            "--fps", f"{clip.fps_num}/{clip.fps_den}",
            "-o", JPBD.output, "-",
            "--frame-threads", "16",
            "--no-sao", "--fades",
            "--preset", "slower",
            "--crf", "15", "--qcomp", "0.70",
            "--bframes", "16",
            "--psy-rd", "2.0", "--psy-rdoq", "1.0",
            "--deblock", "-1:-1",
            "--rc-lookahead", "96",
            "--min-keyint", "23", "--keyint", "360",
            "--aq-mode", "3", "--aq-strength", "1.0"
            ]
        print("Encoder command: ", " ".join(x265_args), "\n")
        process = subprocess.Popen(x265_args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src, '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(JPBD.src_clip, (JPBD.frame_start, JPBD.frame_end), JPBD.a_src, JPBD.a_src_cut.format(1))

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
                '--timestamps', '0:symphogearg_01_timecode.txt',
                '--track-name', '0:HEVC BDRip by Vardë@Kodoku-no-Kawarini', '--language', '0:jpn', JPBD.output,
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1),
                '--chapter-language', 'fra', '--chapters', JPBD.chapter]
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
