"""Symphogear G script"""
__author__ = 'Vardë'

import os
import sys
import subprocess
from typing import NamedTuple
from pathlib import Path
from acsuite import eztrim

import debandshit as dbs
import vardefunc as vdf
import muvsfunc as muvf
import havsfunc as hvf
import G41Fun as gf
import placebo
import xvs

from _assets.symphofunc import hybrid_denoise, single_rate_antialiasing
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

# lsmas doesn't work with some frames of the ending
def infos_bd(path, frame_start, frame_end)-> InfosBD:
    src = path + '.mkv'
    src_clip = lvf.src(src); vdf.set_ffms2_log_level('warning')
    src_cut = src_clip[frame_start:frame_end] if (frame_start or frame_end) else src_clip
    a_src = path + '.wav'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = '_chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)



JPBD = infos_bd(r'戦姫絶唱シンフォギアＧ\[BDMV][131002] 戦姫絶唱シンフォギアG 1\KIXA_90350\BDMV\STREAM\00004', 0, -24)
JPBD_NCOP = infos_bd(r'_assets\symphogearg_ncop01_raw_lossless', None, None)
JPBD_NCED = infos_bd(r'戦姫絶唱シンフォギアＧ\[BDMV][131106] 戦姫絶唱シンフォギアG 2\KIXA_90350\BDMV\STREAM\00009', 24, -24)
JPBD_10 = infos_bd(r'戦姫絶唱シンフォギアＧ\[BDMV][140205] 戦姫絶唱シンフォギアG 5\KIXA_90354\BDMV\STREAM\00004', 0, -24)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    ep10 = JPBD_10.src_cut


    h = 720
    w = get_w(h)
    b, c = vdf.get_bicubic_params('robidoux')
    opstart, opend = 936, 3332
    edstart, edend = 31889, src.num_frames - 1
    opstart_ep10, opend_ep10 = 768, 3164
    full_stuff = [(4438, 4569), (14956, 15111)]


    # Fix compositing error in the OP
    op_src, op_10 = src[opstart:opend+1], ep10[opstart_ep10:opend_ep10+1]
    fix = lvf.rfs(op_src, op_10, [(0, 79), (1035, 1037)])
    fix_src = insert_clip(src, fix, opstart)
    src = depth(fix_src, 32)


    denoise = hybrid_denoise(src, 0.5, 2)
    out = denoise


    luma = get_y(out)
    line_mask = vdf.edge_detect(luma, 'FDOG', 0.05, (1, 1))


    descale = core.descale.Debicubic(luma, w, h, b, c)
    upscale = vdf.fsrcnnx_upscale(descale, None, descale.height*2, '_shaders/FSRCNNX_x2_56-16-4-1.glsl', core.resize.Point)

    antialias = single_rate_antialiasing(upscale, 13, alpha=0.3, beta=0.45, gamma=320, mdis=18)

    scaled = muvf.SSIM_downsample(antialias, src.width, src.height, kernel='Bicubic')
    rescale = core.std.MaskedMerge(luma, scaled, line_mask)
    merged = vdf.merge_chroma(rescale, out)
    out = depth(merged, 16)


    antialias = lvf.rfs(out, lvf.sraa(out, 1.65, 9, alpha=0.3, beta=0.45, gamma=240, nrad=3, mdis=25),
                        [(opstart+840, opstart+881)])
    antialias = lvf.rfs(antialias, lvf.sraa(out, 1.4), [(16701, 16904)])
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
    deband_b = placebo.deband(out, 22, 6, 2)
    deband = lvf.rfs(deband, deband_b, [(opstart+1515, opstart+1603)])
    deband_c = dbs.f3kpf(out, 17, 64, 64)
    deband = lvf.rfs(deband, deband_c, [(31619, 31888)])
    deband = core.std.MaskedMerge(deband, out, deband_mask)

    out = deband


    adg_mask = core.adg.Mask(out.std.PlaneStats(), 20).std.Expr(f'x x {128<<8} - 0.25 * +')
    grain = core.grain.Add(out, 0.2, constant=True)
    grain = core.std.MaskedMerge(out, grain, adg_mask, 0)
    out = grain



    rescale_mask = vdf.drm(luma, b=b, c=c, sw=4, sh=4)
    ref, rescale_mask, src, src_ncop, src_nced = [depth(x, 16) for x in [denoise, rescale_mask, src,
                                                                         JPBD_NCOP.src_cut, JPBD_NCED.src_cut]]

    credit = lvf.rfs(out, core.std.MaskedMerge(out, ref, rescale_mask), full_stuff)
    out = credit



    src_c, src_ncop, src_nced = [c.knlm.KNLMeansCL(a=7, h=35, d=0, device_type='gpu') for c in [src, src_ncop, src_nced]]

    opening_mask = vdf.dcm(out, src_c[opstart:opend+1], src_ncop[:opend-opstart+1], opstart, opend, 4, 4).std.Inflate()
    ending_mask = vdf.dcm(out, src_c[edstart:edend+1], src_nced[:edend-edstart+1], edstart, edend, 4, 4).std.Inflate()
    credit_mask = core.std.Expr([opening_mask, ending_mask], 'x y +')

    credit = lvf.rfs(out, core.std.MaskedMerge(out, ref, credit_mask), [(opstart, opend), (edstart, edend)])
    out = credit


    return depth(out, 10)



def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x265"""
    print('\n\n\nVideo encoding')
    if not os.path.exists(JPBD.output):
        x265_args = [
            "x265", "--y4m", "--frames", f"{clip.num_frames}", "--sar", "1", "--output-depth", "10",
            "--colormatrix", "bt709", "--colorprim", "bt709", "--transfer", "bt709", "--range", "limited",
            "--min-luma", str(16<<2), "--max-luma", str(235<<2),
            "--fps", f"{clip.fps_num}/{clip.fps_den}",
            "-o", JPBD.output, "-",
            "--frame-threads", "4",
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
        ffv1_args = [
            'ffmpeg', '-i', '-', '-vcodec', 'ffv1', '-coder', '1', '-context', '0',
            '-g', '1', '-level', '3', '-threads', '8',
            '-slices', '24', '-slicecrc', '1', JPBD.name + "_lossless.mkv"
        ]
        print("Encoder command: ", " ".join(ffv1_args), "\n")
        process = subprocess.Popen(ffv1_args, stdin=subprocess.PIPE)
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
