"""Symphogear XD script"""
__author__ = 'Vardë'

import sys
import subprocess
from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path

import debandshit as dbs
import vardefunc as vdf
import muvsfunc as muvf
import havsfunc as hvf
import mvsfunc as mvf
import G41Fun as gf
import placebo
import xvs

from vsutil import depth, get_y, get_w
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
    src = path + '.mkv'
    src_clip = lvf.src(src)
    src_cut = src_clip[frame_start:frame_end]
    a_src = path + '.mka'
    a_src_cut = path + '_cut_track_{}.wav'
    a_enc_cut = path + '_track_{}.m4a'
    name = Path(sys.argv[0]).stem
    output = name + '.265'
    chapter = 'chapters/' + name + '.txt'
    output_final = name + '.mkv'
    return InfosBD(path, src, src_clip, frame_start, frame_end,
                   src_cut, a_src, a_src_cut, a_enc_cut,
                   name, output, chapter, output_final)

JPBD = infos_bd(r'シンフォギアライブ2018_t01', 7748, 10279)
OPYT = infos_bd(r'OPアニメーション_戦姫絶唱シンフォギアXD UNLIMITED-1eZ6J4y0Jz4', 0, -24)
X265 = r'C:\Encode Stuff\x265-3.4+12-geff9_vs2015-AVX2\x265.exe'

def hybrid_denoise(clip: vs.VideoNode, knlm_h: float = 0.5, sigma: float = 2,
                    knlm_args: Optional[Dict[str, Any]] = None, bm3d_args: Optional[Dict[str, Any]] = None)-> vs.VideoNode:
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

def single_rate_antialiasing(clip: vs.VideoNode, rep: Optional[int] = None, **eedi3_args: Any)-> vs.VideoNode:
    nnargs: Dict[str, Any] = dict(nsize=0, nns=3, qual=1)
    eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)
    eeargs.update(eedi3_args)

    aa = core.std.Transpose(clip)
    aa = core.eedi3m.EEDI3(aa, 0, False, sclip=core.nnedi3cl.NNEDI3CL(aa, 0, False, False, **nnargs), **eeargs)
    aa = core.std.Transpose(aa)
    aa = core.eedi3m.EEDI3(aa, 0, False, sclip=core.nnedi3cl.NNEDI3CL(aa, 0, False, False, **nnargs), **eeargs)

    if rep:
        aa = core.rgsf.Repair(aa, clip, rep)

    return aa

def w2c_denoise(clip: vs.VideoNode, noise: int, **kwargs)-> vs.VideoNode:
    clip = mvf.ToRGB(clip, depth=32)
    w2c = core.w2xc.Waifu2x(clip, noise, 1, **kwargs)
    return mvf.ToYUV(w2c, css='420', depth=32)

def caffe_denoise(clip: vs.VideoNode, noise: int, scale: int, **kwargs)-> vs.VideoNode:
    return core.caffe.Waifu2x(clip, noise, scale, **kwargs)

def do_filter():
    """Vapoursynth filtering"""
    opstart = 0
    h = 720
    w = get_w(h)
    # getnative script returns 0.2 0.5 as best combo but it introduces too much halos.
    # I think it's plain mitchell but robidoux is good too and very slightly sharp.
    b, c = vdf.get_bicubic_params('robidoux')

    src = JPBD.src_cut + core.vivtc.VDecimate(OPYT.src_clip)[2520:2591]

    src = depth(src, 32)

    denoise = hybrid_denoise(src, 0.65, 2.25)
    denoise_b = w2c_denoise(src, 1)
    denoise = lvf.rfs(denoise, denoise_b, [(opstart+1737, opstart+1747)])
    out = denoise


    luma = get_y(out)
    line_mask = vdf.edge_detect(luma, 'FDOG', 0.05, (1, 1))


    descale = core.descale.Debicubic(luma, w, h, b, c)
    upscale = vdf.fsrcnnx_upscale(descale, descale.height*2, 'shaders/FSRCNNX_x2_56-16-4-1.glsl', core.resize.Point)
    upscale_b = caffe_denoise(descale, 1, scale=2)
    upscale = lvf.rfs(upscale, upscale_b, [(2530, src.num_frames-1)])

    antialias = single_rate_antialiasing(upscale, 13, alpha=0.3, beta=0.45, gamma=320, mdis=18)
    scaled = muvf.SSIM_downsample(antialias, src.width, src.height, kernel='Bicubic')

    rescale = core.std.MaskedMerge(luma, scaled, line_mask)
    merged = vdf.merge_chroma(rescale, out)
    out = depth(merged, 16)


    # Slight sharp though CAS
    sharp = hvf.LSFmod(out, strength=75, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True)
    out = sharp


    dering = gf.HQDeringmod(out, thr=16, darkthr=0.1, show=False)
    out = dering



    warp = xvs.WarpFixChromaBlend(out, thresh=48, depth=8)
    out = warp



    preden = core.knlm.KNLMeansCL(out, d=0, a=3, h=0.6, device_type='GPU', channels='Y')
    deband_mask = detail_mask_func(preden, brz_a=2000, brz_b=800, rad=4)
    deband = dbs.f3kpf(out, 17, 42, 42)
    deband_b = placebo.deband(out, 24, 6.5, 3)
    deband = lvf.rfs(deband, deband_b, [(opstart, opstart+105), (opstart+657, opstart+693)])
    deband = core.std.MaskedMerge(deband, out, deband_mask)
    out = deband



    adg_mask = core.adg.Mask(out.std.PlaneStats(), 20).std.Expr(f'x x {128<<8} - 0.25 * +')
    grain = core.grain.Add(out, 0.2, constant=True)
    grain_b = core.grain.Add(out, 0.8, constant=False)
    grain = core.std.MaskedMerge(out, grain, adg_mask, 0)
    grain = lvf.rfs(grain, grain_b, [(opstart, opstart+105)])
    out = grain


    return depth(out, 10)



def do_encode(clip: vs.VideoNode)-> None:
    """Compression with x265"""
    print('\n\n\nVideo encoding')
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
        "--rc-lookahead", "72",
        "--min-keyint", "23", "--keyint", "240",
        "--aq-mode", "3", "--aq-strength", "1.0"
        ]
    print("Encoder command: ", " ".join(x265_args), "\n")
    process = subprocess.Popen(x265_args, stdin=subprocess.PIPE)
    clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
    process.communicate()


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
