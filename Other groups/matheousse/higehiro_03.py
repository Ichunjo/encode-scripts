"""HigeHiro script"""
__author__ = 'Vardë'

import os
import shlex
import subprocess
from typing import Dict, Any, Optional, List, Tuple

import vardefunc as vdf
import havsfunc as hvf
import lvsfunc as lvf
import G41Fun as gf
import kagefunc as kgf

import init_source

from vsutil import depth, get_y
import vapoursynth as vs

core = vs.core


NUM = __file__[-5:-3]

WEB_CRU = init_source.Infos()
WEB_CRU.set_infos(f'{NUM}/[FeelsBadSubs] Hige wo Soru. Soshite Joshikousei wo Hirou. - {NUM} [1080p].mkv', None, None, preset='web/aac')

WEB_AOD = init_source.Infos()
WEB_AOD.set_infos(f'{NUM}/Higehiro E{NUM} [1080p+][AAC][JapDub][GerSub][Web-DL].mkv', None, None, preset='web/aac')

SUB = f'{NUM}/[FeelsBadSubs] Hige wo Soru. Soshite Joshikousei wo Hirou. - {NUM} [1080p].ass'




class Credit:
    def __init__(self, range_frames: List[Tuple], mask: vs.VideoNode) -> None:
        self.range_frames = range_frames
        self.mask = mask


def do_filter():
    """Vapoursynth filtering"""
    src_cru = WEB_CRU.src_cut
    src_aod = WEB_AOD.src_cut

    _, masksub = core.sub.TextFile(src_aod, SUB, fontdir='fonts', blend=False)
    masksub = core.std.Binarize(masksub, 1)
    masksub = hvf.mt_expand_multi(masksub, 'ellipse', sw=6, sh=4)
    masksub = hvf.mt_inflate_multi(masksub, radius=4).std.Convolution([1] * 9)

    dehardsub = core.std.MaskedMerge(src_aod, src_cru, masksub)
    dehardsub = depth(dehardsub, 16)
    out = dehardsub



    lineart = gf.EdgeDetect(out, 'scharr').std.Maximum().std.Minimum()
    lineart = core.std.Expr(lineart, 'x 4000 < 0 x ? 1.2 *').rgvs.RemoveGrain(3)

    luma = get_y(out)
    ssing = vdf.fsrcnnx_upscale(
        luma, height=1620, shader_file='shaders/FSRCNNX_x2_16-0-4-1.glsl',
        downscaler=lambda c, w, h: core.resize.Bicubic(c, w, h, filter_param_a=-0.5, filter_param_b=0.25),
        profile='fast',
    )
    sraing = sraa_eedi3(ssing, 13, gamma=100, nrad=2, mdis=15)
    down = core.resize.Bicubic(sraing, out.width, out.height, filter_param_a=-0.5, filter_param_b=0.25)
    masked = core.std.MaskedMerge(luma, down, lineart)
    merged = vdf.merge_chroma(masked, out)
    out = merged


    contra = hvf.LSFmod(out, strength=80, Smode=3, edgemode=0, source=dehardsub)
    out = contra


    # I gave up on this
    ending = lvf.rfs(out, dehardsub, [(1368, 3524), (31072, out.num_frames - 1)])
    out = ending



    dehalo = gf.MaskedDHA(out, rx=1.4, ry=1.4, darkstr=0, brightstr=0.8)
    out = dehalo




    dbgra_cru = _dbgra(src_cru)
    deband = core.std.MaskedMerge(out, dbgra_cru, depth(masksub, 16))
    out = deband






    ref = src_cru
    rsc_m = vdf.diff_rescale_mask(ref, 837, mthr=40)
    rsc_m = depth(rsc_m, 16)

    ref = dehardsub
    credit = out


    # Ep Title
    creds = [
        Credit([3691, 3810], vdf.region_mask(rsc_m[3696], 1450, 0, 800, 0)),
    ]
    for cred in creds:
        credit = lvf.rfs(credit, core.std.MaskedMerge(out, ref, cred.mask), cred.range_frames)

    out = credit



    # return dehardsub, vdf.region_mask(rsc_m, 1450, 0, 800, 0)
    # return dehardsub, out


    return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])



def _dbgra(clip: vs.VideoNode) -> vs.VideoNode:
    clip = depth(clip, 16)
    clip = hvf.SMDegrain(clip, tr=1, thSAD=200)
    clip = vdf.dumb3kdb(clip, threshold=45)
    clip = core.std.Expr(clip, ['x 64 -', 'x 32 +', 'x 32 +'])
    clip = kgf.adaptive_grain(clip, 0.4)
    return clip



def sraa_eedi3(clip: vs.VideoNode, rep: Optional[int] = None, **eedi3_args: Any) -> vs.VideoNode:
    """Drop half the field with eedi3+nnedi3 and interpolate them.

    Args:
        clip (vs.VideoNode): Source clip.
        rep (Optional[int], optional): Repair mode. Defaults to None.

    Returns:
        vs.VideoNode: AA'd clip
    """
    nnargs: Dict[str, Any] = dict(nsize=6, nns=2, qual=1)
    eeargs: Dict[str, Any] = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)
    eeargs.update(eedi3_args)

    eedi3_fun, nnedi3_fun = core.eedi3m.EEDI3CL, core.nnedi3cl.NNEDI3CL

    flt = core.std.Transpose(clip)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)
    flt = core.std.Transpose(flt)
    flt = eedi3_fun(flt, 0, False, sclip=nnedi3_fun(flt, 0, False, False, **nnargs), **eeargs)

    if rep:
        flt = core.rgvs.Repair(flt, clip, rep)

    return flt



def do_encode(clip):
    """Compression with x26X"""
    if not os.path.isfile(WEB_AOD.output):
        print('\n\n\nVideo encoding')
        bits = clip.format.bits_per_sample
        x265_cmd = f'x265 -o {WEB_AOD.output} - --y4m' + ' '
        x265_cmd += f'--csv {WEB_AOD.name}_log_x265.csv --csv-log-level 2' + ' '
        x265_cmd += '--preset slower' + ' '
        x265_cmd += f'--frames {clip.num_frames} --fps {clip.fps_num}/{clip.fps_den} --output-depth {bits}' + ' '
        x265_cmd += '--rd 3 --no-rect --no-amp --rskip 1 --tu-intra-depth 2 --tu-inter-depth 2 --tskip' + ' '
        x265_cmd += '--merange 48 --weightb' + ' '
        x265_cmd += '--no-strong-intra-smoothing' + ' '
        x265_cmd += '--psy-rd 1.85 --psy-rdoq 2 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 40 --rc-lookahead 48 --bframes 16' + ' '
        x265_cmd += '--crf 16 --aq-mode 3 --aq-strength 0.85 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=1:-1 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    print('\n\n\nAudio extraction')
    mkv_args = ['mkvextract', WEB_AOD.src, 'tracks', f'1:{WEB_AOD.a_src}']
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', WEB_AOD.output_final,
                '--track-name', '0:HEVC WEBRip by Vardë@Raws-Maji', '--language', '0:jpn', WEB_AOD.output,
                '--track-name', '0:AAC 2.0', '--language', '0:jpn', WEB_AOD.a_src]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
else:
    WEB_CRU.src_cut.set_output(0)
    WEB_AOD.src_cut.set_output(1)

    FILTERED = do_filter()
    # FILTERED.set_output(2)
    FILTERED[0].set_output(2)
    FILTERED[1].set_output(3)
