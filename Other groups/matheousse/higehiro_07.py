"""HigeHiro script"""
__author__ = 'Vardë'


from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, cast

import G41Fun as gf
import havsfunc as hvf
import kagefunc as kgf
import lvsfunc as lvf
import vardefunc as vdf
from vardautomation import (BasicTool, FileInfo, PresetAAC, PresetWEB,
                            X265Encoder)
from vardefunc.mask import Scharr
from vsutil import depth, get_y

import vapoursynth as vs
core = vs.core


NUM = __file__[-5:-3]

WEB_CRU = FileInfo(f'{NUM}/[FeelsBadSubs] Hige wo Soru. Soshite Joshikousei wo Hirou. - {NUM} [1080p].mkv',
                   None, None, preset=[PresetWEB, PresetAAC])
WEB_AOD = FileInfo(f'{NUM}/Higehiro E{NUM} [1080p+][AAC][JapDub][GerSub][Web-DL].mkv',
                   None, None, preset=[PresetWEB, PresetAAC])

SUB = f'{NUM}/[FeelsBadSubs] Hige wo Soru. Soshite Joshikousei wo Hirou. - {NUM} [1080p].ass'




class Credit(NamedTuple):  # noqa: PLC0115
    range_frames: List[Tuple[int, int]]
    mask: vs.VideoNode


class ScharrG41(Scharr):  # noqa: PLC0115
    def _get_divisors(self) -> List[float]:
        return [3, 3]



def filtering() -> Union[vs.VideoNode, Tuple[vs.VideoNode, vs.VideoNode]]:
    """Vapoursynth filtering"""
    src_cru = WEB_CRU.clip_cut
    src_aod = WEB_AOD.clip_cut

    _, masksub = core.sub.TextFile(src_aod, SUB, fontdir='fonts', blend=False)
    masksub = core.std.Binarize(masksub, 1)
    masksub = hvf.mt_expand_multi(masksub, 'ellipse', sw=6, sh=4)
    masksub = hvf.mt_inflate_multi(masksub, radius=4).std.Convolution([1] * 9)

    dehardsub = core.std.MaskedMerge(src_aod, src_cru, masksub)
    dehardsub = depth(dehardsub, 16)
    out = dehardsub


    lineart = ScharrG41().get_mask(get_y(out), 4000, multi=1.2).rgvs.RemoveGrain(3).std.Maximum().std.Minimum()


    luma = get_y(out)
    ssing = vdf.scale.fsrcnnx_upscale(
        luma, height=1620, shader_file='shaders/FSRCNNX_x2_16-0-4-1.glsl',
        downscaler=lambda c, w, h: core.resize.Bicubic(c, w, h, filter_param_a=-0.5, filter_param_b=0.25),
        profile='fast',
    )
    sraing = sraa_eedi3(ssing, 13, gamma=100, nrad=2, mdis=15)
    down = core.resize.Bicubic(sraing, out.width, out.height, filter_param_a=-0.5, filter_param_b=0.25)
    masked = core.std.MaskedMerge(luma, down, lineart)
    merged = vdf.misc.merge_chroma(masked, out)
    out = merged


    contra = hvf.LSFmod(out, strength=80, Smode=3, edgemode=0, source=dehardsub)
    out = contra


    # I gave up on this
    ending = lvf.rfs(out, dehardsub, [(1344, 3500), (31649, out.num_frames - 1)])
    out = ending



    dehalo = gf.MaskedDHA(out, rx=1.4, ry=1.4, darkstr=0, brightstr=0.8)
    out = dehalo




    dbgra_cru = _dbgra(src_cru)
    deband = core.std.MaskedMerge(out, dbgra_cru, depth(masksub, 16))
    out = deband






    ref = src_cru
    rsc_m = vdf.mask.diff_rescale_mask(ref, 837, thr=80)
    rsc_m = depth(rsc_m, 16)

    ref = dehardsub
    credit = out


    # Ep Title
    creds = [
        Credit([(3703, 3822)], vdf.mask.region_mask(rsc_m[3798], 1300, 0, 930, 0)),
    ]
    for cred in creds:
        credit = lvf.rfs(credit, core.std.MaskedMerge(out, ref, cred.mask), cred.range_frames)

    out = credit



    # return dehardsub, vdf.mask.region_mask(rsc_m, 1300, 0, 930, 0)
    # return dehardsub, rsc_m
    # return dehardsub, out


    return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])



def _dbgra(clip: vs.VideoNode) -> vs.VideoNode:
    clip = depth(clip, 16)
    clip = hvf.SMDegrain(clip, tr=1, thSAD=200)
    clip = vdf.deband.dumb3kdb(clip, threshold=45)
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


def do_wizardry() -> None:
    """It's magic"""

    filtered = filtering()
    if isinstance(filtered, vs.VideoNode):
        filtered = cast(vs.VideoNode, filtered)
    else:
        raise ValueError

    if not Path(WEB_AOD.name_clip_output).exists():
        X265Encoder('x265', Path('settings/x265_settings'), filtered, WEB_AOD,
                    progress_update=lambda v, e:
                        print(f"\rVapourSynth: {v}/{e} ~ {100 * v // e}% || Encoder: ", end=""))


    if not Path(WEB_AOD.a_src.format(1)).exists():
        BasicTool('mkvextract', [WEB_AOD.src, 'tracks', f'1:{WEB_AOD.a_src.format(1)}'])


    assert WEB_AOD.a_src is not None
    BasicTool('mkvmerge', ['-o', WEB_AOD.name_file_final,
                           '--track-name', '0:HEVC WEBRip by Vardë@Raws-Maji', '--language', '0:jpn', WEB_AOD.name_clip_output,
                           '--track-name', '0:AAC 2.0', '--language', '0:jpn', WEB_AOD.a_src.format(1)])


if __name__ == '__main__':
    do_wizardry()
else:
    WEB_CRU.clip_cut.set_output(0)
    WEB_AOD.clip_cut.set_output(1)

    FILTERED = filtering()
    # FILTERED.set_output(2)
    FILTERED[0].set_output(2)
    FILTERED[1].set_output(3)
