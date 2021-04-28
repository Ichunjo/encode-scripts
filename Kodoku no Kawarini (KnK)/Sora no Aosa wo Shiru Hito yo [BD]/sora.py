"""Sora no Aosa wo Shiru Hito yo script"""
__author__ = 'Vardë'

import sys
from typing import NamedTuple

from vsutil import core, vs, depth, get_y, join
from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk

import vardefunc as vdf
import kagefunc as kgf
import havsfunc as hvf
import debandshit as dbs
import muvsfunc as muvf
import xvs
import lvsfunc as lvf


class InfosBD(NamedTuple):
    path: str
    src: str
    src_clip: vs.VideoNode
    frame_start: int
    frame_end: int
    src_cut: vs.VideoNode
    name: str
    qpfile: str
    output: str


def infos_bd(path, frame_start, frame_end) -> InfosBD:
    src = path + '.m2ts'
    src_clip = lvf.src(src)
    src_cut = src_clip[frame_start:frame_end]
    name = vdf.Path(sys.argv[0]).stem
    qpfile = name + '_qpfile.log'
    output = name + '.264'
    return InfosBD(path, src, src_clip, frame_start, frame_end, src_cut,
                   name, qpfile, output)

JPBD = infos_bd(r'空の青さを知る人よ\BD\BDMV\STREAM\00004', 0, -24)
X264 = r'C:\Encode Stuff\x264_tmod_r3000\mcf\x64\x264_x64.exe'
X264_ARGS = dict(
    qpfile=JPBD.qpfile, threads=18, ref=16, trellis=2, bframes=16, b_adapt=2,
    direct='auto', deblock='-2:0', me='umh', subme=10, psy_rd='1.0:0.00', merange=32,
    keyint=360, min_keyint=23, rc_lookahead=60, crf=14, qcomp=0.7, aq_mode=3, aq_strength=0.85,
    tcfile_in='sora_timecode.txt'
)

def do_filter():
    """Vapoursynth filtering"""
    def _sraa(clip: vs.VideoNode, nnargs: dict, eeargs: dict) -> vs.VideoNode:
        def _nnedi3(clip):
            return clip.nnedi3.nnedi3(0, False, **nnargs)
        def _eedi3(clip, sclip):
            return clip.eedi3m.EEDI3(0, False, **eeargs, sclip=sclip)
        clip = _eedi3(clip, _nnedi3(clip)).std.Transpose()
        clip = _eedi3(clip, _nnedi3(clip)).std.Transpose()
        return clip

    def _nnedi3(clip: vs.VideoNode, factor: float, args: dict) -> vs.VideoNode:
        upscale = clip.std.Transpose().nnedi3.nnedi3(0, True, **args) \
            .std.Transpose().nnedi3.nnedi3(0, True, **args)
        return core.resize.Spline36(
            upscale, clip.width * factor, clip.height * factor,
            src_top=.5, src_left=.5)

    def area_resize(clip: vs.VideoNode, width: int, height: int) -> vs.VideoNode:
        blank = core.std.BlankClip(clip, format=vs.GRAY16, color=128 << 8)
        clip = join([clip, blank, blank])
        clip = core.area.AreaResize(clip, width, height)
        return get_y(clip)

    src = JPBD.src_cut
    src = depth(src, 16)

    flashback = [(31665, 31987), (68524, 68569)]

    denoise = CoolDegrain(src, tr=1, thsad=48, blksize=8, overlap=4, plane=4)
    denoise = lvf.rfs(denoise, src, flashback)

    dering = hvf.EdgeCleaner(denoise, 10, rmode=3, smode=1, hot=True)


    luma = get_y(dering)
    line_mask = TAAmbk(luma, mtype=2, showmask=1).std.Inflate()
    upscale = _nnedi3(luma, 2, dict(nsize=4, nns=4, qual=2, pscrn=2))
    sraa = _sraa(upscale, dict(nsize=0, nns=3, qual=1, pscrn=1),
                 dict(alpha=0.2, beta=0.5, gamma=40, nrad=3, mdis=20))
    sraa = core.rgvs.Repair(sraa, upscale, 13)
    antialias = area_resize(sraa, src.width, src.height)
    antialias = lvf.rfs(core.std.MaskedMerge(luma, antialias, line_mask),
                        antialias, [(22503, 22789)])
    antialias_merged = vdf.merge_chroma(antialias, denoise)


    deband_mask_a = detail_mask_func(antialias_merged, brz_a=3000, brz_b=1200)
    deband_mask_b = kgf.retinex_edgemask(antialias_merged)
    deband_mask = core.std.Expr([deband_mask_a, deband_mask_b], 'x y +')
    deband = dbs.f3kpf(antialias_merged, 18, 30, 30)
    deband = core.std.MaskedMerge(deband, antialias_merged, deband_mask)
    deband = hvf.ContraSharpening(deband, antialias_merged)
    deband = lvf.rfs(deband, antialias_merged, flashback)


    grain = core.neo_f3kdb.Deband(deband, preset='depth', grainy=24)
    grain_a = kgf.adaptive_grain(grain)
    grain_b = core.grain.Add(grain, 1, 0, constant=True)
    grain = lvf.rfs(grain_a, grain_b, flashback)

    mask_borders = core.std.BlankClip(grain, format=vs.GRAY16, color=(256 << 8) - 1)
    mask_borders = vdf.region_mask(mask_borders, 0, 0, 132, 132).std.Invert()
    final = lvf.rfs(grain, core.std.MaskedMerge(grain, src, mask_borders),
                    flashback + [(65388, 65413), (113621, 113763)])

    smooth = xvs.mvfrc(src[146178:src.num_frames], preset='slow')
    vfr = muvf.VFRSplice([final[:146178], smooth], 'sora_timecode.txt')


    return depth(vfr, 10)

def do_encode(filtered):
    """Compression with x264"""
    print('Qpfile generating')
    vdf.gk(JPBD.src_cut, JPBD.qpfile)

    print('\n\n\nVideo encode')
    vdf.encode(filtered, X264, JPBD.output, **X264_ARGS)

if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
