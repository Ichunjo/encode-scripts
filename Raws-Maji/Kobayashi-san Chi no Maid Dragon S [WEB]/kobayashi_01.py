from functools import partial
from typing import Tuple

import havsfunc as hvf
import kagefunc as kgf
import lvsfunc
import vapoursynth as vs
import vardefunc as vdf
from vardautomation import (AudioCutter, BasicTool, FileInfo, PresetEAC3,
                            PresetWEB, VPath, X265Encoder)
from vardefunc.mask import FDOG, SobelStd
from vardefunc.misc import merge_chroma
from vardefunc.scale import nnedi3_upscale
from vsutil import depth, get_y, iterate, join, split

core = vs.core


NUM = __file__[-5:-3]

WEB_BIL = FileInfo(
    f'{NUM}/[NC-Raws] 小林家的龙女仆S - {NUM} [B-Global][WEB-DL][2160p][AVC AAC][CHS_CHT_ENG_TH_SRT][MKV].mkv',
    None, None, preset=[PresetWEB]
)
WEB_AOD = FileInfo(
    f'{NUM}/Kobayashi-san Chi no Maid Dragon S E{NUM} [1080p+][AAC][JapDub][GerSub][Web-DL].mkv',
    None, None, preset=[PresetWEB]
)
WEB_WAK = FileInfo(
    f'{NUM}/DragonMaid_S2_01_FR_HD.mp4',
    None, None, preset=[PresetWEB]
)
WEB_CRU = FileInfo(
    f'{NUM}/[FeelsBadSubs] Kobayashi-san Chi no Maid Dragon S - {NUM} [1080p].mkv',
    None, None, preset=[PresetWEB]
)

WEB_AMZ_CBR = FileInfo(
    f'{NUM}/Kobayashi-san Chi no Maid Dragon S - {NUM} (Amazon dAnime CBR 1080p).mkv',
    None, None, preset=[PresetWEB, PresetEAC3]
)
WEB_AMZ_VBR = FileInfo(
    f'{NUM}/Kobayashi-san Chi no Maid Dragon S - {NUM} (Amazon dAnime VBR 1080p).mkv',
    None, None, preset=[PresetWEB, PresetEAC3]
)

SUB = f'{NUM}/[FeelsBadSubs] Kobayashi-san Chi no Maid Dragon S - {NUM} [1080p]_Track05.ass'

OPSTART, OPEND = 1534, 3764



class Filtering:
    def main(self) -> vs.VideoNode:
        src_bil = WEB_BIL.clip_cut
        src_aod = WEB_AOD.clip_cut
        src_wak = WEB_WAK.clip_cut
        src_wak = src_wak[0] + src_wak
        src_cru = WEB_CRU.clip_cut


        dehardsub_aod, masksub = self.dehardsub_aod(src_aod, src_cru)
        # return masksub
        # return dehardsub_aod
        # return lvsfunc.comparison.diff(dehardsub_aod, src_cru, height=540)
        deharsub_wak, _ = self.dehardsub_wak(src_wak, src_cru)
        # return deharsub_wak
        # return lvsfunc.comparison.diff(deharsub_wak, src_cru, height=540)
        lineart = FDOG().get_mask(get_y(deharsub_wak))
        lineart = iterate(lineart, core.std.Maximum, 3)
        # return lineart
        dehardsub = core.std.MaskedMerge(dehardsub_aod, deharsub_wak, lineart)
        out = depth(dehardsub, 16)
        # return dehardsub


        dbgra_cru = self._dbgra_cru(src_cru)
        dbgra_wak = self._dbgra_wak(deharsub_wak)
        deband = core.std.MaskedMerge(out, dbgra_cru, depth(masksub, 16))
        deband = merge_chroma(deband, dbgra_wak)
        out = deband


        decsize = vdf.noise.decsiz(out, min_in=128 << 8, max_in=176 << 8)
        out = decsize
        # return lvsfunc.comparison.stack_planes(depth(out, 32))

        planes = split(out)
        planes_ups = planes

        lineart = SobelStd().get_mask(planes[0]).std.Maximum().std.Minimum().resize.Bilinear(3840, 2160)

        nnedi3 = nnedi3_upscale(planes[0])
        planes_ups[1], planes_ups[2] = [
            nnedi3_upscale(p, correct_shift=False).resize.Bicubic(
                src_left=0.5 + vdf.misc.get_chroma_shift(1080, 2160), src_top=0.5
            )
            for p in planes[1:]
        ]
        w2x = core.w2xnvk.Waifu2x(
            core.resize.Point(planes[0], format=vs.RGBS), noise=1, scale=2, model=0
        ).resize.Point(format=vs.GRAY16, matrix=1, dither_type='error_diffusion')

        planes_ups[0] = core.std.MaskedMerge(nnedi3, w2x, lineart)
        diff = core.std.Expr(
            (planes_ups[0].resize.Bilinear(1920, 1080).std.BoxBlur(0, 1, 1, 1, 1).resize.Bilinear(3840, 2160),
             nnedi3.resize.Bilinear(1920, 1080).std.BoxBlur(0, 1, 1, 1, 1).resize.Bilinear(3840, 2160)),
            'x y - abs'
        )

        th_lo, th_hi = 3500, 7000
        strength = f'{th_hi} x - {th_hi} {th_lo} - /'
        planes_ups[0] = core.std.Expr(
            (diff, nnedi3, planes_ups[0]),
            f'x {th_lo} < z x {th_hi} > y z ' + strength + ' * y 1 ' + strength + ' - * + ? ?'
        ).rgvs.Repair(nnedi3, 13).rgvs.Repair(nnedi3, 7)

        upscale = join(planes_ups)
        out = upscale


        out = core.std.Splice([src_bil[:1], out[1:]], mismatch=True)
        out = out.resize.Point(format=vs.YUV420P10, dither_type='error_diffusion')

        while out.num_frames < 34046:
            out += out[-1]

        return out

    @staticmethod
    def dehardsub_aod(src_aod: vs.VideoNode, src_cru: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
        _, masksub = core.sub.TextFile(src_aod, SUB, fontdir='fonts', blend=False)
        masksub = masksub.std.Binarize(1)
        masksub = iterate(masksub, core.std.Maximum, 5)
        masksub = iterate(masksub, partial(core.std.Maximum, coordinates=[0, 0, 0, 1, 1, 0, 0, 0]), 6)
        masksub = iterate(masksub, core.std.Inflate, 5)

        return core.std.MaskedMerge(src_aod, src_cru, masksub), masksub

    @staticmethod
    def dehardsub_wak(src_wak: vs.VideoNode, src_cru: vs.VideoNode) -> Tuple[vs.VideoNode, vs.VideoNode]:
        masksub = kgf.hardsubmask(src_wak, src_cru)
        deharsub_wak = core.std.MaskedMerge(src_wak, src_cru, masksub)

        thr = 7
        diff_mask = core.std.Lut2(
            deharsub_wak.std.Convolution([1]*25), src_cru.std.Convolution([1]*25),
            function=lambda x, y: 255 if abs(x - y) > thr else 0
        )
        diff_mask = core.std.Expr(
            split(diff_mask.resize.Bilinear(format=vs.YUV444P8)),
            'x y z max max'
        )
        diff_mask = iterate(diff_mask, core.std.Maximum, 5)
        diff_mask = lvsfunc.util.replace_ranges(diff_mask, diff_mask.std.BlankClip(), [(OPSTART, OPEND)])

        return core.std.MaskedMerge(deharsub_wak, src_cru, diff_mask), diff_mask

    @staticmethod
    def _dbgra_cru(clip: vs.VideoNode) -> vs.VideoNode:
        clip = depth(clip, 16)
        clip = hvf.SMDegrain(clip, tr=1, thSAD=160)
        clip = vdf.deband.dumb3kdb(clip, threshold=49)
        clip = core.std.Expr(clip, ['x 64 -', 'x 32 +', 'x 32 +'])
        clip = kgf.adaptive_grain(clip, 0.4).grain.Add(0, 0.2, constant=True)
        return clip

    @staticmethod
    def _dbgra_wak(clip: vs.VideoNode) -> vs.VideoNode:
        clip = depth(clip, 16)
        clip = hvf.SMDegrain(clip, tr=1, thSAD=120)
        clip = vdf.deband.dumb3kdb(clip, threshold=[1, 33], grain=32)
        return clip


def do_wizardry() -> None:
    """It's magic"""

    filtered = Filtering().main()

    if not VPath(WEB_AOD.name_clip_output).exists():
        X265Encoder('x265_settings').run_enc(filtered, WEB_AOD)

    if WEB_AMZ_VBR.a_src and not WEB_AMZ_VBR.a_src.format(1).exists():
        BasicTool('mkvextract', [WEB_AMZ_VBR.path.to_str(), 'tracks', f'1:{WEB_AMZ_VBR.a_src.format(1)}']).run()

    if WEB_AMZ_VBR.a_src_cut and not WEB_AMZ_VBR.a_src_cut.format(1).exists():
        WEB_AMZ_VBR.frame_start = 1
        AudioCutter(WEB_AMZ_VBR, track=1).run()

    assert WEB_AMZ_VBR.a_src_cut is not None
    BasicTool('mkvmerge', ['-o', WEB_AOD.name_file_final.to_str(),
                           '--track-name', '0:HEVC WEBRip by Vardë@Meme-Maji', '--language', '0:jpn', WEB_AOD.name_clip_output.to_str(),
                           '--track-name', '0:AAC 2.0', '--language', '0:jpn', WEB_AMZ_VBR.a_src_cut.format(1).to_str()]).run()



def compare():
    from typing import List
    frames: List[int] = [
        11569, 29520, 29811, 31042, 33749, 33564, 1560, 1725, 11036
    ]

    names: List[str] = [
        'Bilibili', 'AoD', 'Wakanim', 'Crunchyroll',
        'Amazon VBR', 'Amazon CBR',
        'Meme-Maji'
    ]
    web_wak = WEB_WAK.clip_cut[0] + WEB_WAK.clip_cut
    mememaji = core.ffms2.Source('kobayashi_01.mkv')
    srcs = [
        WEB_BIL.clip_cut, WEB_AOD.clip_cut, web_wak, WEB_CRU.clip_cut,
        WEB_AMZ_VBR.clip_cut, WEB_AMZ_CBR.clip_cut,
        mememaji
    ]
    srcs = [
        src.std.SetFrameProp('_Matrix', intval=1).resize.Spline36(3840, 2160, vs.RGB24).text.Text(name)
        for src, name in zip(srcs, names)
    ]
    for src, name in zip(srcs, names):
        folder = VPath(name)
        folder.mkdir()
        for frame in sorted(frames):
            core.imwri.Write(src[frame], 'PNG', f'{name}/{name}_%06d.png', frame).get_frame(0)



if __name__ == '__main__':
    do_wizardry()
    # compare()
else:
    WEB_BIL.clip_cut.resize.Spline36(1920, 1080).text.Text('Bilibili').set_output(0)

    # lvsfunc.comparison.stack_planes(depth(WEB_AOD.clip_cut, 32)).set_output(1)
    WEB_AOD.clip_cut.text.Text('AoD').set_output(1)

    wak = WEB_WAK.clip_cut[0] + WEB_WAK.clip_cut
    # lvsfunc.comparison.stack_planes(depth(wak, 32)).set_output(2)
    wak.text.Text('Wakanim').set_output(2)

    # lvsfunc.comparison.stack_planes(depth(WEB_CRU.clip_cut, 32)).set_output(3)
    WEB_CRU.clip_cut.text.Text('Crunchyroll').set_output(3)

    WEB_AMZ_VBR.clip_cut.text.Text('AMZ VBR').set_output(4)
    WEB_AMZ_CBR.clip_cut.text.Text('AMZ CBR').set_output(5)

    FILTERED = Filtering().main()
    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(10)
