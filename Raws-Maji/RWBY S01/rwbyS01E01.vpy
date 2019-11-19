import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import vsTAAmbk as taa
import muvsfunc as muvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
from vsutil import get_w, split, join

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000

src_us_OP = lvf.src(r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00013.m2ts")
src_us = lvf.src(r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00003.m2ts")

src_jp = lvf.src(r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00001.m2ts")
asrc_jp = r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00001.mka"

src_blank = core.std.BlankClip(src_us_OP)
asrc_blank = r"blank_audio.ac3"

ac.eztrim(src_blank, [(0,150)], asrc_blank, "AUDIOS_EN/01a_cut.ac3")
ac.eztrim(src_blank, [(0,1)], asrc_blank, "AUDIOS_EN/01d_cut.ac3")

ac.eztrim(src_jp, [(0,20135), (20135+15,22345)], asrc_jp, "AUDIOS_JAP/01_cut.mka")

#src_jp = src_jp[0:20135]+src_jp[20135+15:22345]
src = src_jp[0:150]+src_us+src_us_OP+src_blank[:1]
src = core.std.AssumeFPS(src, fpsnum=30000, fpsden=1001)
src = fvf.Depth(src, 16)


shift = hvf.FixChromaBleedingMod(src, cx=-0.5, cy=-0.5, thr=48, strength=1, blur=True)

aa = core.std.Transpose(shift)
planes = split(aa)
planes[0] = lvf.upscaled_sraa(planes[0], 2, 13, sharp_downscale=False)
planes[1], planes[2] = [taa.TAAmbk(plane, aatype='Eedi3SangNom', down8=False) for plane in planes[1:]]
aa = join(planes)
aa = core.std.Transpose(aa)

adaptmask = vrdf.adaptive_mask(aa, 25)
linemask = kgf.retinex_edgemask(aa)
deband_a = core.f3kdb.Deband(aa, range=17, y=30, cb=20, cr=20, grainy=0, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(aa, range=17, y=40, cb=30, cr=30, grainy=8, grainc=0, output_depth=16)
deband_c = core.f3kdb.Deband(aa, range=17, y=50, cb=40, cr=40, grainy=16, grainc=0, output_depth=16)
deband_d = core.f3kdb.Deband(aa, range=24, y=110, cb=90, cr=90, grainy=32, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband_b, deband_c, adaptmask)
deband = core.std.MaskedMerge(deband, deband_a, linemask)
deband = fvf.rfs(deband, deband_d, mappings='[1614 1808]')

grain_a = kgf.adaptive_grain(deband, 0.1)
grain_b = kgf.adaptive_grain(deband, 0.3)
grain = fvf.rfs(grain_a, grain_b, mappings='[1614 1808]')

final = fvf.Depth(grain, 10)

final.set_output()