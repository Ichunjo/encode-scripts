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

src_us_OP = 	lvf.src(r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00013.m2ts")
src_us = 		lvf.src(r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00003.m2ts")

src_jp = 		lvf.src(r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00001.m2ts")
asrc_jp = 		r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00001.mka"

src_blank = 	core.std.BlankClip(src_us_OP)
asrc_blank = 	r"blank_audio.ac3"

ac.eztrim(src_blank, 	[(0,150)], 						asrc_blank, "AUDIOS_EN/01a_cut.mka")
ac.eztrim(src_blank, 	[(0,1)], 						asrc_blank, "AUDIOS_EN/01d_cut.mka")

ac.eztrim(src_jp, 		[(0,20135), (20135+15,22345)], 	asrc_jp, 	"AUDIOS_JAP/01_cut.mka")

src = src_jp[0:150]+src_us+src_us_OP+src_blank[:1]
src = core.std.AssumeFPS(src, fpsnum=30000, fpsden=1001)
src = fvf.Depth(src, 16)

opstart = 20135

shift = hvf.FixChromaBleedingMod(src, cx=-0.5, cy=-0.5, thr=48, strength=1, blur=True)

adaptmask = vrdf.adaptive_mask(shift, 20)
linemask = kgf.retinex_edgemask(shift).std.Binarize(9880).std.Inflate()
mergemask = core.std.Expr([adaptmask, linemask], expr="x y -")
denoise = core.knlm.KNLMeansCL(fvf.Depth(shift, 32), a=2, h=0.4, d=0, device_type='gpu', channels='Y')
denoise = core.std.MaskedMerge(shift, fvf.Depth(shift, 16), mergemask)

deringmask = muvf.AnimeMask(denoise, shift=0.2).std.Expr('x 10 < 0 x 3 * ?')
dering = hvf.YAHR(denoise, blur=6)
dering = core.std.MaskedMerge(denoise, dering, deringmask, 0)

aa = core.std.Transpose(dering)
planes = split(aa)
planes[0] = lvf.upscaled_sraa(planes[0], 2, 13, sharp_downscale=False)
planes[1], planes[2] = [taa.TAAmbk(plane, aatype='Eedi3SangNom', down8=False) for plane in planes[1:]]
aa = join(planes)
aa = core.std.Transpose(aa)
aa = fvf.rfs(aa, shift, mappings='[0 150]')

deband_a = core.f3kdb.Deband(aa, range=18, y=42, cb=36, cr=36, grainy=0, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(aa, range=20, y=56, cb=48, cr=48, grainy=8, grainc=0, output_depth=16)
deband_c = core.f3kdb.Deband(aa, range=24, y=72, cb=60, cr=60, grainy=16, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband_b, deband_c, adaptmask)
deband = core.std.MaskedMerge(deband, deband_a, linemask)
deband_d = core.f3kdb.Deband(deband, range=28, y=140, cb=110, cr=110, grainy=32, grainc=0, output_depth=16)
deband_h = core.f3kdb.Deband(deband, range=10, y=64, cb=48, cr=48, grainy=0, grainc=0, output_depth=16)
deband_hh = core.f3kdb.Deband(deband, range=15, y=84, cb=84, cr=84, grainy=32, grainc=0, output_depth=16)
deband = fvf.rfs(deband, deband_a, mappings=f'[{opstart} 21987]')
deband = fvf.rfs(deband, deband_d, mappings='[1614 1808]')
deband = fvf.rfs(deband, deband_h, mappings='[21988 22104]')
deband = fvf.rfs(deband, deband_hh, mappings='[22119 22331]')

grain_a = kgf.adaptive_grain(deband, 0.1)
grain_b = kgf.adaptive_grain(deband, 0.2)
grain_c = kgf.adaptive_grain(deband, 0.3)
grain = fvf.rfs(deband, grain_b, mappings=f'[{opstart} 21987]')
grain = fvf.rfs(deband, grain_c, mappings='[1614 1808]')

final = fvf.Depth(grain, 10)

final.set_output()