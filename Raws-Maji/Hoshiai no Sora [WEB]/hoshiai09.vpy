import vapoursynth as vs
import vsTAAmbk as taa
import muvsfunc as muvf
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
from vsutil import get_w
import acsuite

core = vs.get_core()
core.max_cache_size = 32000
core.num_threads = 14

ac = acsuite.AC()

src_a = lvf.src(r'WKN/Stars Align E09 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'AMZ/Hoshiai no Sora - 09 (Amazon Prime VBR 1080p).mkv')

asrc = r"WKN/Stars Align E09 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,4047)], 						asrc, "Hoshiai09a_cut.mka")
ac.eztrim(src_a, [(0,2),(4047,15171)], 				asrc, "Hoshiai09b_cut.mka")
ac.eztrim(src_a, [(0,2),(15171,34543),(34544,0)], 	asrc, "Hoshiai09c_cut.mka")
ac.eztrim(src_a, [(0,9)], 							asrc, "Hoshiai09d_cut.mka")

blank = core.std.BlankClip(src_a)
src_a = src_a[:4047]+blank[:2]+src_a[4047:15171]+blank[:2]+src_a[15171:34543]+src_a[34544:]+blank[:9]
src_a = core.std.FreezeFrames(src_a, 4047, 4048, 4046)
src_a = core.std.FreezeFrames(src_a, 15173, 15174, 15172)
src_a = core.std.FreezeFrames(src_a, 34637, 34645, 34636)

src_b = src_b+blank[:7]
src_b = core.std.FreezeFrames(src_b, 34639, 34645, 34638)

#scomp = lvf.stack_compare(src_a, src_b, height=540)

src_a = fvf.Depth(src_a, 16)
src_b = fvf.Depth(src_b, 16)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade = kgf.hardsubmask_fades(src_a, src_b, highpass=2000)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.rfs(src, src_fade, mappings="[4075 4180] [34547 34645]")
#scomp2 = lvf.stack_compare(src, src_b, height=540)

src32 = fvf.Depth(src, 32)
src16 = src

opstart, opend = 1894, 4048

rescaled = lvf.smart_descale(src32, [837, 838, 863, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)
rescaled = fvf.Depth(rescaled, 16)

linemask = muvf.AnimeMask(src16, mode=-1).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
aa_light = taa.TAAmbk(src16, aatype='Nnedi3', mclip=linemask, opencl=True)
rescaled = fvf.rfs(rescaled, aa_light, mappings=f"[{opstart} {opend}] [34547 34645]")

deband_mask = kgf.retinex_edgemask(rescaled)
deband = core.f3kdb.Deband(rescaled, range=17, y=36, cb=40, cr=40, grainy=8, grainc=0, output_depth=16)
deband_op = core.f3kdb.Deband(rescaled, range=14, y=24, cb=24, cr=24, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband, deband_op, mappings=f"[{opstart} {opend}]")
deband = core.std.MaskedMerge(deband, rescaled, deband_mask)

grain = kgf.adaptive_grain(deband, 0.2, 14)

final = fvf.Depth(grain, 10)

final.set_output()