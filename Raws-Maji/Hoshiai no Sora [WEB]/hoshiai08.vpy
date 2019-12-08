import vapoursynth as vs
core = vs.get_core()
import vsTAAmbk as taa
import muvsfunc as muvf
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
from vsutil import get_w
import acsuite
ac = acsuite.AC()
core.max_cache_size = 32000
core.num_threads = 12

src_a = lvf.src(r'WKN/Stars Align E08 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'AMZ/Hoshiai no Sora - 08 (Amazon Prime VBR 1080p).mkv')

asrc = r"WKN/Stars Align E08 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,4333)], 					asrc, "Hoshiai08a_cut.mka")
ac.eztrim(src_a, [(0,2),(4333,18644)], 			asrc, "Hoshiai08b_cut.mka")
ac.eztrim(src_a, [(0,2),(18644,34491)], 		asrc, "Hoshiai08c_cut.mka")
ac.eztrim(src_a, [(34501,34560)], 				asrc, "Hoshiai08d_cut.mka")
ac.eztrim(src_a, [(34491,0)], 					asrc, "Hoshiai08e_cut.mka")

blank = core.std.BlankClip(src_a)
src_a = src_a[:4333]+blank[:2]+src_a[4333:18644]+blank[:2]+src_a[18644:34491]+blank[:49]+src_a[34491:]+blank[:12]
src_a = core.std.FreezeFrames(src_a, 4333, 4334, 4332)
src_a = core.std.FreezeFrames(src_a, 18646, 18647, 18645)
src_a = src_a[:34495]+src_b[34495:34496]+src_a[34496:]
src_a = core.std.FreezeFrames(src_a, 34496, 34543, 34495)
src_a = core.std.FreezeFrames(src_a, 34634, 34645, 34633)

src_b = src_b+blank[:10]
src_b = core.std.FreezeFrames(src_b, 34636, 34645, 34635)

#scomp = lvf.stack_compare(src_a, src_b, height=540)

src_a = fvf.Depth(src_a, 32)
src_b = fvf.Depth(src_b, 32)

hardsubmask = kgf.hardsubmask(src_a, src_b)

denoise_src_b = vrdf.KNLMCL(src_b, 5.5, 3)
grain_src_b = core.grain.Add(denoise_src_b, var=80, uvar=20, constant=False)
src_b = fvf.rfs(src_b, grain_src_b, mappings="[0 2181]")

src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade = kgf.hardsubmask_fades(src_a, src_b, highpass=2000)
src_fade = core.std.MaskedMerge(fvf.Depth(src_a, 16), fvf.Depth(src_b, 16), src_fade)

src = fvf.rfs(fvf.Depth(src, 16), src_fade, mappings="[4335 4442] [15893 15988] [24400 24441] [34544 34645]")
#scomp2 = lvf.stack_compare(src, fvf.Depth(src_b, 16), height=540)

src32 = fvf.Depth(src, 32)
src16 = src

opstart, opend = 2182, 4334

rescaled_a = lvf.smart_descale(src32, [837, 838, 863, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)
rescaled_b = fvf.DescaleAA(src32, get_w(748), 748, kernel='bicubic', b=1/5, c=1/2, opencl=True, device=0)
rescaled = fvf.rfs(rescaled_a, rescaled_b, mappings="[28579 28746]")
rescaled = fvf.Depth(rescaled, 16)

linemask = muvf.AnimeMask(src16, mode=-1).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
aa_light = taa.TAAmbk(src16, aatype='Nnedi3', mclip=linemask, opencl=True)
rescaled = fvf.rfs(rescaled, aa_light, mappings=f"[{opstart} {opend}] [34544 34645]")

aa_plus = taa.TAAmbk(rescaled, aatype='Eedi3SangNom', alpha=0.8, cycle=1, mtype=0, aa=48)
rescaled = fvf.ReplaceFramesSimple(rescaled, aa_plus, mappings="[28579 28746]")

deband_mask = kgf.retinex_edgemask(rescaled)
deband = core.f3kdb.Deband(rescaled, range=17, y=36, cb=40, cr=40, grainy=8, grainc=0, output_depth=16)
deband_op = core.f3kdb.Deband(rescaled, range=14, y=24, cb=24, cr=24, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband, deband_op, mappings=f"[{opstart} {opend}]")
deband = core.std.MaskedMerge(deband, rescaled, deband_mask)

grain = kgf.adaptive_grain(deband, 0.2, 14)
grain = fvf.rfs(grain, src16, mappings="[0 2181]")

final = fvf.Depth(grain, 10)

final.set_output()