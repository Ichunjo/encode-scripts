import vapoursynth as vs
core = vs.get_core()
import vsTAAmbk as taa
import muvsfunc as muvf
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import acsuite
ac = acsuite.AC()
core.max_cache_size = 32000
core.num_threads = 12

src_a = lvf.src(r'WKN/Stars Align E05 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'FUNI/[HorribleSubs] Hoshiai no Sora - 05 [1080p].mkv')[241:]

asrc = r"WKN/Stars Align E05 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,3015)], 						asrc, "Hoshiai05a_cut.mka")
ac.eztrim(src_a, [(0,2),(3015,34519)], 				asrc, "Hoshiai05b_cut.mka")
ac.eztrim(src_a, [(0,13)], 							asrc, "Hoshiai05c_cut.mka")
ac.eztrim(src_a, [(0,12)], 							asrc, "Hoshiai05d_cut.mka")
ac.eztrim(src_a, [(34519,0)], 						asrc, "Hoshiai05e_cut.mka")

blank = core.std.BlankClip(src_a)
src_a = src_a[:3015]+blank[:2]+src_a[3015:34519]+blank[:25]+src_a[34519:]+blank[:10]
src_a = core.std.FreezeFrames(src_a, first=[3015], last=[3016], replacement=[3014])
src_a = core.std.FreezeFrames(src_a, first=[34521], last=[34545], replacement=[34520])
src_a = core.std.FreezeFrames(src_a, first=[34636], last=[34645], replacement=[34635])

src_b = src_b+blank[:8]
src_b = core.std.FreezeFrames(src_b, first=[34638], last=[34645], replacement=[34637])

#scomp = lvf.stack_compare(src_a, src_b, height=540)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade = fvf.Depth(kgf.hardsubmask_fades(src_a, src_b, highpass=2000), 8)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.rfs(src, src_fade, mappings="[3024 3133] [17102 17235] [20902 20955] [34546 34645]")
# scomp2 = lvf.stack_compare(src, src_b, height=540)

src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

opstart, opend = 864, 3016

rescaled = lvf.smart_descale(src32, [837, 838, 863, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)
rescaled = fvf.Depth(rescaled, 16)

linemask = muvf.AnimeMask(src16, mode=-1).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
aa_light = taa.TAAmbk(src16, aatype='Nnedi3', mclip=linemask, opencl=True)
rescaled = fvf.rfs(rescaled, aa_light, mappings=f"[{opstart} {opend}] [34546 34645]")

aa_plus = taa.TAAmbk(rescaled, aatype='Eedi3SangNom', mtype=2, aa=24)
rescaled = fvf.rfs(rescaled, aa_plus, mappings="[14334 14465]")

deband_mask = kgf.retinex_edgemask(rescaled)
deband = core.f3kdb.Deband(rescaled, range=17, y=36, cb=40, cr=40, grainy=8, grainc=0, output_depth=16)
deband_op = core.f3kdb.Deband(rescaled, range=14, y=24, cb=24, cr=24, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband, deband_op, mappings=f"[{opstart} {opend}]")
deband = core.std.MaskedMerge(deband, rescaled, deband_mask)

grain = kgf.adaptive_grain(deband, 0.2, 14)

final = fvf.Depth(grain, 10)

final.set_output()