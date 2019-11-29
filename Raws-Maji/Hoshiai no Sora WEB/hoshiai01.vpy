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

src_a = lvf.src(r'WKN/Stars Align E01 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'FUNI/[HorribleSubs] Hoshiai no Sora - 01 [1080p].mkv')[241:]

asrc = r"WKN/Stars Align E01 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,8076)], 				asrc, "Hoshiai01a_cut.mka")
ac.eztrim(src_a, [(0,2),(8076,16394)], 		asrc, "Hoshiai01b_cut.mka")
ac.eztrim(src_a, [(0,2),(16394,34486)], 	asrc, "Hoshiai01c_cut.mka")
ac.eztrim(src_a, [(0,3),(34486,0)], 		asrc, "Hoshiai01d_cut.mka")

blank = core.std.BlankClip(src_a)
src_b = blank[0:2]+src_b
src_a = src_a[:8076]+blank[:2]+src_a[8076:16394]+blank[:2]+src_a[16394:34486]+blank[:3]+src_a[34486:]+blank[0:63]
src_a = core.std.FreezeFrames(src_a, first=[8076], last=[8077], replacement=[8075])
src_a = core.std.FreezeFrames(src_a, first=[16396], last=[16397], replacement=[16395])
src_a = core.std.FreezeFrames(src_a, first=[34490], last=[34492], replacement=[34489])
src_a = core.std.FreezeFrames(src_a, first=[34583], last=[34645], replacement=[34582])
scomp = lvf.stack_compare(src_a, src_b, height=540)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)
#scomp1 = lvf.stack_compare(src, src_b, height=540)

src_fade = fvf.Depth(kgf.hardsubmask_fades(src_a, src_b, highpass=2000), 8)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.ReplaceFramesSimple(src, src_fade, mappings="[8147 8350] [12084 12116] [14896 14931] [15855 15902] [16527 16622] [17535 17582] [17925 17966] [34187 34492]")
#scomp2 = lvf.stack_compare(src, src_b, height=540)

src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

rescaled = lvf.smart_descale(src32, [837, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)
rescaled = fvf.Depth(rescaled, 16)

linemask = muvf.AnimeMask(src16, mode=-1).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
aa_light = taa.TAAmbk(src16, aatype='Nnedi3', mclip=linemask, opencl=True)
rescaled = fvf.rfs(rescaled, aa_light, mappings="[34545 34645]")

deband_mask = kgf.retinex_edgemask(rescaled)
deband = core.f3kdb.Deband(rescaled, range=17, y=36, cb=40, cr=40, grainy=8, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband, rescaled, deband_mask)

grain = kgf.adaptive_grain(deband, 0.2, 14)

final = fvf.Depth(grain, 10)

final.set_output()