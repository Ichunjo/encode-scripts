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

src_a = lvf.src(r'WKN/Stars Align E02 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'FUNI/[HorribleSubs] Hoshiai no Sora - 02 [1080p].mkv')[241:]

asrc = r"WKN/Stars Align E02 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,1623)], 			asrc, "Hoshiai02a_cut.mka")
ac.eztrim(src_a, [(0,3),(1623,0)], 		asrc, "Hoshiai02b_cut.mka")

blank = core.std.BlankClip(src_a)
src_a = src_a[:1623]+blank[:3]+src_a[1623:]+blank[:12]
src_a = core.std.FreezeFrames(src_a, first=[1623], last=[1625], replacement=[1622])
src_a = core.std.FreezeFrames(src_a, first=[34634], last=[34645], replacement=[34633])
#scomp = lvf.stack_compare(src_a, src_b, height=540)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)
#scomp1 = lvf.stack_compare(src, src_b, height=540)

src_fade = fvf.Depth(kgf.hardsubmask_fades(src_a, src_b, highpass=2000), 8)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.ReplaceFramesSimple(src, src_fade, mappings="[97 123] [1637 1745] [34544 34645]")
#scomp2 = lvf.stack_compare(src, src_b, height=540)

src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

rescaled = lvf.smart_descale(src32, [837, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)
rescaled = fvf.Depth(rescaled, 16)

linemask = muvf.AnimeMask(src16, mode=-1).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
aa_light = taa.TAAmbk(src16, aatype='Nnedi3', mclip=linemask, opencl=True)
rescaled = fvf.rfs(rescaled, aa_light, mappings="[34544 34645]")

deband_mask = kgf.retinex_edgemask(rescaled)
deband = core.f3kdb.Deband(rescaled, range=17, y=36, cb=40, cr=40, grainy=8, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband, rescaled, deband_mask)

grain = kgf.adaptive_grain(deband, 0.2, 14)

final = fvf.Depth(grain, 10)

final.set_output()