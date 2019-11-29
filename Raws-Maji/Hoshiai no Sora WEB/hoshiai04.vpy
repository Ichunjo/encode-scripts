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

src_a = lvf.src(r'WKN/Stars Align E04 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'FUNI/[HorribleSubs] Hoshiai no Sora - 04v2 [1080p].mkv')[241:]

asrc = r"WKN/Stars Align E04 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,2848)], 						asrc, "Hoshiai04a_cut.mka")
ac.eztrim(src_a, [(0,1),(2848,16118)], 				asrc, "Hoshiai04b_cut.mka")
ac.eztrim(src_a, [(0,13),(16118,0)], 				asrc, "Hoshiai04c_cut.mka")

blank = core.std.BlankClip(src_a)
src_a = src_a[:2848]+blank[:1]+src_a[2848:16118]+blank[:13]+src_a[16118:]+blank[:11]
src_a = core.std.FreezeFrames(src_a, first=[2848], last=[2848], replacement=[2847])
src_a = core.std.FreezeFrames(src_a, first=[16119], last=[16131], replacement=[16118])
src_a = core.std.FreezeFrames(src_a, first=[34635], last=[34645], replacement=[34634])

src_b = src_b+blank[:9]
src_b = core.std.FreezeFrames(src_b, first=[34635], last=[34645], replacement=[34634])

#scomp = lvf.stack_compare(src_a, src_b, height=540)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade = fvf.Depth(kgf.hardsubmask_fades(src_a, src_b, highpass=2000), 8)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.rfs(src, src_fade, mappings="[2916 3022] [34545 34645]")
#scomp2 = lvf.stack_compare(src, src_b, height=540)

src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

opstart, opend = 696, 2848

rescaled = lvf.smart_descale(src32, [837, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)
rescaled = fvf.Depth(rescaled, 16)

linemask = muvf.AnimeMask(src16, mode=-1).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
aa_light = taa.TAAmbk(src16, aatype='Nnedi3', mclip=linemask, opencl=True)
rescaled = fvf.rfs(rescaled, aa_light, mappings=f"[{opstart} {opend}] [34545 34645]")

deband_mask = kgf.retinex_edgemask(rescaled)
deband = core.f3kdb.Deband(rescaled, range=17, y=36, cb=40, cr=40, grainy=8, grainc=0, output_depth=16)
deband_op = core.f3kdb.Deband(rescaled, range=14, y=24, cb=24, cr=24, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband, deband_op, mappings=f"[{opstart} {opend}]")
deband = core.std.MaskedMerge(deband, rescaled, deband_mask)

grain = kgf.adaptive_grain(deband, 0.2, 14)

final = fvf.Depth(grain, 10)

final.set_output()