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

src_a = lvf.src(r'WKN/Stars Align E03 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'FUNI/[HorribleSubs] Hoshiai no Sora - 03 [1080p].mkv')[241:]

asrc = r"WKN/Stars Align E03 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,3542)], 						asrc, "Hoshiai03a_cut.mka")
ac.eztrim(src_a, [(0,3),(3542,17589)], 				asrc, "Hoshiai03b_cut.mka")
ac.eztrim(src_a, [(0,3),(17589,34515)], 			asrc, "Hoshiai03c_cut.mka")
ac.eztrim(src_a, [(0,12)], 							asrc, "Hoshiai03d_cut.mka")
ac.eztrim(src_a, [(0,12)], 							asrc, "Hoshiai03e_cut.mka")
ac.eztrim(src_a, [(0,1),(34515,0)], 				asrc, "Hoshiai03f_cut.mka")

blank = core.std.BlankClip(src_a)
src_a = src_a[:3542]+blank[:3]+src_a[3542:17589]+blank[:3]+src_a[17589:34515]+blank[:25]+src_a[34515:]+blank[:10]
src_a = core.std.FreezeFrames(src_a, first=[3542], last=[3544], replacement=[3541])
src_a = core.std.FreezeFrames(src_a, first=[17592], last=[17594], replacement=[17591])
src_a = core.std.FreezeFrames(src_a, first=[34521], last=[34545], replacement=[34520])
src_a = core.std.FreezeFrames(src_a, first=[34636], last=[34645], replacement=[34635])
#scomp = lvf.stack_compare(src_a, src_b, height=540)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)
#scomp1 = lvf.stack_compare(src, src_b, height=540)

src_fade = fvf.Depth(kgf.hardsubmask_fades(src_a, src_b, highpass=2000), 8)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.ReplaceFramesSimple(src, src_fade, mappings="[3587 3693] [34546 34645]")
#scomp2 = lvf.stack_compare(src, src_b, height=540)

src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

rescaled = lvf.smart_descale(src32, [837, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)
rescaled = fvf.Depth(rescaled, 16)

linemask = muvf.AnimeMask(src16, mode=-1).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
aa_light = taa.TAAmbk(src16, aatype='Nnedi3', mclip=linemask, opencl=True)
rescaled = fvf.rfs(rescaled, aa_light, mappings="[34546 34645]")

aa_plus = taa.TAAmbk(rescaled, aatype='Eedi3SangNom', mtype=2, aa=36)
rescaled = fvf.ReplaceFramesSimple(rescaled, aa_plus, mappings="[9468 9518]")

deband_mask = kgf.retinex_edgemask(rescaled)
deband = core.f3kdb.Deband(rescaled, range=17, y=36, cb=40, cr=40, grainy=8, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband, rescaled, deband_mask)

grain = kgf.adaptive_grain(deband, 0.2, 14)

final = fvf.Depth(grain, 10)

final.set_output()