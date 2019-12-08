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

src_a = lvf.src(r'WKN/Stars Align E07 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'FUNI/[HorribleSubs] Hoshiai no Sora - 07 [1080p].mkv')[241:]

asrc = r"WKN/Stars Align E07 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,4285)], 					asrc, "Hoshiai07a_cut.mka")
ac.eztrim(src_a, [(0,2),(4285,17398)], 			asrc, "Hoshiai07b_cut.mka")
ac.eztrim(src_a, [(0,2),(17398,34486)], 		asrc, "Hoshiai07c_cut.mka")
ac.eztrim(src_a, [(0,27)], 						asrc, "Hoshiai07d_cut.mka")
ac.eztrim(src_a, [(0,28),(34486,0)], 			asrc, "Hoshiai07e_cut.mka")

blank = core.std.BlankClip(src_a)
src_a = src_a[:4285]+blank[:2]+src_a[4285:17398]+blank[:2]+src_a[17398:34486]+blank[:55]+src_a[34486:]+blank[:11]
src_a = core.std.FreezeFrames(src_a, first=[4285], last=[4286], replacement=[4284])
src_a = core.std.FreezeFrames(src_a, first=[17400], last=[17401], replacement=[17399])
src_a = core.std.FreezeFrames(src_a, first=[34490], last=[34490], replacement=[34489])
src_a = src_a[:34491]+src_b[34491:34492]+src_a[34492:]
src_a = core.std.FreezeFrames(src_a, first=[34492], last=[34544], replacement=[34491])
src_a = core.std.FreezeFrames(src_a, first=[34635], last=[34645], replacement=[34634])

src_b = src_b+blank[:9]
src_b = core.std.FreezeFrames(src_b, first=[34637], last=[34645], replacement=[34636])

#scomp = lvf.stack_compare(src_a, src_b, height=540)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade = fvf.Depth(kgf.hardsubmask_fades(src_a, src_b, highpass=2000), 8)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.rfs(src, src_fade, mappings="[899 925] [1581 1619] [3384 3418] [3689 3721] [3773 3828] [4394 4500] [5241 5270] [5775 5856] [6293 6413] [7333 7362] [7458 7487] [7616 7726] [7742 7768] [7787 7804] [7844 7866] [7993 8022] [8545 8578] [9259 9291] [9723 9764] [10721 10747] [11558 11610] [12206 12244] [12689 12742] [13577 13621] [16415 16489] [34545 34645]")
#scomp2 = lvf.stack_compare(src, src_b, height=540)

src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

opstart, opend = 4143, 4284

rescaled = lvf.smart_descale(src32, [837, 838, 863, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)

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