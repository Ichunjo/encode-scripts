import vapoursynth as vs
core = vs.get_core()
import vsTAAmbk as taa
import muvsfunc as muvf
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
from vsutil import get_w
import acsuite
ac = acsuite.AC()
core.max_cache_size = 32000
core.num_threads = 12

src_a = lvf.src(r'WKN/Stars Align E06 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'FUNI/[HorribleSubs] Hoshiai no Sora - 06 [1080p].mkv')[241:]

asrc = r"WKN/Stars Align E06 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,5892)], 			asrc, "Hoshiai06a_cut.mka")
ac.eztrim(src_a, [(0,2),(5892,0)], 		asrc, "Hoshiai06b_cut.mka")

blank = core.std.BlankClip(src_a)
src_a = src_a[:5892]+blank[:2]+src_a[5892:]+blank[:11]
src_a = core.std.FreezeFrames(src_a, first=[5892], last=[5893], replacement=[5891])
src_a = core.std.FreezeFrames(src_a, first=[34636], last=[34645], replacement=[34634])

src_b = src_b+blank[:9]
src_b = core.std.FreezeFrames(src_b, first=[34637], last=[34645], replacement=[34636])

#scomp = lvf.stack_compare(src_a, src_b, height=540)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade = fvf.Depth(kgf.hardsubmask_fades(src_a, src_b, highpass=2000), 8)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.rfs(src, src_fade, mappings="[5934 6041] [13643 13923] [13987 14020] [16173 16193] [16263 16298] [16374 16408] [16518 16564] [17653 17699] [17991 18023] [22569 22616] [23060 23140] [24260 24287] [24330 24355] [24504 24527] [24557 24599] [25273 25377] [29385 29413] [34545 34645]")
#scomp2 = lvf.stack_compare(src, src_b, height=540)

src32 = fvf.Depth(src, 32)
src16 = fvf.Depth(src, 16)

opstart, opend = 3741, 5893

rescaled_a = lvf.smart_descale(src32, [837, 838, 863, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)
rescaled_b = fvf.DescaleAA(src32, get_w(698), 698, kernel='bicubic', b=1/5, c=1/2, opencl=True, device=0)
rescaled = fvf.rfs(rescaled_a, rescaled_b, mappings="[603 647]")

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