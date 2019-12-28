import vapoursynth as vs
import vsTAAmbk as taa
import muvsfunc as muvf
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
import modfunc as mdf
import acsuite

core = vs.get_core()
core.max_cache_size = 32000
core.num_threads = 16

ac = acsuite.AC()

src_a = lvf.src(r'WKN/Stars Align E12 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
#src_b = lvf.src(r'AMZ/Hoshiai no Sora - 12 (Amazon Prime VBR 1080p).mkv')
src_b = lvf.src(r'Funi/[HorribleSubs] Hoshiai no Sora - 12 [1080p].mkv')[241:]

asrc = r"WKN/Stars Align E12 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,2703)], 			asrc, "Hoshiai12a_cut.mka")
ac.eztrim(src_a, [(0,4),(2703,18070)], 	asrc, "Hoshiai12b_cut.mka")
ac.eztrim(src_a, [(0,2),(18070,0)], 	asrc, "Hoshiai12c_cut.mka")

src_a = src_a[:2703]+(src_a[2702]*3)+src_b[2706]+src_a[2703:18070]+(src_a[18069]*2)+src_a[18070:]+(src_a[-1]*6)

src_b = src_b+(src_b[-1]*4)

vrdf.generate_keyframes(src_a, 'hoshiai12_keyframes.log')

#scomp = lvf.stack_compare(src_a, src_b, height=540)

src_a = fvf.Depth(src_a, 16)
src_b = fvf.Depth(src_b, 16)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade = kgf.hardsubmask_fades(src_a, src_b, highpass=2000)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.rfs(src, src_fade, mappings="[2718 2826] [4863 4886] [5734 5781] [6098 6134] [6325 6401] [8044 8047] [10543 10583] [11334 11371] [11807 11836] [12673 12722] [14144 14179] [15355 15378] [15613 15633] [15844 15863] [16317 16373] [18821 18865] [19329 19363] [20394 20420] [22453 22491] [22859 22894] [23116 23163] [24346 24380] [24514 24555] [25007 25073] [25260 25338] [26984 27025] [31934 32217] [34209 34285]")
#scomp2 = lvf.stack_compare(src, src_b, height=540)

opstart, opend = 1189, 2705

rescaled = mdf.smart_descale_mod(src, [837, 838, 863, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)

linemask = muvf.AnimeMask(src, mode=-1).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
aa_light = taa.TAAmbk(src, aatype='Nnedi3', mclip=linemask, opencl=True)
rescaled = fvf.rfs(rescaled, aa_light, mappings=f"[{opstart} {opend}]")

deband_mask = kgf.retinex_edgemask(rescaled)
deband = core.f3kdb.Deband(rescaled, range=17, y=36, cb=40, cr=40, grainy=8, grainc=0, output_depth=16)
deband_op = core.f3kdb.Deband(rescaled, range=14, y=24, cb=24, cr=24, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband, deband_op, mappings=f"[{opstart} {opend}]")
deband = core.std.MaskedMerge(deband, rescaled, deband_mask)

grain = kgf.adaptive_grain(deband, 0.2)

final = fvf.Depth(grain, 10)

final.set_output()