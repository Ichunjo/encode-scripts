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

src_a = lvf.src(r'WKN/Stars Align E11 [1080p][AAC][JapDub][GerSub][Web-DL].mkv')
src_b = lvf.src(r'AMZ/Hoshiai no Sora - 11 (Amazon Prime VBR 1080p).mkv')

asrc = r"WKN/Stars Align E11 [1080p][AAC][JapDub][GerSub][Web-DL].mka"
ac.eztrim(src_a, [(0,2823)], 				asrc, "Hoshiai11a_cut.mka")
ac.eztrim(src_a, [(0,2),(2823,20469)], 		asrc, "Hoshiai11b_cut.mka")
ac.eztrim(src_a, [(0,1),(20469,34468)], 	asrc, "Hoshiai11c_cut.mka")

src_a = src_a[:2823]+(src_a[2822]*2)+src_a[2823:20469]+src_a[20468]+src_a[20469:34468]+(src_a[34467]*2)+(src_b[34473]*73)+src_a[34468:]+(src_a[-1]*10)

src_b = src_b+(src_b[-1]*8)

vrdf.generate_keyframes(src_a, 'hoshiai11_keyframes.log')

#scomp = lvf.stack_compare(src_a, src_b, height=540)

src_a = fvf.Depth(src_a, 16)
src_b = fvf.Depth(src_b, 16)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade = kgf.hardsubmask_fades(src_a, src_b, highpass=2000)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.rfs(src, src_fade, mappings="[2062 2182] [2838 2944] [3748 3777] [4535 4564] [5028 5057] [5391 5420] [6673 6706] [8663 8692] [9713 9742] [10586 10615] [10790 10796] [10807 10854] [11308 11340] [11717 11763] [12684 12713] [13324 13353] [17344 17513] [29252 29279] [29835 29885] [30022 30051] [30160 30258] [34546 34645]")
#scomp2 = lvf.stack_compare(src, src_b, height=540)

opstart, opend = 2681, 2824

rescaled = mdf.smart_descale_mod(src, [837, 838, 863, 959], 1/5, 1/2, single_rate_upscale=True, rep=13, sharp_downscale=True)

linemask = muvf.AnimeMask(src, mode=-1).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3)
aa_light = taa.TAAmbk(src, aatype='Nnedi3', mclip=linemask, opencl=True)
rescaled = fvf.rfs(rescaled, aa_light, mappings=f"[{opstart} {opend}] [34546 34645]")

deband_mask = kgf.retinex_edgemask(rescaled)
deband = core.f3kdb.Deband(rescaled, range=17, y=36, cb=40, cr=40, grainy=8, grainc=0, output_depth=16)
deband_op = core.f3kdb.Deband(rescaled, range=14, y=24, cb=24, cr=24, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband, deband_op, mappings=f"[{opstart} {opend}]")
deband = core.std.MaskedMerge(deband, rescaled, deband_mask)

grain = kgf.adaptive_grain(deband, 0.2)

final = fvf.Depth(grain, 10)

final.set_output()