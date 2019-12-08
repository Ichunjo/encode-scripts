import vapoursynth as vs
core = vs.get_core()
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
from vsutil import get_y
core.max_cache_size = 26000

#Most of the credit() and edgemask_aa() code has been taken in DescaleAA of fvsfunc
def edgemask_aa(src):
	bits = src.format.bits_per_sample
	sample_type = src.format.sample_type
	
	if sample_type == vs.INTEGER:
		maxvalue = (1 << bits) - 1
	else:
		maxvalue = 1
	
	thrlow = 4 * maxvalue // 0xFF if sample_type == vs.INTEGER else 4 / 0xFF
	thrhigh = 24 * maxvalue // 0xFF if sample_type == vs.INTEGER else 24 / 0xFF
	edgemask = core.std.Prewitt(get_y(src), planes=0)	
	edgemask = core.std.Expr(edgemask, "x {thrhigh} >= {maxvalue} x {thrlow} <= 0 x ? ?".format(thrhigh=thrhigh, maxvalue=maxvalue, thrlow=thrlow))
	mask = edgemask.std.Inflate().std.Deflate()
	return mask

src = lvf.src(r'symphogearxyz_nced01.m2ts')[24:2184]
src = fvf.Depth(src, 16)

edstart = 0

denoise_a = kgf.hybriddenoise(src, sigma=1.25, knl=0.35)
denoise_b = kgf.hybriddenoise(src, sigma=3, knl=0.75)
denoise = fvf.rfs(denoise_a, denoise_b, mappings=f'[{edstart} {edstart+639}] [{edstart+1046} {edstart+1409}]')

aa_mask = edgemask_aa(denoise)
aa = lvf.upscaled_sraa(denoise, 1.5, 13)
sharp = hvf.LSFmod(aa, strength=80, defaults='old', source=denoise)
aa = core.std.MaskedMerge(denoise, sharp, aa_mask)

adaptmask_db = vrdf.adaptive_mask(aa, 120)
edgemask_db = kgf.retinex_edgemask(aa, 1).std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
deband_a = core.f3kdb.Deband(aa, range=17, y=30, cb=30, cr=30, grainy=16, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(aa, range=17, y=40, cb=40, cr=40, grainy=24, grainc=0, output_depth=16)
deband_c = core.f3kdb.Deband(aa, range=17, y=46, cb=46, cr=46, grainy=24, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband_a, deband_b, adaptmask_db)
deband = core.std.MaskedMerge(deband, deband_a, edgemask_db)
deband = fvf.rfs(deband, deband_c, mappings=f'[{edstart} {edstart+639}] [{edstart+1046} {edstart+1409}]')

grain_a = kgf.adaptive_grain(deband, 0.4, luma_scaling=6)
grain_b = kgf.adaptive_grain(deband, 0.4, static=False)
grain = fvf.rfs(grain_a, grain_b, mappings=f'[{edstart} {edstart+639}] [{edstart+1046} {edstart+1409}]')

final = fvf.Depth(grain, 10)

final.set_output()