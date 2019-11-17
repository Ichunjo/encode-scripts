import vapoursynth as vs
core = vs.get_core()
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
from vsutil import get_y, get_w
core.max_cache_size = 26000

#Most of the credit() and edgemask_aa() code has been taken in DescaleAA of fvsfunc
def credit(src, w=get_w(871), h=871, thr=10, expand=3, inflate=3):
	ow = src.width
	oh = src.height
	
	bits = src.format.bits_per_sample
	sample_type = src.format.sample_type
	
	if sample_type == vs.INTEGER:
		maxvalue = (1 << bits) - 1
		thr = thr * maxvalue // 0xFF
	else:
		maxvalue = 1
		thr /= (235 - 16)
	
	luma = get_y(src)
	descale = fvf.Resize(luma, w, h, kernel='bicubic', a1=1/3, a2=1/3, invks=True)
	upscale = fvf.Resize(descale, ow, oh, kernel='bicubic', a1=1/3, a2=1/3)
	diffmask = core.std.Expr([luma, upscale], 'x y - abs')
	for _ in range(expand):
		diffmask = core.std.Maximum(diffmask, planes=0)
	for _ in range(inflate):
		diffmask = core.std.Inflate(diffmask, planes=0)
	
	thrlow = 4 * maxvalue // 0xFF if sample_type == vs.INTEGER else 4 / 0xFF
	thrhigh = 24 * maxvalue // 0xFF if sample_type == vs.INTEGER else 24 / 0xFF
	edgemask = core.std.Prewitt(luma, planes=0)	
	edgemask = core.std.Expr(edgemask, "x {thrhigh} >= {maxvalue} x {thrlow} <= 0 x ? ?".format(thrhigh=thrhigh, maxvalue=maxvalue, thrlow=thrlow))
	
	mask = core.std.Expr([diffmask, edgemask], 'x {thr} >= 0 y ?'.format(thr=thr))
	mask = mask.std.Inflate().std.Deflate()
	
	return mask
	
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
		
src = lvf.src(r'symphogearxyz01.m2ts')
src = fvf.Depth(src, 16)

denoise = kgf.hybriddenoise(src, sigma=1.25, knl=0.35)

aa_mask = edgemask_aa(denoise)
aa_credit_mask = credit(denoise)
luma = get_y(denoise)
aa = taa.TAAmbk(luma, aatype='Eedi3', mclip=aa_mask, sharp=80)
aa_credit = taa.TAAmbk(luma, aatype='Eedi3', mclip=aa_credit_mask, sharp=80)
aa = fvf.rfs(aa, aa_credit, mappings='[0 1934] [18163 18328]')
aa = fvf.rfs(aa, luma, mappings='[31516 34045]')
aa = core.std.ShufflePlanes([aa, denoise], [0, 1, 2], vs.YUV)

adaptmask_db = vrdf.adaptive_mask(aa, 120)
edgemask_db = kgf.retinex_edgemask(aa, 1).rgvs.RemoveGrain(3)
edgemask_db = core.std.Binarize(edgemask_db, 9500).std.Inflate()
deband_a = core.f3kdb.Deband(aa, range=17, y=30, cb=30, cr=30, grainy=16, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(aa, range=17, y=40, cb=40, cr=40, grainy=24, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband_a, deband_b, adaptmask_db)
deband = core.std.MaskedMerge(deband, deband_a, edgemask_db)

grain = kgf.adaptive_grain(deband, 0.4, luma_scaling=6)

final = fvf.Depth(grain, 10)

final.set_output()