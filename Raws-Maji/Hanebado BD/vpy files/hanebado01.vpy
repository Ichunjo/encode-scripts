import vapoursynth as vs
core = vs.get_core()
import muvsfunc as muvf
import mvsfunc as mvf
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
from nnedi3_rpow2 import nnedi3_rpow2
from vsutil import insert_clip
core.max_cache_size = 64000

src = lvf.src(r'hanebado01.m2ts')[:34526]
src = fvf.Depth(src, 16)

opstart, opend = 4531, 6688

luma, chroma_u, chroma_v = kgf.split(src)

denoise_u = hvf.SMDegrain(chroma_u, tr=2, thSAD=200)
denoise_v = hvf.SMDegrain(chroma_v, tr=2, thSAD=200)

fixedges = core.edgefixer.ContinuityFixer(luma, 1, 1, 1, 1)

descaled = fvf.Depth(core.descale.Debicubic(fvf.Depth(fixedges, 32), 1280, 720, 0, 1/2), 16)

edgemask = core.std.Binarize(kgf.retinex_edgemask(descaled, 0.1), 12000)
adaptmask = vrdf.adaptive_mask(descaled, 35)
linemask = core.std.Expr([adaptmask, edgemask], expr="x y -").rgvs.RemoveGrain(3)

denoise_luma_a = mvf.BM3D(descaled, 1)
denoise_luma_b = mvf.BM3D(descaled, 3)
denoise_luma = core.std.MaskedMerge(denoise_luma_a, denoise_luma_b, linemask)
denoise_luma = fvf.rfs(denoise_luma, descaled, mappings='[22133 22334] [24099 24216]')
contra = hvf.ContraSharpening(denoise_luma, descaled)

scaled = nnedi3_rpow2(contra).resize.Spline36(1920, 1080)
merged = kgf.join([scaled, denoise_u, denoise_v])
aa = taa.TAAmbk(merged, aatype='Nnedi3SangNom', mtype=2, repair=2, sharp=100)

##OP##
denoise_op_a = mvf.BM3D(fixedges, 3)
denoise_op_b = mvf.BM3D(fixedges, 4)
denoise_op = fvf.rfs(denoise_op_a, denoise_op_b, mappings=f'[{opstart+338} {opstart+662}] [{opstart+704} {opstart+718}] [{opstart+1556} {opstart+1799}]')
contra_op = hvf.ContraSharpening(denoise_op, fixedges)
contra_op = fvf.rfs(denoise_op, contra_op, mappings=f'[{opstart+704} {opstart+718}]')

merged_op = kgf.join([contra_op, denoise_u, chroma_v])

linemask_op = core.std.Sobel(contra_op).std.Expr("x 2 *").std.Maximum()
aa_op_a = taa.TAAmbk(merged_op, aatype='Nnedi3', mclip=linemask_op)
aa_op_b = taa.TAAmbk(merged_op, aatype='Eedi3', mtype=0)
aa_op_c = taa.TAAmbk(merged_op, aatype='Eedi3', cycle=2, mclip=linemask_op)
aa_op = fvf.rfs(aa_op_a, aa_op_b, mappings=f'[{opstart+719} {opstart+733}]')
aa_op = fvf.rfs(aa_op, aa_op_c, mappings=f'[{opstart+1886} {opstart+1940}]')
##OP##

merged_all = fvf.rfs(aa, aa_op, mappings=f'[{opstart} {opend}]')

c_mask = vrdf.DRM(fixedges, 720, 'Spline36', mthr=120).std.Inflate()
credit = fvf.rfs(merged_all, core.std.MaskedMerge(merged_all, src, c_mask, 0), mappings=f'[464 523] [{opstart} {opend}] [6713 6856] [17814  17885] [28603 30740] [34424 34525]')

deband_a_a = core.f3kdb.Deband(credit, range=17, y=30, cb=24, cr=24, grainy=0, grainc=0, output_depth=16)
deband_a_b = core.f3kdb.Deband(credit, range=17, y=38, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
deband_a = core.std.MaskedMerge(deband_a_a, deband_a_b, core.resize.Bilinear(linemask, 1920, 1080))
deband_b = core.f3kdb.Deband(credit, range=24, y=56, cb=42, cr=42, grainy=0, grainc=0, output_depth=16)
deband_c = core.f3kdb.Deband(credit, range=18, y=48, cb=36, cr=36, grainy=0, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings=f'[{opstart+338} {opstart+662}] [{opstart+1556} {opstart+1799}]')
deband = fvf.rfs(deband, deband_c, mappings=f'[{opstart+704} {opstart+718}] [{opstart+1096} {opstart+1453}] [{opstart+1941} {opend}]')
deband = fvf.rfs(deband, credit, mappings='[22133 22334] [24099 24216]')

grain = kgf.adaptive_grain(deband, 0.2, luma_scaling=16, static=False)
grain = fvf.rfs(grain, deband, mappings='[22133 22334] [24099 24216]')

final = fvf.Depth(grain, 10)
final.set_output()