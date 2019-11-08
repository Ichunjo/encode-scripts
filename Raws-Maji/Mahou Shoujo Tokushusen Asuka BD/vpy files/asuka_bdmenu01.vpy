import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import muvsfunc as muvf
import fvsfunc as fvf
import vsTAAmbk as taa
import havsfunc as hvf
from nnedi3_rpow2CL import nnedi3_rpow2CL
import vardefunc as vrdf
from vsutil import get_y, get_w
core = vs.core
core.max_cache_size = 32000

src = lvf.src(r"asuka_bdmenu01.m2ts")
src = fvf.Depth(src, 16)

end = src.num_frames-1

scaled = fvf.DescaleAA(src, get_w(838), 838, kernel='bicubic', b=0, c=1/2, opencl=True, device=0)

denoise = vrdf.KNLMeansCL(scaled, 0.55, 0, 0)
denoise_mask = vrdf.adaptive_mask(scaled, 10)
denoise = core.std.MaskedMerge(denoise, scaled, denoise_mask)

line = kgf.retinex_edgemask(denoise)

aa = lvf.transpose_aa(denoise, eedi3=True)
aa = core.std.MaskedMerge(denoise, aa, line)

deband = core.f3kdb.Deband(aa, range=17, y=34, cb=26, cr=26, grainy=16, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband, aa, line)

deband = fvf.rfs(deband, src, mappings=f'[192 {end}]')

final = fvf.Depth(deband, 10)
final.set_output()