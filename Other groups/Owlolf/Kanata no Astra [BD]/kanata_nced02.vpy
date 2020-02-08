import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import mvsfunc as mvf
import vsTAAmbk as taa
import vardefunc as vrdf
import descale as dsc
import acsuite
from nnedi3_rpow2CL import nnedi3_rpow2CL
from vsutil import get_w

core = vs.core
core.max_cache_size = 32000

ac = acsuite.AC()

src = lvf.src(r"BDROM Vol 2v2/BDMV/STREAM/00021.m2ts")

asrc = r'BDROM Vol 2v2/BDMV/STREAM/00021.mka'
ac.eztrim(src,[(24,-24)], asrc,"kanata_nced02.mka")

src = src[24:-24]

vrdf.generate_keyframes(src, 'kanata_nced02_keyframes.log')

src = fvf.Depth(src, 32)

desc_y = dsc.Descale(src, get_w(990), 990, 'bicubic', 1/3, 1/3, gray=True)
scaled_y = vrdf.Bicubic(nnedi3_rpow2CL(desc_y, 2, nns=4), 1920, 1080, 'catrom')
scaled_y = fvf.Depth(scaled_y, 16)

den_uv = core.knlm.KNLMeansCL(src, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
den_uv = fvf.Depth(den_uv, 16)

line_m = kgf.retinex_edgemask(scaled_y)

aa_y = lvf.upscaled_sraa(scaled_y, 1.5, 20, sharp_downscale=False)
aa_y = core.std.MaskedMerge(scaled_y, aa_y, line_m)
rep_y = core.edgefixer.ContinuityFixer(aa_y, 4, 0, 4, 0)

aa_uv = taa.TAAmbk(den_uv, aatype=0, aatypeu='Nnedi3', aatypev='Nnedi3', down8=False, mtype=0, opencl=True)

merged = core.std.ShufflePlanes([rep_y, aa_uv], [0, 1, 2], vs.YUV)

db_a = core.neo_f3kdb.Deband(merged, 16, 30, 30, 30, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, merged, line_m)

final = fvf.Depth(db, 10)

final.set_output()