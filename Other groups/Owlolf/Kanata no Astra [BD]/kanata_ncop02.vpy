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

src = lvf.src(r"BDROM Vol 2v2/BDMV/STREAM/00019.m2ts")

asrc = r'BDROM Vol 2v2/BDMV/STREAM/00019.mka'
ac.eztrim(src,[(24,-24)], asrc,"kanata_ncop02.mka")

src = src[24:-24]

vrdf.generate_keyframes(src, 'kanata_ncop02_keyframes.log')

w_op, h_op = get_w(720), 720
b_op, c_op = 1/3, 1/3
opstart = 1

src = fvf.Depth(src, 16)

planes = kgf.split(src)
planes[1], planes[2] = [core.resize.Spline16(plane, src_left=0.5) for plane in planes[1:]]
shift = kgf.join(planes)

fe = core.edgefixer.ContinuityFixer(shift, [2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1])

fe16 = fvf.Depth(fe, 16)
fe32 = fvf.Depth(fe, 32)

desc_y_a = dsc.Descale(fe32, w_op, h_op, 'bicubic', b_op, c_op, gray=True)

den_y_a_a = mvf.BM3D(desc_y_a, 1.25, radius1=1, depth=16)
den_y_a_b = mvf.BM3D(desc_y_a, 2, radius1=1, depth=16)

den_y_a = fvf.rfs(den_y_a_a, den_y_a_b, mappings=f'[{opstart+1138} {opstart+1728}]')

den_uv = core.knlm.KNLMeansCL(fe32, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
den_uv = fvf.Depth(den_uv, 16)

scaled_y_a = vrdf.Bicubic(nnedi3_rpow2CL(den_y_a, 2, nns=4), 1920, 1080, 'robidoux')

aa_uv = taa.TAAmbk(den_uv, aatype=0, aatypeu='Nnedi3', aatypev='Nnedi3', down8=False, mtype=0, opencl=True)

merged = core.std.ShufflePlanes([scaled_y_a, aa_uv], [0, 1, 2], vs.YUV)

line_m = kgf.retinex_edgemask(scaled_y_a)
db_a = core.neo_f3kdb.Deband(merged, 16, 30, 30, 30, 24, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(merged, 22, 80, 60, 60, 32, 0, sample_mode=4)
db_c = core.neo_f3kdb.Deband(merged, 18, 48, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db_a, merged, line_m)

db = fvf.rfs(db, db_b, mappings=f'[{opstart} {opstart+14}]')
db = fvf.rfs(db, db_c, mappings=f'[{opstart+1138} {opstart+1728}]')

grain_a = kgf.adaptive_grain(db, 0.3, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.85, static=False, luma_scaling=18)
grain_d = kgf.adaptive_grain(grain_a, 2, static=False, luma_scaling=6)

grain = fvf.rfs(grain_a, grain_b, mappings=f'[{opstart+1138} {opstart+1728}]')
grain = fvf.rfs(grain, grain_d, f'[{opstart+1019} {opstart+1081}]')

final = fvf.Depth(grain, 10)

src.set_output()