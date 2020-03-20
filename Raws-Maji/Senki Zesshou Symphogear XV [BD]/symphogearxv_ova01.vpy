import vapoursynth as vs
import fvsfunc as fvf
import lvsfunc as lvf
import modfunc as mdf
from cooldegrain import CoolDegrain
from nnedi3_rpow2CL import nnedi3_rpow2CL

core = vs.core
core.max_cache_size = 8192

src = lvf.src(r'SYMPHOGEAR_XV_VOL3/BDMV/STREAM/00007.m2ts')
import acsuite
ac = acsuite.AC()
asrc = r'SYMPHOGEAR_XV_VOL3/BDMV/STREAM/00007.mka'
ac.eztrim(src,[(24,-24)], 	asrc,"symphogearxv_ova01.mka")

desc_w, desc_h = 1551, 873
y_shift = 0.5*(src.height/desc_h)

src = src[24:-24]
src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=1, thsad=48, blksize=8, overlap=4, plane=4)
den = fvf.Depth(den, 32)
y = lvf.get_y(den)

desc_y = core.fmtc.resample(y, desc_w, desc_h, sh=src.height+y_shift, kernel='bicubic', a1=0, a2=1/2, invks=True)
upsc_y = nnedi3_rpow2CL(desc_y, 2, nsize=4)
scaled_y = core.fmtc.resample(upsc_y, 1920, 1080, sh=upsc_y.height-1, kernel='spline64')
scaled_a = core.std.ShufflePlanes([scaled_y, den], [0, 1, 2], vs.YUV)

desc_y = core.fmtc.resample(y, lvf.get_w(800), 800, kernel='bicubic', a1=0, a2=1/2, invks=True)
scaled_y = nnedi3_rpow2CL(desc_y, 2, 1920, 1080, kernel='spline64', nsize=4)
scaled_b = core.std.ShufflePlanes([scaled_y, den], [0, 1, 2], vs.YUV)

scaled = fvf.rfs(scaled_a, scaled_b, mappings='[8952 9695]')
scaled = fvf.Depth(scaled, 16)

line_m = mdf.retinex_edgemask_mod(scaled, opencl=True)
db = core.neo_f3kdb.Deband(scaled, 17, 24, 24, 24, 0, 0, sample_mode=4)

final = core.fmtc.bitdepth(db, bits=10, dmode=3)


final.set_output()