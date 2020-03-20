import vapoursynth as vs
import fvsfunc as fvf
import lvsfunc as lvf
import modfunc as mdf
from cooldegrain import CoolDegrain
from nnedi3_rpow2 import nnedi3_rpow2

core = vs.core
core.max_cache_size = 8192

src = lvf.src(r'SYMPHOGEAR_XV_VOL5/BDMV/STREAM/00007.m2ts')
import acsuite
ac = acsuite.AC()
asrc = r'SYMPHOGEAR_XV_VOL5/BDMV/STREAM/00007.mka'
ac.eztrim(src,[(24,-24)], 	asrc,"symphogearxv_ova03.mka")

desc_w, desc_h = 1551, 873
y_shift = (src.height/desc_h)*0.25

src = src[24:-24]
src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=1, thsad=48, blksize=8, overlap=4, plane=4)
den = fvf.Depth(den, 32)
planes = lvf.split(den)


desc_y = core.fmtc.resample(planes[0], desc_w, desc_h, sy=-y_shift, sh=-y_shift, kernel='bicubic', a1=0, a2=1/2, invks=True)
upsc_y = nnedi3_rpow2(desc_y, 2, 1920, 1080, kernel='spline64', nsize=4, nns=4, qual=2)
planes[1], planes[2] = [core.fmtc.resample(plane, sy=-y_shift/2, sh=-y_shift/2, kernel='spline64') for plane in planes[1:]]
scaled_a = lvf.join([upsc_y, planes[1], planes[2]])

desc_y = core.fmtc.resample(planes[0], lvf.get_w(800), 800, kernel='bicubic', a1=0, a2=1/2, invks=True)
upsc_y = nnedi3_rpow2(desc_y, 2, 1920, 1080, kernel='spline64', nsize=4, nns=4, qual=2)
scaled_b = core.std.ShufflePlanes([upsc_y, den], [0, 1, 2], vs.YUV)

scaled = fvf.rfs(scaled_a, scaled_b, mappings='[9312 10055]')
scaled = fvf.Depth(scaled, 16)

line_m = mdf.retinex_edgemask_mod(scaled, opencl=True)
db = core.neo_f3kdb.Deband(scaled, 17, 24, 24, 24, 0, 0, sample_mode=4)

final = core.fmtc.bitdepth(db, bits=10, dmode=3)


final.set_output()