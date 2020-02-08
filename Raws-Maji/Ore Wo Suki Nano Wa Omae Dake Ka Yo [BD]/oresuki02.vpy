import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as hvf
import fvsfunc as fvf
import vardefunc as vrdf
import modfunc as mdf
import descale as dsc
import acsuite
from cooldegrain import CoolDegrain
from vsTAAmbk import TAAmbk

core = vs.core
core.max_cache_size = 32000

ac = acsuite.AC()

src = 		core.dgdecodenv.DGSource(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00002.dgi")
src_ncop = 	core.dgdecodenv.DGSource(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00004.dgi")
src_nced = 	core.dgdecodenv.DGSource(r"[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00005.dgi")

asrc = r'[BDMV][191225][Ore Wo Suki Nano Wa Omae Dake Ka Yo][Vol.01]/BDMV/STREAM/00002.mka'
ac.eztrim(src,[(0,34045)], asrc,"oresuki02.mka")

src = 		src[0:34045]
src = 		src+src[-1]
src_ncop = 	src_ncop[24:-24]
src_nced = 	src_nced[24:-24]

opstart, opend = 456, 2613
edstart, edend = 31767, 33924
desc_str = 0.5
desc_w, desc_h = 1280, 720
vrdf.generate_keyframes(src, 'oresuki02_keyframes.log')

fe = core.edgefixer.ContinuityFixer(src, [2,1,1], [2,1,1], [2,1,1], [2,1,1])

y = kgf.get_y(fvf.Depth(fe, 32))

desc_sharp =    dsc.Descale(y, desc_w, desc_h, 'bilinear')
desc_regular =  dsc.Descale(y, desc_w, desc_h, 'bicubic', 0, 1/2)
desc_y =        core.std.Expr([desc_sharp, desc_regular], f'x {desc_str} * y 1 {desc_str} - * +')

den_y = CoolDegrain(desc_y, tr=1, thsad=48, bits=16, blksize=8, overlap=4, plane=0)
den_uv = CoolDegrain(fe, tr=1, thsad=0, thsadc=60, bits=16, blksize=8, overlap=4, plane=3)

line_m = mdf.retinex_edgemask_mod(den_y, opencl=True).std.Binarize(9500)
line_m = line_m.rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).std.Inflate()
darken = hvf.FastLineDarkenMOD(den_y, strength=48, protection=5, luma_cap=191, threshold=4, thinning=0)
aa = TAAmbk(darken, aatype='Nnedi3', mtype=0, down8=False, opencl=True)
aa = core.std.MaskedMerge(den_y, aa, line_m)

u, v = vrdf.to444(den_uv, 1280, 720, False)
scaled = kgf.join([den_y, u, v])

db = core.neo_f3kdb.Deband(scaled, 16, 30, 30, 30, 12, 0, sample_mode=4)
grain = kgf.adaptive_grain(db)

# 1080p things
# Downscale Episode / NCs
ref =    core.resize.Spline36(kgf.get_y(src), desc_w, desc_h)
ref_op = core.resize.Spline36(kgf.get_y(src_ncop), desc_w, desc_h)
ref_ed = core.resize.Spline36(kgf.get_y(src_nced), desc_w, desc_h)

# Credit mask for OP and ED
credit_op_m = vrdf.DCM(grain, ref[opstart:opend+1], ref_op[:opend-opstart+1], opstart, opend, 4, 4).std.Deflate()
credit_ed_m = vrdf.DCM(grain, ref[edstart:edend+1], ref_ed[:edend-edstart+1], edstart, edend, 4, 4).std.Deflate()
credit_m = core.std.Expr([credit_op_m, credit_ed_m], 'x y +')

# Merging 
fe16 = fvf.Depth(fe, 16)
ref = core.resize.Spline36(kgf.get_y(fe16), desc_w, desc_h)
credit = core.std.MaskedMerge(kgf.get_y(grain), ref, credit_m, 0)

# Other 1080p things
credit_m = core.resize.Spline36(vrdf.DRM(fe16, desc_h, 'bicubic', 0, 1/2).std.Inflate(), desc_w, desc_h)
credit = fvf.rfs(credit, core.std.MaskedMerge(credit, ref, credit_m, 0), mappings='[2660 2733] [33925 34045]')

# Merging the chroma
credit = core.std.ShufflePlanes([credit, grain], [0, 1, 2], vs.YUV)

final = fvf.Depth(credit, 10)
final.set_output()