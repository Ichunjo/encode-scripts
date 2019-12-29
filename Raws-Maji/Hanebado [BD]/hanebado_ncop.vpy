import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
import modfunc as mdf
import atomchtools as atf
import descale as dsc
import acsuite
from edi_rpow2 import znedi3_rpow2
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 26000
core.num_threads = 18
ac = acsuite.AC()

src_ncop = lvf.src(r'[BDMV][Hanebado!][Vol.1]/BDMV/STREAM/00007.m2ts')

asrc = r'[BDMV][Hanebado!][Vol.1]/BDMV/STREAM/00007.mka'
ac.eztrim(src_ncop,[(24,-24)], asrc,"hanebado_ncop.mka")

src_ncop = src_ncop[24:-24]

opstart, opend = 0, src_ncop.num_frames-1
dw, dh = 1280, 720
uw, uh = src_ncop.width, src_ncop.height
opzone_a = f'[{opstart+1096} {opstart+1453}]'
opzone_b = f'[{opstart+338} {opstart+578}] [{opstart+1556} {opstart+1799}]'

#Generate keyframe used in x264.exe
vrdf.generate_keyframes(src_ncop, 'hanebado_ncop_keyframes.log')

src16 = fvf.Depth(src_ncop, 16)

#Light denoise and strong denoise then.
den_pre = CoolDegrain(src16, tr=1, thsad=72, thsadc=48, bits=16, blksize=8, overlap=4)
den_a = mdf.hybriddenoise_mod(den_pre, knl=0.6, sigma=2.5, radius1=1)
den_b = mdf.hybriddenoise_mod(den_pre, knl=0.8, sigma=3.5, radius1=1)

#Protected lineart.
edge_m = kgf.retinex_edgemask(den_pre, 0.1).std.Binarize(12000)
den = core.std.MaskedMerge(den_a, den_pre, edge_m)

#Scenefiltering.
den = fvf.rfs(den, den_a, mappings=opzone_a)
den = fvf.rfs(den, den_b, mappings=opzone_b)

#Descale and upscale. Since it’s bilinear kernel, I prefer decsale after the denoise. 
desc_y = dsc.Descale(den, dw, dh, 'bilinear', gray=True)
scaled_y = znedi3_rpow2(desc_y, 2, nns=4).resize.Spline16(uw, uh)
scaled = core.std.ShufflePlanes([scaled_y, den], [0, 1, 2], vs.YUV)

#AA for opening since op seems 1080p native on some frames.
aa_a = taa.TAAmbk(den, aatype='Eedi3', mtype=1, gamma=500, alpha=0.4, beta=0.6, sharp=-1)
aa_b = taa.TAAmbk(den, aatype='Nnedi3SangNom', mtype=2, sharp=80, aa=24)

aa = fvf.rfs(scaled, aa_a, mappings=f'[{opstart} {opstart+337}] [{opstart+579} {opstart+662}] [{opstart+719} {opstart+1095}] [{opstart+1454} {opstart+1555}] [{opstart+1800} {opend}]')
aa = fvf.rfs(aa, aa_b, mappings=opzone_b)

#Deband performed with neo_f3kdb
db_a = core.neo_f3kdb.Deband(aa, 17, 36, 36, 36, 24, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(aa, 18, 42, 42, 42, 48, 0, sample_mode=4)
db_c = core.neo_f3kdb.Deband(aa, 18, 48, 48, 48, 0, 0, sample_mode=4)
db_d = core.neo_f3kdb.Deband(aa, 17, 48, 36, 36, 48, 0, sample_mode=4)
db = fvf.rfs(db_a, db_b, mappings=opzone_b)
db = fvf.rfs(db, db_d, mappings=opzone_a)

#Of course regraining
grain_a = kgf.adaptive_grain(db, 0.25, luma_scaling=8)
grain_b = kgf.adaptive_grain(db, 0.45, luma_scaling=8)
grain = fvf.rfs(grain_a, db_c, mappings=f'[{opstart} {opstart+337}]')
grain = fvf.rfs(grain, grain_b, mappings=opzone_a)

final = fvf.Depth(grain, 10)

final.set_output()