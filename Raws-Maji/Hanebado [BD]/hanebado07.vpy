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

src = 		lvf.src(r'[BDMV][181114][Hanebado!][Vol.3]/BDMV/STREAM/00003.m2ts')
src_op = 	lvf.src(r'[BDMV][180919][Hanebado!][Vol.1]/BDMV/STREAM/00006.m2ts')
src_ncop = 	lvf.src(r'[BDMV][180919][Hanebado!][Vol.1]/BDMV/STREAM/00007.m2ts')
src_nced = 	lvf.src(r'[BDMV][180919][Hanebado!][Vol.1]/BDMV/STREAM/00008.m2ts')

asrc = r'[BDMV][181114][Hanebado!][Vol.3]/BDMV/STREAM/00003.mka'
ac.eztrim(src,[(0,34525)], asrc,"hanebado07.mka")

src = src[:34525]
src = src+src[-1]
src_ncop = src_ncop[24:-24]
src_op = src_op[24:-24]
src_nced = src_nced[24:-24]

opstart, opend = 3861, 6016
edstart, edend = 32367, src.num_frames-1
dw, dh = 1280, 720
uw, uh = src.width, src.height
opzone_a = f'[{opstart+1096} {opstart+1453}]'
opzone_b = f'[{opstart+338} {opstart+578}] [{opstart+1556} {opstart+1799}]'
flashback = '[3354 3446] [15909 15965] [16402 16526] [20962 21116]'

#Generate keyframe used in x264.exe
vrdf.generate_keyframes(src, 'hanebado07_keyframes.log')

src16 = fvf.Depth(src, 16)

#Always fix edges first
fixedges = core.edgefixer.ContinuityFixer(src16, [1,0,0], [1,0,0], [1,0,0], [1,0,0])
fixedges = fvf.rfs(fixedges, src16, mappings=f'[{opstart} {opend}] [{edstart} {edend}]')

#Light denoise and strong denoise then.
den_pre = CoolDegrain(fixedges, tr=1, thsad=72, thsadc=48, bits=16, blksize=8, overlap=4)
den_a = mdf.hybriddenoise_mod(den_pre, knl=0.6, sigma=2.5, radius1=1)
den_b = mdf.hybriddenoise_mod(den_pre, knl=0.8, sigma=3.5, radius1=1)

#Protected lineart.
edge_m = kgf.retinex_edgemask(den_pre, 0.1).std.Binarize(12000)
den = core.std.MaskedMerge(den_a, den_pre, edge_m)

#Scenefiltering.
den = fvf.rfs(den, den_a, mappings=opzone_a)
den = fvf.rfs(den, den_b, mappings=opzone_b)
den = fvf.rfs(den, den_pre, mappings=flashback)

#Descale and upscale. Since it’s bilinear kernel, I prefer decsale after the denoise. 
desc_y = dsc.Descale(den, dw, dh, 'bilinear', gray=True)
scaled_y = znedi3_rpow2(desc_y, 2, nns=4).resize.Spline16(uw, uh)
scaled = core.std.ShufflePlanes([scaled_y, den], [0, 1, 2], vs.YUV)

#AA for opening since op seems 1080p native on some frames.
#Srong AA for ED especially this cursed chroma
aa_a = taa.TAAmbk(den, aatype='Eedi3', mtype=1, gamma=500, alpha=0.4, beta=0.6, sharp=-1)
aa_b = taa.TAAmbk(den, aatype='Nnedi3SangNom', mtype=2, sharp=80, aa=24)
planes = kgf.split(src16)
planes[0] = lvf.upscaled_sraa(planes[0], 2, 6)
planes[1], planes[2] = [lvf.upscaled_sraa(plane, 1.5, 6) for plane in planes[1:]]
aa_c = kgf.join(planes)

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
grain = fvf.rfs(grain, aa_c, mappings=f'[{edstart} {edend}]')
grain = fvf.rfs(grain, grain_b, mappings=opzone_a)

#Creating and applying credit mask in OP & ED
credit_op_m = vrdf.DCM(grain, src_op[:opend-opstart+1], src_ncop[:opend-opstart+1], opstart, opend, 4, 4)
credit_op = core.std.MaskedMerge(grain, den_pre, credit_op_m)

credit_ed_m = vrdf.DCM(grain, src[edstart:edend+1], src_nced[:edend-edstart+1], edstart, edend, 4, 4)
credit_ed = core.std.MaskedMerge(credit_op, src16, credit_ed_m)

#Creating and applying credit mask in some random scenes
credit_m = vrdf.DRM(fixedges, 720, 'bilinear', mthr=120).std.Inflate()
credit = fvf.rfs(credit_ed, core.std.MaskedMerge(credit_ed, den_pre, credit_m), mappings='[436 543] [21553 21624]')

final = fvf.Depth(credit, 10)

final.set_output()