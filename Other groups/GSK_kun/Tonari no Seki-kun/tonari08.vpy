import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import atomchtools as atf
import vsTAAmbk as taa
import acsuite

core = vs.get_core()
core.max_cache_size = 32000
core.num_threads = 14

ac = acsuite.AC()

src = lvf.src(r'BDMV/STREAM/00059.m2ts')
ncop = lvf.src(r'BDMV/STREAM/00073.m2ts')
nced = lvf.src(r'BDMV/STREAM/00049.m2ts')

asrc = r'BDMV/STREAM/00059.mka'
ac.eztrim(src, [(24,11027)], asrc, "AUDIOS/08/08_cut.mka")

src = src[24:11027]

opstart, opend = 0, 1055
edstart, edend = 9926, src.num_frames-1

ncop = ncop[24:-24]
ncop = src[:1]+ncop
nced = nced[24:-24]
nced = src[edstart:edstart+1]+nced

src = core.std.CropRel(src, left=4, top=0, right=4, bottom=0)
src = core.edgefixer.ContinuityFixer(src, [3,0,0], [4,0,0], [3,0,0], [4,0,0])

ncop = core.std.CropRel(ncop, left=4, top=0, right=4, bottom=0)
ncop = core.edgefixer.ContinuityFixer(ncop, [3,0,0], [4,0,0], [3,0,0], [4,0,0])

nced = core.std.CropRel(nced, left=4, top=0, right=4, bottom=0)
nced = core.edgefixer.ContinuityFixer(nced, [3,0,0], [4,0,0], [3,0,0], [4,0,0])

denoise = fvf.Depth(core.knlm.KNLMeansCL(fvf.Depth(src, 32), a=2, h=0.4, d=3, device_type='gpu'), 16)

aa = taa.TAAmbk(denoise, aatype='Nnedi3SangNom', mtype=1, mpand=(3,2), sharp=60)

###
op = src[opstart:opend+1]
ed = src[edstart:edend+1]

mask_op = hvf.mt_expand_multi(atf.DiffCreditlessMask(op, ncop[:op.num_frames]), sw=4, sh=4)
mask_ed = hvf.mt_expand_multi(atf.DiffCreditlessMask(ed, nced[:ed.num_frames]), sw=4, sh=4)

op = fvf.Depth(op, 16)
ed = fvf.Depth(ed, 16)
dering_op = hvf.HQDeringmod(op, mrad=4, msmooth=3, mthr=110, drrep=1, thr=128, darkthr=128, elast=1)
dering_ed = hvf.HQDeringmod(ed, mrad=4, msmooth=3, mthr=110, drrep=1, thr=128, darkthr=128, elast=1)
dehalo_op = hvf.FineDehalo(dering_op)
dehalo_ed = hvf.FineDehalo(dering_ed)
derh_op = core.std.MaskedMerge(aa[opstart:opend+1], dehalo_op, fvf.Depth(mask_op, 16))
derh_ed = core.std.MaskedMerge(aa[edstart:edend+1], dehalo_ed, fvf.Depth(mask_ed, 16))
###

fixcredits = derh_op+aa[opend+1:edstart]+derh_ed

scaled = core.resize.Spline36(fixcredits, 1920, 1080)

deband = core.f3kdb.Deband(scaled, range=4, y=36, cb=30, cr=30, grainy=0, grainc=0, output_depth=16)
deband = core.f3kdb.Deband(deband, range=17, y=38, cb=36, cr=36, grainy=24, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(deband, range=17, y=44, cb=36, cr=36, grainy=24, grainc=0, output_depth=16)
deband = fvf.rfs(deband, deband_b, mappings='[4375 4506] [5885 5999]')

grain = kgf.adaptive_grain(deband, 0.3)

final = fvf.Depth(grain, 10)
final = final+final[-1]+final[-1]

final.set_output()