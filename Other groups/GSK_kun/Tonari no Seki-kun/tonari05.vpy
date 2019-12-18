import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import atomchtools as atf
import vsTAAmbk as taa
import vardefunc as vrdf
import acsuite

core = vs.get_core()
core.max_cache_size = 32000
core.num_threads = 14

ac = acsuite.AC()

src = lvf.src(r'BDMV/STREAM/00056.m2ts')
ncop = lvf.src(r'BDMV/STREAM/00044.m2ts')
nced = lvf.src(r'BDMV/STREAM/00049.m2ts')

asrc = r'BDMV/STREAM/00056.mka'
ac.eztrim(src, [(24,11027)], asrc, "AUDIOS/05/05_cut.mka")

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
denoise_plus = vrdf.KNLMCL(fvf.Depth(src, 32), 0.85, 0.75, 0, 16)
denoise = fvf.rfs(denoise, denoise_plus, mappings='[9198 9328] [9475 9613]')

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

linemask = kgf.retinex_edgemask(scaled)
deband = core.f3kdb.Deband(scaled, range=4, y=36, cb=30, cr=30, grainy=0, grainc=0, output_depth=16)
deband = core.f3kdb.Deband(deband, range=17, y=38, cb=36, cr=36, grainy=24, grainc=0, output_depth=16)
deband_b = fvf.GradFun3(deband, smode=2, radius=19, thr=2)
deband_b = core.grain.Add(deband_b, 1)
deband_c = fvf.GradFun3(deband, smode=2, radius=14, thr=1)
deband_c = core.grain.Add(deband_c, 0.6)
deband_d = fvf.GradFun3(deband, smode=2, radius=17, thr=2)
deband_d = core.grain.Add(deband_d, 0.6)
deband = fvf.rfs(deband, core.std.MaskedMerge(deband, deband_b, linemask), mappings='[1702 1772]')
deband = fvf.rfs(deband, core.std.MaskedMerge(deband, deband_c, linemask), mappings='[9198 9328]')
deband = fvf.rfs(deband, core.std.MaskedMerge(deband, deband_d, linemask), mappings='[9475 9613]')

grain = kgf.adaptive_grain(deband, 0.3)
grain_b = kgf.adaptive_grain(deband, 0.45, 6)
grain = fvf.rfs(grain, grain_b, mappings='[9198 9328] [9475 9613]')

final = fvf.Depth(grain, 10)

blank = core.std.BlankClip(final)
final = final+blank[:2]
final = core.std.FreezeFrames(final, first=[11003], last=[11004], replacement=[11002])

final.set_output()