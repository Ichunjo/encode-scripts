from  vsTAAmbk import TAAmbk
from nnedi3_rpow2 import nnedi3_rpow2
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import mvsfunc as mvf
import debandshit as dbs
import atomchtools as atf
import vardefunc as vrf
import xvs
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

def _sraa(clip: vs.VideoNode) -> vs.VideoNode:
    nnargs = dict(nsize=4, nns=3, qual=1, pscrn=2)
    eeargs = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)

    def _znedi3(clip):
        return clip.znedi3.nnedi3(0, False, **nnargs)

    def _eedi3(clip, zclip):
        return clip.eedi3m.EEDI3(0, False, **eeargs, sclip=zclip)

    aa = _eedi3(clip, _znedi3(clip)).std.Transpose()
    aa = _eedi3(aa, _znedi3(aa)).std.Transpose()
    return aa

src = lvf.src(r'[BDMV][191127] Fate／kaleid liner Prisma☆Illya Prisma☆Phantasm\PRISMAPHANTASM_SP\BDMV\STREAM\00001.m2ts')

src = src[48:-24]

part_810p = '[49179 53688]'
part_720p_bc = '[22569 22958]'

src = core.std.FreezeFrames(src, 3485, 3485, 3484)

ef_a = lvf.edgefixer(src, [1,0,0])
ef_b = core.fb.FillBorders(src, 0, 0, 1, 1)
ef_b = lvf.edgefixer(ef_b, [1,1,1], [1,1,1], [2,1,1], [2,1,1], [10,5,5])
ef = fvf.rfs(ef_a, ef_b, mappings='[9775 10117] [80412 80699]')


opstart, opend = 2789, 4946

ef = fvf.Depth(ef, 32)

planes = lvf.split(ef)
planes[1], planes[2] = [core.resize.Spline36(plane, src_left=-0.25) for plane in planes[1:]]
shift = lvf.join(planes)

#den = core.knlm.KNLMeansCL(shift, a=2, h=0.4, d=3, device_type='gpu', channels='UV')
den_a = mvf.BM3D(shift, [1.5, 1], radius1=1)
den_b = mvf.BM3D(shift, [2.5, 1], radius1=1)
den = fvf.rfs(den_a, den_b, mappings='[82330 82412] [83236 83318] [84097 84767]')

"""
"""
# 720p
y = lvf.get_y(den)
desc_a = core.descale.Debilinear(y, 1280, 720)
desc_b = core.descale.Debicubic(y, 1280, 720, 0.3782, 0.3109)
desc = fvf.rfs(desc_a, desc_b, mappings=part_720p_bc)

doubled = nnedi3_rpow2(fvf.Depth(desc, 16), nns=4, qual=2)
sraa = _sraa(doubled).resize.Spline36(1920, 1080)

sharp = xvs.ssharp(sraa, mask=True)
sharp = core.std.Merge(sraa, sharp, 0.4)

scaled_a = fvf.rfs(sharp, sraa, mappings=part_720p_bc)

# 810p
desc_c = core.descale.Despline36(y, 1440, 810)
scaled_b = nnedi3_rpow2(fvf.Depth(desc_c, 16), 2, 1920, 1080, nns=4, qual=2)


# Opening
scaled_op = lvf.sraa(fvf.Depth(y, 16), 2, 3)

# Merging
scaled = fvf.rfs(scaled_a, scaled_b, mappings=part_810p)
scaled = fvf.rfs(scaled, scaled_op, mappings=f'[{opstart} {opend}]')
scaled = core.std.ShufflePlanes([scaled, fvf.Depth(den, 16)], [0, 1, 2], vs.YUV)

aa_extra = TAAmbk(scaled, aatype='Eedi3')
scaled = fvf.rfs(scaled, aa_extra, mappings='[80700 80770]')

"""
"""
warp = xvs.WarpFixChromaBlend(scaled, thresh=64, blur=2)

db_m = atf.retinex_edgemask(warp, draft=True, opencl=True) \
   .std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
db = core.neo_f3kdb.Deband(warp, 17, 36, 36, 36, 32, 0, sample_mode=4)
db = core.std.MaskedMerge(db, warp, db_m, 0)
db_b = dbs.f3kbilateral(db, 17, 48, 48)
db = fvf.rfs(db, db_b, mappings='[4216 4243]')

grain = kgf.adaptive_grain(db, 0.2, luma_scaling=4)

out = grain
ref = fvf.Depth(ef, 16)
credit_m = vrf.drm(ref, 720, 'bilinear', 0, 0, 75).std.Inflate()
cred = fvf.rfs(out, core.std.MaskedMerge(out, ref, credit_m), mappings='[9775 10117] [80412 80699]')
cred = fvf.rfs(cred, ref, mappings='[84700 87043]')


final = fvf.Depth(cred, 10)


final.set_output()
