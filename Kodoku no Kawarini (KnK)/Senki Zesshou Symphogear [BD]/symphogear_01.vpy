from acsuite import eztrim
from vsTAAmbk import TAAmbk
import fvsfunc as fvf
import mvsfunc as mvf
import lvsfunc as lvf
import atomchtools as atf
import vardefunc as vrf
import G41Fun as gf
import muvsfunc as muvf
import xvs
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

def _sraa(clip: vs.VideoNode, znedi: bool = True) -> vs.VideoNode:
    nnargs = dict(nsize=4, nns=3, qual=1, pscrn=1)
    eeargs = dict(alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)

    def _znedi3(clip):
        if znedi:
            clip = clip.znedi3.nnedi3(0, False, **nnargs)
        else:
            clip = clip.nnedi3.nnedi3(0, False, **nnargs)
        return clip

    def _eedi3(clip, zclip):
        return clip.eedi3m.EEDI3(0, False, **eeargs, sclip=zclip)

    clip = _eedi3(clip, _znedi3(clip)).std.Transpose()
    clip = _eedi3(clip, _znedi3(clip)).std.Transpose()
    return clip

def _fix_ghosting(clip: vs.VideoNode, mask: vs.VideoNode, image_fix: vs.VideoNode) -> vs.VideoNode:
    return core.std.MaskedMerge(clip, image_fix, core.resize.Point(mask, format=vs.GRAYS,
                                                                   matrix_s='709').std.Invert())

path = '戦姫絶唱シンフォギア/戦姫絶唱シンフォギア 1/KIXA_90182/BDMV/STREAM/00001'
src = lvf.src(path + '.m2ts')
asrc = path + '.mka'
mask_fix = lvf.src('戦姫絶唱シンフォギア/fix_ghosting.png')
eztrim(src, (24, -24), asrc, "symphogear_01_cut.wav")


src = src[24:-24]

ef = lvf.edgefixer(src, [2, 1, 1])
ef = fvf.Depth(ef, 32)


# Fix random production errors
clip_fix = _fix_ghosting(ef[13859:13889], mask_fix, ef[13889])
ef = vrf.insert_clip(ef, clip_fix, 13859)




y = vrf.get_y(ef)
b, c = 1/3, 1/3
h = 720
w = vrf.get_w(h)
desc_y = core.descale.Debicubic(y, w, h, b, c)
upsc_y = core.resize.Bicubic(desc_y, y.width, y.height, filter_param_a=b, filter_param_b=c)


den_y = mvf.BM3D(desc_y, 1.55, radius1=1)
contra_y = gf.ContraSharpening(den_y, desc_y, radius=2, planes=0)



double = vrf.nnedi3cl_double(contra_y, False, nsize=0, nns=4, qual=2, pscrn=1)
sraa = _sraa(double, False)


scaled_y = core.resize.Spline64(sraa, y.width, y.height)
scaled_y = fvf.Depth(scaled_y, 16)

# Scenefiltering AA
aa_y_a = lvf.sraa(fvf.Depth(y, 16))
aa_y_b = TAAmbk(scaled_y, aatype='Eedi3SangNom', mtype=1)
aa_y_b = core.std.Merge(aa_y_b, xvs.ssharp(aa_y_b), 0.6)
aa_y_c = core.rgvs.Repair(_sraa(scaled_y), scaled_y, 3)
aa_y = lvf.rfs(scaled_y, aa_y_a, [(336, 473)])
aa_y = lvf.rfs(aa_y, aa_y_b, [(20291, 20379)])
aa_y = lvf.rfs(aa_y, aa_y_c, [(28211, 28262)])
aa_y = vrf.fade_filter(aa_y, aa_y, aa_y_b, 20277, 20290)
aa_y = fvf.Depth(aa_y, 32)

# Perform credit mask for OP / ED / Random shits
cred_m = core.std.Expr([y, upsc_y], 'x y - abs').std.Binarize(0.05)
cred_m = vrf.iterate(cred_m, core.std.Maximum, 4)
cred_m = vrf.iterate(cred_m, core.std.Inflate, 2)

cred_y = lvf.rfs(aa_y, core.std.MaskedMerge(aa_y, y, cred_m, 0), [(36, 335), (474, 1966)])



# Chroma stuff & merge
cred = core.std.ShufflePlanes([cred_y, ef], [0, 1, 2], vs.YUV)
den_uv = core.knlm.KNLMeansCL(cred, a=2, h=0.5, d=3, device_type='gpu', channels='UV')
warp_uv = lvf.util.quick_resample(den_uv, xvs.WarpFixChromaBlend, thresh=128)

merged = warp_uv




# Deband masks
db_den = core.knlm.KNLMeansCL(merged, a=2, h=0.5, d=0, device_type='gpu', channels='Y')
db_ma = lvf.util.quick_resample(db_den, atf.retinex_edgemask, opencl=True) \
    .std.Binarize((9400 - (16 << 8)) / ((235 << 8) - (16 << 8))).std.Median().std.Inflate()

db_mb = core.std.Prewitt(db_den).std.Binarize(0.0325) \
    .std.Median().std.Maximum().std.Deflate().std.Deflate()
db_mb = vrf.get_y(db_mb)

db_m = core.std.Expr([db_mb, db_ma], 'x y max')



# Deband stages
db_a = core.placebo.Deband(merged, radius=16, threshold=5.5, iterations=1, grain=0, planes=1|2|4)
db_a = core.std.MaskedMerge(db_a, merged, db_m)

db_b = core.placebo.Deband(merged, radius=17, threshold=9.5, iterations=2, grain=0, planes=1|2|4)
db_b = core.std.MaskedMerge(db_b, merged, db_mb)

db_c = core.placebo.Deband(merged, radius=24, threshold=16, iterations=3, grain=0, planes=1|2|4)
db_c = core.std.MaskedMerge(db_c, merged, db_mb)

db = lvf.rfs(db_a, db_b, [(9990, 10097), (12370, 12710), (14550, 14640), (16850, 17051),
                          (19433, 19516), (21857, 21928), (28335, 28470), (28825, 28881),
                          (29558, 29663)])
db = lvf.rfs(db, db_c, [(13145, 13154)])
grain = core.placebo.Deband(db, iterations=0, grain=12, planes=1)




# Motion interpolation ending to fix the stuttering
# This is not the best interpolation you can see here because the scrolling credit
# is non linear at the beginning.
out, src = [fvf.Depth(c, 16) for c in [grain, src]]
cred = lvf.rfs(out, src, [(32310, 34046)]) + src.std.BlankClip(length=1)


smooth = xvs.mvfrc(cred[32399:33954], num=60000, den=1001, preset='slow')
vfr = muvf.VFRSplice([cred[:32399], smooth, cred[33954:]], 'symphogear_01_timecode.txt')


final = fvf.Depth(vfr, 10)


final.set_output(0)
