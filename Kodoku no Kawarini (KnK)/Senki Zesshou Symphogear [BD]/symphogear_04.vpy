from acsuite import eztrim
import fvsfunc as fvf
import mvsfunc as mvf
import lvsfunc as lvf
import atomchtools as atf
import vardefunc as vrf
import G41Fun as gf
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

path = '戦姫絶唱シンフォギア/戦姫絶唱シンフォギア 2/KIXA_90183/BDMV/STREAM/00002'
src = lvf.src(path + '.m2ts')
asrc = path + '.mka'
eztrim(src, (0, 34046), asrc, "symphogear_04_cut.wav")

src = src[:34046]
src += src[-1]
opstart, opend = 0, 1990
edstart, edend = 30867, 33037



ef = lvf.edgefixer(src, [2, 1, 1])
ef = fvf.Depth(ef, 32)




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


# Perform credit mask for OP / ED / Random shits
cred_m = core.std.Expr([y, upsc_y], 'x y - abs').std.Binarize(0.05)
cred_m = vrf.iterate(cred_m, core.std.Maximum, 4)
cred_m = vrf.iterate(cred_m, core.std.Inflate, 2)

cred_y = lvf.rfs(scaled_y, core.std.MaskedMerge(scaled_y, y, cred_m, 0),
                 [(opstart, opend), (edstart, edend), (18701, 18844)])


# Chroma stuff & merge
cred = core.std.ShufflePlanes([cred_y, ef], [0, 1, 2], vs.YUV)
den_uv = core.knlm.KNLMeansCL(cred, a=2, h=0.5, d=3, device_type='gpu', channels='UV')
warp_uv = lvf.util.quick_resample(den_uv, xvs.WarpFixChromaBlend, thresh=64)

merged = warp_uv



# Deband masks
db_den_a = core.knlm.KNLMeansCL(merged, a=2, h=0.5, d=0, device_type='gpu', channels='Y')
db_den_b = mvf.BM3D(merged, [10, 5], profile1='vn')
db_den = lvf.rfs(db_den_a, db_den_b, [(4798, 4906)])

db_ma = lvf.util.quick_resample(db_den, atf.retinex_edgemask, opencl=True) \
    .std.Binarize((9400 - (16 << 8)) / ((235 << 8) - (16 << 8))).std.Median().std.Inflate()

db_mb = core.std.Prewitt(db_den).std.Binarize(0.0325) \
    .std.Median().std.Maximum().std.Deflate().std.Deflate()
db_mb = vrf.get_y(db_mb)

db_m = core.std.Expr([db_mb, db_ma], 'x y max')


mask = lvf.src('戦姫絶唱シンフォギア/symphogear_04_1655_mask.png', db_mb)
mask = vrf.iterate(mask, core.std.Deflate, 4)
db_mc = core.std.Expr([mask, db_m], 'x y max')



# Deband stages
db_a = core.placebo.Deband(merged, radius=16, threshold=5.5, iterations=1, grain=0, planes=1|2|4)
db_a = core.std.MaskedMerge(db_a, merged, db_m)

db_b = core.std.MaskedMerge(db_a, merged, db_mc)

db_c = core.placebo.Deband(db_den_b, radius=20, threshold=7, iterations=1, grain=10, planes=1|2|4)
db_c = core.std.MaskedMerge(db_c, db_den_b, db_m)


db = lvf.rfs(db_a, db_b, [(1637, 1672)])
db = lvf.rfs(db, db_c, [(4798, 4906)])
db = vrf.fade_filter(db, db_c, db_a, 4907, 4928)
grain = core.placebo.Deband(db, iterations=0, grain=12, planes=1)


final = lvf.rfs(grain, fvf.Depth(src, 32), [(7178, 7285), (30867, 30992), (32990, 33038)])
final = fvf.Depth(final, 10)


final.set_output(0)
