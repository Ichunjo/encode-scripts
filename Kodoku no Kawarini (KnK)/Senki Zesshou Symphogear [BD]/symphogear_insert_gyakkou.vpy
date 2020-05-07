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

path = '戦姫絶唱シンフォギア/戦姫絶唱シンフォギア 1/KIXA_90182/BDMV/STREAM/00007'
src = lvf.src(path + '.m2ts')
asrc = path + '.mka'
eztrim(src, (0, -24), asrc, "symphogear_insert_gyakkou_cut.wav")


src = src[:-24]

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


# Chroma stuff & merge
scaled = core.std.ShufflePlanes([scaled_y, ef], [0, 1, 2], vs.YUV)
den_uv = core.knlm.KNLMeansCL(scaled, a=2, h=0.5, d=3, device_type='gpu', channels='UV')
warp_uv = lvf.util.quick_resample(den_uv, xvs.WarpFixChromaBlend, thresh=64)

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

grain = core.placebo.Deband(db_a, iterations=0, grain=12, planes=1)



final = fvf.Depth(grain, 10)

final.set_output(0)
