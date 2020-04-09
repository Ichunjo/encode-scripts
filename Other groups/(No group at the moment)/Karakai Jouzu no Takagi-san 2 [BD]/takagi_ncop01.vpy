import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import havsfunc as hvf
import modfunc as mdf
import xvs
from cooldegrain import CoolDegrain
from functools import partial
from acsuite import eztrim
core = vs.core
core.max_cache_size = 16384

"""Robidoux values"""
b, c = 0.3782, 0.3109
descale_res = range(764, 769)

src = lvf.src(r'Vol1/00008.m2ts')
asrc = r'Vol1/00008.wav'
eztrim(src,[(0,-24)], asrc,"takagi_ncop_cut.mka")

src = src[:-24]
src = fvf.Depth(src, 16)

den = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=0)

descale = lvf.smarter_descale(lvf.get_y(src), descale_res, 
			partial(core.descale.Debicubic, b=b, c=c), 
			partial(core.resize.Bicubic, filter_param_a=b, filter_param_b=c),
			thr=0.005, rescale=False)
	
descale = fvf.Depth(descale, 16)

"""Transposing for each frame"""
def _transpose_shift(n, f, clip):
	h = f.props['descaleResolution']
	w = lvf.get_w(h)
	clip = core.resize.Bicubic(clip, w, h*2, src_top=0.5)
	clip = core.std.Transpose(clip)
	return clip

"""Doubling and downscale to 1080p"""
znargs = dict(field=0, dh=True, nsize=4, nns=4, qual=2, pscrn=4)
upsc = core.znedi3.nnedi3(descale, **znargs)
upsc = core.std.FrameEval(upsc, partial(_transpose_shift, clip=upsc), prop_src=upsc)
upsc = core.znedi3.nnedi3(upsc, **znargs)
upsc = core.resize.Bicubic(upsc, src.height, src.width, src_top=0.5).std.Transpose()

aa = lvf.sraa(upsc, 1.4, 13, sharp_downscale=True)

den = core.knlm.KNLMeansCL(den, a=2, h=0.4, d=3, device_type='gpu', channels='UV')
scaled = core.std.ShufflePlanes([aa, den], [0, 1, 2], vs.YUV)

dering = hvf.EdgeCleaner(scaled, 5, rmode=1, smode=1)
warp = xvs.WarpFixChromaBlend(dering, thresh=96, blur=2)

line_m = mdf.retinex_edgemask_mod(warp, opencl=True).std.Binarize(9500).std.Inflate()
db = core.neo_f3kdb.Deband(warp, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, warp, line_m)

final = fvf.Depth(db, 10)

final.set_output()



# vspipe --y4m takagi_ncop01.vpy - | x264 --demuxer y4m -o takagi_ncop01.264 - 
# --input-res 1920x1080 --sar 1:1 --fps 24000/1001 --input-depth 10 --output-depth 10 --input-csp i420 --output-csp i420 
# --partitions all --colormatrix bt709 --colorprim bt709 --transfer bt709 --threads 27 
# --ref 16 --deblock -1:-1 --me umh --subme 10 --psy-rd 0.85:0.0 --merange 24 --trellis 2 --no-dct-decimate --no-fast-pskip 
# --bframes 16 --b-adapt 2 --direct auto --keyint 360 --min-keyint 23 --rc-lookahead 60 --crf 15 --qcomp 0.7 --aq-mode 3 --aq-strength 0.8