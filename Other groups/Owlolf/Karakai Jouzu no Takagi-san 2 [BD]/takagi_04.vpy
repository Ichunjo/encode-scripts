from acsuite import eztrim
from cooldegrain import CoolDegrain
import lvsfunc as lvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrf
import xvs
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

path = 'TAKAGISAN2_2/BDMV/STREAM/00004'
src = lvf.src(path + '.m2ts')

eztrim(src, (0, -24), path + '.mka', "takagi_04.wav")

src = src[:-24]
src += src[-1]

opstart, opend = 0, 2157
edstart, edend = 31886, src.num_frames - 1
b, c = 0.3782, 0.3109
desc_res = range(764, 769)

den = CoolDegrain(fvf.Depth(src, 16), tr=2, thsad=48, blksize=8, overlap=4, plane=4)


y = fvf.Depth(vrf.get_y(den), 32)
descale = lvf.smart_descale(y, desc_res, b=b, c=c, thr=0.05)
rescale = lvf.smart_reupscale(fvf.Depth(descale, 16))
aa = lvf.sraa(rescale, 1.4, 13, sharp_downscale=True)


scaled = core.std.ShufflePlanes([aa, den], [0, 1, 2], vs.YUV)

dering = hvf.EdgeCleaner(scaled, 5, rmode=1, smode=1)


upscale = core.resize.Bicubic(descale, src.width, src.height, filter_param_a=b, filter_param_b=c)
cred_m = core.std.Expr([y, upscale], 'x y - abs').std.Binarize(0.05)
cred_m = vrf.iterate(cred_m, core.std.Maximum, 4)
cred_m = vrf.iterate(cred_m, core.std.Inflate, 2)
cred = lvf.rfs(dering, core.std.MaskedMerge(dering, den, fvf.Depth(cred_m, 16), 0),
			   [(opstart, opend), (edstart, edend)])

warp = xvs.WarpFixChromaBlend(cred, thresh=128, blur=2)

line_m = lvf.denoise.detail_mask(warp, brz_a=3000, brz_b=1500)
db = core.std.MaskedMerge(
	 core.neo_f3kdb.Deband(warp, 17, 30, 36, 36, 12, 0, sample_mode=4),
	 core.neo_f3kdb.Deband(warp, 17, 42, 48, 48, 24, 0, sample_mode=4),
	 core.adg.Mask(warp.std.PlaneStats(), 12)
)
db = core.std.MaskedMerge(db, warp, line_m)

db_b = core.std.BoxBlur(warp, [0, 1, 2], 4, 2, 4, 2)
db_b = core.placebo.Deband(db_b, radius=22, threshold=12, iterations=2, grain=6, planes=1|2|4)
line_m2 = vrf.iterate(line_m, core.std.Maximum, 4)
line_m2 = vrf.iterate(line_m2, core.std.Inflate, 12)
line_m2 = vrf.iterate(line_m2, core.std.Minimum, 2).std.Binarize(8400)
line_m2 = vrf.iterate(line_m2, core.std.Deflate, 4)
db_b = core.std.MaskedMerge(db_b, warp, line_m2)

db = lvf.rfs(db, db_b, [(29697, 29946), (29977, 30182)])


final = fvf.Depth(db, 10)

final.set_output(0)
