from acsuite import eztrim
from cooldegrain import CoolDegrain
from nnedi3_rpow2 import nnedi3_rpow2
import fvsfunc as fvf
import vardefunc as vrf
import havsfunc as hvf
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core

src = lvf.src('Fragtime (WOWOW Cinema).d2v')
src = fvf.Depth(src, 16)
ivtc = fvf.JIVTC(src, 2, tff=True)

# The audio seems slightly out of sync with the video.
eztrim(ivtc, (479, 0), 'Fragtime (WOWOW Cinema).aac', 'fragtime_audio_cut.mka')

ivtc = ivtc[481:]

# https://down.7086.in/lgd/
delogo = core.delogohd.DelogoHD(ivtc, 'WOWOW Cinema 1920x1080.lgd', top=-2, cutoff=100)
delogo_smooth = hvf.logoNR(delogo, ivtc, l=1700, t=30, r=60, b=1000, h=30)

den = CoolDegrain(delogo_smooth, tr=2, thsad=60, blksize=8, overlap=4, plane=0)
den = core.knlm.KNLMeansCL(den, a=2, h=0.5, d=3, device_type='gpu', channels='UV')


desc_y = fvf.DebicubicM(vrf.get_y(den), vrf.get_w(844), 844, b=1/3, c=1/3)

scaled_y = nnedi3_rpow2(desc_y).resize.Spline36(1920, 1080)
merged = core.std.ShufflePlanes([scaled_y, den], [0, 1, 2], vs.YUV)

db_m = lvf.denoise.detail_mask(merged, brz_a=3000, brz_b=1500)
db_a = core.placebo.Deband(merged, radius=16, threshold=4.5, iterations=1, grain=0, planes=1|2|4)
db_b = core.placebo.Deband(merged, radius=17, threshold=7.5, iterations=1, grain=0, planes=1|2|4)
db_c = core.placebo.Deband(merged, radius=20, threshold=7, iterations=2, grain=0, planes=1|2|4)
db = lvf.rfs(db_a, db_b, [(14037, 141790), (32661, 32797), (69470, 69709),
                          (73072, 73091), (75918, 75971), (76494, 76667)])
db = lvf.rfs(db, db_c, [(73251, 73303), (73835, 73857)])
db = vrf.fade_filter(db, db, db_c, 73813, 73834)
db = core.std.MaskedMerge(db, merged, db_m)
grain = core.placebo.Deband(db, iterations=0, grain=7, planes=1)

cred = lvf.rfs(grain, delogo_smooth, [(0, 401), (79616, 86158)]) + core.std.BlankClip(grain, length=1)

final = fvf.Depth(cred, 10)

final.set_output(0)
