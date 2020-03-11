import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import muvsfunc as muvf
import modfunc as mdf
import mvsfunc as mvf
import havsfunc as hvf
import vardefunc as vrdf
import acsuite
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrainSF, CoolDegrain

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src = lvf.src(r'[BDMV][200226] 劇場版 響け!ユーフォニアム~誓いのフィナーレ~/BDMV/STREAM/00000.m2ts')
asrc = r'[BDMV][200226] 劇場版 響け!ユーフォニアム~誓いのフィナーレ~/BDMV/STREAM/audio.mka'
ac.eztrim(src,[(720,-24)], asrc,"hibike movie 3.mka")

src = src[720:-24]

crop = core.std.Crop(src, 0, 0, 22, 22)


adapt_m = vrdf.adaptive_mask(crop, 20)
line_m = mdf.retinex_edgemask_mod(crop).std.Binarize(50).rgvs.RemoveGrain(3).rgvs.RemoveGrain(3).std.Inflate()

den_a = CoolDegrainSF(crop, tr=1, thsad=48, blksize=8, overlap=4, plane=0)
den_b = CoolDegrainSF(crop, tr=1, thsad=24, blksize=8, overlap=4, plane=0)
den_y = core.std.MaskedMerge(den_a, den_b, fvf.Depth(adapt_m, 32))
den_y = core.std.MaskedMerge(den_y, fvf.Depth(crop, 32), fvf.Depth(line_m, 32))

den_yy = mvf.BM3D(fvf.Depth(crop, 32), [5, 2.5])

den_y = fvf.rfs(den_y, den_yy, mappings='[1992 2156] [143483 143663]')


den = core.knlm.KNLMeansCL(den_y, a=2, h=0.4, d=3, device_type='gpu', channels='UV')
den = fvf.Depth(den, 16)

dering_m = muvf.AnimeMask(den, shift=0).std.Expr('x 10 < 0 x 3 * ?').rgvs.RemoveGrain(3)
dering = hvf.HQDeringmod(den, ringmask=dering_m, drrep=24, thr=24, planes=[0, 1, 2])


aa = TAAmbk(dering, aatype='Nnedi3', nsize=1, nns=3)

ddiff = CoolDegrain(aa, tr=3, thsad=120, blksize=8, overlap=4, plane=0)
line_m = mdf.retinex_edgemask_mod(ddiff).std.Binarize(9500).rgvs.RemoveGrain(3).std.Inflate()
diff = core.std.MakeDiff(ddiff, aa)

db = core.neo_f3kdb.Deband(ddiff, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, ddiff, fvf.Depth(line_m, 16))
db = core.std.MergeDiff(db, diff)

grain_a = kgf.adaptive_grain(db, 0.15)
grain_b = core.grain.Add(db, var=2, uvar=1, constant=False)
grain = fvf.rfs(grain_a, grain_b, mappings='[37812 37951]' + '[11565 11648] [11763 11822]' +
 '[11861 11931] [14264 14425] [18682 18755] [18798 18895] [22030 22161] [28194 28263]' + 
 '[30893 31036] [34096 34150] [36584 36858] [58319 58360] [60897 60932] [143101 143482]')


bd = core.std.CropRel(grain, left=672, top=0, right=672, bottom=0)
bd = core.std.AddBorders(bd, left=672, top=0, right=672, bottom=0)
bd = fvf.rfs(grain, bd, mappings='[11565 11648] [11763 11822] [11861 11931]' +
 '[14264 14425] [18682 18755] [18798 18895] [22030 22161] [28194 28263] [30893 31036]' + 
 '[34096 34150] [36584 36858] [58319 58360] [60897 60932] [143101 143482]')

bd = core.std.CropRel(bd, left=182, top=0, right=182, bottom=0)
bd = core.std.AddBorders(bd, left=182, top=0, right=182, bottom=0)
bd = fvf.rfs(grain, bd, mappings='[37812 37951]')
 

fill = core.fb.FillBorders(fvf.Depth(crop, 16), 0, 205, 0, 0)
final = fvf.rfs(bd, fill, mappings='[134982 141272]')

final = fvf.Depth(final, 10)

final = final + core.std.BlankClip(final)[0]*193

final.set_output()