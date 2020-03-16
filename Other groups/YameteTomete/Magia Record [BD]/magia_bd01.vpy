import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import modfunc as mdf
import acsuite
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain
from adptvgrnMod import adptvgrnMod

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src = lvf.src('[BDMV][200304][Magia Record][Vol.1]/BD_VIDEO/BDMV/STREAM/00001.m2ts')
endcard = lvf.src('[Raws-Maji] Magia Record Mahou Shoujo Madoka☆Magica Gaiden - 01v2 [WEB 1920x1080 x264 AAC].mkv')[-1]

asrc = r"[BDMV][200304][Magia Record][Vol.1]/BD_VIDEO/BDMV/STREAM/00001.mka"
blank = r"blank.wav"
ac.eztrim(src, [(0,-48)], asrc, "magia_bd01a.mka")
ac.eztrim(src, [(0,119)], blank, "magia_bd01b.mka")


src = src[:-48]
src = fvf.Depth(src, 16)

den_y = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=0)
den = core.knlm.KNLMeansCL(den_y, a=2, h=0.25, d=3, device_type='gpu', channels='UV')

aa_a = lvf.nneedi3_clamp(den, ret_mask=True, nns=4)
aa_b = lvf.upscaled_sraa(den, rep=6, sharp_downscale=False)
aa = fvf.rfs(aa_a, aa_b, mappings="[29453 29476] [29510 29532] [29640 29663] [29775 29798] [29866 29889] [30011 30034]")


ref = fvf.Depth(aa, 8)
line_m_a = TAAmbk(ref, mtype=1, showmask=1).std.Binarize(10).rgvs.RemoveGrain(3).std.Inflate()
line_m_b = mdf.retinex_edgemask_mod(ref).std.Binarize(42).rgvs.RemoveGrain(3).rgvs.RemoveGrain(4).std.Inflate()
line_m = core.std.Expr([line_m_a, line_m_b], 'x y +')
line_m = core.std.Expr(line_m, 'x 30 < 0 x 3 * ?').std.Inflate()

db = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 18, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, fvf.Depth(line_m, 16))

grain_a = kgf.adaptive_grain(db, 0.25, luma_scaling=10)
grain_b = adptvgrnMod(db, 1, size=1.5, luma_scaling=0, grain_chroma=False)
grain = fvf.rfs(grain_a, grain_b, mappings="[5149 5598] [8691 10137]")

crop = core.std.Crop(grain, 240, 240, 0, 0)
borders = core.std.AddBorders(crop, 240, 240, 0, 0)
borders = fvf.rfs(grain, borders, mappings="[5149 5598]")

final = fvf.Depth(borders, 10)
final = final + endcard*119

final.set_output()
