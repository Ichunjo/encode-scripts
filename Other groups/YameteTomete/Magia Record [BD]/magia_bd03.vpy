import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import modfunc as mdf
import acsuite
from vsTAAmbk import TAAmbk
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 16384

ac = acsuite.AC()

src = lvf.src('[BDMV][200304][Magia Record][Vol.1]/BD_VIDEO/BDMV/STREAM/00003.m2ts')
endcard = lvf.src('[Raws-Maji] Magia Record Mahou Shoujo Madoka☆Magica Gaiden - 03 [WEB 1920x1080 x264 AAC].mkv')[-1]

asrc = r"[BDMV][200304][Magia Record][Vol.1]/BD_VIDEO/BDMV/STREAM/00003.mka"
blank = r"blank.wav"
ac.eztrim(src, [(0,-48)], asrc, "magia_bd03a.mka")
ac.eztrim(src, [(0,117)], blank, "magia_bd03b.mka")

src = src[:-48]
src = fvf.Depth(src, 16)
edstart, edend = 32012, 34168

den_y_a = CoolDegrain(src, tr=2, thsad=48, blksize=8, overlap=4, plane=0)
den_y_b = CoolDegrain(src, tr=3, thsad=96, blksize=8, overlap=4, plane=0)
den_y = fvf.rfs(den_y_a, den_y_b, mappings=f'[{edstart+1870} {edstart+1900}]')
den = core.knlm.KNLMeansCL(den_y, a=2, h=0.25, d=3, device_type='gpu', channels='UV')

aa_a = lvf.nneedi3_clamp(den, ret_mask=True, nns=4)
aa_b = lvf.upscaled_sraa(den, 2, rep=1)
aa = fvf.rfs(aa_a, aa_b, mappings='[14667 14696]')

ref = fvf.Depth(aa, 8)
line_m_a = TAAmbk(ref, mtype=1, showmask=1).std.Binarize(10).rgvs.RemoveGrain(3).std.Inflate()
line_m_b = mdf.retinex_edgemask_mod(ref).std.Binarize(42).rgvs.RemoveGrain(3).rgvs.RemoveGrain(4).std.Inflate()
line_m = core.std.Expr([line_m_a, line_m_b], 'x y +')
line_m = core.std.Expr(line_m, 'x 30 < 0 x 3 * ?').std.Inflate()

db = core.neo_f3kdb.Deband(aa, 17, 42, 42, 42, 18, 0, sample_mode=4)
db = core.std.MaskedMerge(db, aa, fvf.Depth(line_m, 16))
db_b = core.neo_f3kdb.Deband(db, 18, 48, 48, 48, 0, 0)
db_c = core.neo_f3kdb.Deband(db_b, 12, 260, 260, 260, 20, 0, sample_mode=4)

db = fvf.rfs(db, db_b, mappings=f'[{edstart+1870} {edstart+1900}] [13041 13180] [13932 13949] [14121 14168] [14647 14666]')
db = fvf.rfs(db, db_c, mappings='[14667 14696]')


grain = kgf.adaptive_grain(db, 0.25, luma_scaling=10)

crop = core.std.Crop(grain, 0, 0, 132, 132)
fe = core.edgefixer.ContinuityFixer(crop, 0, [1,0,0], 0, 0)
bd = core.std.AddBorders(fe, 0, 0, 132, 132)
bd = fvf.rfs(grain, bd, mappings="[0 2905]")


final = fvf.Depth(bd, 10)
final = final + endcard*117

final.set_output()
