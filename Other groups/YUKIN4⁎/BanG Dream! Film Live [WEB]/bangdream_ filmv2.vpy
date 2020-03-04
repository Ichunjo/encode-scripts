import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
import modfunc as mdf
import debandshit as dbs
from functools import partial
from adptvgrnMod import adptvgrnMod
from cooldegrain import CoolDegrainSF

core = vs.core
core.max_cache_size = 16384

src = lvf.src('Film/[nao-R] BanG Dream! FILM LIVE (AbemaTV 1080p) [D8DA9AAE].mkv')

# Code of smart_descale in lvsfunc 
def _descaling(clip: vs.VideoNode, h: int):
    down = clip.descale.Debilinear(lvf.get_w(h), h)
    up = down.resize.Bilinear(clip.width, clip.height)
    diff = core.std.Expr([clip, up], 'x y - abs').std.PlaneStats()
    return down, diff

def _select(n, y, debic_list, f):
    errors = [x.props.PlaneStatsAverage for x in f]
    y_deb = debic_list[errors.index(min(errors))]
    dmask = core.std.Expr([y, y_deb.resize.Bicubic(clip.width, clip.height)], 'x y - abs 0.025 > 1 0 ?') \
    .std.Maximum().std.SetFrameProp("_descaled_resolution", intval=y_deb.height)

    up = lvf.upscaled_sraa(y_deb, 1.75, h=clip.height, ar=16/9, sharp_downscale=False) \
    .resize.Bicubic(clip.width, clip.height)
    return core.std.ClipToProp(up, dmask)

def _square():
    top = core.std.BlankClip(length=len(y), format=vs.GRAYS, height=4, width=10, color=[1])
    side = core.std.BlankClip(length=len(y), format=vs.GRAYS, height=2, width=4, color=[1])
    center = core.std.BlankClip(length=len(y), format=vs.GRAYS, height=2, width=2, color=[0])
    t1 = core.std.StackHorizontal([side, center, side])
    return core.std.StackVertical([top, t1, top])

def _restore_original(n, f, clip: vs.VideoNode, orig: vs.VideoNode, thresh_a: float, thresh_b: float):
    if f.props.PlaneStatsAverage < thresh_a:
        return clip.std.SetFrameProp("_descaled", intval=1)
    elif f.props.PlaneStatsAverage > thresh_b:
        return orig.std.SetFrameProp("_descaled", intval=0)
    return core.std.Merge(clip, orig, (f.props.PlaneStatsAverage - thresh_a) * 20).std.SetFrameProp("_descaled", intval=2)


clip = src
clip32 = fvf.Depth(clip, 32)
y = lvf.get_y(clip32)

# Season 2 & 3 are produced in 806p
# Dunno why there is a mix of both for the movie
debic_listp = [_descaling(y, h) for h in [806, 810]]
debic_list = [a[0] for a in debic_listp]
debic_props = [a[1] for a in debic_listp]

y_deb = core.std.FrameEval(y, partial(_select, y=y, debic_list=debic_list), prop_src=debic_props)
dmask = core.std.PropToClip(y_deb)
line = core.std.StackHorizontal([_square()]*192)
full_squares = core.std.StackVertical([line]*108)

artifacts = core.misc.Hysteresis(dmask.resize.Bicubic(clip32.width, clip32.height, _format=vs.GRAYS),
                                    core.std.Expr([lvf.get_y(clip32).tcanny.TCanny(sigma=3), full_squares], 'x y min'))

ret_raw = mdf.retinex_edgemask_mod(fvf.Depth(clip, 16))
ret = ret_raw.std.Binarize(30).rgvs.RemoveGrain(3)

mask = core.std.Expr([ret.resize.Point(_format=vs.GRAYS), lvf.iterate(artifacts, core.std.Maximum, 3)], 'x y -').std.Binarize(0.4)
mask = mask.std.Inflate().std.Convolution(matrix=[1]*9).std.Convolution(matrix=[1]*9)

y = core.std.MaskedMerge(y, y_deb, mask)
merged = core.std.ShufflePlanes([y, clip32], [0, 1, 2], vs.YUV)

denoise = core.knlm.KNLMeansCL(merged, a=2, h=0.55, d=3, device_type='gpu', channels='UV')
denoise = CoolDegrainSF(denoise, tr=1, thsad=80, blksize=8, overlap=4, plane=0)
denoise = fvf.Depth(denoise, 16)

adapt_m = vrdf.adaptive_mask(denoise, 20)
line_m = mdf.retinex_edgemask_mod(denoise).std.Binarize(9500).std.Inflate()
db_a = core.neo_f3kdb.Deband(denoise, 17, 42, 36, 36, 32, 0, sample_mode=4)
db_b = core.neo_f3kdb.Deband(dbs.f3kbilateral(denoise, 17, 48, 42), grainy=48, preset='depth')
db = core.std.MaskedMerge(db_a, db_b, adapt_m)
db = core.std.MaskedMerge(db, denoise, line_m)

grain_a = adptvgrnMod(db, 0.4, size=1.25, grain_chroma=False, luma_scaling=8)
grain_b = hvf.GrainFactory3(db, g1str=0.8, g2str=0.6, g3str=0.4, g1size=1.5, g2size=1.35, g3size=1.25)
grain = lvf.limit_dark(grain_a, grain_b, threshold=0.25)

grain = fvf.rfs(grain, fvf.Depth(clip, 16), mappings='[0 311] [99305 104615]')

blank = core.std.BlankClip(grain)

final = grain + blank[:399]

final = fvf.Depth(final, 10)

final.set_output()