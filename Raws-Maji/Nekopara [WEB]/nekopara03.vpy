import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
import descale as dsc
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 26000
core.num_threads = 18

src_ = [
	core.dgdecodenv.DGSource(r'WKN/NekoparA - 03 (Wakanim SC 1080p).dgi'),
	core.dgdecodenv.DGSource(r'AMZ/NekoparA - 03 (Amazon dAnime VBR 1080p).dgi')
]

src_[0] = src_[0][0] + src_[0] + src_[0][-1] * 2
src_[1] = src_[1] + src_[1][-1] * 2
#scomp = lvf.stack_compare(src_a, src_b, height=540)

vrdf.generate_keyframes(src_[1], 'nekopara03_keyframes.log')

src_ = [fvf.Depth(src, 16) for src in src_]

hardsubmask = kgf.hardsubmask(src_[0], src_[1])
src = core.std.MaskedMerge(src_[0], src_[1], hardsubmask)

src_fade = kgf.hardsubmask_fades(src_[0], src_[1], highpass=2000)
src_fade = core.std.MaskedMerge(src_[0], src_[1], src_fade)
src = fvf.rfs(src, src_fade, mappings='[5967 6004] [10853 10907] [15657 15779] [33326 33446]')
src_fade = kgf.hardsubmask_fades(src_[0], src_[1], 20, 2000)
src_fade = core.std.MaskedMerge(src_[0], src_[1], src_fade)
src = fvf.rfs(src, src_fade, mappings='[16382 16428]')
#scomp2 = lvf.stack_compare(src, src_[1], height=540)

opstart, opend = 792, 2948
edstart, edend = 31169, 33325
desc_str = 1/3
desc_w, desc_h = kgf.get_w(846), 846
b, c = 0.2620, 0.3690

# Chroma shift
planes = kgf.split(src)
planes[1], planes[2] = [core.resize.Bicubic(plane, src_left=0.25, src_top=-0.125) for plane in planes[1:]]
src = kgf.join(planes)

# Descale / downscale
desc_y = dsc.Descale(src, desc_w, desc_h, 'bicubic', b, c, gray=True)
down_y = core.resize.Spline36(kgf.get_y(src), desc_w, desc_h)
desc_y = core.std.Expr([desc_y, down_y], f'x {desc_str} * y 1 {desc_str} - * +')

# Luma denoise
den_m = vrdf.adaptive_mask(desc_y, 20)
den_a = CoolDegrain(desc_y, tr=1, thsad=72, thsadc=0, bits=16, blksize=8, overlap=4)
den_b = CoolDegrain(desc_y, tr=1, thsad=48, thsadc=0, bits=16, blksize=8, overlap=4)
den_y = core.std.MaskedMerge(den_a, den_b, den_m)

# Chroma things
den_uv = core.knlm.KNLMeansCL(src, a=2, h=0.3, d=3, device_type='gpu', channels='UV')
aa_uv = taa.TAAmbk(den_uv, aatype=0, aatypeu='Nnedi3', aatypev='Nnedi3', down8=False, mtype=0, opencl=True)

# Luma AA + upscale
scaled_l_a = lvf.upscaled_sraa(den_y, 2, None, src.height, 16/9, False)
scaled_l_b = lvf.upscaled_sraa(den_y, 1.25, None, src.height, 16/9, False)
scaled_l = fvf.rfs(scaled_l_a, scaled_l_b, mappings='[0 77]')

scaled_t = vrdf.Bicubic(den_y, src.width, src.height, 'catrom')

line_m = vrdf.Bicubic(kgf.retinex_edgemask(den_y), src.width, src.height).std.Inflate()
scaled_y = core.std.MaskedMerge(scaled_t, scaled_l, line_m)

scaled = core.std.ShufflePlanes([scaled_y, aa_uv], [0, 1, 2], vs.YUV)

# Deband + grain
db = core.neo_f3kdb.Deband(scaled, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, scaled, line_m)

grain = kgf.adaptive_grain(db, 0.2)

# Credits AA
aa_cred = taa.TAAmbk(src, aatype='Nnedi3', nsize=4, nns=4, mtype=1, down8=False, opencl=True)
credit_m = vrdf.DRM(src, desc_h, 'bicubic', b, c, 100).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, aa_cred, credit_m), mappings=f'[{opstart} {opend}] [{edstart} {edend}]')

# Endcard things
endcard = lvf.upscaled_sraa(src, 2, None, src.height, False)
endcard = fvf.rfs(credit, endcard, mappings=f'[{edend+1} {src.num_frames-1}]')

final = fvf.Depth(endcard, 10)
final.set_output()