import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vardefunc as vrdf
import descale as dsc
import acsuite
from vsutil import get_w
from nnedi3_rpow2CL import nnedi3_rpow2CL
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 32000

ac = acsuite.AC()

src_a = lvf.src(r'WKN/Nekopara_01_EN_HD.mp4')
src_b = lvf.src(r'AMZ/NekoparA - 01v2 (Amazon dAnime VBR 1080p).mkv')

asrc_a = r"WKN/Nekopara_01_EN_HD.mka"
asrc_b = r"AMZ/NekoparA - 01v2 (Amazon dAnime VBR 1080p).mka"
ac.eztrim(src_a, [(0,1)], 				asrc_a, "nekopara01a_cut.mka")
ac.eztrim(src_a, [(0,959)], 			asrc_a, "nekopara01b_cut.mka")
ac.eztrim(src_b, [(960,3117)], 			asrc_b, "nekopara01c_cut.mka")
ac.eztrim(src_a, [(959,0)], 			asrc_a, "nekopara01d_cut.mka")
ac.eztrim(src_a, [(0,2)], 			    asrc_a, "nekopara01e_cut.mka")

src_a = src_a[0]+src_a[:959]+src_b[960:3117]+src_a[959:]+(src_a[-1]*2)
src_b = src_b+(src_b[-1]*2)
#scomp = lvf.stack_compare(src_a, src_b, height=540)

vrdf.generate_keyframes(src_b, 'nekopara01_keyframes.log')

src_a = fvf.Depth(src_a, 16)
src_b = fvf.Depth(src_b, 16)

hardsubmask = kgf.hardsubmask(src_a, src_b)
src = core.std.MaskedMerge(src_a, src_b, hardsubmask)

src_fade = kgf.hardsubmask_fades(src_a, src_b, highpass=2000)
src_fade = core.std.MaskedMerge(src_a, src_b, src_fade)
src = fvf.rfs(src, src_fade, mappings='[20371 20400] [33326 33446]')
#scomp2 = lvf.stack_compare(src, src_b, height=540)

opstart, opend = 960, 3116
edstart, edend = 31169, 33325

desc_y = dsc.Descale(src, get_w(846), 846, 'bicubic', 1/3, 1/3, gray=True)

den_m = vrdf.adaptive_mask(desc_y, 20)
den_a = CoolDegrain(desc_y, tr=1, thsad=72, thsadc=0, bits=16, blksize=8, overlap=4)
den_b = CoolDegrain(desc_y, tr=1, thsad=48, thsadc=0, bits=16, blksize=8, overlap=4)
den_y = core.std.MaskedMerge(den_a, den_b, den_m)

den_uv = core.knlm.KNLMeansCL(src, a=2, h=0.3, d=3, device_type='gpu', channels='UV')

aa_uv = vrdf.to444(den_uv, 960, 540, False)
aa_uv[0], aa_uv[1] = [fvf.Depth(plane, 16) for plane in aa_uv[0:]]

line_m = kgf.retinex_edgemask(src).std.Inflate()
scaled_line = lvf.upscaled_sraa(den_y, 2, None, 1080, False)
scaled_text = nnedi3_rpow2CL(den_y, 2).resize.Spline16(1920, 1080)
scaled_y = core.std.MaskedMerge(scaled_text, scaled_line, line_m)

scaled = kgf.join([scaled_y, aa_uv[0], aa_uv[1]])

db = core.neo_f3kdb.Deband(scaled, 17, 36, 36, 36, 24, 0, sample_mode=4)
db = core.std.MaskedMerge(db, scaled, line_m)

grain = kgf.adaptive_grain(db, 0.2)

aa_cred = taa.TAAmbk(src, aatype='Nnedi3', nsize=4, nns=4, mtype=1, down8=False, opencl=True)
credit_m = vrdf.DRM(src, 846, 'bicubic', 1/3, 1/3, 100).std.Inflate()
credit = fvf.rfs(grain, core.std.MaskedMerge(grain, aa_cred, credit_m), mappings=f'[{opstart} {opend}] [{edstart} {edend}]')

endcard = fvf.rfs(credit, src, mappings=f'[[{edend+1} {src.num_frames-1}]')

final = fvf.Depth(endcard, 10)

final.set_output()