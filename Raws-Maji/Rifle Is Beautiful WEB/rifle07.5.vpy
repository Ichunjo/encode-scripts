import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
from vsutil import get_w

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000
core.num_threads = 12

src = lvf.src(r"VRV Hidive/[Varde-rip] Rifle Is Beautiful - 07.5 [1080p].mkv")

src_audio = lvf.src(r"Ohys/[Ohys-Raws] Rifle Is Beautiful - 07.5 (MX 1280x720 x264 AAC).mkv")
asrc = r"Ohys/[Ohys-Raws] Rifle Is Beautiful - 07.5 (MX 1280x720 x264 AAC).mka"
ac.eztrim(src_audio, [(0,2156),(2398,34283)], asrc, "07.5_cut.mka")

src = src[24:33950]
src = fvf.Depth(src, 32)

Y, U, V = kgf.split(src)
desc_y = kgf.inverse_scale(Y, height=900, kernel='bicubic', b=0, c=1/2, mask_detail=True)

downs_u = core.resize.Spline36(U, get_w(450), 450)
downs_v = core.resize.Spline36(V, get_w(450), 450)

merged = kgf.join([desc_y, downs_u, downs_v])

blank = core.std.BlankClip(merged)
preview = core.resize.Bicubic(fvf.Depth(src_audio, 32), 1600, 900)
merged = merged+preview[34167:34283]+blank[:4]
merged = core.std.FreezeFrames(merged, 34042, 34045, 34041)

denoise = fvf.Depth(vrdf.KNLMCL(merged, 0.45, 0.25), 16)

aa = lvf.upscaled_sraa(denoise, 1.5, 13, sharp_downscale=False)

deband = core.f3kdb.Deband(aa, range=16, y=36, cb=30, cr=30, grainy=64, grainc=0, output_depth=16)

grain = kgf.adaptive_grain(deband, 0.2, static=False)
grain = kgf.adaptive_grain(grain, 0.4, static=False, luma_scaling=6)

final = fvf.Depth(grain, 10)

final.set_output()