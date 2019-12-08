import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
from vsutil import get_w
import vsTAAmbk as taa

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000

src = lvf.src(r"VRV Hidive/[Erai-Raws] Rifle Is Beautiful - 07 [1080p].mkv")

src_audio = lvf.src(r"Ohys/[Ohys-Raws] Rifle Is Beautiful - 07 (MX 1280x720 x264 AAC).mkv")
asrc = r"Ohys/[Ohys-Raws] Rifle Is Beautiful - 07 (MX 1280x720 x264 AAC).mka"
ac.eztrim(src_audio, [(0,2636),(2877,34285)], asrc, "ep07_cut.mka")

src = src[24:34070]
src = fvf.Depth(src, 32)

Y, U, V = kgf.split(src)
desc_y = kgf.inverse_scale(Y, height=900, kernel='bicubic', b=0, c=1/2, mask_detail=True)

downs_u = core.resize.Spline36(U, get_w(450), 450)
downs_v = core.resize.Spline36(V, get_w(450), 450)

merged = kgf.join([desc_y, downs_u, downs_v])

denoise = fvf.Depth(vrdf.KNLMCL(merged, 0.45, 0.25), 16)

aa = lvf.upscaled_sraa(denoise, 1.5, 13, sharp_downscale=False)
aa_b = taa.TAAmbk(aa, aatype='Eedi3SangNom', cycle=1, mtype=2, sharp=40, down8=False)
aa = fvf.rfs(aa, aa_b, mappings='[19956 20183]')

deband = core.f3kdb.Deband(aa, range=16, y=36, cb=30, cr=30, grainy=64, grainc=0, output_depth=16)

grain = kgf.adaptive_grain(deband, 0.2, static=False)
grain = kgf.adaptive_grain(grain, 0.4, static=False, luma_scaling=6)

final = fvf.Depth(grain, 10)

final.set_output()