import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import muvsfunc as muvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
from vsutil import get_w

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000

src = lvf.src(r"VRV Hidive/[Erai-Raws] Rifle Is Beautiful - 05 [1080p].mkv")[24:34071]

#Audio
#src_audio = lvf.src(r"Azure/[Azure-Raws] Rifle Is Beautiful - 05 (MX 1280x720 x264 AAC).mkv")
src_audio = lvf.src(r"Ohys/[Ohys-Raws] Rifle Is Beautiful - 05 (MX 1280x720 x264 AAC).mkv")
#src_audio = src_audio[0:3188]+src_audio[3428:34284]
#asrc = r"Azure/[Azure-Raws] Rifle Is Beautiful - 05 (MX 1280x720 x264 AAC).mka"
asrc = r"Ohys/[Ohys-Raws] Rifle Is Beautiful - 05 (MX 1280x720 x264 AAC).mka"
ac.eztrim(src_audio, [(0,3188),(3428,34284)], asrc, "ep05_cut.mka")
#scomp = lvf.stack_compare(src, src_audio, height=480)

src = fvf.Depth(src, 32)

#Classic filterchain
Y, U, V = kgf.split(src)
desc_y = kgf.inverse_scale(Y, height=900, kernel='bicubic', b=0, c=1/2, mask_detail=True)

downs_u = core.resize.Spline36(U, get_w(450), 450)
downs_v = core.resize.Spline36(V, get_w(450), 450)

merged = kgf.join([desc_y, downs_u, downs_v])

denoise = fvf.Depth(vrdf.KNLMCL(merged, 0.45, 0.25), 16)

aa = lvf.upscaled_sraa(denoise)

deband_a = core.f3kdb.Deband(aa, range=16, y=36, cb=30, cr=30, grainy=64, grainc=0, output_depth=16)
deband_b = core.f3kdb.Deband(aa, range=18, y=64, cb=56, cr=56, grainy=64, grainc=0, output_depth=16)
deband = fvf.rfs(deband_a, deband_b, mappings='[4336 4502]')

grain_a = kgf.adaptive_grain(deband, 0.2, static=False)
grain_a = kgf.adaptive_grain(grain_a, 0.4, static=False, luma_scaling=6)
grain_b = core.grain.Add(deband, var=0.85, constant=False)
grain = fvf.rfs(grain_a, grain_b, mappings='[20346 20759]')

final = fvf.Depth(grain, 10)

final.set_output()