import vapoursynth as vs
import kagefunc as kgf
import fvsfunc as fvf
import lvsfunc as lvf
import vsTAAmbk as taa
import acsuite

core = vs.get_core()
core.max_cache_size = 32000
core.num_threads = 14

ac = acsuite.AC()

src = lvf.src(r'BDMV/STREAM/00075.m2ts')

asrc = r'BDMV/STREAM/00075.mka'
ac.eztrim(src, [(24,1094)], asrc, "AUDIOS/nc/ncop01_cut.mka")

src = src[24:1094]

src = core.std.CropRel(src, left=4, top=0, right=4, bottom=0)
src = core.edgefixer.ContinuityFixer(src, [3,0,0], [4,0,0], [3,0,0], [4,0,0])

denoise = fvf.Depth(core.knlm.KNLMeansCL(fvf.Depth(src, 32), a=2, h=0.4, d=3, device_type='gpu'), 16)

aa = taa.TAAmbk(denoise, aatype='Nnedi3SangNom', mtype=1, mpand=(3,2), sharp=60)

scaled = core.resize.Spline36(aa, 1920, 1080)

deband = core.f3kdb.Deband(scaled, range=4, y=36, cb=30, cr=30, grainy=0, grainc=0, output_depth=16)
deband = core.f3kdb.Deband(deband, range=17, y=38, cb=36, cr=36, grainy=24, grainc=0, output_depth=16)

grain = kgf.adaptive_grain(deband, 0.3)

final = fvf.Depth(grain, 10)

final.set_output()