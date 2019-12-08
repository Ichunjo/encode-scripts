import vapoursynth as vs
core = vs.get_core()
import mvsfunc as mvf
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
from vsutil import get_w, split, join
core.max_cache_size = 32000

#Since there are somes broken frames due to the Mega Distortion Blocky Grain Of Hell I try to cover with a bit of blur :^)
#Code had stolen from Gradfun3 of fvsfunc
def BlurOfHellMotherFucker(src, radius, thr, planes):
        thr_1 = max(thr * 4.5, 1.25)
        r4 = max(radius * 4 / 3, 4.0)
        last = src
        last = core.bilateral.Bilateral(last, ref=last, sigmaS=r4 / 2, sigmaR=thr_1 / 255, planes=planes, algorithm=0)
        return last

src = lvf.src(r'island07.m2ts')
src = fvf.Depth(src, 16)

#I don’t want to destroy all the frames with the BlurOfHellMotherFucker so I cheat with FreezeFrames when it’s possible
src = core.std.FreezeFrames(src, first=[18751], last=[18752], replacement=[18753])
src = core.std.FreezeFrames(src, first=[18819], last=[18819], replacement=[18818])
src = core.std.FreezeFrames(src, first=[22264], last=[22275], replacement=[22263])

opstart, opend = 1320, 3476
edstart, endend = 31769, 33926

flashback = '[0 714] [14209 14559] [17789 23180] [28125 28269] [28762 28864]'
ed_filter = f'[{edstart} {edstart+1508}] [{edstart+1578} {endend}]'
ed_filter2 = f'[{edstart+1509} {edstart+1578}]'
blocky = '[18735 18752] [18820 18831] [19521 19532] [19969 19980]'

deblock = core.deblock.Deblock(src, quant=60, aoffset=8, boffset=6)
deblock = fvf.rfs(src, deblock, mappings=blocky)

y, u, v = split(deblock)

descaled = kgf.inverse_scale(y, height=720, kernel='bicubic', b=1/5, c=1/2, mask_detail=True)
descaled = fvf.Depth(descaled, 16)

adaptmask = vrdf.adaptive_mask(descaled, 300)
edgemask = kgf.retinex_edgemask(descaled, 1)
denoise_a = mvf.BM3D(descaled, 1.2)
denoise_b = mvf.BM3D(descaled, 2.1)
denoise_ed = hvf.SMDegrain(descaled, tr=2, thSAD=150)

denoise = core.std.MaskedMerge(denoise_a, denoise_b, adaptmask)
denoise = core.std.MaskedMerge(denoise, descaled, edgemask)

denoise = fvf.rfs(denoise, denoise_b, mappings=flashback + ed_filter2)
denoise = fvf.rfs(denoise, denoise_a, mappings=f'[{opstart} {opstart+106}]')
denoise = fvf.rfs(denoise, denoise_ed, mappings=ed_filter)

aa = lvf.upscaled_sraa(denoise, 2, 13)

scaled_u = core.resize.Bicubic(u, get_w(720), 720, src_left=0.25)
scaled_v = core.resize.Bicubic(v, get_w(720), 720, src_left=0.25)

merged = join([aa, scaled_u, scaled_v])

adaptmask = vrdf.adaptive_mask(merged, 300)
edgemask = kgf.retinex_edgemask(merged, 1)
deband_a = fvf.GradFun3(merged, smode=2, thr=0.15,radius=13)
deband_b = fvf.GradFun3(merged, smode=2, thr=0.35,radius=15)
deband_c = core.f3kdb.Deband(merged, range=15, y=34, cb=26, cr=26, grainy=16, grainc=0, output_depth=16)
deband_c = core.std.MaskedMerge(deband_c, merged, edgemask)
deband_d = core.f3kdb.Deband(merged, range=24, y=90, cb=80, cr=80, grainy=32, grainc=0, output_depth=16)
deband_ed = core.f3kdb.Deband(merged, range=12, y=18, cb=12, cr=12, grainy=8, grainc=0, output_depth=16)
deband_ed2 = core.f3kdb.Deband(merged, range=9, y=72, cb=64, cr=64, grainy=32, grainc=0, output_depth=16)
deband = core.std.MaskedMerge(deband_a, deband_b, adaptmask)
deband = core.std.MaskedMerge(deband, merged, edgemask)
deband = fvf.rfs(deband, deband_c, mappings=flashback)
deband = fvf.rfs(deband, deband_d, mappings=blocky)
deband = fvf.rfs(deband, deband_a, mappings=f'[{opstart} {opstart+106}]')
deband = fvf.rfs(deband, deband_ed, mappings=ed_filter)
deband = fvf.rfs(deband, deband_ed2, mappings=ed_filter2)

#Fade effect to simulate a certain transition... Maybe?
blur = BlurOfHellMotherFucker(deband, radius=23, thr=10, planes=[0, 1, 2])
deband = vrdf.fade_filter(deband, blur, deband, start=18735, end=18765)
deband = vrdf.fade_filter(deband, deband, blur, start=18810, end=18831)
deband = vrdf.fade_filter(deband, deband, blur, start=19517, end=19532)
deband = vrdf.fade_filter(deband, deband, blur, start=19960, end=19980)

grain_a = kgf.adaptive_grain(deband, 0.3, luma_scaling=16)
grain_b = core.grain.Add(deband, var=0.85, constant=False)
grain_c = hvf.GrainFactory3(deband, 4, 3, 2, g1size=2, g2size=1.5, g3size=1) #Time to recreate details!
grain_ed = kgf.adaptive_grain(deband, 0.2, luma_scaling=16)

grain = fvf.rfs(grain_a, grain_b, mappings=flashback + f"[{opstart+1624} {opstart+1647}]")
grain = fvf.rfs(grain, grain_ed, mappings=ed_filter)
grain = vrdf.fade_filter(grain, grain_c, grain, start=18735, end=18765)
grain = vrdf.fade_filter(grain, grain, grain_c, start=18810, end=18831)
grain = vrdf.fade_filter(grain, grain, grain_c, start=19517, end=19532)
grain = vrdf.fade_filter(grain, grain, grain_c, start=19960, end=19980)

final = fvf.Depth(grain, 10)
final.set_output()