import vapoursynth as vs
import fvsfunc as fvf
import lvsfunc as lvf

core = vs.core

src = lvf.src(r'SYMPHOGEAR_XV_VOL5/BDMV/STREAM/00011.m2ts')
src = fvf.Depth(src, 16)
aa = lvf.upscaled_sraa(src, rep=6, sharp_downscale=False)
db = core.neo_f3kdb.Deband(aa, 18, 48, 48, 48, 24, 0, sample_mode=4)
freeze = core.std.FreezeFrames(db, 170, src.num_frames-1, src.num_frames-1)
final = fvf.Depth(freeze, 10)
final.set_output()