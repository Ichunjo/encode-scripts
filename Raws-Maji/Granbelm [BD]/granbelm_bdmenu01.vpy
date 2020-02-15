import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf

core = vs.core
core.max_cache_size = 26000

src = core.dgdecodenv.DGSource(r'DMPXA047A/BDMV/STREAM/00007.dgi')
src = fvf.Depth(src, 16)

aa = lvf.upscaled_sraa(src, 2, 13)

db = core.neo_f3kdb.Deband(aa, 17, 36, 36, 36, 24, 0, sample_mode=4)

freeze = core.std.FreezeFrames(db, 50, src.num_frames-1, 50)

final = fvf.Depth(freeze, 10)

final.set_output()