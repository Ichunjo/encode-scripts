import fvsfunc as fvf
import muvsfunc as muvf
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

path = '戦姫絶唱シンフォギア/戦姫絶唱シンフォギア 6/KIXA_90187/BDMV/STREAM/00008'
src = lvf.src(path + '.m2ts')


src = fvf.Depth(src, 16)

src = core.neo_f3kdb.Deband(src, 17, 36, 36, 36, 24, 0, sample_mode=4)

clip_a = src[:480]
clip_b = src[480:]
clip_b = clip_b[480::int(clip_b.num_frames/2)]

vfr = muvf.VFRSplice([clip_a, clip_b], 'symphogear_bdmenu06_timecode.txt', False)
vfr = fvf.Depth(vfr, 10)

vfr.set_output(0)
