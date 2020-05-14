from acsuite import eztrim
import havsfunc as hvf
import vardefunc as vrf
import lvsfunc as lvf
import vapoursynth as vs

core = vs.core
core.max_cache_size = 16384

path = r"[BDMV] Ao-chan Can't Study!\Ao-chan Cant Study BD-1\BDMV\STREAM\00004"
src = lvf.src(path + '.m2ts')
eztrim(src, (24, 0), path + '.wav', "aochan_bdmenu.mka", mkvextract_path="mkvextract")

src = src[24:]

src = core.resize.Point(src, format=vs.YUV420P16, range_in=0, range=1, dither_type='error_diffusion')

full = vrf.to444(src)
dering = hvf.HQDeringmod(full, mrad=2, thr=96, darkthr=0, planes=[0, 1, 2])
db = core.neo_f3kdb.Deband(dering, 17, 30, 30, 30, 6, 0, sample_mode=4, keep_tv_range=False)
final = core.resize.Point(db, format=vs.YUV444P10, dither_type='error_diffusion', range=1)


final[::int(src.num_frames)].set_output(0)