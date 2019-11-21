import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import vsTAAmbk as taa
import muvsfunc as muvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
from vsutil import get_w, split, join

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000

src_us_OP = 	lvf.src(r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00039.m2ts")
src_us = 		lvf.src(r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00004.m2ts")
src_us_ED = 	lvf.src(r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00014.m2ts")

asrc_us_OP = 	r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00039_T2_Audio - English.ac3"
asrc_us = 		r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00004_T2_Audio - English.ac3"
asrc_us_ED = 	r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00014_T2_Audio - English.ac3"

src_jp_OP = 	lvf.src(r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00019.m2ts")
src_jp = 		lvf.src(r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00001.m2ts")

asrc_jp_OP = 	r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00019.mka"
asrc_jp = 		r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00001.mka"

src_blank = 	core.std.BlankClip(src_us_OP)
asrc_blank = 	r"blank_audio.ac3"

ac.eztrim(src_us_OP, 	[(0,0)], 								asrc_us_OP, 	"AUDIOS_EN/02a_cut.mka")
ac.eztrim(src_us, 		[(0,0)], 								asrc_us, 		"AUDIOS_EN/02b_cut.mka")
ac.eztrim(src_us_ED, 	[(0,0)], 								asrc_us_ED, 	"AUDIOS_EN/02c_cut.mka")
ac.eztrim(src_blank, 	[(0,1)], 								asrc_blank, 	"AUDIOS_EN/02d_cut.mka")

ac.eztrim(src_jp_OP,	[(30,30+src_us_OP.num_frames)], 		asrc_jp_OP, 	"AUDIOS_JAP/02a_cut.mka")
ac.eztrim(src_jp, 		[(22361,22361+src_us.num_frames)], 		asrc_jp, 		"AUDIOS_JAP/02b_cut.mka")
ac.eztrim(src_jp, 		[(42257,42257+src_us_ED.num_frames)], 	asrc_jp, 		"AUDIOS_JAP/02c_cut.mka")
ac.eztrim(src_blank, 	[(0,1)], 								asrc_blank, 	"AUDIOS_JAP/02d_cut.mka")

#src_jp_OP = src_jp_OP[30:30+src_us_OP.num_frames]
#src_jp_ep = src_jp[22361:22361+src_us.num_frames]
#src_jp_ED = src_jp[42257:42257+src_us_ED.num_frames]
#src_jp = src_jp_OP+src_jp_ep+src_jp_ED+src_blank[:1]

"""
src = src_us_OP+src_us+src_us_ED+src_blank[:1]
src = core.std.AssumeFPS(src, fpsnum=30000, fpsden=1001)
#src = fvf.Depth(src, 16)

#Filtering

final = fvf.Depth(grain, 10)

final.set_output()
"""