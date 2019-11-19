import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import vsTAAmbk as taa
import muvsfunc as muvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
from vsutil import get_w

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000

src_us_OP = lvf.src(r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00013.m2ts")
src_us = lvf.src(r"[BDMV] RWBY Vol.1 [US]/BDMV/STREAM/00003.m2ts")

src_jp = lvf.src(r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00001.m2ts")
asrc_jp = r"[BDMV][151209] RWBY Vol.1 [JP]/RWBY_VOLUME1/BDMV/STREAM/00001.mka"

src_blank = core.std.BlankClip(src_us_OP)
asrc_blank = r"blank_audio.ac3"

ac.eztrim(src_blank, [(0,150)], asrc_blank, "AUDIOS_EN/01a_cut.ac3")
ac.eztrim(src_blank, [(0,1)], asrc_blank, "AUDIOS_EN/01d_cut.ac3")

ac.eztrim(src_jp, [(0,20135), (20135+15,22345)], asrc_jp, "AUDIOS_JAP/01_cut.mka")

src = src_jp[0:150]+src_us+src_us_OP+src_blank[:1]
#src_jp = src_jp[0:20135]+src_jp[20135+15:22345]

src = fvf.Depth(src, 32)

"""
Filtering
"""
final = fvf.Depth(src, 10)
final.set_output()