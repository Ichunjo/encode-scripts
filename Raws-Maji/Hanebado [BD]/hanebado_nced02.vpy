import vapoursynth as vs
import vsTAAmbk as taa
import kagefunc as kgf
import fvsfunc as fvf
import havsfunc as hvf
import lvsfunc as lvf
import vardefunc as vrdf
import modfunc as mdf
import atomchtools as atf
import descale as dsc
import acsuite
from edi_rpow2 import znedi3_rpow2
from cooldegrain import CoolDegrain

core = vs.core
core.max_cache_size = 26000
core.num_threads = 18
ac = acsuite.AC()

src_nced = lvf.src(r'[BDMV][181017][Hanebado!][Vol.2]/BDMV/STREAM/00006.m2ts')

asrc = r'[BDMV][181017][Hanebado!][Vol.2]/BDMV/STREAM/00006.mka'
ac.eztrim(src_nced,[(24,-24)], asrc,"hanebado_nced02.mka")

src_nced = src_nced[24:-24]

vrdf.generate_keyframes(src_nced, 'hanebado_nced02_keyframes.log')

src16 = fvf.Depth(src_nced, 16)

planes = kgf.split(src16)
planes[0] = lvf.upscaled_sraa(planes[0], 2, 6)
planes[1], planes[2] = [lvf.upscaled_sraa(plane, 1.5, 6) for plane in planes[1:]]
aa_c = kgf.join(planes)

final = fvf.Depth(aa_c, 10)

final.set_output()