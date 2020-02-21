import vapoursynth as vs
import lvsfunc as lvf
import fvsfunc as fvf
import vardefunc as vrdf
import vsTAAmbk as taa
import debandshit as dbs
import havsfunc as hvf
from cooldegrain import CoolDegrain
from mvsfunc import GetMatrix
import acsuite

ac = acsuite.AC()

core = vs.core
core.max_cache_size = 16384

def get_h(width: int, aspect_ratio: float=16/9, only_even: bool=True) -> int:
    height = width / aspect_ratio
    height = int(round(height))
    if only_even:
        height = height // 2 * 2
    return height



src = lvf.src("IsekaiQuartet_Season1/BDMV/STREAM/00018.m2ts")
src_ncop = lvf.src("IsekaiQuartet_Season1/BDMV/STREAM/00029.m2ts")[24:-24]
src_jp = lvf.src("jp/[Isekai Quartet][05][BDRIP][1080P][H264_FLAC].mkv")[:-24]
endcard = core.imwri.Read('yande.re 538736 cleavage crossover dress emilia_(re_zero) endcard isekai_quartet megumin pointy_ears re_zero_kara_hajimeru_isekai_seikatsu rimukoro thighhighs.jpg', float_output=True)
endcard = core.std.AssumeFPS(endcard, src)

asrc = r'IsekaiQuartet_Season1/BDMV/STREAM/05.w64'
ac.eztrim(src,[(24,-24)], asrc,"isekai05-cut.mka")

src = src[24:-24]

opstart, opend, = 539, 2685
edstart, edend = 15154, 17023 # Première frame où Emilia bouge

src = lvf.insert_clip(src, src_ncop[:opend-opstart+1], opstart)
src = lvf.insert_clip(src, src_jp[edstart:], edstart)

# Credit JAP
src_jp = fvf.Depth(src_jp, 16)
src = fvf.Depth(src, 16)
credit_m = vrdf.DCM(
	src, 
	core.knlm.KNLMeansCL(src_jp, a=10, h=50, d=0, device_type='gpu'), 
	core.knlm.KNLMeansCL(src, a=10, h=50, d=0, device_type='gpu'), 
	0, src.num_frames-1, 2, 2)

credit_m = hvf.mt_expand_multi(credit_m, mode='ellipse', sw=30, sh=30)
credit_m = hvf.mt_inflate_multi(credit_m, 0, 4)

credit_jp = fvf.rfs(src, core.std.MaskedMerge(src, src_jp, credit_m), mappings=f'[{opstart} {opend}]')


shift = core.fmtc.resample(credit_jp, sx=[0, 0.5], kernel='spline36')
den = CoolDegrain(shift, tr=1, thsad=80, thsadc=60, bits=16, blksize=8, overlap=4)
aa = taa.TAAmbk(den, aatype='Nnedi3', mtype=0)
db = dbs.f3kbilateral(aa, 17, 76, 76)
db = core.neo_f3kdb.Deband(db, grainy=32, preset='depth')
grain = core.grain.Add(db, 0.3, constant=True)



# Endcard
endcard = core.std.CropRel(endcard, left=0, top=0, right=1, bottom=10)

endcard = core.resize.Spline36(endcard, grain.width, get_h(grain.width, endcard.width/endcard.height), vs.YUV444PS, matrix_s=GetMatrix(src), dither_type='error_diffusion')

endcard = core.std.Crop(endcard, 0, 0, (endcard.height-grain.height)/2, (endcard.height-grain.height)/2)
endcard = core.resize.Spline36(endcard, format=vs.YUV420P16)

endcard = core.neo_f3kdb.Deband(endcard, 18, 36, 36, 36, 32, 0)
endcard = endcard*239

merged = grain + endcard



final = fvf.Depth(merged, 10)

final.set_output()
