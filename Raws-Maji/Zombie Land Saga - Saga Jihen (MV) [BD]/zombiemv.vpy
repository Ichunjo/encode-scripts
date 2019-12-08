import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import mvsfunc as mvf
import fvsfunc as fvf
import havsfunc as hvf
import vardefunc as vrdf
from nnedi3_rpow2CL import nnedi3_rpow2CL
from vsutil import get_y, get_w, split, join

import acsuite
ac = acsuite.AC()

core = vs.core
core.max_cache_size = 32000
core.num_threads = 14

src = lvf.src(r"Zombie Land Saga Franchouchou The Best [BDREMUX] [923224C3].mkv")
asrc = r"Zombie Land Saga Franchouchou The Best [BDREMUX] [923224C3].mka"
ac.eztrim(src, [(24,8368)], asrc, "audio-cut.mka")

src = src[24:8368]
src = fvf.Depth(src, 16)

deblock = hvf.Deblock_QED(src, quant1=24, quant2=26)
deblock = fvf.Depth(deblock, 32)

descaled = core.descale.Debicubic(get_y(deblock), get_w(844), 844, 0, 1/2)

denoise_y = mvf.BM3D(descaled, sigma=1.5, radius1=1, depth=16)
denoise_c = core.knlm.KNLMeansCL(deblock, a=2, h=0.4, d=3, device_type='gpu', channels='UV')
denoise_c = fvf.Depth(denoise_c, 16)

db_mask = kgf.retinex_edgemask(denoise_y)
db_y, db_c = vrdf.F3kdbSep(denoise_y, denoise_c, 17, 44, 36, 0, 0, mask=db_mask)
db_y_b, db_c_b = vrdf.F3kdbSep(denoise_y, denoise_c, 17, 48, 42, 0, 0, mask=db_mask)
db_y_c, db_c_c = vrdf.F3kdbSep(denoise_y, denoise_c, 17, 64, 56, 0, 0)
db_y_d, db_c_d = vrdf.F3kdbSep(denoise_y, denoise_c, 22, 82, 82, 0, 0)
db_y_e, db_c_e = vrdf.F3kdbSep(db_y_d, db_c_d, 10, 64, 64, 0, 0)

db_y = fvf.rfs(db_y, db_y_b, mappings="[2429 2459] [2586 2606] [2651 2735] [3801 3896] [3909 4301] [5333 5348] [5361 5374] [5385 5398] [5411 5422]")
db_c = fvf.rfs(db_c, db_c_b, mappings="[2429 2459] [2586 2606] [2651 2735] [3801 3896] [3909 4301] [5333 5348] [5361 5374] [5385 5398] [5411 5422]")
db_y = fvf.rfs(db_y, db_y_c, mappings="[3160 3211] [1215 1342] [5228 5283] [5308 5319]")
db_c = fvf.rfs(db_c, db_c_c, mappings="[3160 3211] [1215 1342] [5228 5283] [5308 5319]")
db_y = fvf.rfs(db_y, db_y_d, mappings="[1343 1375]")
db_c = fvf.rfs(db_c, db_c_d, mappings="[1343 1375]")
db_y = fvf.rfs(db_y, db_y_e, mappings="[3330 3383]")
db_c = fvf.rfs(db_c, db_c_e, mappings="[3330 3383]")

scaled = nnedi3_rpow2CL(db_y, nns=4, device=0).resize.Spline36(1920, 1080)

grain_a = core.f3kdb.Deband(scaled, grainy=24, output_depth=16, preset='depth')
grain_b = core.f3kdb.Deband(scaled, grainy=48, output_depth=16, preset='depth')
grain = fvf.rfs(grain_a, grain_b, mappings="[1343 1375] [3330 3383] [3160 3211] [1215 1342] [5228 5283] [5308 5319]")
grain = kgf.adaptive_grain(grain, 0.35, 6)

planes = split(db_c)

merged = join([grain, planes[1], planes[2]])

final = fvf.Depth(merged, 10)

final.set_output()