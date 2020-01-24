import vapoursynth as vs
import lvsfunc as lvf
import kagefunc as kgf
import fvsfunc as fvf
import vardefunc as vrdf
import modfunc as mdf
import vsTAAmbk as taa
import acsuite
from cooldegrain import CoolDegrain
from vsutil import insert_clip
from typing import List

core = vs.core
core.max_cache_size = 64000
ac = acsuite.AC()

src_amz = lvf.src(r"Made in Abyss - Movie 1 (Amazon dAnime VBR 720p).mkv")
src_ru = lvf.src(r"Gekijouban Soushuuhen Made in Abyss - 01 (WEB-DL 1920x1080 x264 AAC Rus).mkv")
src_fr = lvf.src(r"Made in Abyss - Movie 01 (Wakanim FR 1080p) [!].mkv")
src_ = (
    lvf.src(r"[BDMV][アニメ] メイドインアビス/[BDMV][アニメ][171025] メイドインアビス 上巻/DISC 1/BDMV/STREAM/00004.m2ts"),
    lvf.src(r"[BDMV][アニメ] メイドインアビス/[BDMV][アニメ][171025] メイドインアビス 上巻/DISC 1/BDMV/STREAM/00005.m2ts"),
    lvf.src(r"[BDMV][アニメ] メイドインアビス/[BDMV][アニメ][171025] メイドインアビス 上巻/DISC 1/BDMV/STREAM/00006.m2ts"),
    lvf.src(r"[BDMV][アニメ] メイドインアビス/[BDMV][アニメ][171025] メイドインアビス 上巻/DISC 1/BDMV/STREAM/00007.m2ts"),
    lvf.src(r"[BDMV][アニメ] メイドインアビス/[BDMV][アニメ][171025] メイドインアビス 上巻/DISC 2/BDMV/STREAM/00005.m2ts"),
    lvf.src(r"[BDMV][アニメ] メイドインアビス/[BDMV][アニメ][171025] メイドインアビス 上巻/DISC 2/BDMV/STREAM/00006.m2ts"),
    lvf.src(r"[BDMV][アニメ] メイドインアビス/[BDMV][アニメ][171025] メイドインアビス 上巻/DISC 2/BDMV/STREAM/00007.m2ts"),
    lvf.src(r"[BDMV][アニメ] メイドインアビス/[BDMV][アニメ][171222] メイドインアビス 下巻/DISC 1/BDMV/STREAM/00004.m2ts"),
    lvf.src(r"[BDMV][アニメ] メイドインアビス/[BDMV][アニメ][171222] メイドインアビス 下巻/DISC 1/BDMV/STREAM/00005.m2ts")
)
asrc_amz = r'Made in Abyss - Movie 1 (Amazon dAnime VBR 720p).mka'
asrc_rus = r'Gekijouban Soushuuhen Made in Abyss - 01 (WEB-DL 1920x1080 x264 AAC Rus).mka'

def Depth(clip: vs.VideoNode, bits: int) -> vs.VideoNode:
    return fvf.Depth(clip, bits)

# Using znedi3 instead of nnedi3
# Thanks to kageru and LightArrowsEXE for create this
# Original function: https://github.com/Irrational-Encoding-Wizardry/lvsfunc
def rescale(clip: vs.VideoNode, res: List[int]) -> vs.VideoNode:
    return mdf.smart_descale_mod(clip, res, 1/3, 1/3)
    # return lvf.smart_descale(clip, res, 1/3, 1/3)
	
def denoise_uv(clip: vs.VideoNode, strength: int) -> vs.VideoNode:
    return core.knlm.KNLMeansCL(clip, a=2, h=strength, d=3, device_type='gpu', channels='UV')
    # return mvf.BM3D(clip, sigma=strength)

def denoise_y(clip: vs.VideoNode, strength: int) -> vs.VideoNode:
	return CoolDegrain(clip, tr=1, thsad=strength, thsadc=0, bits=16, blksize=8, overlap=4)

def deband(clip: vs.VideoNode, range: int, strength_a: int, strength_b: int, grain: int, luma_scaling: int) -> vs.VideoNode:
    return core.std.MaskedMerge(
        core.neo_f3kdb.Deband(clip, range, strength_a, strength_a, strength_a, grain, 0, sample_mode=4), 
        core.neo_f3kdb.Deband(clip, range, strength_b, strength_b, strength_b, grain, 0, sample_mode=4), 
        vrdf.adaptive_mask(clip, luma_scaling))

def aa(clip: vs.VideoNode, aatype: str, alpha: float, beta: float, gamma: int) -> vs.VideoNode:
    return taa.TAAmbk(Depth(clip, 16), aatype=aatype, alpha=alpha, beta=beta, gamma=gamma, mtype=0, down8=False)

def adapt_grain(clip: vs.VideoNode, strength: int, luma_scaling: int) -> vs.VideoNode:
    return kgf.adaptive_grain(clip, strength, luma_scaling=luma_scaling)

def web_filter(clip: vs.VideoNode, rescale_bool: bool = True) -> vs.VideoNode:
    clip = Depth(clip, 32)
    clip = denoise_uv(clip, 0.5)
    # clip = denoise_uv(clip, [0, 1.5])
    if rescale_bool:
        clip = rescale(clip, [846, 847, 848, 849])
    clip = aa(clip, 'Eedi3', 0.4, 0.5, 200)
    clip = deband(clip, 17, 36, 48, 32, 20)
    clip = adapt_grain(clip, 0.4, 8)
    return clip

def bd_filter(clip: vs.VideoNode) -> vs.VideoNode:
    clip = Depth(clip, 32)
    clip = denoise_uv(clip, 0.5)
    clip = denoise_y(clip, 48)
    clip = rescale(clip, [846, 847, 848, 849])
    clip = aa(clip, 'Eedi3', 0.3, 0.5, 500)
    clip = deband(clip, 17, 30, 36, 24, 20)
    clip = adapt_grain(clip, 0.2, 8)
    return clip

def dehardsub(clip_a: vs.VideoNode, clip_b: vs.VideoNode, 
                left: int = 0, right: int = 0, top: int = 0, bottom: int = 0, 
                highpass: int = 2000) -> vs.VideoNode:
    mask = kgf.hardsubmask_fades(clip_a, clip_b, highpass=highpass)
    if left > 0 or right > 0 or top > 0 or bottom > 0:
        mask = vrdf.RegionMask(mask, left, right, top, bottom)
    return core.std.MaskedMerge(clip_a, clip_b, mask)

# Slicing audios & videos
# I prefer remove the intros and others things.
# The third line is here to export the missing part and concat to the 2.0 channel after.
# Command in settings file
ac.eztrim(src_amz,  [(480,170712)],                    asrc_amz, "mia_movie1_2.0.mka")
ac.eztrim(src_ru,   [(480,170712),(170880,171239)],    asrc_rus, "mia_movie1_5.1.mka")
ac.eztrim(src_ru,   [(170880,171239)],                 asrc_rus, "mia_movie1_2.0+.mka")

# Remove studio ads and (I think) operating certificate
src_ru = src_ru[480:170712] + src_ru[170880:171239]
src_fr = src_fr[480:170712] + src_fr[170880:171239]
ref = src_ru

# Filtering chain
src_ = [bd_filter(src) for src in src_]
src_.insert(0, None)

# The post credit return an error known
src_fr_norescale = web_filter(src_fr, False)
src_ru_norescale = web_filter(src_ru, False)
src_fr = web_filter(src_fr)
src_ru = web_filter(src_ru)

# Some random dehardsub clips for credits or signs
clip_dehard_a = dehardsub(src_[2][16874:16952], src_ru[36645:36723])
clip_dehard_b = dehardsub(src_[3][19321:19398], src_ru[59966:60043])
clip_dehard_c = dehardsub(src_[3][27743:28043] + src_[3][28205 + 13:28355], src_ru[66165:66165 + 437])
clip_dehard_d = dehardsub(src_[3][28937:29213], src_ru[66872:67148])
clip_dehard_e = dehardsub(src_[3][29417:29543], src_ru[67352:67478])
clip_dehard_f = dehardsub(src_[3][33611:33741], src_ru[71278:71408])
clip_dehard_g = dehardsub(src_[4][16643:16736], src_ru[80227:80320], 0, 0, 500, 0)
clip_dehard_h = dehardsub(src_fr[89053:89209], src_ru[89053:89209], 0, 0, 0, 500)
clip_dehard_i = dehardsub(src_fr[145183:145399], src_ru[145183:145399], 0, 0, 0, 750)

sq_m = kgf.squaremask(src_fr_norescale, 1920, 600, 0, 197).std.Inflate()
clip_dehard_j = core.std.MaskedMerge(src_ru_norescale, src_fr_norescale, sq_m)

clip_dehard_k = dehardsub(src_ru_norescale[170484:170534], src_fr_norescale[170484:170534], 0, 0, 195, 0, 1300)
clip_dehard_l = dehardsub(src_ru_norescale[170534:src_ru.num_frames], src_fr_norescale[170534:src_ru.num_frames], 0, 0, 150, 0, 1300)

clip_a = src_ru[0:5430]
clip_b = src_fr[0:5430]
clip_dehard_m = dehardsub(clip_a, clip_b, 0, 0, 200, 200, 1300)
clip_dehard_n = dehardsub(clip_a, clip_b, 0, 0, 200, 0, 1300)


# The fucking slicing smh
final = (
    clip_dehard_m[:5130] + clip_dehard_n[5130:5298] + clip_dehard_m[5298:] + src_[1][180:519] + src_[1][540:735] + 
    src_[1][747:1029] + src_[1][1047:1580] + src_[1][2542:2733] + src_ru[6970:7046] + src_[1][2821:3053] + src_[1][3080:3513] + 
    src_[1][3878:4496] + src_[1][4495] * 6 + src_[1][4496:5153] + src_[1][5152] * 12 + src_[1][5153:5519] + src_ru[9370:9472] + 
    src_[1][5633:5909] + src_ru[9748:9796] + src_[1][5957:6050] + src_ru[9889:10071] + src_[1][6289:8917] + src_[1][8923:8938] +
    src_[1][8944:8962] + src_[1][8968:9800] + src_[1][14486:16105] + src_[1][16111:16477] + src_[1][19420 + 24:21922] +
    src_[1][21925:21934] + src_[1][21941:21943] + src_[1][21942] + src_[1][21943:21955] + src_[1][21958:22000] +
    src_[1][22003:24882] + src_ru[20972:21116] + src_[1][25050:27947] + src_[1][27952:27975] + src_[1][27974] * 5 +
    src_[1][27975:28923] + src_ru[24989:25001] + src_[1][28923:30231] + src_ru[26309:27194] + src_[2][3861:6165] +
    src_[2][6204:6246] + src_[2][6273:10998] + src_[2][14494:15165] + src_ru[34936:35164] + src_[2][15393:16874] +
    clip_dehard_a + src_ru[36723:36741] + src_[2][16952 + 24:17621] + src_[2][18681:18789] + src_ru[37494:37596] +
    src_[2][18915:19380] + src_[2][20064:20319] + src_ru[38316:38490] + src_[2][20493:22113] + src_[2][22200:22386] +
    src_[2][22385] * 9 + src_[2][22452:25995] + src_[2][25995 + 37:26727] + src_[2][26877:27663] + src_ru[45329:45427] +
    src_[2][27735:28113] + src_[2][29475:30069] + src_[2][30153:31521] + src_ru[47767:48109] + src_[3][361:1978] +
    src_ru[49726:49935] + src_[3][5751:8529] + src_[3][8523] * 3 + src_[3][8519] * 9 + src_[3][8616:10185] +
    src_[3][12253:14585] + src_[3][14624:15266] + src_[3][15296:15518] + src_ru[57490:57715] + src_[3][15662:15670] +
    src_[3][15669] + src_[3][15673] * 3 + src_[3][15674:15695] + src_[3][15776:16784] + src_ru[58756:58972] +
    src_[3][18317 + 10:19321] + clip_dehard_b + src_[3][19398:21201] + src_[3][21816:21978] + src_[3][22392:23967] +
    src_[3][24039:26621] + clip_dehard_c + src_[3][28355:28625] + clip_dehard_d + src_[3][29213:29417] + clip_dehard_e +
    src_ru[67478:67544] + src_[3][29609:29957] + src_[3][29999:30941] + src_ru[68834:68930] + src_[3][31037:31301] +
    src_ru[69194:69362] + src_[3][31469:31787] + src_ru[69680:69728] + src_[3][31835:32015] + src_ru[69908:69962] +
    src_[3][32069:32100] + src_ru[69993:70081] + src_[3][32188:32212] + src_ru[70105:70207] + src_[3][32392:32524] +
    src_ru[70339:70723] + src_[3][32908:33016] + src_ru[70831:70861] + src_[3][33046:33082] + src_ru[70897:71120] +
    src_[3][33453:33611] + clip_dehard_f + src_[3][33741:33831] + src_[4][3237:3442] + src_[4][3731:4077] +
    src_ru[72049:72073] + src_[4][4101:4311] + src_[4][4383:5373] + src_[4][5439:6165] + src_[4][6429:6708] +
    src_[4][6795:7077] + src_[4][7242 + 27:7404] + src_[4][7416 + 12:7529] + src_[4][7943:8315] + src_ru[75168:75348] +
    src_[4][8495:8594] + src_ru[75447:75519] + src_[4][8666:8738] + src_ru[75591:75693] + src_[4][8840:8906] +
    src_[4][8996:9044] + src_[4][9044 + 12:9629] + src_ru[76380:76509] + src_[4][9841:9894] + src_[4][9916:10014] +
    src_[4][10087:10117] + src_ru[76690:76720] + src_[4][10189:10291] + src_[4][10297:10582] + src_[4][10687:11041] +
    src_[4][11074:12034] + src_[4][12106:12355] + src_[4][12367:12425] + src_[4][12449:12679] + src_[4][12874:13078] +
    src_[4][13450 + 12:13894] + src_ru[79594:79822] + src_[4][15842:16196] + src_[4][16592:16643] + clip_dehard_g +
    src_[4][16814:17426] + src_ru[80932:81879] + src_[4][20081:20102] + src_ru[81900:81930] + src_[4][20132:20593] +
    src_[4][20597:20687] + src_[4][20813:21675] + src_[4][21684:21714] + src_[4][21738:22198] + src_[4][22260:22563] +
    src_ru[84136:84352] + src_[4][22887:23376] + src_ru[84841:85072] + src_[4][23736:23910] + src_[4][23952:24057] +
    src_[4][24102:24186] + src_[4][24185] * 12 + src_[4][24234:24507] + src_ru[85720:85963] + src_[4][24864:25218] +
    src_[4][25506 + 11:25608] + src_ru[86408:86774] + src_[4][25974:26106] + src_[4][26320:26959] + src_[4][27004:27103] +
    src_[4][27112 + 45:27660] + src_ru[88147:88249] + src_[4][29083:29215] + src_[4][29215 + 60:29479] +
    src_[4][29509:29785] + src_[4][29833:30025] + clip_dehard_h + src_[4][30181:30424] + src_[4][30499:31370] +
    src_[4][31369] * 36 + src_ru[90359:90551] + src_[5][3087:3233] + src_[5][3297:3366] + src_[5][3390:3483] +
    src_[5][3531:3783] + src_[5][4037:4259] + src_[5][4259 + 24:4499] + src_[5][4516:4706] + src_[5][4718:5192] +
    src_[5][5192 + 18:5249] + src_[5][5261:5342] + src_[5][5357:5393] + src_[5][5405:5687] + src_[5][5798:5876] +
    src_[5][8176:8236] + src_ru[92789:92822] + src_[5][8278 + 10:8323] + src_[5][8330:8368] + src_[5][8374:8557] +
    src_ru[93078:93087] + src_[5][8572:8675] + src_[5][8686 + 18:8764] + src_[5][8782:8821] + src_[5][8830:8884] +
    src_[5][8884 + 12:8986] + src_[5][8986+6:9322] + src_[5][9346:9358] + src_[5][9367:9461] + src_[5][9473:9556] + 
    src_[5][9580+53:9922] + src_[5][9946:10060] + src_[5][10066:10204] + src_ru[94493:94541] + src_[5][10204:10660] +
    src_[5][10672:10810] + src_[5][10816:10972] + src_[5][11266:11602] + src_[5][11608:11827] + src_[5][11842-3:11914] +
    src_[5][12004:12058] + src_ru[95975:96192] + src_[5][12293:12856] + src_ru[96755:96830] + src_[5][12946:13270] + 
    src_[5][13300:13498] + src_[5][13510:13555] + src_[5][13564:13948] + src_ru[97781:98039] + src_[5][14206:14536] + 
    src_ru[98369:98474] + src_[5][14656:15193] + src_[5][15202:15694] + src_[5][15706:15754] + src_ru[99551:99752] + 
    src_[5][16064:16799] + src_[5][16835:17375] + src_[5][17447:17771] + src_ru[101351:101825] + src_[5][18563:18845] +
    src_[5][22121:22283] + src_ru[102269:102383] + src_[5][22397:22760] + src_[5][23258:24314] + src_[5][24560+16:25484] + 
    src_[5][27787+16:28663] + src_[5][28663+93:28951] + src_[5][29739:30162] + src_[5][30161] * 3 + src_[5][30177:30303] + 
    src_[5][30456:31146] + src_[6][2149+9:2233] + src_[6][2239+23:2323] + src_[6][2449:2521] + src_[6][2593:2743] + src_ru[107365:107425] + 
    src_[6][3055:3175] + src_[6][3223:3979] + src_[6][3994:4183] + src_[6][4183+21:4279] + src_[6][4279+6:4771] + src_ru[109051:109249] + 
    src_[6][7242:7362] + src_[6][7422:7716] + src_[6][7788:9123] + src_[6][9147:9348] + src_[6][9348+12:12036] + src_[6][13029:14751] + 
    src_[6][15708:15780] + src_[6][16464:17877] + src_[6][21456:22659] + src_[6][22827:23290] + src_[6][23307:23367] + 
    src_ru[118808:118922] + src_[6][24332:24940] + src_ru[119530:119824] + src_[6][25276:25474] + src_[6][25854:25986] + 
    src_[6][25986+12:26106] + src_[6][26118:26718] + src_ru[120862:120886] + src_[6][27312:27384] + src_[6][27468:27726] + 
    src_ru[121216:121278] + src_[6][27845:28830] + src_[6][28974:29322] + src_[6][29412:29982] + src_[6][30066:30186] + 
    src_[6][30318:30366] + src_[6][30366+48:30603] + src_[6][30630+3:30852] + src_[6][30852+12:30996] + src_[6][31104:31572] + 
    src_ru[124357:124447] + src_[7][5541:5900] + src_[7][5957:6177] + src_ru[125026:125176] + src_[7][6378:8949] + 
    src_[7][8967:8979] + src_ru[127759:127783] + src_[7][9012:9324] + src_[7][9348:9720] + src_ru[128467:128497] + 
    src_[7][9759:10359] + src_ru[129097:129400] + src_[7][10689:10845] + src_[7][10863:11121] + src_[7][11217:11802] + 
    src_[7][11811:11859] + src_[7][11865:12216] + src_[7][12225+9:12273] + src_[7][12279:12568] + src_ru[131126:131191] + 
    src_[7][12642:12690] + src_ru[131239:131275] + src_[7][12726:13161] + src_ru[131710:131839] + src_[7][13290:13314] + 
    src_ru[131863:131923] + src_[7][13374:13443] + src_[7][13461:13506] + src_[7][13509+9:13539] + src_[7][13545:13755] + 
    src_[7][13797:13972] + src_[7][13978:14011] + src_[7][14017:14031] + src_[7][14038:14107] + src_[7][14116:14209] + 
    src_[7][14218:14596] + src_[7][14608:14959] + src_[7][14971+3:15064] + src_ru[133471:133522] + src_[7][15151:15350] + 
    src_[7][15416:15444] + src_ru[133749:133833] + src_[7][15528:15971] + src_[7][16025:16533] + src_[7][16533+9:16701] + 
    src_[7][16713:16749] + src_ru[134979:135119] + src_[7][16882:17085] + src_ru[135322:135393] + src_[7][17372:17852] + 
    src_ru[135873:136017] + src_[7][18014:18047] + src_[7][18062:18248] + src_[7][18347:18419] + src_[7][18479:18515] + 
    src_[7][18518:18820] + src_[7][18825+31:19442] + src_ru[137232:137495] + src_[7][19850:19916] + src_ru[137561:137607] + 
    src_[7][19988:20210] + src_[7][20234:23258] + src_[7][23330:23369] + src_[7][23603:24676] + src_[7][27385:27685] + 
    src_[7][27733:28099] + src_[8][105:132] + src_[8][311:387] + src_[7][29038:31487] + clip_dehard_i + src_[8][7731:10281] + 
    src_[8][10377+8:10457] + src_[8][10456] * 8 + src_[8][10457:11123] + src_[7][24897:24930] + src_[7][24952:25219] + 
    src_[7][25267:25782] + src_[7][25891:26359] + src_ru[149978:149997] + src_[8][11148:13493] + src_ru[152342:152365] + 
    src_[8][13955:14213] + src_[8][14231:14525] + src_[8][14831:16388] + src_[8][16448:16538] + src_[8][16550:16604] +
    src_ru[154618:154720] + src_[8][16706:17741] + src_[8][17765+72:18215] + src_ru[156133:156625] + src_[8][18965:20084] + 
    src_ru[157744:157921] + src_[8][20654:20903] + src_[8][20948:21353] + src_[8][21449:22349] + src_ru[159475:159622] + 
    src_[8][22625:22715] + src_ru[159712:160011] + src_[8][23111:23432] + src_[8][23431] * 18 + src_[8][24803:27743] + 
    src_[8][27827:29776] + src_ru[165239:165335] + src_[8][29872:31660] + src_ru[167123:169991] + src_ru_norescale[169991:170101] + 
    clip_dehard_j[170101:170484] + clip_dehard_k + clip_dehard_l
    )

# Recover labial synchro
final = insert_clip(final, final[87653:87659], 87647)
final = insert_clip(final, final[87659:87661], 87656)
final = insert_clip(final, final[87657], 87658)

# Some black frames
final = fvf.rfs(final, src_ru, mappings="[13556 13563]")

# Credits
final = fvf.rfs(final, Depth(ref, 16), mappings="[168058 169990]")

final = Depth(final, 10)
final.set_output()