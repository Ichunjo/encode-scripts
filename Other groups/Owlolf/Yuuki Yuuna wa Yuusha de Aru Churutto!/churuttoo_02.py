"""YuYuYu Churutto script"""
__author__ = 'Vardë'

import os
import shlex
import subprocess
from acsuite import eztrim

import vardefunc as vdf
import havsfunc as hvf
import lvsfunc as lvf
import G41Fun as gf
import xvs

import init_source

from vsutil import depth, get_y
import vapoursynth as vs

core = vs.core


NUM = __file__[-5:-3]
WEB = init_source.Infos()
WEB.set_infos(f'{NUM}/Yuuki Yuuna wa Yuusha de Aru Churutto! - {NUM} (Amazon Rental VBR 1080p).mkv', 24, -23, preset='web/eac3')



def do_filter():
    """Vapoursynth filtering"""
    src = WEB.src_cut
    src = src.std.AssumeFPS(src)


    src = depth(src, 16)
    out = src


    denoise = hvf.SMDegrain(out, tr=1, thSAD=100, thSADC=100)
    out = denoise


    dering = hvf.EdgeCleaner(out, 15, smode=1, hot=True)
    dering = gf.MaskedDHA(dering, darkstr=0.05, brightstr=0.75)
    out = dering


    aaa = vdf.nnedi3_upscale(get_y(out), pscrn=1, correct_shift=False)
    aaa = aaa.resize.Bicubic(1920, 1080, src_left=0.5, src_top=0.5, filter_param_a=-0.5, filter_param_b=0.25)
    out = vdf.merge_chroma(aaa, out)


    cwarp = xvs.WarpFixChromaBlend(out, 64, depth=4)
    out = cwarp


    detail_mask = detail_mask_func(out, brz_a=2250, brz_b=1200)
    deband = vdf.dumb3kdb(out, 16, threshold=33, grain=(24, 0))
    deband = core.std.MaskedMerge(deband, out, detail_mask)
    out = deband


    return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])



def do_encode(clip):
    """Compression with x26X"""
    if not os.path.isfile(WEB.output):
        print('\n\n\nVideo encoding')
        bits = clip.format.bits_per_sample
        x265_cmd = f'x265 -o {WEB.output} - --y4m' + ' '
        x265_cmd += f'--csv {WEB.name}_log_x265.csv --csv-log-level 2' + ' '
        x265_cmd += '--preset slower' + ' '
        x265_cmd += f'--frames {clip.num_frames} --fps {clip.fps_num}/{clip.fps_den} --output-depth {bits}' + ' '
        x265_cmd += '--rd 3 --no-rect --no-amp --rskip 1 --tu-intra-depth 2 --tu-inter-depth 2 --tskip' + ' '
        x265_cmd += '--merange 48 --weightb' + ' '
        # x265_cmd += '--no-strong-intra-smoothing' + ' '
        x265_cmd += '--psy-rd 1.65 --psy-rdoq 0.85 --no-open-gop --keyint 240 --min-keyint 23 --scenecut 40 --rc-lookahead 48 --bframes 8' + ' '
        x265_cmd += '--crf 16 --aq-mode 3 --aq-strength 0.60 --cbqpoffs -2 --crqpoffs -2 --qcomp 0.70' + ' '
        x265_cmd += '--deblock=0:0 --no-sao --no-sao-non-deblock' + ' '
        x265_cmd += f'--sar 1 --range limited --colorprim 1 --transfer 1 --colormatrix 1 --min-luma {str(16<<(bits - 8))} --max-luma {str(235<<(bits - 8))}'  # + ' '

        print("Encoder command: ", " ".join(shlex.split(x265_cmd)), "\n")
        process = subprocess.Popen(shlex.split(x265_cmd), stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    print('\n\n\nAudio extraction')
    mkv_args = ['mkvextract', WEB.src, 'tracks', f'1:{WEB.a_src}']
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    print('\n\n\nAudio cutting')
    eztrim(WEB.src_clip, (WEB.frame_start, WEB.frame_end), WEB.a_src, WEB.a_src_cut.format(1))

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', WEB.output_final,
                '--track-name', '0:HEVC WEBRip by Vardë@Owlolf', '--language', '0:jpn', WEB.output,
                '--track-name', '0:EAC3 2.0', '--language', '0:jpn', WEB.a_src_cut.format(1)]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
else:
    WEB.src_cut.set_output(0)

    FILTERED = do_filter()
    FILTERED.set_output(1)
