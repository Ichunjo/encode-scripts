"""DAB script"""
__author__ = 'Vardë'

import os
import subprocess

import vardefunc as vdf
from init_source import Infos

from vsutil import depth
import vapoursynth as vs

core = vs.core


JPBD = Infos()
JPBD.set_infos(r'[BDMV] Date a Bullet\BDROM\BDMV\STREAM\00000.m2ts', None, None)


def do_filter():
    """Vapoursynth filtering"""
    src = JPBD.src_cut
    src = depth(src, 32)
    out = src


    clips = [out[f] for f in range(out.num_frames)]

    mean = core.average.Mean(clips).std.AssumeFPS(fpsnum=1, fpsden=1)
    full = vdf.to_444(mean, znedi=False)
    out = depth(full, 16)


    decz = vdf.decsiz(out, sigmaS=50, min_in=0, max_in=1)
    out = decz


    return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])




def do_encode(clip):
    """Compression with x26X"""
    if not os.path.isfile(JPBD.output):
        print('\n\n\nVideo encoding')
        ffv1_args = [
            'ffmpeg', '-i', '-', '-vcodec', 'ffv1', '-coder', '1', '-context', '0',
            '-g', '1', '-level', '3', '-threads', '8',
            '-slices', '24', '-slicecrc', '1', '-color_range', 'tv', JPBD.name + "_lossless.mkv"
        ]
        print("Encoder command: ", " ".join(ffv1_args), "\n")
        process = subprocess.Popen(ffv1_args, stdin=subprocess.PIPE)
        clip.output(process.stdin, y4m=True, progress_update=lambda value, endvalue:
                    print(f"\rVapourSynth: {value}/{endvalue} ~ {100 * value // endvalue}% || Encoder: ", end=""))
        process.communicate()

    print('\n\n\nAudio extraction')
    eac3to_args = ['eac3to', JPBD.src, '2:', JPBD.a_src.format(1), '-log=NUL']
    subprocess.run(eac3to_args, text=True, check=True, encoding='utf-8')


    print('\n\n\nAudio encoding')
    qaac_args = ['qaac', JPBD.a_src.format(1), '-V', '127', '--no-delay', '-o', JPBD.a_enc_cut.format(1)]
    subprocess.run(qaac_args, text=True, check=True, encoding='utf-8')

    ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder', '-print_format', 'default=nokey=1:noprint_wrappers=1', JPBD.a_enc_cut.format(1)]
    encoder_name = subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')
    f = open("tags_aac.xml", 'w')
    f.writelines(['<?xml version="1.0"?>', '<Tags>', '<Tag>', '<Targets>', '</Targets>',
                  '<Simple>', '<Name>ENCODER</Name>', f'<String>{encoder_name}</String>', '</Simple>',
                  '</Tag>', '</Tags>'])
    f.close()

    print('\nFinal muxing')
    mkv_args = ['mkvmerge', '-o', JPBD.output_final,
                '--track-name', '0:HEVC BDRip by Vardë@Natsumi-no-Sekai', '--language', '0:jpn', JPBD.name + "_lossless.mkv",
                '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', JPBD.a_enc_cut.format(1)]
    subprocess.run(mkv_args, text=True, check=True, encoding='utf-8')

    # Clean up
    files = [JPBD.a_src, JPBD.a_src_cut.format(1),
             JPBD.a_enc_cut.format(1), 'tags_aac.xml']
    for file in files:
        if os.path.exists(file):
            os.remove(file)


if __name__ == '__main__':
    FILTERED = do_filter()
    do_encode(FILTERED)
