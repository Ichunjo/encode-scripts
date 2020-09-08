"""Script conversion image"""
import os
import subprocess
import shlex
import glob
from multiprocessing.pool import ThreadPool
from vsutil import is_image

def conversion(args):
    file = args
    # https://github.com/6o6o/fft-descreen
    os.system(f'python descreen.py \"{file}\" \"{file[:-3]}png\"')

    webp_cmd = f'cwebp -q 100 -m 6 -f 0 -lossless -quiet -mt \"{file[:-3]}png\" -o \"{file[:-3]}webp\"'
    subprocess.run(shlex.split(webp_cmd), check=True, encoding='utf-8')

    os.remove(f'\"{file[:-3]}png\"')


def main():
    files = glob.glob('**', recursive=True)

    num_threads = 12
    pool = ThreadPool(num_threads)

    for file in files:
        if is_image(file):
            pool.apply_async(conversion, ((file),))
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
