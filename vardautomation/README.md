# Vardautomation

Package I used for my automation stuff

# How to install vardautomation

```
python -m pip install git+https://github.com/Ichunjo/encode-scripts.git#subdirectory=vardautomation/ -U
```

# Requirements
* Python 3.9 or newer
* VapourSynth r53
* Other modules in `requirements.txt`


# Usage
## File indexing
My core module is based on `vardautomation.FileInfo`.


Let's say in my file `bangdream_film_live_1.py`:
```py
from vardautomation import FileInfo

JPBD = FileInfo('Bang Dream Live BD-1/BDMV/STREAM/00006.m2ts')

print(JPBD)
```

These are the default values for the `vardautomation.NoPreset` preset:
```
vardautomation.config.FileInfo({
    'path': 'Bang Dream Live BD-1\\BDMV\\STREAM\\00006', 
    'src': 'Bang Dream Live BD-1/BDMV/STREAM/00006.m2ts',
    'idx': lvsfunc.misc.source,  # function
    'name': 'bangdream_film_live_1',
    'a_src': '',
    'a_src_cut': '',
    'a_enc_cut': '',
    'chapter': '',
    'preset': [
        <vardautomation.presets.Preset object at 0x000001C4453F58E0>
    ],
    'clip': <vapoursynth.VideoNode object at 0x000001C4111188B0>,
    'frame_start': None,
    'frame_end': None,
    'clip_cut': <vapoursynth.VideoNode object at 0x000001C4111188B0>,
    'name_clip_output': 'bangdream_film_live_1.265',
    'name_file_final': 'bangdream_film_live_1.mkv',
    'name_clip_output_lossless': 'bangdream_film_live_1_lossless.mkv',
    'do_lossless': False,
    'qpfile': 'bangdream_film_live_1_qpfile.log',
    'do_qpfile': False
})
```

Several presets are available in `vardautomation.presets.py`.


## Basic usage
Let's take a situation where a full show has to be encoded.
First, import your file and do your filtering:
```py
import vapoursynth as vs
from vardautomation import FileInfo, PresetAAC, PresetBD

JPBD = FileInfo(r'[BDMV][210526][TBR31116D][ウマ娘 プリティーダービー Season 2][Vol.1]\UMAMUSUME2_1\BDMV\STREAM\00002.m2ts', 24, -24,
                preset=[PresetBD, PresetAAC])

class Filtering():
    def main(self) -> vs.VideoNode:
        src = JPBD.clip_cut
        ... # do things

        return final_clip
```

Now that you have your filtering done, the next step will be to encode it, extract audio(s), chapters and mux into a usable file.
These steps must therefore be prepared:
```py
ENCODER = X265Encoder('x265', Path('uma_common/x265_settings'))

A_EXTRACTER = BasicTool('eac3to', [JPBD.src, '2:', JPBD.a_src.format(1), '3:', JPBD.a_src.format(2), '-log=NUL'])
A_CUTTER = [AudioCutter(JPBD, track=1), AudioCutter(JPBD, track=2)]
A_ENCODER = [AudioEncoder('qaac', Path('uma_common/qaac_settings'), JPBD, track=1),
             AudioEncoder('qaac', Path('uma_common/qaac_settings'), JPBD, track=2)]
```

TODO
