"""Automation module"""
import subprocess
from typing import Dict, List, Tuple

import vapoursynth as vs

from .status import Status
from .types import AnyPath

core = vs.core


class Properties:
    """Collection of static methods to get some properties from the parameters and/or the clip"""
    @staticmethod
    def get_color_range(params: List[str], clip: vs.VideoNode, bits: int) -> Tuple[int, int]:
        """[summary]

        Args:
            params (List[str]): [description]
            clip (vs.VideoNode): [description]
            bits (int): [description]

        Raises:
            ValueError: [description]
            vs.Error: [description]
            ValueError: [description]

        Returns:
            Tuple[int, int]: [description]
        """
        if '--range' in params:
            rng_param = params[params.index('--range') + 1]
            if rng_param == 'limited':
                min_luma = 16 << (bits - 8)
                max_luma = 235 << (bits - 8)
            elif rng_param == 'full':
                min_luma = 0
                max_luma = (1 << bits) - 1
            else:
                Status.fail('Wrong range in parameters!', exception=ValueError)
        elif '_ColorRange' in clip.get_frame(0).props:
            color_rng = clip.get_frame(0).props['_ColorRange']
            if color_rng == 1:
                min_luma = 16 << (bits - 8)
                max_luma = 235 << (bits - 8)
            elif color_rng == 0:
                min_luma = 0
                max_luma = (1 << bits) - 1
            else:
                Status.fail('Wrong "_ColorRange" prop in the clip!', exception=vs.Error)
        else:
            Status.fail('Cannot guess the color range!', exception=ValueError)

        return min_luma, max_luma

    @staticmethod
    def get_csp(clip: vs.VideoNode) -> str:
        """[summary]

        Args:
            clip (vs.VideoNode): [description]

        Returns:
            str: [description]
        """
        def _get_csp_subsampled(format_clip: vs.Format) -> str:
            sub_w, sub_h = format_clip.subsampling_w, format_clip.subsampling_h
            csp_yuv_subs: Dict[Tuple[int, int], str] = {(0, 0): 'i444', (1, 0): 'i422', (1, 1): 'i420'}
            return csp_yuv_subs[(sub_w, sub_h)]

        assert clip.format is not None

        csp_avc = {
            vs.GRAY: 'i400',
            vs.YUV: _get_csp_subsampled(clip.format),
            vs.RGB: 'rgb'
        }
        return csp_avc[clip.format.color_family]

    @staticmethod
    def get_encoder_name(path: AnyPath) -> str:
        ffprobe_args = ['ffprobe', '-loglevel', 'quiet', '-show_entries', 'format_tags=encoder',
                        '-print_format', 'default=nokey=1:noprint_wrappers=1', str(path)]
        return subprocess.check_output(ffprobe_args, shell=True, encoding='utf-8')
