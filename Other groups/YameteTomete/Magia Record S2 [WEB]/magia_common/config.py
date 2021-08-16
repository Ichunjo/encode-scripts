import os
from typing import Any, Dict, List, Optional, Tuple, Union

import vapoursynth as vs
from vardautomation import (JAPANESE, AudioStream, EztrimCutter, FileInfo, Mux,
                            Patch, RunnerConfig, SelfRunner, VideoStream,
                            X265Encoder)
from vardautomation.tooling import MKVAudioExtracter
from vardefunc.types import Range

core = vs.core


class Encoding:
    v_encoder: X265Encoder

    def __init__(self, file: FileInfo, clip: vs.VideoNode, num: str) -> None:
        self.file = file
        self.clip = clip
        self.num = num

    def run(self, *, zones: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None, upload_ftp: bool = False) -> None:
        assert self.file.a_src
        assert self.file.a_src_cut

        self.v_encoder = X265Encoder('magia_common/x265_settings', zones=zones)
        a_extracter = MKVAudioExtracter(self.file, track_in=1, track_out=1)
        a_cutter = EztrimCutter(self.file, track=1)

        muxer = Mux(
            self.file,
            streams=(
                VideoStream(self.file.name_clip_output, 'HEVC WEBRip by Vardë@Raws-Maji', JAPANESE),
                AudioStream(self.file.a_src_cut.set_track(1), 'EAC3 2.0', JAPANESE),
                None
            ), merge_args={'--ui-language': 'en'}
        )
        # muxer = Mux(self.file)

        config = RunnerConfig(
            self.v_encoder, None,
            a_extracter, a_cutter, None,
            muxer
        )

        runner = SelfRunner(self.clip, self.file, config)
        runner.run()

        self.file.name_file_final = self.file.name_file_final.replace(f'magireco_s2_{self.num}_premux.mkv')


        rclone_cmd = (f'rclone copy --progress --sftp-set-modtime=false "{self.file.name_file_final.absolute().as_posix()}" '
                      + f"YametoTomato:files/ongoing/magireco_s2/{self.num}/")
        print(rclone_cmd)
        if upload_ftp:
            os.system(rclone_cmd)

    def do_patch(self, ranges: Union[Range, List[Range]]) -> None:
        p = Patch(self.v_encoder, self.clip, self.file, ranges)
        p.run()
        p.do_cleanup()
