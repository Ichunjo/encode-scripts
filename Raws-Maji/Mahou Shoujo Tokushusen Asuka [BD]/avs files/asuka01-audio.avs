LoadPlugin("C:\Program Files (x86)\AviSynth+\plugins64\ffms2.dll")
FFAudioSource("asuka01.w64")
__film = last
__just_audio = last
__blank = BlankClip(length=99999, fps=23.976)
__film = AudioDub(__blank, __film)
__final = __film.trim(24, 34837)
__final 
AudioDubEx(__just_audio, last)
AudioBits(last)>24?ConvertAudioTo24bit(last):last 
return last