from vsutil import core, vs, depth, join, split, get_depth


def Deband(clip: vs.VideoNode, radius: int = 17, threshold: float = 4,
           iterations: int = 1, grain: float = 4, chroma: bool = True)-> vs.VideoNode:
    """Wrapper for placebo.Deband because at the moment, processing one plane is faster.

    Args:
        clip (vs.VideoNode):
        radius (int, optional): Defaults to 17.
        threshold (float, optional): Defaults to 4.
        iterations (int, optional): Defaults to 1.
        grain (float, optional): Defaults to 4.
        chroma (bool, optional): Defaults to True.

    Returns:
        vs.VideoNode
    """
    if get_depth(clip) != 16:
        clip = depth(clip, 16)
    if chroma is True:
        clip = join([core.placebo.Deband(x, 1, iterations, threshold, radius, grain)
                     for x in split(clip)])
    else:
        clip = core.placebo.Deband(clip, 1, iterations, threshold, radius, grain)
    return clip
