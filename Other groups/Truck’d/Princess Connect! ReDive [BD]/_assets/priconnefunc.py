"""Priconne functions"""
__author__ = 'Vardë'

from typing import Optional
import havsfunc as hvf

import vapoursynth as vs
core = vs.core


def line_darkening(clip: vs.VideoNode, strength: float = 0.2, **kwargs)-> vs.VideoNode:
    """Darken lineart through Toon

    Args:
        clip (vs.VideoNode): Source clip.
        strength (float, optional): Strength in Toon. Defaults to 0.2.

    Returns:
        vs.VideoNode: Darked clip.
    """
    darken = hvf.Toon(clip, strength, **kwargs)
    darken_mask = core.std.Expr(
        [core.std.Convolution(clip, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False),
         core.std.Convolution(clip, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False)],
        ['x y max {neutral} / 0.86 pow {peak} *'.format(neutral=1 << (clip.format.bits_per_sample-1),
                                                        peak=(1 << clip.format.bits_per_sample)-1)])
    return core.std.MaskedMerge(clip, darken, darken_mask)


def stabilization(orig: vs.VideoNode, sharp: vs.VideoNode, tradius: int = 2, aapel: int = 1,
                  aaov: Optional[int] = None, aablk: Optional[int] = None)-> vs.VideoNode:
    """Stabilize edges

    Args:
        orig (vs.VideoNode): Source clip. Should be in GRAY.

        sharp (vs.VideoNode): Sharp clip.

        tradius (int, optional): (1~3) 1 = Degrain1 / 2 = Degrain2 / 3 = Degrain3. Defaults to 2.

        aapel (int, optional): Accuracy of the motion estimation. Defaults to 1.
            (Value can only be 1, 2 or 4.
            1 means a precision to the pixel.
            2 means a precision to half a pixel,
            4 means a precision to quarter a pixel,
            produced by spatial interpolation (better but slower).)

        aaov (Optional[int], optional): Block overlap value (horizontal). Defaults to None.
            Must be even and less than block size.(Higher = more precise & slower)

        aablk (Optional[int], optional): Size of a block (horizontal). Defaults to None.
            Larger blocks are less sensitive to noise, are faster, but also less accurate.

    Returns:
        vs.VideoNode: Stabilized clip.
    """
    bits = orig.format.bits_per_sample
    neutral = 1 << (bits-1)

    if aaov is None:
        aaov = 8 if orig.width > 1100 else 4

    if aablk is None:
        aablk = 16 if orig.width > 1100 else 8

    diff = core.std.MakeDiff(orig, sharp)

    orig_super = hvf.DitherLumaRebuild(orig, s0=1).mv.Super(pel=aapel)
    diff_super = core.mv.Super(diff, pel=aapel)

    if tradius == 3:
        fv3 = core.mv.Analyse(orig_super, isb=False, delta=3, overlap=aaov, blksize=aablk)
        bv3 = core.mv.Analyse(orig_super, isb=True, delta=3, overlap=aaov, blksize=aablk)
    if tradius >= 2:
        fv2 = core.mv.Analyse(orig_super, isb=False, delta=2, overlap=aaov, blksize=aablk)
        bv2 = core.mv.Analyse(orig_super, isb=True, delta=2, overlap=aaov, blksize=aablk)
    if tradius >= 1:
        fv1 = core.mv.Analyse(orig_super, isb=False, delta=1, overlap=aaov, blksize=aablk)
        bv1 = core.mv.Analyse(orig_super, isb=True, delta=1, overlap=aaov, blksize=aablk)


    if tradius == 1:
        diff_degrain = core.mv.Degrain1(diff, diff_super, bv1, fv1)
    elif tradius == 2:
        diff_degrain = core.mv.Degrain2(diff, diff_super, bv1, fv1, bv2, fv2)
    elif tradius == 3:
        diff_degrain = core.mv.Degrain3(diff, diff_super, bv1, fv1, bv2, fv2, bv3, fv3)
    else:
        raise ValueError('stabilization: valid values of \"tradius\" are 1, 2 and 3!')

    diff_degrain = core.std.Expr([diff, diff_degrain], f'x {neutral} - abs y {neutral} - abs < x y ?').std.Merge(diff_degrain, 0.6)

    return core.std.MakeDiff(orig, diff_degrain)
