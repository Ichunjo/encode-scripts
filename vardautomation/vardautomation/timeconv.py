"""Conversion time module"""

from fractions import Fraction

from .status import Status


class Convert:
    """Collection of static method to perform time conversion"""
    @classmethod
    def ts2f(cls, ts: str, fps: Fraction, /) -> int:
        s = cls.ts2seconds(ts)
        f = cls.seconds2f(s, fps)
        return f

    @classmethod
    def f2ts(cls, f: int, fps: Fraction, /, *, precision: int = 3) -> str:
        s = cls.f2seconds(f, fps)
        ts = cls.seconds2ts(s, precision=precision)
        return ts

    @classmethod
    def seconds2ts(cls, s: float, /, *, precision: int = 3) -> str:
        m = s // 60
        s %= 60
        h = m // 60
        m %= 60
        return cls.composets(h, m, s, precision=precision)

    @staticmethod
    def f2seconds(f: int, fps: Fraction, /) -> float:
        if f == 0:
            s = 0.0

        t = round(float(10 ** 9 * f * fps ** -1))
        s = t / 10 ** 9
        return s

    @staticmethod
    def ts2seconds(ts: str, /) -> float:
        h, m, s = map(float, ts.split(':'))
        return h * 3600 + m * 60 + s

    @staticmethod
    def seconds2f(s: float, fps: Fraction, /) -> int:
        return round(s * fps)

    @staticmethod
    def composets(h: float, m: float, s: float, /, *, precision: int = 3) -> str:
        if precision == 0:
            out = f"{h:02.0f}:{m:02.0f}:{round(s):02}"
        elif precision == 3:
            out = f"{h:02.0f}:{m:02.0f}:{s:06.3f}"
        elif precision == 6:
            out = f"{h:02.0f}:{m:02.0f}:{s:09.6f}"
        elif precision == 9:
            out = f"{h:02.0f}:{m:02.0f}:{s:012.9f}"
        else:
            Status.fail('precision must be <= 9 and >= 0', exception=ValueError)
        return out
