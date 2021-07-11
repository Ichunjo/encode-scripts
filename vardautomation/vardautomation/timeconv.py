"""Conversion time module"""

from fractions import Fraction

from .status import Status


class Convert:
    """Collection of static method to perform time conversion"""
    @staticmethod
    def f2seconds(f: int, fps: Fraction, /) -> float:  # noqa
        if f == 0:
            s = 0.0  # noqa

        t = round(float(10 ** 9 * f * fps ** -1))  # noqa
        s = t / 10 ** 9  # noqa
        return s

    @staticmethod
    def f2ts(f: int, fps: Fraction, /, *, precision: int = 3) -> str:  # noqa
        s = Convert.f2seconds(f, fps)  # noqa
        ts = Convert.seconds2ts(s, precision=precision)  # noqa
        return ts

    @staticmethod
    def ts2seconds(ts: str, /) -> float:  # noqa
        h, m, s = map(float, ts.split(':'))  # noqa
        return h * 3600 + m * 60 + s

    @staticmethod
    def seconds2ts(s: float, /, *, precision: int = 3) -> str:  # noqa
        m = s // 60  # noqa
        s %= 60  # noqa
        h = m // 60  # noqa
        m %= 60  # noqa

        return Convert.composets(h, m, s, precision=precision)

    @staticmethod
    def seconds2f(s: float, fps: Fraction, /) -> int:  # noqa
        return round(s * fps)

    @staticmethod
    def ts2f(ts: str, fps: Fraction, /) -> int:  # noqa
        s = Convert.ts2seconds(ts)  # noqa
        f = Convert.seconds2f(s, fps)  # noqa
        return f

    @staticmethod
    def composets(h: float, m: float, s: float, /, *, precision: int = 3) -> str:  # noqa
        if precision == 0:  # noqa
            return f"{h:02.0f}:{m:02.0f}:{round(s):02}"
        elif precision == 3:
            return f"{h:02.0f}:{m:02.0f}:{s:06.3f}"
        elif precision == 6:
            return f"{h:02.0f}:{m:02.0f}:{s:09.6f}"
        elif precision == 9:
            return f"{h:02.0f}:{m:02.0f}:{s:012.9f}"
        else:
            Status.fail('precision must be <= 9 and >= 0', exception=ValueError)
