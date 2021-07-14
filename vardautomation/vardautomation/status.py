"""Logger module"""
from typing import NoReturn, Optional, Type

import colorama

colorama.init()


class Colors:
    FAIL: str = f'{colorama.Back.RED}{colorama.Fore.BLACK}'
    INFO: str = f'{colorama.Back.BLUE}{colorama.Fore.WHITE}{colorama.Style.BRIGHT}'
    RESET: str = colorama.Style.RESET_ALL


class Status:
    @staticmethod
    def fail(string: str, /, *, exception: Type[Exception] = Exception, chain_err: Optional[Exception] = None) -> NoReturn:
        raise exception(f'{Colors.FAIL}{string}{Colors.RESET}') from chain_err

    @staticmethod
    def info(string: str, /, raise_error: bool = False, *,
             exception: Optional[Type[Exception]] = Exception, chain_err: Optional[Exception] = None) -> None:
        if not raise_error:
            print(f'{Colors.INFO}{string}{Colors.RESET}')
        else:
            if exception:
                raise exception(f'{Colors.INFO}{string}{Colors.RESET}') from chain_err
