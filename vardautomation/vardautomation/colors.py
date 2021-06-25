"""Colors module"""
import colorama

colorama.init()


class Colors:
    FAIL: str = f'{colorama.Back.RED}{colorama.Fore.BLACK}'
    INFO: str = f'{colorama.Back.BLUE}{colorama.Fore.WHITE}{colorama.Style.BRIGHT}'
    RESET: str = colorama.Style.RESET_ALL
