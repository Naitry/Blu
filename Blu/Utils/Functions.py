import inspect


def getCurrentFunctionName() -> str:
    return inspect.currentframe().f_back.f_code.co_name
