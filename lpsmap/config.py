import os

import lpsmap

def get_include():
    d = os.path.join(os.path.dirname(lpsmap.__file__), 'core', 'include')
    return d


def get_libdir():
    d = os.path.join(os.path.dirname(lpsmap.__file__), 'core', 'lib')
    return d
