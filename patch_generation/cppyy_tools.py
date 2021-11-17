import cppyy
import os
import sys


def get_naoth_dir():
    """
    TODO add readme
    TODO use pathlib
    """
    return os.path.abspath(os.environ["NAOTH_REPO"])


def get_toolchain_dir():
    """
    TODO add readme
    TODO use pathlib
    """
    toolchain_path = os.path.join(
        os.path.abspath(os.environ["NAO_CTC"]), "../")
    return toolchain_path


def setup_shared_lib(naoth_dir, toolchain_dir):
    """
    load shared lib and general includes
    """
    shared_lib_name = "libscriptsim.so"
    if sys.platform.startswith("win32"):
        shared_lib_name = "scriptsim.dll"
    elif sys.platform.startswith("darwin"):
        shared_lib_name = "libscriptsim.dylib"
    
    cppyy.load_library(os.path.join(
        naoth_dir, "NaoTHSoccer/dist/Native/" + shared_lib_name))

    # add relevant include paths to allow mapping our code
    cppyy.add_include_path(os.path.join(
        naoth_dir, "Framework/Commons/Source"))

    # add relevant include paths to allow mapping our code
    cppyy.add_include_path(os.path.join(
        naoth_dir, "Framework/Commons/Source"))
    cppyy.add_include_path(os.path.join(naoth_dir, "NaoTHSoccer/Source"))
    cppyy.add_include_path(os.path.join(
        toolchain_dir, "toolchain_native/extern/include"))
    cppyy.add_include_path(os.path.join(
        toolchain_dir, "toolchain_native/extern/include/glib-2.0"))
    cppyy.add_include_path(os.path.join(
        toolchain_dir, "toolchain_native/extern/lib/glib-2.0/include"))

