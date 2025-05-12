####################
### Start Checks ###
####################
import os, sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def print_seperator(error = False, linelength=150):
    """
    prints a seperator for better seperation
    """
    if error:
        eprint("_" * linelength)
    else:
        print("_" * linelength)

if sys.version_info.major != 3 or sys.version_info.minor < 7:
    eprint("This framework requires Python version 3.7 or higher")
    sys.exit(1)

#######################
### Check Debugging ###
#######################
try:
    import pydevd
    DEBUGGING = True
except ImportError:
    DEBUGGING = False

############################
### Load Parameter Flags ###
############################
from utils.flags import flags
FLAGS = flags.FLAGS

def handleError(func, path, exc_info):
    import stat
    eprint('Handling Error for file ', path)
    eprint(exc_info)
    # Check if file access issue
    if not os.access(path, os.W_OK):
        # Try to change the permision of file
        os.chmod(path, stat.S_IWUSR)
        # call the calling function again
        func(path)

