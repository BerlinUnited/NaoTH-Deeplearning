"""
    This should be a simplified version of patch tools.

    First version should output patch data in non-ball and ball folders
"""
import cppyy.ll
from PatchStuff import PatchExecutor

if __name__ == "__main__":
    evaluator = PatchExecutor()

    with cppyy.ll.signals_as_exception():
        pass