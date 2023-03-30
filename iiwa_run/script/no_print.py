import os
import sys


class NoPrint:
    def __init__(self, stdout=True, stderr=False):
        self.stdout = stdout
        self.stderr = stderr

    def __enter__(self):
        if self.stdout:
            sys.stdout = open(os.devnull, 'w')
        if self.stderr:
            sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stdout:
            sys.stdout = sys.__stdout__
        if self.stderr:
            sys.stderr = sys.__stderr__
