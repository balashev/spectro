# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:36:40 2016

@author: Serj
"""
from PyQt6.QtWidgets import (QApplication)


import sys
from . import sviewer as sv

def main():
    app = QApplication(sys.argv)
    window = sv.sviewer()
    return app.exec()

if __name__ == '__main__':
    raise SystemExit(main())
