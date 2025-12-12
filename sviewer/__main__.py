# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:36:40 2016

@author: Serj
"""
#from PyQt6.QtGui import QScreen, QIcon
from PyQt6.QtWidgets import (QApplication)

import sys, os
sys.path.append(os.sep.join(os.path.normpath(os.path.realpath(__file__)).split(os.sep)[:-3]))
import spectro.sviewer.sviewer as sv


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ex = sv.sviewer()

    sys.exit(app.exec())