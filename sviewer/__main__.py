# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:36:40 2016

@author: Serj
"""
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__))[:-16])
#sys.path.append('C:/science/python')
#sys.path.append('/media/serj/3078FE3678FDFB04/science/python')
import spectro.sviewer.sviewer as sv

from PyQt5.QtWidgets import (QApplication)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = sv.sviewer()
    sys.exit(app.exec_())