# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:17:12 2020

@author: justi
"""

import os

def clear_all_ckpt(ckptdir = './nn_models'):
    for f in os.listdir(ckptdir):
        os.remove(ckptdir+'/{}'.format(f))
        
def clear_saved(ckptdir = './nn_models'):
    if os.path.exists(ckptdir):
        clear_all_ckpt()
        os.rmdir(ckptdir)

def clear_ckpt(name, ckptdir = './nn_models'):
    for f in os.listdir(ckptdir):
        if f[:len(name)] == name: 
            os.remove(ckptdir+'/{}'.format(f))