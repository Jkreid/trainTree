# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:15:45 2020

@author: justi
"""
import util
import tensorflow as tf

def train(function):
    def train_function(tfg, *args,
                       src_branch='main',
                       dest_branch='main',
                       **kwargs):
        
        session = tfg._load(src_branch)
        value   = function(tfg, session, *args, **kwargs)
        tfg._save(session, dest_branch)
        return value
    return train_function


def run(function):
    def run_function(tfg, *args,
                     branch='main',
                     **kwargs):
        
        with tfg._load(branch) as session:
            value = function(tfg, session, *args, **kwargs)
        return value
    return run_function
    

def tfg(cls):
    
    class TensorFlowGraph(cls):
        
        def __init__(self, name, *args,
                     ckptdir='../../data',
                     branch='main',
                     **kwargs):
            
            tf.reset_default_graph()
            self.graph         = tf.Graph()
            self.name          = name
            self.ckptdir       = ckptdir
            with tf.variable_scope(name) and self.graph.as_default():
                value = cls.__init__(self, *args, **kwargs)
                self._initialize(branch)
            return value
                
    
        def get_ckpt(self, branch='main'):
            return self.ckptdir+f'/{self.name}-{branch}.ckpt'
        
        def _clr_ckpt(self, branch='main'):
            util.clear_ckpt(f'{self.name}-{branch}')
        
        def _reset(self, *args, branch='main', **kwargs):
            self._clr_ckpt(branch)
            self.__init__(
                self.name,
                *args,
                ckptdir=self.ckptdir,
                branch=branch
                **kwargs
            )
            
        def _initialize(self, branch='main'):
            self.saver = tf.train.Saver(tf.global_variables())
            with tf.Session(graph=self.graph) as session:
                session.run(tf.global_variables_initializer())
                self.saver.save(
                    sess=session,
                    save_path=self.get_ckpt(branch=branch)
                )
        
        def _save(self, session, branch='main'):
            self.saver.save(
                sess=session,
                save_path=self.get_ckpt(branch=branch)
            )
            session.close()
        
        def _load(self, branch='main'):
            tf.reset_default_graph()
            session = tf.Session(graph=self.graph)
            self.saver.restore(
                sess=session,
                save_path=self.get_ckpt(branch=branch)
            )
            return session
    
    return TensorFlowGraph

    