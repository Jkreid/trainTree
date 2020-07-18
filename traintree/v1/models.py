# -*- coding: utf-8 -*-
"""
@author: Justin Reid
"""

""" Neural Network Classes and Functions """

import graph
import random
import statistics
import numpy      as np
import tensorflow as tf


#==============================================================================
# Basic Data Management functions
#==============================================================================

class OneHot:
    
    CHARS = ["'"]+list('abcdefghijklmnopqrstuvwxyz "1234567890,.+-=*&!@#$~`;\
                       \:^()<>/\|[]{}?_ABCDEFGHIJKLMNOPQRSTUVWXYZ%')
    #row oriented one-hot vectors
    def __init__(self, key=None):
        if key == None: 
            self.key = self.CHARS
        else: 
            self.key = list(key)
        self.veclen = len(self.key)
   
    def make_key(self,data_list,give_key=True): 
        self.key = sorted(list(set(data_list)))
        self.veclen = len(self.key)
        if give_key: 
            return self.key
    
    def encode(self,strg):
        """ encode entire string of characters into one-hot matrix """
        ohmat = np.zeros((len(strg),self.veclen))
        for i in range(len(strg)):
            ohmat[i,self.key.index(strg[i])] = 1
        return ohmat
    
    def decode(self, ohmat):
        dec_list = []
        for i in range(ohmat.size[0]):
            dec_list.append(self.key[np.where(ohmat[i]==1.0)])
        return dec_list

def flip(probs): 
    return (probs > np.random.random_sample(probs.shape))*1.0

#==============================================================================
# Basic Neural Network Classes (Classification & Regression)
#==============================================================================

@graph.tfg
class Neural_Network:
    
    def __init__(self, 
                 input_size    = 1, 
                 layer_sizes   = None, 
                 output_size   = 1, 
                 learning_rate = 0.01, 
                 in_act_f      = tf.identity, 
                 lay_act_f     = None, 
                 out_act_f     = tf.identity, 
                 loss_f        = tf.losses.mean_squared_error, 
                 optimizer     = tf.train.AdamOptimizer):
        
        self.in_act_f      = in_act_f
        self.lay_act_f     = lay_act_f
        self.out_act_f     = out_act_f
        self.loss_f        = loss_f
        self.input_size    = input_size, 
        self.layer_sizes   = layer_sizes, 
        self.output_size   = output_size, 
        self.learning_rate = learning_rate, 
        self.optimizer     = optimizer
        
        self.x = in_act_f(
            tf.placeholder(
                tf.float32, 
                shape=(None,self.input_size)
            )
        )
        if self.layer_sizes:
            self.layers = [tf.layers.dense(self.x, self.layer_sizes[0])]
            for size in self.layer_sizes[1:]:
                self.layers.append(
                    tf.layers.dense(
                        self.layers[-1], 
                        size, 
                        activation=lay_act_f
                    )
                )
            
            self.output_no_act = tf.layers.dense(
                self.layers[-1], 
                self.output_size
            )
        else: 
            self.output_no_act = tf.layers.dense(self.x, self.output_size)
        self.output = self.out_act_f(self.output_no_act)
        # training
        self.y = tf.placeholder(
            tf.float32, 
            shape=(None,self.output_size)
        )
        self.loss = self.loss_f(
            self.y, self.output, 
            reduction=tf.losses.Reduction.MEAN
        )
        self.train_op = self.optimizer.minimize(self.loss)
    
    
    def reset(self, branch='main'):
        return self._reset(
            name          = self.name,
            ckptdir       = self.ckptdir,
            input_size    = self.input_size, 
            layer_sizes   = self.layer_sizes, 
            output_size   = self.output_size, 
            learning_rate = self.learning_rate, 
            in_act_f      = self.in_act_f, 
            lay_act_f     = self.lay_act_f, 
            out_act_f     = self.out_act_f, 
            loss_f        = self.loss_f, 
            optimizer     = self.optimizer,
            branch        = branch
        )
        
    
    @graph.train
    def train(self,
              sess,
              x_data, 
              y_data, 
              n_train_epochs = 500, 
              batch_size     = 1, 
              loss           = False):
        
        xdata  = x_data.reshape(x_data.shape[0],-1)
        ydata  = y_data.reshape(y_data.shape[0],-1)
        losses = []
        for i in range(n_train_epochs):
            batch_ids = random.sample(range(xdata.shape[0]-1), batch_size)
            x_batch, y_batch = xdata[batch_ids,:], ydata[batch_ids,:]
            loss_vals, _ = sess.run(
                [self.loss, self.train_op], 
                feed_dict={
                    self.x: x_batch, 
                    self.y: y_batch
                }
            )
            losses.append(loss_vals)
        return losses if loss else None
    
    
    @graph.run
    def get_nn_output(self, sess, xs):
        return sess.run(
            self.output, 
            feed_dict={self.x:xs.reshape(xs.shape[0],-1)}
        )

# Neural Network Subclasses

class Regression(Neural_Network): 
    
    def __init__(self, 
                 name, 
                 input_size    = 1, 
                 output_size   = 1, 
                 layer_sizes   = None, 
                 learning_rate = 0.01, 
                 in_act_f      = tf.identity, 
                 lay_act_f     = None, 
                 optimizer     = tf.train.AdamOptimizer,
                 branch        = 'main',
                 **kwargs):
        
        super().__init__(
            name          = name, 
            input_size    = input_size, 
            layer_sizes   = layer_sizes, 
            output_size   = output_size, 
            learning_rate = learning_rate, 
            in_act_f      = in_act_f, 
            lay_act_f     = lay_act_f, 
            out_act_f     = tf.identity, 
            loss_f        = tf.losses.mean_squared_error, 
            optimizer     = optimizer,
            branch        = branch,
            **kwargs
        )
    
    def predict(self, xs, branch='main'):
        return self.get_nn_output(xs, branch=branch)


class Single_Classifier(Neural_Network):
    
    def __init__(self, 
                 name, 
                 input_size, 
                 output_size, 
                 layer_sizes   = None, 
                 learning_rate = 0.01, 
                 lay_act_f     = None, 
                 optimizer     = tf.train.AdamOptimizer,
                 branch        = 'main',
                 **kwargs):
        
        super().__init__(
            name          = name, 
            input_size    = input_size, 
            layer_sizes   = layer_sizes, 
            output_size   = output_size, 
            learning_rate = learning_rate, 
            lay_act_f     = lay_act_f, 
            out_act_f     = tf.nn.softmax, 
            loss_f        = tf.losses.softmax_cross_entropy, 
            optimizer     = optimizer,
            branch        = branch,
            **kwargs
        )
    
    def predict(self, xs, branch='main'):
        ys = self.get_nn_output(xs, branch=branch)
        choices = np.zeros(ys.shape)
        for i in range(ys.shape[0]): 
            choices[i,np.random.choice(self.output_size,p=ys[i])] = 1.0
        return choices


class Multi_Classifier(Neural_Network):
    
    def __init__(self, 
                 name, 
                 input_size, 
                 output_size, 
                 layer_sizes   = None, 
                 learning_rate = 0.01, 
                 lay_act_f     = None, 
                 optimizer     = tf.train.AdamOptimizer,
                 branch        = 'main',
                 **kwargs):
        
        super().__init__(
            name          = name, 
            input_size    = input_size, 
            layer_sizes   = layer_sizes, 
            output_size   = output_size, 
            learning_rate = learning_rate, 
            lay_act_f     = lay_act_f, 
            out_act_f     = tf.nn.sigmoid, 
            loss_f        = tf.losses.sigmoid_cross_entropy, 
            optimizer     = optimizer,
            branch        = branch,
            **kwargs
        )

    def predict(self, xs, branch='main'): 
        return flip(self.get_nn_output(xs, branch=branch))


#==============================================================================
# Auto Encocder Class
#==============================================================================

@graph.tfg
class AutoEncoder:
    
    def __init__(self, 
                 input_size, 
                 code_size, 
                 hidden_sizes  = None, 
                 learning_rate = 0.01,
                 optimizer     = tf.train.AdamOptimizer):
        
        self.input_size    = input_size
        self.output_size   = code_size
        self.layer_sizes   = hidden_sizes
        self.learning_rate = learning_rate
        self.optimizer     = optimizer
        
        self.x = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_size)
        )
        self.code_x = tf.placeholder(
            tf.float32, 
            shape=(None, self.output_size)
        )
        self.output_layer = tf.layers.Dense(
            units=self.input_size, 
            activation=tf.nn.relu
        )
        if self.layer_sizes:
            self.encode_layers = [
                tf.layers.dense(
                    self.encode_layers[-1], 
                    self.layer_sizes[0], 
                    activation=tf.nn.relu
                )
            ]
            for size in self.layer_sizes[1:]:
                self.encode_layers.append(
                    tf.layers.dense(
                        self.encode_layers[-1], 
                        size, 
                        activation=tf.nn.relu
                    )
                )
            self.code = tf.layers.dense(
                self.encode_layers[-1], 
                self.output_size, 
                activation=tf.nn.relu
            )
            
            self.decode_layers = [
                tf.layers.Dense(
                    units=i, 
                    activation=tf.nn.relu
                ) for i in reversed(self.layer_sizes)
            ]
            #straight through (training)
            self.decode_outputs_train = [self.decode_layers[0](self.code)]
            for layer in self.decode_layers[1:]:
                self.decode_outputs_train.append(
                    layer(self.decode_outputs_train[-1])
                )
            self.approx_train = self.ouput_layer(self.decode_outputs_train[-1])
            #from encoding
            self.decode_outputs = [self.decode_layers[0](self.code_x)]
            for layer in self.decode_layers[1:]:
                self.decode_outputs.append(
                    layer(self.decode_outputs[-1])
                )
            self.approx = self.output_layer(self.decode_outputs[-1])
        
        else:
            self.code = tf.layers.dense(
                self.x, 
                self.output_size, 
                activation=tf.nn.relu
            )
            #straight through (training)
            self.appox_train = self.output_layer(self.code)
            #from encoding
            self.approx = self.output_layer(self.code_x)

        #training
        self.loss = tf.losses.mean_squared_error(
            self.x, 
            self.approx_train, 
            reduction=tf.losses.Reduction.MEAN
        )
        self.train_op = self.optimizer.minimize(self.loss)

    
    def reset(self, branch='main'):
        return self._reset(
            name = self.name,
            input_size    = self.input_size, 
            code_size     = self.output_size, 
            hidden_sizes  = self.layer_sizes, 
            learning_rate = self.learning_rate, 
            branch        = branch
        )
        
    
    @graph.train
    def train(self,
              sess,
              xs,
              n_train_epochs = 500,
              batch_size     = 1,
              loss           = False):
        
        batch_size = min(batch_size, xs.shape[0])
        xs=xs.reshape(xs.shape[0],-1)
        losses=[]
        for i in range(n_train_epochs):
            batch_ids = random.sample(range(xs.shape[0]-1), batch_size)
            x_batch = xs[batch_ids,:]
            loss_vals, _ = sess.run([self.loss, self.train_op], 
                                    feed_dict={self.x:x_batch}
                                    )
            losses.append(loss_vals)
        return losses if loss else None
    
    @graph.run
    def encode(self, sess, xs):
        return sess.run(
            self.code, 
            feed_dict={self.x:xs.reshape(xs.shape[0],-1)}
        )
    
    def decode(self, sess, code):
        return sess.run(
            self.approx, 
            feed_dict={self.code_x:code.reshape(code.shape[0],-1)}
        )
    
    def approximate(self, sess, xs):
        return sess.run(
            self.approx_train, 
            feed_dict={self.code_x:xs.reshape(xs.shape[0],-1)}
        )

#==============================================================================
# RNN & LSTM Classes
#==============================================================================

@graph.tfg
class RNN:
    
    def __init__(self, 
                 input_size, 
                 layer_sizes, 
                 output_size, 
                 act_n_loss, 
                 learning_rate = 0.003, 
                 cell          = tf.nn.rnn_cell.LSTMCell,
                 optimizer     = tf.train.RMSPropOptimizer,
                 **kwargs):
        
        self.input_size    = input_size
        self.layer_sizes   = layer_sizes
        self.output_size   = output_size
        self.learning_rate = learning_rate
        self.cell          = cell
        self.optimizer     = optimizer
        self.act_n_loss    = act_n_loss
        self.state         = None
        
        if act_n_loss == 'sftmx': 
            self.act_out = tf.nn.softmax, 
            self.loss_f  = tf.losses.softmax_cross_entropy
        elif act_n_loss == 'sigmd': 
            self.act_out = tf.nn.sigmoid, 
            self.loss_f  = tf.losses.sigmoid_cross_entropy
        elif act_n_loss == 'reg': 
            self.act_out = tf.identity, 
            self.loss_f  = tf.losses.mean_squared_error
        else: 
            self.act_out, self.loss_f = act_n_loss 
            #ex. act_n_loss = (tf.nn.relu, tf.losses.mean_squared_error)
        
        self.x_input = tf.placeholder(
            tf.float32,
            shape=(None, None, self.input_size)
        )
        self.cells = tf.nn.rnn_cell.MultiRNNCell(
            [cell(layer_size) for layer_size in layer_sizes]
        )
        outputs, self.new_state = tf.nn.dynamic_rnn(
            self.cells, 
            self.x_input, 
            dtype=tf.float32, 
            initial_state=self.state
        )
        network_output = tf.layers.dense(outputs, units=self.output_size)
        self.output = self.act_out(network_output)
        # training
        self.y_batch = tf.placeholder(tf.float32,network_output.shape)
        self.loss = self.loss_f(
            self.y_batch, 
            network_output, 
            reduction=tf.losses.Reduction.MEAN
        )
        self.train_op  = self.optimizer.minimize(self.loss)

    
    def reset(self, branch='main'): 
        """ Reset Network to pre-training conditions """
        return self._reset(
            name          = self.name, 
            input_size    = self.input_size, 
            layer_sizes   = self.layer_sizes, 
            output_size   = self.output_size, 
            act_n_loss    = self.act_n_loss, 
            learning_rate = self.learning_rate, 
            cell          = self.cell,
            optimizer     = self.optimizer,
            branch        = branch
        )
        
    @graph.train   
    def seq_lrn_rout(self,
                     sess,
                     data,
                     n_train_epochs = 100,
                     batch_size     = 64,
                     time_steps     = 100,
                     loss           = False):
        """ Train Network on Given Sequences of Data with size >= batch_size """
        data    = data.reshape(data.shape[0],self.input_size)
        x_batch = np.zeros((batch_size, time_steps, self.input_size))
        y_batch = np.zeros((batch_size, time_steps, self.input_size))
        losses  = []
        for i in range(n_train_epochs):
            batch_ids = random.sample(
                range(data.shape[0] - time_steps - 1),
                batch_size
            )
            for j in range(time_steps):
                xinds            = [k + j for k in batch_ids]
                yinds            = [k + j + 1 for k in batch_ids]
                x_batch[:, j, :] = data[xinds, :]
                y_batch[:, j, :] = data[yinds, :]
            self.state = self.cells.zero_state(batch_size,tf.float32)
            loss_val, _ = sess.run(
                [self.loss, self.train_op], 
                feed_dict={
                    self.x_input: x_batch, 
                    self.y_batch: y_batch
                }
            )
            losses.append(loss_val)
        return losses if loss else None
    
    @graph.train
    def xinput_lrn_rout(self,
                        sess,
                        xdata, 
                        ydata, 
                        n_train_epochs = 100, 
                        batch_size     = 64, 
                        time_steps     = 100, 
                        loss           = False):
        
        x_batch = np.zeros((batch_size, time_steps, self.input_size))
        y_batch = np.zeros((batch_size, time_steps, self.output_size))
        losses  = []
        for i in range(n_train_epochs):
            batch_ids = random.sample(
                range(xdata.shape[0] - time_steps - 1),
                batch_size
            )
            for j in range(time_steps):
                ind              = [k + j for k in batch_ids]
                x_batch[:, j, :] = xdata[ind, :]
                y_batch[:, j, :] = ydata[ind, :]
            self.state  = self.cells.zero_state(batch_size,tf.float32)
            loss_val, _ = sess.run(
                [self.loss, self.train_op], 
                feed_dict={
                    self.x_input: x_batch, 
                    self.y_batch: y_batch
                }
            )
            losses.append(loss_val)
        return losses if loss else None
    
# RNN Subclasses

class Symbolic_output_to_input_LSTM(RNN):
    
    def __init__(self, 
                 name, 
                 key           = None, 
                 layer_sizes   = [128], 
                 learning_rate = 0.003,
                 branch        = 'main',
                 **kwargs):
        
        self.oh = OneHot(key)
        self.x_init = None
        super().__init__(
            name          = name,
            input_size    = len(self.oh.key),
            layer_sizes   = layer_sizes,
            output_size   = len(self.oh.key),
            act_n_loss    = 'sftmx',
            learning_rate = learning_rate,
            branch        = branch,
            **kwargs
        )
  
    
    def train(self, 
              data, 
              n_train_epochs = 100,
              batch_size     = 64,
              time_steps     = 100,
              loss           = False,
              src_branch     = 'main',
              dest_branch    = 'main'):
        
        self.x_init = statistics.mode(data)
        return self.seq_lrn_rout(
            self.oh.enca(data), 
            n_train_epochs = n_train_epochs,
            batch_size     = batch_size,
            time_steps     = time_steps,
            loss           = loss,
            src_branch     = src_branch,
            dest_branch    = dest_branch
        )
    
    
    @graph.run
    def generate(self,
                 sess,
                 length, 
                 seed_symb = None, 
                 rand      = False):
        """ Generate fixed length sequence """
        gen_seq = [
            seed_symb or
            np.random.choice(self.oh.key) if (rand or not self.x_init)
            else self.x_init
        ]
        self.state = self.cells.zero_state(1,tf.float32)
        for i in range(length-1):
            output_probs, self.state = sess.run(
                [self.output, self.new_state], 
                feed_dict={self.x_input:[self.oh.enc(gen_seq[-1])]}
            )
            gen_seq.append(
                np.random.choice(
                    self.oh.key,
                    p=output_probs[0,0]
                )
            )        
        return gen_seq


class Vector_output_to_input_LSTM(RNN):
    
    def __init__(self, 
                 name, 
                 vec_size, 
                 layer_sizes   = [128], 
                 learning_rate = 0.003,
                 branch        = 'main',
                 **kwargs):
        
        super().__init__(
            name          = name, 
            input_size    = vec_size, 
            layer_sizes   = layer_sizes, 
            output_size   = vec_size, 
            act_n_loss    = 'sigmd', 
            learning_rate = learning_rate,
            branch        = branch,
            **kwargs
        )
        
    def train(self, 
              data, 
              n_train_epochs = 100, 
              batch_size     = 64, 
              time_steps     = 100,
              loss           = False,
              src_branch     = 'main',
              dest_branch    = 'main'):
        
        """ Train Network on Given Sequences of Data with size >= batch_size """
        return self.seq_lrn_rout(
            data,
            n_train_epochs = n_train_epochs,
            batch_size     = batch_size,
            time_steps     = time_steps,
            loss           = loss,
            src_branch     = src_branch,
            dest_branch    = dest_branch
        )
    
    @graph.run
    def generate(self,
                 sess,
                 length,
                 seed_vec=None):
        """ Generate fixed length sequence """
        gen_seq = np.zeros((length, self.input_size))
        if seed_vec: 
            gen_seq[0,:] = seed_vec
        self.state = self.cells.zero_state(1,tf.float32)
        for i in range(1,length):
            output_probs, self.state = sess.run(
                [self.output, self.new_state], 
                feed_dict={
                    self.x_input:(
                        gen_seq[i-1].reshape((1,1,self.output_size))
                    )
                }
            )
            gen_seq[i,:] = flip(output_probs[0,0])
        return gen_seq


class Series_output_to_input_LSTM(RNN):
    
    def __init__(self, 
                 name, 
                 vec_size      = 1, 
                 layer_sizes   = [128], 
                 learning_rate = 0.003,
                 branch        = 'main',
                 **kwargs):
        
        self.x_init = np.zeros((1,vec_size))
        super().__init__(
            name          = name, 
            input_size    = vec_size, 
            layer_sizes   = layer_sizes, 
            output_size   = vec_size, 
            act_n_loss    = 'reg', 
            learning_rate = learning_rate,
            branch        = branch,
            **kwargs
        )
    
    def train(self, 
              data, 
              n_train_epochs = 100,
              batch_size     = 64,
              time_steps     = 100,
              stable         = False,
              loss           = False,
              src_branch     = 'main',
              dest_branch    = 'main'):
        """ Train Network on Given Sequences of Data with size >= batch_size """
        self.x_init = np.mean(data, axis=0) if stable else data[0,:]
        return self.seq_lrn_rout(
            data, 
            n_train_epochs = n_train_epochs,
            batch_size     = batch_size,
            time_steps     = time_steps,
            loss           = loss,
            src_branch     = src_branch,
            dest_branch    = dest_branch
        )
    
    @graph.run
    def generate(self,
                 sess,
                 length,
                 seed_val=None):
        """ Generate fixed length sequence """
        gen_seq = np.zeros((length, self.input_size))
        gen_seq[0,:] = seed_val or self.x_init 
        self.state = self.cells.zero_state(1,tf.float32)
        for i in range(1,length):
            output, self.state = sess.run(
                [self.output, self.new_state], 
                feed_dict={
                    self.x_input:(
                        gen_seq[i-1].reshape((1,1,self.output_size))
                    )
                }
            )
            gen_seq[i,:] = output[0,0]
        return gen_seq


class Xinput_SingClass_LSTM(RNN):
    
    def __init__(self, 
                 name, 
                 input_size, 
                 output_size, 
                 layer_sizes   = [128], 
                 learning_rate = 0.003,
                 branch        = 'main',
                 **kwargs):
        
        super().__init__(
            name          = name, 
            input_size    = input_size, 
            layer_sizes   = layer_sizes, 
            output_size   = output_size, 
            act_n_loss    = 'sftmx', 
            learning_rate = learning_rate,
            branch        = branch,
            **kwargs
        )
        
    def train(self, 
              xdata, 
              ydata, 
              n_train_epochs = 100,
              batch_size     = 64,
              time_steps     = 100,
              loss           = False,
              src_branch     = 'main',
              dest_branch    = 'main'):
        
        return self.xinput_lrn_rout(
            xdata          = xdata,
            ydata          = ydata,
            n_train_epochs = n_train_epochs,
            batch_size     = batch_size, 
            time_steps     = time_steps,
            loss           = loss,
            src_branch     = src_branch,
            dest_branch    = dest_branch
        )
    
    @graph.run
    def predict(self, sess, xs):
        self.state = self.cells.zero_state(1,tf.float32)
        predictions = np.zeros((xs.shape[0], self.output_size))
        for i in range(xs.shape[0]):
            output_probs, self.state = sess.run(
                [self.output, self.new_state], 
                feed_dict={self.x_input:(xs[i].reshape((1,1,xs.shape[1])))}
            )
            predictions[
                i, np.random.choice(self.output_size, p=output_probs[0,0])
            ] = 1
        return predictions


class Xinput_MultClass_LSTM(RNN):
    
    def __init__(self, 
                 name, 
                 input_size, 
                 output_size, 
                 layer_sizes   = [128], 
                 learning_rate = 0.003,
                 branch        = 'main',
                 **kwargs):
        
        super().__init__(
            name          = name, 
            input_size    = input_size, 
            layer_sizes   = layer_sizes, 
            output_size   = output_size, 
            act_n_loss    = 'sigmd', 
            learning_rate = learning_rate,
            branch        = branch,
            **kwargs
        )
        
    def train(self, 
              xdata, 
              ydata, 
              n_train_epochs = 100,
              batch_size     = 64,
              time_steps     = 100,
              loss           = False,
              src_branch     = 'main',
              dest_branch    = 'main'):
        
        return self.xinput_lrn_rout(
            xdata          = xdata,
            ydata          = ydata,
            n_train_epochs = n_train_epochs,
            batch_size     = batch_size,
            time_steps     = time_steps,
            loss           = loss,
            src_branch     = src_branch,
            dest_branch    = dest_branch
        )
    
    @graph.run
    def predict(self, sess, xs):
        self.state = self.cells.zero_state(1,tf.float32)
        predictions = np.zeros((xs.shape[0], self.output_size))
        for i in range(xs.shape[0]):
            output_probs, self.state = sess.run(
                [self.output, self.new_state], 
                feed_dict={self.x_input:(xs[i].reshape((1,1,xs.shape[1])))}
            )
            predictions[i,:] = flip(output_probs[0,0])
        return predictions
        

class Xinput_Reg_LSTM(RNN):
    
    def __init__(self, 
                 name, 
                 input_size, 
                 output_size, 
                 layer_sizes   = [128], 
                 learning_rate = 0.003,
                 branch        = 'main',
                 **kwargs):
        
        super().__init__(
            name          = name, 
            input_size    = input_size, 
            layer_sizes   = layer_sizes, 
            output_size   = output_size, 
            act_n_loss    = 'reg', 
            learning_rate = learning_rate,
            branch        = branch,
            **kwargs
        )
        
    def train(self, 
              xdata, 
              ydata, 
              n_train_epochs = 100, 
              batch_size     = 64,
              time_steps     = 100,
              loss           = False,
              src_branch     = 'main',
              dest_branch    = 'main'):
        
        return self.xinput_lrn_rout(
            xdata          = xdata, 
            ydata          = ydata, 
            n_train_epochs = n_train_epochs,
            batch_size     = batch_size,
            time_steps     = time_steps,
            loss           = False,
            src_branch     = src_branch,
            dest_branch    = dest_branch
            )
    
    @graph.run
    def predict(self, sess, xs):
        self.state = self.cells.zero_state(1,tf.float32)
        predictions = np.zeros((xs.shape[0], self.output_size))
        for i in range(xs.shape[0]):
            output, self.state = sess.run(
                [self.output, self.new_state], 
                feed_dict={self.x_input:(xs[i].reshape((1,1,xs.shape[1])))}
            )
            predictions[i,:] = output[0,0]
        return predictions
    
# Bidirectional LSTM ### Under Construction ###

#==============================================================================
# Model Aliases
#==============================================================================
NN       = Neural_Network
Reg      = Regression
SingCls  = Single_Classifier
MultCls  = Multi_Classifier
AE       = AutoEncoder
LSTM     = RNN
Seq      = Symbolic_output_to_input_LSTM
VecSeq   = Vector_output_to_input_LSTM
NumSeq   = Series_output_to_input_LSTM
SingLstm = Xinput_SingClass_LSTM
MultLstm = Xinput_MultClass_LSTM
RegLstm  = Xinput_Reg_LSTM