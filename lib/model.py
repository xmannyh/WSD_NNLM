import numpy as np
import tensorflow as tf

class BasicLSTM():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 5

        # Apply correct f function as hidden layer output
        if args.func == 'relu':
            activ_func = tf.nn.relu
        elif args.func == 'sigmoid':
            activ_func = tf.nn.sigmoid
        else:
            activ_func = tf.nn.tanh 
        
        self.x = tf.placeholder(tf.int32, [None, args.seq_length-1], name='x')
        self.y = tf.placeholder(tf.int32, name='y')
        self.keep_prob = tf.placeholder(tf.float32)

        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(args.rnn_size)
            cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell_dropout
        
        self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell()] * args.num_layers)
        self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)

	
        with tf.name_scope('embed'):
            embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.x)
            inputs = tf.nn.dropout(inputs, self.keep_prob)
        
        outputs = []
        state = self.initial_state
        with tf.variable_scope('LSTM'):
            for time_step in range(args.seq_length-1):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        
        self.final_state = state
        # Output shape: [batch_size, rnn_size]
        output = outputs[-1]
        #print('output shape {}'.format(output.get_shape().as_list()))

        with tf.name_scope('context_layer'):
            output = activ_func(output)

            cont_w = tf.get_variable("cont_w", [args.rnn_size, args.wrd_embedding_size], \
             initializer=tf.glorot_uniform_initializer())
            cont_b = tf.get_variable("cont_b", [args.wrd_embedding_size])
            cont_layer = tf.get_variable("cont_layer", [args.batch_size, args.wrd_embedding_size])

            # Output shape: [batch_size, wrd_embedding_size]
            cont_layer = tf.matmul(output, cont_w) + cont_b
            self.cont_logits = cont_layer
            # print('cont_logits shape {}'.format(cont_layer.get_shape().as_list()))
            
            
            # testing
        with tf.name_scope('softmax'):
            proj_w = tf.get_variable("proj_w", [args.wrd_embedding_size, args.verb_size])
            proj_b = tf.get_variable("proj_b", [args.verb_size]) 
            
            softmax_w = proj_w
            
            # Output shape: [batch_size, verb_size]
            blah = tf.matmul(self.cont_logits, softmax_w, name="context_layer") + proj_b
            self.logits = blah
            #print('logits shape {}'.format(self.logits.get_shape().as_list()))
            # Output shape: [batch_size, verb_size]
            self.probs = tf.nn.softmax(self.logits)
            #print('probs shape {}'.format(self.probs.get_shape().as_list()))
            
            labels = self.y
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits, name='softmax_loss')
            self.cost = tf.reduce_sum(self.loss) / args.batch_size
            
        with tf.name_scope('accuracy'):
            # Output shape: [batch_size]
            output_words = tf.argmax(self.probs, axis=1)
            #print('output_words shape {}'.format(output_words.get_shape().as_list()))
            output_words = tf.cast(output_words, tf.int32)
            self.equal = tf.equal(output_words, tf.reshape(self.y, [-1]))

