
import tensorflow as tf
import numpy as np

class LSTM_CNN_Model(object):



    def __init__(self,config,is_training=True):

        self.keep_prob=config.keep_prob
        self.batch_size = 64

        num_step=config.num_step
        self.input_data=tf.placeholder(tf.int32,[None,num_step])
        self.target = tf.placeholder(tf.int64,[None])
        self.mask_x = tf.placeholder(tf.float32,[num_step,None])

        class_num=config.class_num
        hidden_neural_size=config.hidden_neural_size
        vocabulary_size=config.vocabulary_size
        embed_dim=config.embed_dim
        hidden_layer_num=config.hidden_layer_num

        #build LSTM network

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size,forget_bias=0.0,state_is_tuple=True)
        if self.keep_prob<1:
            lstm_cell =  tf.contrib.rnn.DropoutWrapper(
                lstm_cell,output_keep_prob=self.keep_prob
            )

        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*hidden_layer_num,state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size,tf.float32)

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            embedding = tf.get_variable("embedding",[vocabulary_size,embed_dim],dtype=tf.float32)
            inputs=tf.nn.embedding_lookup(embedding,self.input_data)

        if self.keep_prob<1:
            inputs = tf.nn.dropout(inputs,self.keep_prob)

        out_put=[]
        state=self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(num_step):
                if time_step>0: tf.get_variable_scope().reuse_variables()
                (cell_output,state)=cell(inputs[:,time_step,:],state)
                out_put.append(cell_output)
        out_put=out_put*self.mask_x[:,:,None]

        with tf.name_scope("Conv_layer"):
            out_put = tf.transpose(out_put,[1,2,0])
            out_put = tf.reshape(out_put , [self.batch_size,hidden_neural_size,num_step,-1])

            print(out_put)
            W_conv = tf.get_variable(name="conv_w" , initializer=tf.truncated_normal(shape=[600,5,1,200],stddev=0.1))
            B_conv = tf.get_variable(name="conv_b", initializer=tf.constant(0.1,shape=[200]))

            conv_output = tf.nn.relu(tf.nn.conv2d(out_put , W_conv , strides=[1,1,1,1],padding='VALID') + B_conv)
            conv_output = tf.reshape(conv_output,[self.batch_size,36,200,1])
            max_pool_out = tf.nn.max_pool(conv_output,ksize=[1,36,1,1],strides=[1,1,1,1],padding='VALID')
            max_pool_out = tf.reshape(max_pool_out,[self.batch_size,200])


        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w",[200,class_num],dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b",[class_num],dtype=tf.float32)
            self.logits = tf.matmul(max_pool_out,softmax_w)+softmax_b

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits+1e-10,labels=self.target)
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(self.prediction,self.target)
            self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

        #add summary
        loss_summary = tf.summary.scalar("loss",self.cost)
        #add summary
        accuracy_summary=tf.summary.scalar("accuracy_summary",self.accuracy)

        if not is_training:
            return

        self.globle_step = tf.Variable(tf.constant(0),dtype=tf.int32,name="globle_step",trainable=False)
        self.lr = tf.Variable(tf.constant(0.8),dtype=tf.float32,trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                      config.max_grad_norm)


        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(grad_summaries)

        self.summary =tf.summary.merge([loss_summary,accuracy_summary,self.grad_summaries_merged])



        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        self.train_op=optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr,self.new_lr)

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})




















