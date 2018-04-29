"""
# python -2.7 
# Format taken from stanford course project website
# Modified by: Chaikesh, Prakash and Sonu
# added many modules like Highway Maxout network adapted from Xiong et.el 2017
# added Modified DCN encoder
"""



"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _maybe_mask_score

class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


## added by me
def maybe_mask_affinity(affinity, sequence_length, affinity_mask_value=float('-inf')):
    """ Masks affinity along its third dimension with `affinity_mask_value`.
    Used for masking entries of sequences longer than `sequence_length` prior to 
    applying softmax.  
    Args:  
        affinity: Tensor of rank 3, shape [N, D or Q, Q or D] where attention logits are in the second dimension.  
        sequence_length: Tensor of rank 1, shape [N]. Lengths of second dimension of the affinity.  
        affinity_mask_value: (optional) Value to mask affinity with.  
    
    Returns:  
        Masked affinity, same shape as affinity.
    """
    if sequence_length is None:
        return affinity
    score_mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(affinity)[1])
    score_mask = tf.tile(tf.expand_dims(score_mask, 2), (1, 1, tf.shape(affinity)[2]))
    affinity_mask_values = affinity_mask_value * tf.ones_like(affinity)
    return tf.where(score_mask, affinity, affinity_mask_values)


def _maybe_mask_to_start(score, start, score_mask_value):
    score_mask = tf.sequence_mask(start, maxlen=tf.shape(score)[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(~score_mask, score, score_mask_values)


class RNNEncoder1(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder1"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class RNNEncoder2(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder2"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


def convert_gradient_to_tensor(x):
    """Identity operation whose gradient is converted to a `Tensor`.
    Currently, the gradient to `tf.concat` is particularly expensive to
    compute if dy is an `IndexedSlices` (a lack of GPU implementation
    forces the gradient operation onto CPU).  This situation occurs when
    the output of the `tf.concat` is eventually passed to `tf.gather`.
    It is sometimes faster to convert the gradient to a `Tensor`, so as
    to get the cheaper gradient for `tf.concat`.  To do this, replace
    `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.
    Args:
    x: A `Tensor`.
    Returns:
    The input `Tensor`.
    """
    return x

def start_and_end_encoding(encoding, answer):
    """ Gathers the encodings representing the start and end of the answer span passed
    and concatenates the encodings.
    Args:  
        encoding: Tensor of rank 3, shape [N, D, xH]. Query-document encoding.  
        answer: Tensor of rank 2. Answer span.  
    
    Returns:
        Tensor of rank 2 [N, 2xH], containing the encodings of the start and end of the answer span
    """
    batch_size = tf.shape(encoding)[0]
    start, end = answer[:, 0], answer[:, 1]
    encoding_start = tf.gather_nd(encoding, tf.stack([tf.range(batch_size), start], axis=1))  # May be causing UserWarning
    encoding_end = tf.gather_nd(encoding, tf.stack([tf.range(batch_size), end], axis=1))
    return convert_gradient_to_tensor(tf.concat([encoding_start, encoding_end], axis=1))



def dcn_decode(encoding, document_length, state_size=100, pool_size=4, max_iter=4, keep_prob=1.0):
    """ DCN+ Dynamic Decoder.
    Builds decoder graph that iterates over possible solutions to problem
    until it returns same answer in two consecutive iterations or reaches `max_iter` iterations.  
    Args:  
        encoding: Tensor of rank 3, shape [N, D, xH]. Query-document encoding.  
        state_size: Scalar integer. Size of state and highway network.  
        pool_size: Scalar integer. Number of units that are max pooled in maxout network.  
        max_iter: Scalar integer. Maximum number of attempts for answer span start and end to settle.  
        keep_prob: Scalar float. Probability of keeping units during dropout.
    Returns:  
        A tuple containing  
            TensorArray of answer span logits for each iteration.  
            TensorArray of logit masks for each iteration.
    """

    with tf.variable_scope('decoder_loop', reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(encoding)[0]
        lstm_dec = tf.contrib.rnn.LSTMCell(num_units=state_size)
        lstm_dec = tf.contrib.rnn.DropoutWrapper(lstm_dec, input_keep_prob=keep_prob)

        # initialise loop variables
        start = tf.zeros((batch_size,), dtype=tf.int32)
        end = document_length - 1
        answer = tf.stack([start, end], axis=1)
        state = lstm_dec.zero_state(batch_size, dtype=tf.float32)
        not_settled = tf.tile([True], (batch_size,))
        logits = tf.TensorArray(tf.float32, size=max_iter, clear_after_read=False)

        def calculate_not_settled_logits(not_settled, answer, output, prev_logit):
            enc_masked = tf.boolean_mask(encoding, not_settled)
            output_masked = tf.boolean_mask(output, not_settled)
            answer_masked = tf.boolean_mask(answer, not_settled)
            document_length_masked = tf.boolean_mask(document_length, not_settled)
            new_logit = decoder_body(enc_masked, output_masked, answer_masked, state_size, pool_size, document_length_masked, keep_prob)
            new_idx = tf.boolean_mask(tf.range(batch_size), not_settled)
            logit = tf.dynamic_stitch([tf.range(batch_size), new_idx], [prev_logit, new_logit])  # TODO test that correct
            return logit

        for i in range(max_iter):
            if i > 1:
                names = 'not_settles_iter_'+ str(i+1)
                #tf.summary.scalar(f'not_settled_iter_{i+1}', tf.reduce_sum(tf.cast(not_settled, tf.float32)))
                tf.summary.scalar(names, tf.reduce_sum(tf.cast(not_settled, tf.float32)))
            
            output, state = lstm_dec(start_and_end_encoding(encoding, answer), state)
            if i == 0:
                logit = decoder_body(encoding, output, answer, state_size, pool_size, document_length, keep_prob)
            else:
                prev_logit = logits.read(i-1)
                logit = tf.cond(
                    tf.reduce_any(not_settled),
                    lambda: calculate_not_settled_logits(not_settled, answer, output, prev_logit),
                    lambda: prev_logit
                )
            start_logit, end_logit = logit[:, :, 0], logit[:, :, 1]
            start = tf.argmax(start_logit, axis=1, output_type=tf.int32)
            end = tf.argmax(end_logit, axis=1, output_type=tf.int32)
            new_answer = tf.stack([start, end], axis=1)
            if i == 0:
                not_settled = tf.tile([True], (batch_size,))
            else:
                not_settled = tf.reduce_any(tf.not_equal(answer, new_answer), axis=1)
            not_settled = tf.reshape(not_settled, (batch_size,))  # needed to establish dimensions
            answer = new_answer
            logits = logits.write(i, logit)

    return logits

def decoder_body(encoding, state, answer, state_size, pool_size, document_length, keep_prob=1.0):
    """ Decoder feedforward network.  
    Calculates answer span start and end logits.  
    Args:  
        encoding: Tensor of rank 3, shape [N, D, xH]. Query-document encoding.  
        state: Tensor of rank 2, shape [N, D, C]. Current state of decoder state machine.  
        answer: Tensor of rank 2, shape [N, 2]. Current iteration's answer.  
        state_size: Scalar integer. Hidden units of highway maxout network.  
        pool_size: Scalar integer. Number of units that are max pooled in maxout network.  
        keep_prob: Scalar float. Input dropout keep probability for maxout layers.
    
    Returns:  
        Tensor of rank 3, shape [N, D, 2]. Answer span logits for answer start and end.
    """
    maxlen = tf.shape(encoding)[1]
    
    def highway_maxout_network(answer):
        span_encoding = start_and_end_encoding(encoding, answer)
        r_input = convert_gradient_to_tensor(tf.concat([state, span_encoding], axis=1))
        r_input = tf.nn.dropout(r_input, keep_prob)
        r = tf.layers.dense(r_input, state_size, use_bias=False, activation=tf.tanh)
        r = tf.expand_dims(r, 1)
        r = tf.tile(r, (1, maxlen, 1))
        highway_input = convert_gradient_to_tensor(tf.concat([encoding, r], 2))
        logit = highway_maxout(highway_input, state_size, pool_size, keep_prob)
        #alpha = two_layer_mlp(highway_input, state_size, keep_prob=keep_prob)
        logit = _maybe_mask_score(logit, document_length, -1e30)
        return logit

    with tf.variable_scope('start'):
        alpha = highway_maxout_network(answer)

    with tf.variable_scope('end'):
        updated_start = tf.argmax(alpha, axis=1, output_type=tf.int32)
        updated_answer = tf.stack([updated_start, answer[:, 1]], axis=1)
        beta = highway_maxout_network(updated_answer)
    
    return tf.stack([alpha, beta], axis=2)


def two_layer_mlp(inputs, hidden_size, keep_prob=1.0):
    """ Two layer MLP network.
    Args:  
        inputs: Tensor of rank 3, shape [N, D, ?]. Inputs to network.  
        hidden_size: Scalar integer. Hidden units of network.  
        keep_prob: Scalar float. Input dropout keep probability.  
    Returns:  
        Tensor of rank 2, shape [N, D]. Logits.
    """
    
    inputs = tf.nn.dropout(inputs, keep_prob)
    layer1 = tf.layers.dense(inputs, hidden_size, activation=tf.nn.relu)
    output = tf.layers.dense(layer1, 1)
    output = tf.squeeze(output, -1)
    return output


def highway_maxout(inputs, hidden_size, pool_size, keep_prob=1.0):
    """ Highway maxout network.
    Args:  
        inputs: Tensor of rank 3, shape [N, D, ?]. Inputs to network.  
        hidden_size: Scalar integer. Hidden units of highway maxout network.  
        pool_size: Scalar integer. Number of units that are max pooled in maxout layer.  
        keep_prob: Scalar float. Input dropout keep probability for maxout layers.  
    Returns:  
        Tensor of rank 2, shape [N, D]. Logits.
    """
    layer1 = maxout_layer(inputs, hidden_size, pool_size, keep_prob)
    layer2 = maxout_layer(layer1, hidden_size, pool_size, keep_prob)
    
    highway = convert_gradient_to_tensor(tf.concat([layer1, layer2], -1))
    output = maxout_layer(highway, 1, pool_size, keep_prob)
    output = tf.squeeze(output, -1)
    return output


def mixture_of_experts():
    pass


def maxout_layer(inputs, outputs, pool_size, keep_prob=1.0):
    """ Maxout layer
    Args:  
        inputs: Tensor of rank 3, shape [N, D, ?]. Inputs to layer.  
        outputs: Scalar integer, number of outputs.  
        pool_size: Scalar integer, number of units to max pool over.  
        keep_prob: Scalar float, input dropout keep probability.  
    
    Returns:  
        Tensor, shape [N, D, outputs]. Result of maxout layer.
    """
    
    inputs = tf.nn.dropout(inputs, keep_prob)
    pool = tf.layers.dense(inputs, outputs*pool_size)
    pool = tf.reshape(pool, (-1, tf.shape(inputs)[1], outputs, pool_size))
    output = tf.reduce_max(pool, -1)
    return output

def maybe_dropout(keep_prob, is_training):
    return tf.cond(tf.convert_to_tensor(is_training), lambda: keep_prob, lambda: 1.0)