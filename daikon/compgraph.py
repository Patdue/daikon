#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import tensorflow as tf

from daikon import constants as C


def compute_lengths(sequences):
    """
    This solution is similar to:
    https://danijar.com/variable-sequence-lengths-in-tensorflow/
    """
    used = tf.sign(tf.abs(sequences))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    lengths = tf.cast(lengths, tf.int32)
    return lengths


def define_computation_graph(source_vocab_size: int, target_vocab_size: int, batch_size: int):

    tf.reset_default_graph()

    # Placeholders for inputs and outputs
    encoder_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='encoder_inputs')

    decoder_targets = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_targets')
    decoder_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_inputs')

    with tf.variable_scope("Embeddings"):
        source_embedding = tf.get_variable('source_embedding', [source_vocab_size, C.EMBEDDING_SIZE])
        target_embedding = tf.get_variable('target_embedding', [source_vocab_size, C.EMBEDDING_SIZE])

        encoder_inputs_embedded = tf.nn.embedding_lookup(source_embedding, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(target_embedding, decoder_inputs)

    with tf.variable_scope("Encoder"):
        # Construct forward and backward cells
        forward_cell = tf.contrib.rnn.LSTMCell(C.HIDDEN_SIZE/2)
        backward_cell = tf.contrib.rnn.LSTMCell(C.HIDDEN_SIZE/2)

        bi_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
            forward_cell, backward_cell, encoder_inputs_embedded, dtype=tf.float32)

        encoder_outputs = tf.concat(bi_outputs, -1)

    with tf.variable_scope("Decoder"):
        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(C.HIDDEN_SIZE, encoder_outputs)

        decoder_cell = tf.contrib.rnn.LSTMCell(C.HIDDEN_SIZE)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
            attention_mechanism, attention_layer_size=C.HIDDEN_SIZE)

        decoder_initial_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)

        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                                 decoder_inputs_embedded,
                                                                 initial_state=decoder_initial_state,
                                                                 dtype=tf.float32)

    with tf.variable_scope("Logits"):
        decoder_logits = tf.contrib.layers.linear(decoder_outputs, target_vocab_size)

    with tf.variable_scope("Loss"):
        one_hot_labels = tf.one_hot(decoder_targets, depth=target_vocab_size, dtype=tf.float32)
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_labels,
            logits=decoder_logits)

        # mask padded positions
        target_lengths = compute_lengths(decoder_targets)
        target_weights = tf.sequence_mask(lengths=target_lengths, maxlen=None, dtype=decoder_logits.dtype)
        weighted_cross_entropy = stepwise_cross_entropy * target_weights
        loss = tf.reduce_mean(weighted_cross_entropy)

    with tf.variable_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=C.LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return encoder_inputs, decoder_targets, decoder_inputs, loss, train_step, decoder_logits, summary
