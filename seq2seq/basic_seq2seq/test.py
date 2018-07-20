# encoding = utf-8
import tensorflow as tf
import pickle

flags = tf.app.flags

flags.DEFINE_string("checkpoint_path", './run/trained_model.cpkt', "checkpoint_path")
flags.DEFINE_integer("batch_size", 32, "The size of train images [32]")

FLAGS = flags.FLAGS

vocab_to_idx = pickle.load(open('./data/vocab_to_idx.txt', mode='rb'))
idx_to_vocab = pickle.load(open('./data/idx_to_vocab.txt', mode='rb'))


def seq_to_idx(text):
    return [vocab_to_idx[word] for word in text], len(text)


def idx_to_seq(idxs):
    return ''.join([idx_to_vocab[idx]for idx in idxs])


def test(input_word):
    # prepare data
    idx, seq_length = seq_to_idx(input_word)
    input_x_idx = [idx] * FLAGS.batch_size
    seq_length = [seq_length] * FLAGS.batch_size
    start_tokens_idx = [vocab_to_idx['<GO>']] * FLAGS.batch_size
    end_token_idx = vocab_to_idx['<EOS>']

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # restore the graph
        loader = tf.train.import_meta_graph(FLAGS.checkpoint_path+'.meta')
        loader.restore(sess, FLAGS.checkpoint_path)

        input_x = loaded_graph.get_tensor_by_name('input_x:0')
        encoder_sequence_length = loaded_graph.get_tensor_by_name('encoder_sequence_length:0')
        start_tokens = loaded_graph.get_tensor_by_name('decoder/predicting_decoder/start_tokens:0')
        end_token = loaded_graph.get_tensor_by_name('decoder/predicting_decoder/end_token:0')
        predictions = loaded_graph.get_tensor_by_name('decoder/predicting_decoder/predictions:0')

        predictions_idx = sess.run(predictions, feed_dict={input_x: input_x_idx,
                                                           encoder_sequence_length: seq_length,
                                                           start_tokens: start_tokens_idx,
                                                           end_token: end_token_idx})

        output_word = idx_to_seq(predictions_idx[0][:-1])
        print("输入：", input_word)
        print("输出：", output_word)
if __name__ == '__main__':
    test('历尽')
