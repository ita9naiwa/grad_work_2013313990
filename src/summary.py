import tensorflow as tf
import numpy as np

class summary(object):
    def __init__(self, sess, dir, **kwargs):
        self.sess = sess
        self.step = 0
        self.writer = tf.summary.FileWriter(dir, self.sess.graph)
        self.summary_dict = {}
        for name, dtype in kwargs.items():
            var = tf.Variable(dtype(0))
            tf.summary.scalar(name, var)
            self.summary_dict[name] = var
        summary_ops = tf.summary.merge_all()
        self.summary_ops = summary_ops
        print(self.summary_dict)

    def write_log(self, **kwargs):
        feed_dict = {}
        for key, item in kwargs.items():
            feed_dict[self.summary_dict[key]] = item
        summary_str = self.sess.run(self.summary_ops, feed_dict=feed_dict)
        self.writer.add_summary(summary_str, self.step)
        self.step += 1


if __name__ == "__main__":

    d = {
        "name": int,
        "flat": int,
    }
    summary(d)
