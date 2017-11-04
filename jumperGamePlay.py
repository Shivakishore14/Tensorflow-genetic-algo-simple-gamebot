import tensorflow as tf
from optparse import OptionParser
from JumperGame import JumperGame
import time

parser = OptionParser()
parser.add_option("-n", "--num-model", dest="MODEL_NUM",
                  help="The Model Number to User")

(options, args) = parser.parse_args()
print options, args
if options.MODEL_NUM == None:
    print "Give -n Model Number"
    exit()
MODEL_NUM = options.MODEL_NUM
MODEL_PREFIX = "game_checkpoints/my_test_model-{}.meta"


sess = tf.Session()
new_saver = tf.train.import_meta_graph(MODEL_PREFIX.format(MODEL_NUM))
new_saver.restore(sess,MODEL_PREFIX.format(MODEL_NUM)[:-5])

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
game = JumperGame()
done = False
while not done:
    ip = game.get_input_to_algo()
    # print ip
    y_ = sess.run([tf.argmax(Y,1)], feed_dict={X:[ip]})[0]
    done = game.step(action=y_[0])
    game.render()
    time.sleep(1)
    # done = True
print game.get_fitness()
sess.close()
