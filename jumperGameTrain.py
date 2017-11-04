from JumperGame import JumperGame
import tensorflow as tf
import random
import numpy as np

def mutate_w_with_percent_change(p, add_sub_rand=True):
    #considering its 2d array
    new_p = []
    for i in p:
        row = []
        for j in i:
            temp = j
            delta = np.random.random_sample() + 0.5
            if np.random.random_sample() > 0.5:
                temp = temp * delta
            if add_sub_rand == True:
                if np.random.random_sample() > 0.5:
                    if np.random.random_sample() > 0.5:
                        temp = temp - np.random.random_sample()
                    else:
                        temp = temp + np.random.random_sample()
            row.append(temp)
        new_p.append(row)
    return new_p
def mutate_b_with_percent_change(p, add_sub_rand=True):
    #considering its 1d array
    new_p = []
    for i in p:
        temp = i
        delta = np.random.random_sample() + 0.5
        if np.random.random_sample() > 0.5:
            temp = temp * delta
        if add_sub_rand == True:
            if np.random.random_sample() > 0.5:
                if np.random.random_sample() > 0.5:
                    temp = temp - np.random.random_sample()
                else:
                    temp = temp + np.random.random_sample()
        new_p.append(temp)
    return new_p

def cross_over(w11, w12, b11, b12, w21, w22, b21, b22):
    new_w1 = []
    for i in range(len(w11)):
        row = []
        for j in range(len(w11[0])):
            if np.random.random_sample() > 0.5:
                row.append(w11[i][j])
            else:
                row.append(w21[i][j])
        new_w1.append(row)
    new_w2 = []
    for i in range(len(w12)):
        row = []
        for j in range(len(w12[0])):
            if np.random.random_sample() > 0.5:
                row.append(w12[i][j])
            else:
                row.append(w22[i][j])
        new_w2.append(row)
    new_b1 = []
    for i in range(len(b11)):
        if np.random.random_sample() > 0.5:
            new_b1.append(b11[i])
        else:
            new_b1.append(b21[i])

    new_b2 = []
    for i in range(len(b12)):
        if np.random.random_sample() > 0.5:
            new_b2.append(b12[i])
        else:
            new_b2.append(b22[i])

    return (new_w1, new_w2, new_b1, new_b2)
graph = tf.Graph()
with graph.as_default():
    num_input = 10
    hidden_units = 6
    num_class = 2

    X = tf.placeholder(tf.float32, shape=[1, num_input], name='X')

    W1 = tf.Variable(tf.random_normal([num_input, hidden_units], stddev=1.0), name="W1")
    B1 = tf.Variable(tf.random_normal([hidden_units], stddev=1.0) , name="B1")
    A1 = tf.nn.softmax(tf.matmul(X, W1) + B1,  name="A1")

    W2 = tf.Variable(tf.random_normal([hidden_units, num_class], stddev=1.0), name="W2")
    B2 = tf.Variable(tf.random_normal([num_class], stddev=1.0), name="B2")
    Y_ = tf.nn.softmax(tf.matmul(A1, W2) + B2, name="Y")
    Y_index = tf.argmax(Y_,1)

    W1_placeholder = tf.placeholder(tf.float32, shape=[num_input, hidden_units])
    W2_placeholder = tf.placeholder(tf.float32, shape=[hidden_units, num_class])
    W1_assign = tf.assign(W1, W1_placeholder)
    W2_assign = tf.assign(W2, W2_placeholder)

    B1_placeholder = tf.placeholder(tf.float32, shape=[hidden_units])
    B2_placeholder = tf.placeholder(tf.float32, shape=[num_class])
    B1_assign = tf.assign(B1, B1_placeholder)
    B2_assign = tf.assign(B2, B2_placeholder)
    # W3 = tf.Variable(tf.random_normal([hidden_units2, num_class]))
    # B3 = tf.Variable(tf.random_normal([num_class]))
    # Y_ = tf.nn.softmax(tf.matmul(A2, W3) + B3, name="Y_")

    init = tf.global_variables_initializer()
    saver = tf.train.Saver( max_to_keep=150 )

POPULATION_SIZE = 20
MUTATION_PROBABILITY = 0.8
W_MUTATION_PROBABILITY = 0.5
B_MUTATION_PROBABILITY = 0.5
MAX_GEN = 10000
N_EPISODE = 10
is_training_finished = False
sessions = [tf.Session(graph=graph) for _ in range(POPULATION_SIZE)]
for sess in sessions:
    sess.run(init)
for generation in range(MAX_GEN):
    fitness_data = []
    for sess in sessions:

        fitness = 0
        for _ in range(N_EPISODE):
            fitness_episode = 0
            game = JumperGame()
            flag = True
            save_threshold = 0
            while True:
                inputs = game.get_input_to_algo()
                # print inputs
                predicted_action, p ,w1,b1,w2,b2= sess.run([Y_index, Y_,W1,B1,W2,B2], feed_dict={X:[inputs]})
                # print predicted_action, p,w1,b1,w2,b2
                done = game.step(action=predicted_action[0])
                if game.get_fitness() > 9000:
                    if flag:
                        print " it's OVER 9000!!!!!!!!!!"
                        flag = False
                    save_threshold = save_threshold + 1
                    if save_threshold > 100:
                        saver.save(sess, 'game_checkpoints/my_test_model',global_step=generation)
                        is_training_finished = True
                        print "Saved Ultimate CheckPoint"
                        break
                if done:
                    fitness_episode = fitness_episode + game.get_fitness()
                    break
            if not flag:
                break
            # print fitness_episode
        fitness = fitness_episode / N_EPISODE
        fitness_data.append(fitness)
    # print loss_data
    if is_training_finished :
        break
    sess_fit = zip(sessions, fitness_data)
    sess_fit = sorted(sess_fit, key=lambda tup: tup[1], reverse=True)
    sessions = [sess for sess, _ in sess_fit]
    fitness_data = [f for _, f in sess_fit]
    print "{} : {}".format(generation, fitness_data[:5])
    saver.save(sessions[0], 'game_checkpoints/my_test_model',global_step=generation)

    for sess in sessions[POPULATION_SIZE/2:]:
        sess.close()
    del sessions[POPULATION_SIZE/2:]
    for index in range(0,POPULATION_SIZE/4):
        sess = tf.Session(graph=graph)
        sess.run(init)
        if np.random.random_sample() > MUTATION_PROBABILITY:
            if np.random.random_sample() < W_MUTATION_PROBABILITY:
                w1_, w2_ = sessions[index].run([W1, W2])
                w1_ = mutate_w_with_percent_change(w1_)
                w2_ = mutate_w_with_percent_change(w2_)
                # print w1_, w2
                sess.run([W1_assign, W2_assign],feed_dict={W1_placeholder:w1_, W2_placeholder:w2_})

            if  np.random.random_sample() < B_MUTATION_PROBABILITY:
                b1_, b2_ = sessions[index].run([B1, B2])
                b1_ = mutate_b_with_percent_change(b1_)
                b2_ = mutate_b_with_percent_change(b2_)
                sess.run([B1_assign, B2_assign],feed_dict={B1_placeholder:b1_, B2_placeholder:b2_})
        # print sessions[index].run([loss], feed_dict={x:[1,2,3,4], y:[-2,-3,-4,-5]} )[0]
        sessions.append(sess)
    for index in range(0, POPULATION_SIZE/4):
        sess = tf.Session(graph=graph)
        sess.run(init)
        w11_, w12_, b11_, b12_ = sessions[index].run([W1, W2, B1, B2])
        w21_, w22_, b21_, b22_ = sessions[index+1].run([W1, W2, B1, B2])

        w1_,w2_,b1_,b2_ = cross_over(w11=w11_, w12=w12_, b11=b11_, b12=b12_, w21 = w21_, w22 = w22_, b21=b21_, b22=b22_)
        sess.run([W1_assign, W2_assign, B1_assign, B2_assign],feed_dict={W1_placeholder:w1_, W2_placeholder:w2_, B1_placeholder:b1_, B2_placeholder:b2_})
        sessions.append(sess)
