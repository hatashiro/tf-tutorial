import random
import tensorflow as tf
import matplotlib.pyplot as plt
from lib import load_data, convert_to_squared_grayscale


train_data_dir = "data/Training"
test_data_dir = "data/Testing"

# load training data
images, labels = load_data(train_data_dir)
images28 = convert_to_squared_grayscale(images)

# initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# create layers for learning
images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 62)
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("Layers:")
print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)

# prepare session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
for i in range(201):
    _, accuracy_val = sess.run([train_op, accuracy],
                               feed_dict={
                                   x: images28,
                                   y: labels
                               })
    if i % 10 == 0:
        print("Loss: ", loss)

# check with sample images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(
        40,
        10,
        "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
        fontsize=12,
        color=color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()

# check with test data
test_images, test_labels = load_data(test_data_dir)
test_images28 = convert_to_squared_grayscale(test_images)

predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))
