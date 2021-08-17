import jax
import jax.numpy as jnp
from jax import grad, jit
import haiku as hk
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_preprocessed_mnist():
	import tensorflow as tf
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
	x_train = x_train / 255
	x_test = x_test / 255
	return x_train, y_train, x_test, y_test

class LeNet(hk.Module):
	def __init__(self, output_size, name=None):
		super().__init__(name=name)
		self.output_size = output_size
		
	def __call__(self, x):
		x = hk.Reshape((28,28,1))(x)
		x = hk.Conv2D(32,5,padding="SAME")(x)
		x = jax.nn.relu(x)
		x = hk.MaxPool(window_shape=(2,2), strides=2, padding="VALID")(x)
		x = hk.Conv2D(48,5,padding="VALID")(x)
		x = jax.nn.relu(x)
		x = hk.MaxPool(window_shape=(2,2), strides=2, padding="VALID")(x)
		x = hk.Flatten()(x),
		x = hk.Linear(256)(x[0])
		x = hk.Linear(84)(x)
		x = hk.Linear(self.output_size)(x)
		return x

def net_fn(images):
	cnn = LeNet(10)	
	return cnn(images)
	
x_train, y_train, x_test, y_test = get_preprocessed_mnist()
model = hk.without_apply_rng(hk.transform(net_fn))
params = model.init(jax.random.PRNGKey(0), jnp.array(x_train[:128]))
optimizer = optax.adam(3e-4)
opt_state = optimizer.init(params)

@jit
def softmax_cross_entropy(logits, labels):
	one_hot = jax.nn.one_hot(labels, logits.shape[-1])
	return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

@jit
def loss_fn(params, images, labels):
	logits = model.apply(params, images)
	crossentropy = jnp.mean(softmax_cross_entropy(logits, labels))
	l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
	return crossentropy + 0.01 * l2_loss
	
@jit
def update(params, opt_state, images, labels):
	grads = grad(loss_fn)(params, images, labels)
	updates, opt_state = optimizer.update(grads, opt_state)
	new_params = optax.apply_updates(params, updates)
	return new_params, opt_state
	
@jit
def accuracy(params, images, labels):
	preds = model.apply(params, images)
	return jnp.mean(jnp.argmax(preds, axis=-1) == labels)
	
STEP_SIZE = 128	
history = {"train_acc": [], "test_acc": []}
for epoch in range(30):
	for i in tqdm(range(0, len(y_train), STEP_SIZE)):
		params, opt_state = update(params, opt_state, x_train[i:i+STEP_SIZE], y_train[i:i+STEP_SIZE])
	train_accuracy = accuracy(params, x_train, y_train)
	test_accuracy = accuracy(params, x_test, y_test)
	train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
	print(f'epoch: {epoch+1}, train acc: {train_accuracy*100:.2f}%, test acc: {test_accuracy*100:.2f}%')
	history["train_acc"].append(train_accuracy)
	history["test_acc"].append(test_accuracy)
plt.plot(history["train_acc"])
plt.plot(history["test_acc"])
plt.title("accuracy")
plt.xlabel("step")
plt.ylabel("accuracy")
plt.legend(["train acc", " test acc"])
plt.show()