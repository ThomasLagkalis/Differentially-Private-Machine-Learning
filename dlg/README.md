# Gradient Leakage Attacks
Experiments based on the Deep Leakage from Gradients attack (DLG). Specifically, we experimented with the original Euclidean distance cost function, as well as normalized Euclidean distance and cosine similarity. All experiments were performed on the [**MNIST**](http://yann.lecun.com/exdb/mnist/) and [**CIFAR10**](https://www.cs.toronto.edu/~kriz/cifar.html) ataset. Finally, several experiments are performed on different batch sizes and additive noise injected in the gradient.

## Usage 

Install required packages using the following command:

```
pip install -r requirements.txt
```

Note: The '.py' files may be outdated. All experiments are conducted in the [notebooks](./notebooks), so I suggest using those instead.
