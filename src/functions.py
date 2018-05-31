import mxnet as mx
from mxnet import nd, autograd
import numpy as np

ctx = mx.cpu(0)
#ctx = mx.gpu(0)

def one_hots(numerical_list, vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result


def textify(embedding,character_list):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result


def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition =nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition


def simple_rnn(inputs, state, params, temperature=1.0):
    [Wxh, Whh, bh, Why, by] = params
    outputs = []
    h = state
    for X in inputs:
        h_linear = nd.dot(X, Wxh) + nd.dot(h, Whh) + bh
        h = nd.tanh(h_linear)
        yhat_linear = nd.dot(h, Why) + by
        yhat = softmax(yhat_linear, temperature=temperature)
        outputs.append(yhat)
    return (outputs, h)


def cross_entropy(yhat, y):
    return - nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))


def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = 0.
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def make_dict(character_list):
    character_dict = {}
    for k, word in enumerate(character_list):
        character_dict[word] = k
    return character_dict

def sample(prefix, num_chars,character_list,num_hidden,params,temperature=1.0):

    # Initialize the string that we'll return to the supplied prefix
    string = prefix

    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    vocab_size = len(character_list)
    character_dict = make_dict(character_list)
    prefix_numerical = [character_dict[char] for char in prefix]
    input = one_hots(prefix_numerical,vocab_size)

    # Set the initial state of the hidden representation ($h_0$) to the zero vector
    sample_state = nd.zeros(shape=(1, num_hidden), ctx=ctx)

    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    for i in range(num_chars):
        outputs, sample_state = simple_rnn(input, sample_state, params,temperature=temperature)
        choice = np.random.choice(vocab_size, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input = one_hots([choice],vocab_size)
    return string