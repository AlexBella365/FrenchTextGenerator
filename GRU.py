
# coding: utf-8

# In[4]:


from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
import pickle
import re

mx.random.seed(1)
#ctx = mx.gpu(0)
ctx = mx.cpu(0)


# In[5]:


with open("data/moliere_1_clean.txt",encoding='utf-8') as f1:
    moliere_1 = f1.read()
with open("data/moliere_2_clean.txt",encoding='utf-8') as f2:
    moliere_2 = f2.read()
with open("data/moliere_3_clean.txt",encoding='utf-8') as f3:
    moliere_3 = f3.read()


# In[36]:


text = moliere_1 + ' ' + moliere_2 + ' ' + moliere_3


# In[37]:


len(text)


# In[38]:


text = text.replace('œ','oe')
text = text.replace('æ','ae')
text = text.replace('î','i')
text = text.replace('ï','i')
text = text.replace('º','')
text = text.replace('_','')
text = text.replace('ñ','n')
text = text.replace('λ','')
text = text.replace('ο','')
text = text.replace('ρ','')
text = text.replace('ς','')
text = text.replace('φ','')
text = text.replace('β','')
text = text.replace('ε','')
text = text.replace('ι','')

for (old,new) in [('û','u'),('é','e'),('è','e'),('ê','e'),('à','a'),('â','a'),('ç','c'),('Ç','C'),('É','E'), ('Ê','E'), ('ë','e'), ('ô','o'), ('ù','u')]:
    text = text.replace(old,new)

text = text.replace('\n',' ')

text = re.sub(r'( )+',' ',text) #remove multiple spaces

text[:200]


# In[39]:


text = text#.lower().split()
character_list = list(set(text))
vocab_size = len(character_list)
character_list.sort()
print("Length of vocab: %s" % vocab_size)
print(character_list)


# In[40]:


character_dict = {}
for k, word in enumerate(character_list):
    character_dict[word] = k
#print(character_dict)


# In[41]:


text_numerical = [character_dict[word] for word in text]

print(text_numerical[:20])
print("".join([character_list[idx] for idx in text_numerical[:20]]))


# In[42]:


def one_hots(numerical_list, vocab_size=vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result


# In[43]:


def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result


# In[44]:


print(one_hots(text_numerical[:2]))


# In[45]:


textify(one_hots(text_numerical[200:380]))


# In[ ]:


seq_length = 64
# -1 here so we have enough characters for labels later
num_samples = (len(text_numerical) - 1) // seq_length
dataset = one_hots(text_numerical[:seq_length*num_samples]).reshape((num_samples, seq_length, vocab_size))
textify(dataset[0])


# In[ ]:


batch_size = 32
print('# of sequences in dataset: ', len(dataset))
num_batches = len(dataset) // batch_size
print('# of batches: ', num_batches)
train_data = dataset[:num_batches*batch_size].reshape((num_batches, batch_size, seq_length, vocab_size))
# swap batch_size and seq_length axis to make later access easier
train_data = nd.swapaxes(train_data, 1, 2)
print('Shape of data set: ', train_data.shape)


# In[ ]:


labels = one_hots(text_numerical[1:seq_length*num_samples+1])
train_label = labels.reshape((num_batches, batch_size, seq_length, vocab_size))
train_label = nd.swapaxes(train_label, 1, 2)
print(train_label.shape)


# In[ ]:


powerof2 = np.ceil(np.log2(vocab_size)).astype(int)
powerof2


# In[ ]:


num_inputs = vocab_size
num_hidden = 2**(powerof2+1)
num_outputs = vocab_size

num_hidden


# In[21]:


num_inputs = vocab_size
num_hidden = 256
num_outputs = vocab_size

########################
#  Weights connecting the inputs to the hidden layer
########################
Wxz = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxr = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
Wxh = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01

########################
#  Recurrent weights connecting the hidden layer across time steps
########################
Whz = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whr = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
Whh = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01

########################
#  Bias vector for hidden layer
########################
bz = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
br = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
bh = nd.random_normal(shape=num_hidden, ctx=ctx) * .01

########################
# Weights to the output nodes
########################
Why = nd.random_normal(shape=(num_hidden,num_outputs), ctx=ctx) * .01
by = nd.random_normal(shape=num_outputs, ctx=ctx) * .01


# In[22]:


params = [Wxz, Wxr, Wxh, Whz, Whr, Whh, bz, br, bh, Why, by]

for param in params:
    param.attach_grad()


# In[23]:


def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition =nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition


# In[24]:


def gru_rnn(inputs, h, temperature=1.0):
    outputs = []
    for X in inputs:
        z = nd.sigmoid(nd.dot(X, Wxz) + nd.dot(h, Whz) + bz)
        r = nd.sigmoid(nd.dot(X, Wxr) + nd.dot(h, Whr) + br)
        g = nd.tanh(nd.dot(X, Wxh) + nd.dot(r * h, Whh) + bh)
        h = z * h + (1 - z) * g

        yhat_linear = nd.dot(h, Why) + by
        yhat = softmax(yhat_linear, temperature=temperature)
        outputs.append(yhat)
    return (outputs, h)


# In[25]:


def cross_entropy(yhat, y):
    return - nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))


# In[26]:


def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = 0.
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)


# In[27]:


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


# In[28]:


def sample(prefix, num_chars,include_prefix=True, temperature=1.0):
    #####################################
    # Initialize the string that we'll return to the supplied prefix
    #####################################
    string = prefix if include_prefix else ''

    #####################################
    # Prepare the prefix as a sequence of one-hots for ingestion by RNN
    #####################################
    prefix_numerical = [character_dict[char] for char in prefix]
    input = one_hots(prefix_numerical)

    #####################################
    # Set the initial state of the hidden representation ($h_0$) to the zero vector
    #####################################
    h = nd.zeros(shape=(1, num_hidden), ctx=ctx)
    c = nd.zeros(shape=(1, num_hidden), ctx=ctx)

    #####################################
    # For num_chars iterations,
    #     1) feed in the current input
    #     2) sample next character from from output distribution
    #     3) add sampled character to the decoded string
    #     4) prepare the sampled character as a one_hot (to be the next input)
    #####################################
    for i in range(num_chars):
        outputs, h = gru_rnn(input, h, temperature=temperature)
        choice = np.random.choice(vocab_size, p=outputs[-1][0].asnumpy())
        string += character_list[choice]
        input = one_hots([choice])
    return string


# In[29]:


moving_loss = 0.0
learning_rate = 0.5

losses = []


# In[30]:


'''
with open("output/saveState.pickle","rb") as open_file:
    dic = pickle.load(open_file)
params = dic['params']
'''


# In[33]:


epochs = 500


# In[34]:


for e in range(epochs):
    ############################
    # Attenuate the learning rate by a factor of 2 every 50 epochs.
    ############################
    if ((e+1) % 50 == 0):
        learning_rate = learning_rate / 2.0
        
        
    h = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for i in range(num_batches):
        data_one_hot = train_data[i]
        label_one_hot = train_label[i]
        with autograd.record():
            outputs, h = gru_rnn(data_one_hot, h)
            loss = average_ce_loss(outputs, label_one_hot)
            loss.backward()
        SGD(params, learning_rate)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        if (i == 0) and (e == 0):
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = 0.99 * moving_loss + 0.01 * nd.mean(loss).asscalar()
        
    losses.append(moving_loss)

    if e%10 == 0:
        print("******** Epoch %s | Loss: %s" % (e, moving_loss))
        print(sample("Bonjou", 80, temperature=0.5))
        print(sample("Bonjou", 80, temperature=0.1))
        print(sample("Bonjour cher ami, je su", 80, temperature=0.1))
        
        with open("output/saveState.pickle","wb") as save_file:
            to_save = {'loss':losses,'params': params,'num_hidden':num_hidden,'character_list':character_list}
            pickle.dump(to_save, save_file)
            print('==== Model saved ====')
        
        print('')


# In[35]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(24, 8), dpi= 100)
plt.xscale('log')
#plt.yscale('log')
plt.plot(range(len(losses)),losses,'.-')
plt.grid()

