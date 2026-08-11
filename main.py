import torch
words = open("names.txt", "r").read().splitlines()

initial = torch.zeros((27,27), dtype=torch.float32)
#char mapping
chars =  sorted(set(list(("".join(words)))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

#creating dataset
block_size = 3
xs, ys = [], []
for w in words:
    context = [0]*block_size
    
    for ch in w + ".":#for each pair of characters in the word
        ix = stoi[ch] #char to index
        xs.append(context) #append the index of the first character to xs and ys
        ys.append(ix)
        context = context[1:] + [ix] # crop and append the new character index to the context
    
        
xs = torch.tensor(xs) #convert it to tensor
ys = torch.tensor(ys)
num = xs.nelement()

#one hot encoding
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()
#print(xenc.shape)

# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
for k in range(100): # 100 iterations of training
    xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
    logits = xenc @ W # predict log-counts
    #counts = logits.exp() # counts, equivalent to N
    #probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # btw: the last 2 lines here are together called a 'softmax'
    cross_entropy = torch.nn.functional.cross_entropy(logits, ys) # built-in cross-entropy loss, for comparison
   #loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() # L2 regularization
    loss = cross_entropy + 0.01*(W**2).mean() # L2 regularization 
    print(loss.item())

    # backward pass
    W.grad = None # set to zero the gradient
    loss.backward()

    # update
    W.data += -50 * W.grad

# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
  
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))