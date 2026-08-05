import torch
words = open("names.txt", "r").read().splitlines()

initial = torch.zeros((27,27), dtype=torch.float32)
#char mapping
chars =  sorted(set(list(("".join(words)))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

#creating dataset
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."] # adding start and end tokens 
    for ch1, ch2 in zip(chs,chs[1:]):#for each pair of characters in the word
        ix1 = stoi[ch1] #char to index
        ix2 = stoi[ch2]
        xs.append(ix1) #append the index of the first character to xs and ys
        ys.append(ix2)
xs = torch.tensor(xs) #convert it to tensor
ys = torch.tensor(ys)
print(xs, ys)