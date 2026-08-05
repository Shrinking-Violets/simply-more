import torch
words = open("names.txt", "r").read().splitlines()

initial = torch.zeros((27,27), dtype=torch.float32)
sorted_words =  sorted(set(list(("".join(words)))))
stoi = {s:i+1 for i,s in enumerate(sorted_words)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)