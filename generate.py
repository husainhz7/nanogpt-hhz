import torch

from gpt import GPTModel, decode

# params
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTModel()
m = model.to(device)
m.load_state_dict(torch.load("nano-shkspr-gpt.pth"))

context = torch.zeros((1, 1), dtype=torch.long, device=device)
for _ in range(5000):
    context = m.generate(context, max_new_tokens=2)
    text = decode(context[0].tolist())
    print(text[-2:], end='')


open('generated.txt', 'w').write(decode(context[0].tolist()))

