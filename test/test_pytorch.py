import torch

print('torch file:', torch.__file__)
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print(' MPS available:', torch.backends.mps.is_available())
