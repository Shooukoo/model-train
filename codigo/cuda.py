import torch, torchvision, torchaudio
print("CUDA disponible:", torch.cuda.is_available())
print(torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)