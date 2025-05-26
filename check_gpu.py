import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print('GPU detected!')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print('You can use GPU for training.')
else:
    print('No GPU detected. Training will use CPU.')
    print('If you have a GPU, ensure CUDA is installed and PyTorch is configured for GPU support.') 