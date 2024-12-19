import torch
import torchvision
import torchvision.transforms as transforms
from torch_geometric.datasets import Planetoid
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.datasets import FEMNIST
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# General transform for image datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 1. CIFAR-10 Dataset
def load_cifar10():
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# 2. CIFAR-100 Dataset
def load_cifar100():
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# 3. IMDB4K (Text Dataset)
def load_imdb4k():
    tokenizer = get_tokenizer("basic_english")
    train_iter, test_iter = IMDB(root='./data', split=('train', 'test'))
    vocab = torchtext.vocab.build_vocab_from_iterator(
        [tokenizer(x[1]) for x in train_iter], specials=["<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    def text_pipeline(text):
        return vocab(tokenizer(text))
    return train_iter, test_iter, text_pipeline

# 4. Cora (Graph Dataset)
def load_cora():
    dataset = Planetoid(root='./data', name='Cora')
    return dataset[0]

# 5. FEMNIST Dataset
def load_femnist():
    femnist_train = FEMNIST(root="./data/FEMNIST", train=True, download=True, transform=transform)
    femnist_test = FEMNIST(root="./data/FEMNIST", train=False, download=True, transform=transform)
    train_loader = DataLoader(femnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(femnist_test, batch_size=64, shuffle=False)
    return train_loader, test_loader

# 6. MVTec AD Dataset , Download form here: https://www.mvtec.com/company/research/datasets/mvtec-ad % due to restriction, we can't upload the datasets, 
def load_mvtec_ad(category='bottle'):
   mvtec_root = "./data/MVTecAD"
    mvtec_path = os.path.join(mvtec_root, category)
    dataset = ImageFolder(mvtec_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader

# Load and test each dataset
{if __name__ == "__main__":
    # Load CIFAR-10
    cifar10_train_loader, cifar10_test_loader = load_cifar10()
    print("CIFAR-10 Loaded Successfully")

    # Load CIFAR-100
    cifar100_train_loader, cifar100_test_loader = load_cifar100()
    print("CIFAR-100 Loaded Successfully")

    # Load IMDB4K
    imdb_train_iter, imdb_test_iter, imdb_text_pipeline = load_imdb4k()
    print("IMDB4K Loaded Successfully")

    # Load Cora
    cora_data = load_cora()
    print("Cora Loaded Successfully")

    # Load FEMNIST
    femnist_train_loader, femnist_test_loader = load_femnist()
    print("FEMNIST Loaded Successfully")

    # Load MVTec AD (Example for "bottle" category)
    mvtec_loader = load_mvtec_ad(category='bottle')
    print("MVTec AD Loaded Successfully (Category: bottle)")
