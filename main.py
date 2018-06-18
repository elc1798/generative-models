from train import train_SimpleVAE
import dataset

def main_SimpleVAE():
    train_SimpleVAE(dataset.get_dataset_mnist(), img_dir="simple_vae", zdim=10)

if __name__ == "__main__":
    main_SimpleVAE()
