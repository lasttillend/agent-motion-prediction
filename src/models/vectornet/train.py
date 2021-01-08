import logging
import os
import json
import yaml
from tqdm import tqdm

import numpy as np
from zarr import convenience

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from l5kit.data import ChunkedDataset, LocalDataManager

from preprocess.vectornet.dataset import LyftDataset
from preprocess.vectornet.custom_map import CustomMapAPI
from preprocess.vectornet.vectorizer import Vectorizer
from models.vectornet.vectornet import VectorNet
from models.vectornet.utils import pytorch_neg_multi_log_likelihood_batch

DATA_ROOT = "/home/han/study/projects/agent-motion-prediction/data/lyft_dataset"
os.environ["L5KIT_DATA_FOLDER"] = DATA_ROOT
MODEL_OUTPUT_FOLDER = "/home/han/study/projects/agent-motion-prediction/models"

def main():
    # Load config file.
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_filepath = os.path.join(file_dir, "agent_motion_config.yaml")  # under the same directory as this script
    with open(cfg_filepath) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg) 

    # Load semantic map and dataset meta.
    semantic_map_path = os.path.join(DATA_ROOT, cfg["vector_params"]["semantic_map_key"])
    dataset_meta_path = os.path.join(DATA_ROOT, cfg["vector_params"]["dataset_meta_key"])
    with open(dataset_meta_path, "rb") as f:
        dataset_meta = json.load(f)
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    # Create custom map and vectorizer 
    map_api = CustomMapAPI(semantic_map_path, world_to_ecef)
    vectorizer = Vectorizer(map_api)

    # Initiate zarr dataset.
    dm = LocalDataManager()
    dataset_path = dm.require(cfg["train_data_loader"]["key"])
    print("dataset_path:", dataset_path)
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()
    
    save_path = os.path.join(MODEL_OUTPUT_FOLDER, cfg["output_params"]["save_path"])
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    tensorboard_path = os.path.join(MODEL_OUTPUT_FOLDER, cfg["output_params"]["tensorboard_path"])
    writer = SummaryWriter(tensorboard_path)
    logger = init_logger(cfg)
    
    lyft_dataset = LyftDataset(cfg, zarr_dataset, map_api, vectorizer)
    
    train_loader = DataLoader(dataset=lyft_dataset,
                              batch_size=cfg["train_data_loader"]["batch_size"],
                              shuffle=cfg["train_data_loader"]["shuffle"],
                              num_workers=cfg["train_data_loader"]["num_workers"])
                            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device

    model = VectorNet(cfg, traj_features=6, map_features=5, num_modes=3)
    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), rho=0.9)
    
    logger.info("Start Training...")
    do_train(model, cfg, train_loader, optimizer, scheduler=None, writer=writer, logger=logger)

def do_train(model, cfg, train_loader, optimizer, scheduler, writer, logger):
    device = cfg["device"] 
    print_every = cfg["train_params"]["print_every"]
    save_every = cfg["train_params"]["save_every"]
    save_path = os.path.join(MODEL_OUTPUT_FOLDER, cfg["output_params"]["save_path"])
    num_epochs = cfg["train_params"]["num_epochs"]
    criterion = pytorch_neg_multi_log_likelihood_batch
   
    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(train_loader)):
            model.train()
            
            # Inputs.
            map_feature = data["map_feature"].to(device)
            traj_feature = data["traj_feature"].to(device)
            
            # Forward pass.
            preds, confidences = model(traj_feature, map_feature)
            
            targets = data["target_positions"].to(device)
            targets_availabiliteis = data["target_availabilities"].to(device)

            # # Calculate loss.
            loss = criterion(targets, preds, confidences, targets_availabiliteis)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % print_every == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}: Iteration {i}, loss = {loss.item()}")
                writer.add_scalar("training_loss", loss.item(), epoch)
        if (epoch + 1) % save_every == 0:
            file_path = os.path.join(save_path, f"model_epoch{epoch + 1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss
            }, file_path)
            logger.info(f"Save model {file_path}")

    torch.save(model.state_dict(), os.path.join(save_path, "model_final.pth"))
    logger.info("Save final model " + os.path.join(save_path, "model_final.path"))
    logger.info("Finish Training")

def init_logger(cfg: dict):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    log_file = os.path.join(MODEL_OUTPUT_FOLDER, cfg["output_params"]["log_file"])
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.DEBUG)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
    
if __name__ == "__main__":
    main()    

