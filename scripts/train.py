import torch
import numpy as np
import pickle
import os
from typing import Dict
import yaml

from models.ddpm import DDPM
from models.unet import UNetModel
from datasets.grasp_anything import GraspAnyting

import argparse

def collate_fn_squeeze_pcd_batch(batch_data) -> Dict:
    """ General collate function used for dataloader.
    This collate function is used for point-transformer
    """
    offset, count = [], 0
    for item in batch_data['pos']:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)
    batch_data['offset'] = offset

    batch_data['pos'] = torch.func.rearrange(batch_data['pos'], 'b n c -> (b n) c')
    batch_data['feat'] = torch.func.rearrange(batch_data['feat'], 'b n c -> (b n) c')
    
    return batch_data

def saveWeights(model, URL = './test0.pth', saveStateOnly = True):
    print("Saving weights...")
    if saveStateOnly:
        torch.save(model.state_dict(), URL)
    else:
        torch.save(model, URL)
    print(f"Saved model {URL}")

def loadWeights(model, URL = './test0.pth', loadOnlyWeights = True):
    print(f"Loading weights from {URL}...")
    
    if loadOnlyWeights:
        state_dict = torch.load(URL, weights_only=loadOnlyWeights)
        model.load_state_dict(state_dict)
    else:
        model = torch.load(URL, weights_only=loadOnlyWeights)
    # model.eval()
    print('Model loaded')
    return model

def load_data(path = 'data/Realdex/Realdex'):
    print('Loading data')
    scene_path = os.path.join(path, 'object_pcds_nors.pkl')
    data_path = os.path.join(path, 'realdex_shadowhand_downsample.pt')

    object_pcds = pickle.load(open(scene_path, 'rb'))

    with open("data/item_list.txt", "w") as file:
        file.writelines(line + '\n' for line in list(object_pcds.keys()))

    grasp_data = torch.load(data_path, weights_only=False)

    print('Done loading')
    return grasp_data, object_pcds

def train(model, dataloader, model_URL, learning_rate = 0.0001, epoch_rate = 5, device=None):
    print('Started training')

    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # hl = HandParams()
    # _joint_angle_lower, _joint_angle_upper = hl.get_joint_limits()
    # _global_trans_lower, _global_trans_upper = hl.get_global_limits()
    # model.set_joint_limits(_joint_angle_lower, _joint_angle_upper)
    # model.set_global_limits(_global_trans_lower, _global_trans_upper)
    # model.optimizer.set_joint_limits(_joint_angle_lower, _joint_angle_upper)
    # model.optimizer.set_global_limits(_global_trans_lower, _global_trans_upper)
    # criterion = nn.CrossEntropyLoss()
    # dataloader = tqdm.tqdm(dataloader)
    try:
        for epoch in range(epoch_rate):
            running_loss = 0.0
            # p = 1
            for i, data in enumerate(dataloader):
                data = collate_fn_squeeze_pcd_batch(data)
                for key in data:
                    if torch.is_tensor(data[key]):
                        data[key] = data[key].to(device)

                optimizer.zero_grad()

                outputs = model(data)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # if (len(dataloader) * (0.2 * p)) < i:
                #     print(f'data prog: {0.2 * p}')
                #     p += 1
                
            print(f'epoch: {epoch} loss: {running_loss/len(dataloader)}')
            if (epoch + 1) % 10 == 0:
                saveWeights(model, f'{model_URL[:-4]}_epoch{model_URL[-4:]}')
    except KeyboardInterrupt:
        saveWeights(model, f'{model_URL[:-4]}_intr{model_URL[-4:]}')

    print('Done training')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="train",
        usage="train.py [--trainData TRAIN-DATA] [--paramsPath PARAMS-PATH] [--modelLoadPath MODEL-LOAD-PATH] [--modelSavePath MODEL-SAVE-PATH]",
    )

    parser.add_argument(
        "--trainData",
        default="/home/gustavskrasnikovs/repos/off_shute/data/Realdex/Realdex",
    )

    parser.add_argument(
        "--paramsPath",
        default="data/dex_hand_params/all.yaml",
    )

    parser.add_argument(
        "--modelLoadPath",
        default="./data/test-model-0.pth",
    )

    parser.add_argument(
        "--modelSavePath",
        default="models/26_10_16/model-09_30.pth",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Using device: {torch.cuda.get_device_name(device)}')
        print(f'Using device: {device}')
    else:
        device = torch.device("cpu")
        print('Using cpu')

    grasp_data, object_pcds = load_data(args.trainData)

    # currated_obj_list = list(object_pcds.keys())[:int(len(list(object_pcds.keys())))]

    # grasp_data['metadata'] = [data for data in grasp_data['metadata'] if data['object_name'] in currated_obj_list]

    params = yaml.safe_load(open(args.paramsPath, 'r'))

    train_params = params['grasp_gen_ur']

    data = GraspAnyting(grasp_data, object_pcds)

    dataloader = torch.utils.data.DataLoader(data, batch_size=train_params['train']['batch_size'], shuffle=True)

    model = UNetModel(params['UNetModel'], device).to(device)
    model = DDPM(model, False, device, params['DDPM']).to(device)

    load_model = False
    if load_model:
        model = loadWeights(model, URL=args.modelLoadPath, loadOnlyWeights=True)

    model = train(model, dataloader, device=device, model_URL=args.modelSavePath, epoch_rate=train_params['train']['epochs'], learning_rate=float(train_params['lr']))
    
    saveWeights(model, URL=args.modelSavePath, saveStateOnly=True)
    print('done')