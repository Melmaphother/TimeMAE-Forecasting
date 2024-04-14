import torch
import warnings
import os

warnings.filterwarnings('ignore')
from args import args, Test_data, Train_data_all, Train_data
try:
    from args import VAL_data
except:
    print('No validation data')

try:
    from args import Train_data_forecasting, VAL_data_forecasting, Test_data_forecasting
except:
    print('No forecasting data')

from dataset import Dataset, ETTForecastingDataset
from model.TimeMAE import TimeMAE
from process import Trainer
import torch.utils.data as Data
from forecasting import TimeMAEForecasting


def main():
    torch.set_num_threads(12)
    torch.cuda.manual_seed(3407)
    train_dataset = Dataset(device=args.device, mode='pretrain', data=Train_data_all, wave_len=args.wave_length)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = train_dataset.shape()
    train_linear_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data, wave_len=args.wave_length)
    train_linear_loader = Data.DataLoader(train_linear_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataset = Dataset(device=args.device, mode='test', data=Test_data, wave_len=args.wave_length)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    print(args.data_shape)
    print('dataset initial ends')

    model = TimeMAE(args)

    print('model initial ends')
    trainer = Trainer(args, model, train_loader, train_linear_loader, test_loader, verbose=True)

    trainer.pretrain()
    # trainer.finetune()

def forecasting():
    torch.set_num_threads(12)
    torch.cuda.manual_seed(3407)
    #pred_lens = [24, 48, 96, 168, 192, 336, 720]
    pred_lens = [288, 672]
    test_loss_all = {}
    for pred_len in pred_lens:
        args.pred_len = pred_len
        print('pred_len:', pred_len)
        train_dataset = ETTForecastingDataset(args, 'train')
        train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        args.data_shape = train_dataset.shape()
        val_dataset = ETTForecastingDataset(args, 'val')
        val_loader = Data.DataLoader(val_dataset, batch_size=args.test_batch_size, drop_last=True)
        test_dataset = ETTForecastingDataset(args, 'test')
        test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size, drop_last=True)

        print(args.data_shape)
        print('dataset initial ends')

        model = TimeMAE(args)
        state_dict = torch.load(args.save_path + '/pretrain_model.pkl', map_location=args.device)
        model.load_state_dict(state_dict)
        model.init_forecasting(args, pred_len)
        print('model initial ends')

        args.save_path_each_pred = args.save_path + '/' +str(pred_len) + '/'
        if not os.path.exists(args.save_path_each_pred):
            os.makedirs(args.save_path_each_pred)
        data_loader = (train_loader, val_loader, test_loader)
        test_loss = TimeMAEForecasting(args, model, data_loader).forecasting_finetune()
        test_loss_all[pred_len] = test_loss
    
    for key, value in test_loss_all.items():
        print("pred_len:", key, "test_loss:", value)


if __name__ == '__main__':
    # main()
    forecasting()
