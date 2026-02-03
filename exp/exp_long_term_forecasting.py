import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.polynomial import (chebyshev_torch, hermite_torch, laguerre_torch,
                              leg_torch)
from utils.tools import (EarlyStopping, adjust_learning_rate, ensure_path,
                         visual)

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.model_id = args.model_id
        self.model_name = args.model
        self.pred_len = args.pred_len

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        ensure_path(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs = model_output
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]

                        loss = 0
                        if self.args.rec_lambda:
                            loss_rec = criterion(outputs, batch_y_cut)
                            loss += self.args.rec_lambda * loss_rec
                            if (i + 1) % 100 == 0:
                                print(f"\tloss_rec: {loss_rec.item()}")

                        if self.args.auxi_lambda > 0:
                            if self.args.auxi_mode == "mae":
                                loss_auxi = F.l1_loss(outputs, batch_y_cut)
                            elif self.args.auxi_mode == "rfft":
                                fft_out = torch.fft.rfft(outputs, dim=1)
                                fft_y   = torch.fft.rfft(batch_y_cut, dim=1)
                                if self.args.auxi_type == 'complex':
                                    diff = fft_out - fft_y
                                elif self.args.auxi_type == 'mag':
                                    diff = fft_out.abs() - fft_y.abs()
                                else:
                                    raise NotImplementedError
                                if self.args.auxi_loss == "MAE":
                                    loss_auxi = diff.abs().mean() if self.args.module_first else diff.mean().abs()
                                elif self.args.auxi_loss == "RMSE":
                                    loss_auxi = (diff.abs()**2).mean() if self.args.module_first else (diff**2).mean().abs()
                                    loss_auxi = torch.sqrt(loss_auxi + 1e-8)
                                else:
                                    raise NotImplementedError
                            else:
                                raise NotImplementedError
                            loss += self.args.auxi_lambda * loss_auxi
                            if (i + 1) % 100 == 0:
                                print(f"\tloss_auxi: {loss_auxi.item()}")
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    outputs = model_output
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss = 0
                    if self.args.rec_lambda:
                        loss_rec = criterion(outputs, batch_y_cut)
                        loss += self.args.rec_lambda * loss_rec
                        if (i + 1) % 100 == 0:
                            print(f"\tloss_rec: {loss_rec.item()}")

                    if self.args.auxi_lambda > 0:
                        if self.args.auxi_mode == "mae":
                            loss_auxi = F.l1_loss(outputs, batch_y_cut)
                        elif self.args.auxi_mode == "rfft":
                            fft_out = torch.fft.rfft(outputs, dim=1)
                            fft_y   = torch.fft.rfft(batch_y_cut, dim=1)
                            if self.args.auxi_type == 'complex':
                                diff = fft_out - fft_y
                            elif self.args.auxi_type == 'mag':
                                diff = fft_out.abs() - fft_y.abs()
                            else:
                                raise NotImplementedError
                            if self.args.auxi_loss == "MAE":
                                loss_auxi = diff.abs().mean() if self.args.module_first else diff.mean().abs()
                            elif self.args.auxi_loss == "RMSE":
                                loss_auxi = (diff.abs()**2).mean() if self.args.module_first else (diff**2).mean().abs()
                                loss_auxi = torch.sqrt(loss_auxi + 1e-8)
                            else:
                                raise NotImplementedError
                        else:
                            raise NotImplementedError
                        loss += self.args.auxi_lambda * loss_auxi
                        if (i + 1) % 100 == 0:
                            print(f"\tloss_auxi: {loss_auxi.item()}")

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss
            ))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj == 'TST':
                scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                    steps_per_epoch=train_steps,
                                                    pct_start=self.args.pct_start,
                                                    epochs=self.args.train_epochs,
                                                    max_lr=self.args.learning_rate)
                lr_adjust = {(epoch + 1): scheduler.get_last_lr()[0]}
                if epoch in lr_adjust.keys():
                    lr = lr_adjust[epoch]
                    for param_group in model_optim.param_groups:
                        param_group['lr'] = lr
                    print('Updating learning rate to {}'.format(lr))
                scheduler.step()
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            torch.cuda.empty_cache()

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        torch.save(self.model.state_dict(), best_model_path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)

        self.model.eval()
        metric_collector = create_metric_collector(device=self.device)
        criterion = self._select_criterion()
        total_loss = 0.0
        inputs, preds, trues = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and self.args.inverse:
                    out_np = outputs.cpu().numpy()
                    y_np   = batch_y.cpu().numpy()
                    shape = out_np.shape
                    inv_out = test_data.inverse_transform(out_np.squeeze(0)).reshape(shape)
                    inv_y   = test_data.inverse_transform(y_np.squeeze(0)).reshape(shape)
                    outputs = torch.from_numpy(inv_out).to(self.device)
                    batch_y = torch.from_numpy(inv_y).to(self.device)

                outputs = outputs[:, :, f_dim:].contiguous()
                batch_y = batch_y[:, :, f_dim:].contiguous()

                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                metric_collector.update(outputs, batch_y)

                if self.output_pred:
                    inputs.append(batch_x.cpu().numpy())
                    preds.append(outputs.cpu().numpy())
                    trues.append(batch_y.cpu().numpy())

        avg_test_loss = total_loss / len(test_loader)
        m = metric_collector.compute()
        mae, mse, rmse, mape, mspe = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"]

        if self.output_pred:
            inputs = np.array(inputs)
            preds  = np.array(preds)
            trues  = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
            preds  = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues  = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, mse, mae))

        data_dict = {'metrics': np.array([mae, mse, rmse, mape, mspe])}
        if self.output_pred:
            data_dict['input'] = inputs
            data_dict['pred']  = preds
            data_dict['true']  = trues
        np.savez_compressed(os.path.join(res_path, 'results.npz'), **data_dict)
        return

