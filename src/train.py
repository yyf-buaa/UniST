import torch
from torch.optim import AdamW
import random
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import time
from tqdm import tqdm

class TrainLoop:
    def __init__(self, args, writer, model, data, test_data, val_data, device, early_stop = 5):
        self.args = args
        self.writer = writer
        self.model = model
        self.data = data
        self.test_data = test_data
        self.val_data = val_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=self.weight_decay)
        self.log_interval = args.log_interval
        self.best_rmse_random = 1e9
        self.warmup_steps=5
        self.min_lr = args.min_lr
        self.best_rmse = 1e9
        self.early_stop = early_stop
        
        self.mask_list = {'random':[0.5],'temporal':[0.5],'tube':[0.5],'block':[0.5]}


    def run_step(self, batch, step, mask_ratio, mask_strategy,index, name):
        self.opt.zero_grad()
        loss, num, loss_real, num2 = self.forward_backward(batch, step, mask_ratio, mask_strategy,index=index, name = name)

        self._anneal_lr()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
        self.opt.step()
        return loss, num, loss_real, num2

    def Sample(self, test_data, step, mask_ratio, mask_strategy, seed=None, dataset='', index=0, Type='val'):
        target_1_list = []
        target_2_list = []
        pred_1_list = []
        pred_2_list = []
        with torch.no_grad():
            error_ch1, error_ch2, error_mae_ch1, error_mae_ch2, error_norm, num, num_1, num_2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            pred_fire_num = 0
            target_fire_num = 0
            for _, batch in tqdm(enumerate(test_data[index])):
                
                loss, _, pred, target, mask = self.model_forward(batch, self.model, mask_ratio, mask_strategy, seed=seed, data = dataset, mode='forward')

                pred = torch.clamp(pred, min=-1, max=1)
                pred_ch1 = pred[..., ::2] 
                pred_ch2 = pred[..., 1::2] 

                target_ch1 = target[..., ::2]
                target_ch2 = target[..., 1::2]

                pred_mask_ch1 = pred_ch1[mask == 1].reshape(-1, 1).detach().cpu().numpy()
                pred_mask_ch2 = pred_ch2[mask == 1].reshape(-1, 1).detach().cpu().numpy()

                target_mask_ch1 = target_ch1[mask == 1].reshape(-1, 1).detach().cpu().numpy()
                target_mask_ch2 = target_ch2[mask == 1].reshape(-1, 1).detach().cpu().numpy()

                
                scaler1 = self.args.scaler[dataset][0]  # 第一个通道的 scaler
                scaler2 = self.args.scaler[dataset][1]  # 第二个通道的 scaler

                pred_inv_ch1 = scaler1.inverse_transform(pred_mask_ch1)
                target_inv_ch1 = scaler1.inverse_transform(target_mask_ch1)

                pred_inv_ch2 = scaler2.inverse_transform(pred_mask_ch2)
                target_inv_ch2 = scaler2.inverse_transform(target_mask_ch2)

                # target_list.append(target_inv_ch1)
                # target_list.append(target_inv_ch2)
                pred_1_list.append(pred_inv_ch1)
                pred_2_list.append(pred_inv_ch2)
                target_1_list.append(target_inv_ch1)
                target_2_list.append(target_inv_ch2)
                #筛选大于 0.001 的样本
                mask_1 = (target_inv_ch1 > 0.01).flatten()
                pred_inv_ch1 = pred_inv_ch1[mask_1]
                target_inv_ch1 = target_inv_ch1[mask_1]

                mask_2 = (target_inv_ch2 > 0.01).flatten()
                pred_inv_ch2 = pred_inv_ch2[mask_2]
                target_inv_ch2 = target_inv_ch2[mask_2]

                pred_fire_num += (pred_inv_ch2 > 0.01).flatten().sum()
                target_fire_num += (target_inv_ch2 > 0.01).flatten().sum()
                if len(pred_inv_ch1) == 0 or len(pred_inv_ch2) == 0:
                    continue

                error_ch1 += mean_squared_error(target_inv_ch1, pred_inv_ch1, squared=True) * mask_1.sum().item()
                error_mae_ch1 += mean_absolute_error(target_inv_ch1, pred_inv_ch1) * mask_1.sum().item()
                error_ch2 += mean_squared_error(target_inv_ch2, pred_inv_ch2, squared=True) * mask_1.sum().item()
                error_mae_ch2 += mean_absolute_error(target_inv_ch2, pred_inv_ch2) * mask_2.sum().item()

                error_norm += loss.item() * mask.sum().item()
                num += mask.sum().item()
                num_1 += mask_1.sum().item()
                num_2 += mask_2.sum().item()

        rmse_ch1 = np.sqrt(error_ch1 / num_1)
        rmse_ch2 = np.sqrt(error_ch2 / num_2)
        mae_ch1 = error_mae_ch1 / num_1
        mae_ch2 = error_mae_ch2 / num_2
        loss_test = error_norm / num
        print(f'pred_fire_num={pred_fire_num}, target_fire_num={target_fire_num}')
        return (rmse_ch1, rmse_ch2), (mae_ch1, mae_ch2), loss_test, np.concatenate(pred_1_list, axis=0), np.concatenate(pred_2_list, axis=0), np.concatenate(target_1_list, axis=0), np.concatenate(target_2_list, axis=0)


    def Evaluation(self, test_data, epoch, seed=None, best=True, Type='val'):
        old_mask_strategy_random = self.args.mask_strategy_random
        old_mask_strategy = self.args.mask_strategy
        old_mask_ratio = self.args.mask_ratio
        if Type == 'test' or Type == 'val':
            self.args.mask_strategy_random = 'none'
            self.args.mask_strategy = 'temporal'
            self.args.mask_ratio = (self.args.pred_len+0.0) / (self.args.pred_len+self.args.his_len)
        loss_list = []

        rmse_list = []
        rmse_key_result = {}
        target = []
        pred = []
        for index, dataset_name in enumerate(self.args.dataset.split('*')):

            rmse_key_result[dataset_name] = {}

            if self.args.mask_strategy_random != 'none':
                for s in self.mask_list:
                    for m in self.mask_list[s]:
                        result, mae, loss_test, pred_1, pred_2, target_1, target_2 = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy = s, seed=seed, dataset = dataset_name, index=index, Type=Type)
                        rmse_list.append(result)
                        loss_list.append(loss_test)
                        if s not in rmse_key_result[dataset_name]:
                            rmse_key_result[dataset_name][s] = {}
                        rmse_key_result[dataset_name][s][m] = result
                        
                        if Type == 'val':
                            self.writer.add_scalar('Evaluation_RMSE/{}-{}-{}-prob'.format(dataset_name.split('_C')[0], s, m), result[0], epoch)
                            self.writer.add_scalar('Evaluation_RMSE/{}-{}-{}-brightness'.format(dataset_name.split('_C')[0], s, m), result[1], epoch)
                        elif Type == 'test':
                            self.writer.add_scalar('Test_RMSE/{}-{}-{}-prob'.format(dataset_name.split('_C')[0], s, m), result[0], epoch)
                            self.writer.add_scalar('Test_RMSE/{}-{}-{}-brightness'.format(dataset_name.split('_C')[0], s, m), result[1], epoch)
                            self.writer.add_scalar('Test_MAE/MAE-{}-{}-{}-prob'.format(dataset_name.split('_C')[0], s, m), mae[0], epoch)
                            self.writer.add_scalar('Test_MAE/MAE-{}-{}-{}-brightness'.format(dataset_name.split('_C')[0], s, m), mae[1], epoch)

            else:
                s = self.args.mask_strategy
                m = self.args.mask_ratio
                result, mae,  loss_test, pred_1, pred_2, target_1, target_2 = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy = s, seed=seed, dataset = dataset_name, index=index, Type=Type)
                rmse_list.append(result)
                loss_list.append(loss_test)
                if s not in rmse_key_result[dataset_name]:
                    rmse_key_result[dataset_name][s] = {}
                rmse_key_result[dataset_name][s][m] = {'rmse':result, 'mae':mae}
                
                if Type == 'val':
                    self.writer.add_scalar('Evaluation/{}-{}-{}-prob'.format(dataset_name.split('_C')[0], s, m), result[0], epoch)
                    self.writer.add_scalar('Evaluation/{}-{}-{}-brightness'.format(dataset_name.split('_C')[0], s, m), result[1], epoch)
                elif Type == 'test':
                    self.writer.add_scalar('Test_RMSE/Stage-{}-{}-{}-{}-prob'.format(self.args.stage, dataset_name.split('_C')[0], s, m), result[0], epoch)
                    self.writer.add_scalar('Test_RMSE/Stage-{}-{}-{}-{}-brightness'.format(self.args.stage, dataset_name.split('_C')[0], s, m), result[1], epoch)
                    self.writer.add_scalar('Test_MAE/Stage-MAE-{}-{}-{}-{}-prob'.format(self.args.stage, dataset_name.split('_C')[0], s, m), mae[0], epoch)
                    self.writer.add_scalar('Test_MAE/Stage-MAE-{}-{}-{}-{}-brightness'.format(self.args.stage, dataset_name.split('_C')[0], s, m), mae[1], epoch)
        
        loss_test = np.mean(loss_list)
        if Type == 'test' or Type == 'val':
            self.args.mask_strategy_random = old_mask_strategy_random 
            self.args.mask_strategy = old_mask_strategy
            self.args.mask_ratio = old_mask_ratio
        if best:
            is_break = self.best_model_save(epoch, loss_test, rmse_key_result, pred_1, pred_2, target_1, target_2)
            return is_break

        else:
            return loss_test, rmse_key_result

    def best_model_save(self, step, rmse, rmse_key_result, pred_1, pred_2, target_1, target_2):
        if rmse < self.best_rmse:
            self.early_stop = 0
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best_stage_{}.pkl'.format(self.args.stage))
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best.pkl')
            np.save(self.args.model_path+'model_save/pred_1.npy', pred_1)
            np.save(self.args.model_path+'model_save/pred_2.npy', pred_2)
            np.save(self.args.model_path+'model_save/target_1.npy',target_1)
            np.save(self.args.model_path+'model_save/target_2.npy',target_2)
            self.best_rmse = rmse
            self.writer.add_scalar('Evaluation/RMSE_best', self.best_rmse, step)
            print('\nRMSE_best:{}\n'.format(self.best_rmse))
            print(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result.txt', 'w') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, best rmse: {}\n'.format(self.args.stage, step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            return 'save'

        else:
            self.early_stop += 1
            print('\nRMSE:{}, RMSE_best:{}, early_stop:{}\n'.format(rmse, self.best_rmse, self.early_stop))
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('RMSE:{}, not optimized, early_stop:{}\n'.format(rmse, self.early_stop))
            if self.early_stop >= self.args.early_stop:
                print('Early stop!')
                with open(self.args.model_path+'result.txt', 'a') as f:
                    f.write('Early stop!\n')
                with open(self.args.model_path+'result_all.txt', 'a') as f:
                    f.write('Early stop!\n')
                    exit()

    def mask_select(self):
        if self.args.mask_strategy_random == 'none': #'none' or 'batch'
            mask_strategy = self.args.mask_strategy
            mask_ratio = self.args.mask_ratio
        else:
            mask_strategy=random.choice(['random','temporal','tube','block'])
            mask_ratio=random.choice(self.mask_list[mask_strategy])

        return mask_strategy, mask_ratio

    def run_loop(self):
        step = 0
        if self.args.mode == 'testing':
            self.Evaluation(self.val_data, 0, best=True, Type='val')
            exit()
        
        self.Evaluation(self.val_data, 0, best=True, Type='val')
        
        for epoch in range(self.args.total_epoches):
            print('Training')

            self.step = epoch
            
            loss_all, num_all, loss_real_all, num_all2 = 0.0, 0.0,0.0, 0.0
            start = time.time()
            for name, batch in tqdm(self.data):
                mask_strategy, mask_ratio = self.mask_select()
                loss, num, loss_real, num2  = self.run_step(batch, step, mask_ratio=mask_ratio, mask_strategy = mask_strategy,index=0, name = name)
                step += 1
                loss_all += loss * num
                #loss_real_all += loss_real * num
                num_all += num
                num_all2 += num2
            
            end = time.time()
            print('training time:{} min'.format(round((end-start)/60.0,2)))
            print('epoch:{}, training loss:{}'.format(epoch, loss_all / num_all))
            self.writer.add_scalar('Training/Loss_epoch', loss_all / num_all, step)
            if epoch % self.log_interval == 0 and epoch > 0 or epoch == 10 or epoch == self.args.total_epoches-1:
                print('Evaluation')
                eval_result = self.Evaluation(self.val_data, epoch, best=True, Type='val')

                if eval_result == 'save':
                    print('test evaluate!')
                    rmse_test, rmse_key_test = self.Evaluation(self.test_data, epoch, best=False, Type='test')
                    print('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                    print(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result_all.txt', 'a') as f:
                        f.write('stage:{}, epoch:{}, test rmse: {}\n'.format(self.args.stage, epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')

    def model_forward(self, batch, model, mask_ratio, mask_strategy, seed=None, data=None, mode='backward'):

        batch = [i.to(self.device) for i in batch]

        loss, loss2, pred, target, mask = self.model(
                batch,
                mask_ratio=mask_ratio,
                mask_strategy = mask_strategy, 
                seed = seed, 
                data = data,
                mode = mode, 
            )
        return loss, loss2, pred, target, mask 

    def forward_backward(self, batch, step, mask_ratio, mask_strategy,index, name=None):

        loss, _, pred, target, mask = self.model_forward(batch, self.model, mask_ratio, mask_strategy, data=name, mode='backward')

        pred_mask = pred.squeeze(dim=2)[mask==1]
        target_mask = target.squeeze(dim=2)[mask==1]
        #loss_real = mean_squared_error(self.args.scaler[name].inverse_transform(pred_mask.reshape(-1,1).detach().cpu().numpy()), self.args.scaler[name].inverse_transform(target_mask.reshape(-1,1).detach().cpu().numpy()), squared=True)
    
        loss.backward()

        self.writer.add_scalar('Training/Loss_step', loss, step)
        return loss.item(), mask.sum().item(), None, (1-mask).sum().item()

    def _anneal_lr(self):
        if self.step < self.warmup_steps:
            lr = self.lr * (self.step+1) / self.warmup_steps
        elif self.step < self.lr_anneal_steps:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (self.step - self.warmup_steps)
                    / (self.lr_anneal_steps - self.warmup_steps)
                )
            )
        else:
            lr = self.min_lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        self.writer.add_scalar('Training/LR', lr, self.step)
        return lr

