import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):

    # switch to train mode
    model.train() 
 
    loss_list = []
    total_iters = len(train_loader)

    # 使用tqdm显示进度
    pbar = tqdm(total=total_iters, desc=f'Epoch {epoch} Training')
    
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    accumulation_step = 0
    
    for iter, data in enumerate(train_loader):
        # 适配MAMA-MIA数据集格式 (image, mask, meta)
        if len(data) == 3:
            images, targets, meta = data
            # 【新增】如果不需要meta，可以立即释放
            del meta
        else:
            images, targets = data
            
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets) / gradient_accumulation_steps  
            scaler.scale(loss).backward()
        else:
            out = model(images)
            loss = criterion(out, targets) / gradient_accumulation_steps  
            loss.backward()
        
        # 这些张量已经完成了前向传播和梯度计算
        del images, targets, out
        
        accumulation_step += 1
        

        if accumulation_step % gradient_accumulation_steps == 0:
            if config.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)  
            accumulation_step = 0
            

            if iter % 20 == 0:
                torch.cuda.empty_cache()
        
        loss_list.append(loss.item() * gradient_accumulation_steps) 
        

        del loss

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = np.mean(loss_list)
        

        if torch.cuda.is_available() and iter % 10 == 0:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{current_lr:.6f}',
                'GPU Mem': f'{memory_allocated:.2f}GB'
            })
        else:
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
        pbar.update(1)

        # 定期打印日志
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}/{total_iters}, loss: {avg_loss:.4f}, lr: {current_lr:.6f}'
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                log_info += f', GPU Mem: {memory_allocated:.2f}GB'
            print(log_info)
            logger.info(log_info)
            
    pbar.close()
    

    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # 每个epoch结束后更新学习率
    scheduler.step() 
    
    epoch_loss = np.mean(loss_list)
    print(f'Epoch {epoch} Training - Average Loss: {epoch_loss:.4f}')
    
    return epoch_loss


def val_one_epoch(test_loader,
                  model,
                  criterion, 
                  epoch, 
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    
    print(f'Validating...')
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), desc=f'Epoch {epoch} Validation')
        for i, data in enumerate(test_loader):
            # 适配MAMA-MIA数据集格式
            if len(data) == 3:
                img, msk, meta = data  # 忽略meta数据
                del meta  
            else:
                img, msk = data
                
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            
            # 【新增】记录loss后立即清理中间张量
            loss_value = loss.item()
            loss_list.append(loss_value)
            del loss
            
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out_np = out.squeeze(1).cpu().detach().numpy()
            preds.append(out_np)

            del img, msk, out
            
            if i % 20 == 0:
                torch.cuda.empty_cache()
                
            pbar.update(1)
        pbar.close()

    avg_loss = np.mean(loss_list)
    
    # 计算指标
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds>=config.threshold, 1, 0)
    y_true = np.where(gts>=0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    log_info = f'val epoch: {epoch}, loss: {avg_loss:.4f}, dice: {f1_or_dsc:.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}'
    print(f'Validation Results: {log_info}')
    logger.info(log_info)
    
    print(f'Confusion Matrix:\n{confusion}')
    
    del preds, gts, confusion
    torch.cuda.empty_cache()
    
    return avg_loss, f1_or_dsc


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    
    print(f'Testing...')
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), desc='Testing')
        for i, data in enumerate(test_loader):
            # 适配MAMA-MIA数据集格式
            if len(data) == 3:
                img, msk, meta = data  # 保留meta用于保存
            else:
                img, msk = data
                meta = None
                
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            
            # 适配MAMA-MIA数据保存
            if meta is not None:
                # 处理meta数据格式
                if isinstance(meta, dict):
                    patient_id = meta.get('patient_id', ['unknown'])[0] if isinstance(meta.get('patient_id'), list) else meta.get('patient_id', 'unknown')
                    slice_idx = meta.get('slice_idx', [i])[0] if isinstance(meta.get('slice_idx'), list) else meta.get('slice_idx', i)
                else:
                    patient_id = 'unknown'
                    slice_idx = i
                    
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, 
                         test_data_name=test_data_name, patient_id=patient_id, slice_idx=slice_idx)
            else:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, 
                         test_data_name=test_data_name)
            
            pbar.update(1)
        pbar.close()

    avg_loss = np.mean(loss_list)
    
    # 计算指标
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds>=config.threshold, 1, 0)
    y_true = np.where(gts>=0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)
        
    log_info = f'test of best model, loss: {avg_loss:.4f}, dice: {f1_or_dsc:.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}'
    print(f'Test Results: {log_info}')
    logger.info(log_info)
    
    print(f'Confusion Matrix:\n{confusion}')


    return avg_loss
