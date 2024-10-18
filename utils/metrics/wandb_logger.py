import wandb

def wandb_logger(epoch,
                 mode,
                 recorder,
                 metrics_dict,
                 loss_dict,
                 lr,
                 mapped_cls_name, trainer_global_steps):

    # Metrics
    mean_acc, class_acc = metrics_dict['mean_acc'], metrics_dict['class_acc']
    mean_recall, class_recall = metrics_dict['mean_recall'], metrics_dict['class_recall']
    mean_iou, class_iou = metrics_dict['mean_iou'], metrics_dict['class_iou']
    
    # Losses
    loss_meter_avg = loss_dict['loss_meter_avg']
    loss_focal = loss_dict['loss_focal']
    loss_lovasz = loss_dict['loss_lovasz']

    # Log scalars with wandb for general metrics and losses
    recorder.wandb_logger.log({
        f'{mode}/Loss': loss_meter_avg,
        f'{mode}/LossSoftmax': loss_focal.item(),
        f'{mode}/LossLovasz': loss_lovasz.item(),
        f'{mode}/meanAcc': mean_acc.item(),
        f'{mode}/meanIOU': mean_iou.item(),
        f'{mode}/meanRecall': mean_recall.item(),
        f'{mode}/lr': lr,
        'epoch': epoch
    }, step=trainer_global_steps)

    # Log class-specific metrics
    for i, (_, class_name) in enumerate(mapped_cls_name.items()):
        recorder.wandb_logger.log({
            f'{mode}/Acc/{class_name}_{i:02d}': class_acc[i].item(),
            f'{mode}/Recall/{class_name}_{i:02d}': class_recall[i].item(),
            f'{mode}/IOU/{class_name}_{i:02d}': class_iou[i].item()
        }, step=trainer_global_steps)
