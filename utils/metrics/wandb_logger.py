import wandb

def wandb_logger(epoch,
                 mode,
                 recorder,
                 metrics_dict,
                 loss_dict,
                 lr,
                 mapped_cls_name):

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
        f'{mode}_Loss': loss_meter_avg,
        f'{mode}_LossSoftmax': loss_focal.item(),
        f'{mode}_LossLovasz': loss_lovasz.item(),
        f'{mode}_meanAcc': mean_acc.item(),
        f'{mode}_meanIOU': mean_iou.item(),
        f'{mode}_meanRecall': mean_recall.item(),
        f'{mode}_lr': lr
    }, step=epoch)

    # Log class-specific metrics
    for i, (_, class_name) in enumerate(mapped_cls_name.items()):
        recorder.wandb_logger.log({
            f'{mode}_{i:02d}_{class_name}_Acc': class_acc[i].item(),
            f'{mode}_{i:02d}_{class_name}_Recall': class_recall[i].item(),
            f'{mode}_{i:02d}_{class_name}_IOU': class_iou[i].item()
        }, step=epoch)
