from __future__ import  absolute_import
import os
import time

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset_PIPE, TestDataset_PIPE, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

import numpy as np
import json
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
# import resource

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)[:, np.newaxis, np.newaxis]
        self.std = np.array(std)[:, np.newaxis, np.newaxis]

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
            # for t, m, s in zip(tensor, self.mean, self.std):
            #     t.mul_(s).add_(m)
            #     # The normalize code -> t.sub_(m).div_(s)
        tensor = tensor * self.std + self.mean
        return tensor

def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    # print(pred_bboxes, pred_labels, pred_scores)
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=False)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset_PIPE(opt, split = 'train')
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    valset = TestDataset_PIPE(opt, split='val')
    val_dataloader = data_.DataLoader(valset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False
                                       # pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    early_stop_count = 0

    with open(os.path.join(opt.pipe_data_dir, 'stds.json'), 'r') as f:
            mnstds = json.load(f)
    unorm = UnNormalize(mean=mnstds[0], std=mnstds[1])

    for epoch in range(opt.epoch):
        print(f'training epoch {epoch}')
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = unorm(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict(
                    # [ori_img_],
                    img,
                    visualize=True,
                    pipe=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(val_dataloader, faster_rcnn, test_num=opt.test_num)
        # print(eval_result)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        early_stop_count += 1
        print(f"Epoch {epoch} eval_result{eval_result['map']}")
        if eval_result['map'] > best_map:
            early_stop_count = 0
            best_map = eval_result['map']

            timestr = time.strftime('%m%d%H%M')
            save_str = 'fasterrcnn_%s' % timestr
            save_path = os.path.join(opt.save_path, save_str)
            print('save best map model')
            best_path = trainer.save(save_optimizer=True,
                                     save_path=save_path,
                                     best_map=best_map)
        if early_stop_count >= opt.early_stop:
            break

        if epoch % opt.step_size == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay


if __name__ == '__main__':
    import fire

    fire.Fire()
