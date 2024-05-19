"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B # TODO rename to num_boxes
        self.C = C # TODO rename to num_classes

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        box_predictions = predictions[..., self.C:].reshape(-1, self.S, self.S, self.B, 5)
        #print(predictions.shape, box_predictions.shape)
        #print(torch.equal(predictions[..., (self.C + 1):(self.C +5)], box_predictions[...,0,1:5])) 
        #print(torch.equal(predictions[..., (self.C + 6):(self.C +10)], box_predictions[...,1,1:5])) 
        

        box_target = target[..., self.C:].reshape(-1, 7, 7, self.B, 5)
        print("box_target shape: ", box_target.shape)
        print("new input 1:", box_predictions[..., 0, 1:].shape)
        print("new input 2:", box_target[..., 0, 1:].shape)

        new_iou1 = intersection_over_union(box_predictions[..., 0, 1:], box_target[..., 0, 1:])
        print("new iou output shape:", new_iou1.shape)
        #quit()
        iou = torch.cat(
            [
                intersection_over_union(
                    box_predictions[..., i, 1:], 
                    box_target[..., 0, 1:]
                    ).unsqueeze(0)
                for i in range(self.B)
            ],
            dim = 0
            )

        best_iou, best_box = torch.max(iou, dim = 0)
        print("new ious shape:", iou.shape)
        
        iou_b1 = intersection_over_union(predictions[..., (self.C + 1):(self.C +5)], target[..., (self.C + 1):(self.C +5)])
        print("old iou1 output shape: ",iou_b1.shape)
        print("bbox 1 iou: ", torch.equal(iou_b1, new_iou1)) 


        #iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        #ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        #print("old iou shape:", ious.shape)
        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(iou, dim=0)
        exists_box = target[..., self.C].unsqueeze(3)  # in paper this is Iobj_i
        #print(torch.equal(iou, ious)) 
        quit()
        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
    
