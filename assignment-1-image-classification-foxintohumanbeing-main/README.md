# assignment-1-image-classification-foxintohumanbeing
assignment-1-image-classification-foxintohumanbeing created by GitHub Classroom

Author: Huihan YANG - 120090438

## EX1

* The configuration file `mobilenet_v3_flower_120090438_upload.py` should be put under the configs file in mmclassification.

* To run my model, please first switch the path to EX1/mmclassification, then run
 `python tools/train.py configs/mobilenet_v3_flower_120090438_upload.py`

* The original configuration file is in `EX1/mmclassification/configs/mobilenet_v3_large-3ea3c186.pth`, which can also be download.

* The trained model is stored in `EX1/mmclassification/work_dirs/flower_dataset/EX1_best.pth`, which is submitted through bb.

## EX2

* The complete code with original model is in `main.py`.

* The code with improved model is in `main_improvement.py`.

* The best models are submitted through bb. File `EX2_best.pt` stores the best training model with original model and parameters. `EX2_best_improved` stores the best training model with modified model and tuned parameters.

## utils

* This folder stores the code to organize the data into the required format.

## Report

* Please the note that the report is in DDA4220_Report_1_120090438_UG.pdf