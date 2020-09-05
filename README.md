# Saliency based Adversarial Training (SAT)
Code for ECML-PKDD 2020 Paper: "On Saliency Maps and Adversarial Robustness"

### Usage:
Train Teacher: ```python3 teacher.py --dataset [cifar10/cifar100] --model [ResNet34/ResNet10] --method [adv
std] --bs [128] [--resume]```

Train Student: ```python3 student.py --dataset [cifar10/cifar100] --model [ResNet34/ResNet10] --method [adv
std] --bs [128] [--resume] --teacher [ResNet34_std/ResNet34_adv] --exp [gcam++/gbp,sgrad]```

*Note: There are some comments in file thats need to be uncommented for certain uses. Please go through them before running. More organised code to be followed soon.*

### Files:
- ```teacher.py```: Training a teacher network in both non-robust or robust fashion.
- ```student.py```: Training student network guided by teacher
- ```student_adv.py```: Training student network adversarially guided by teacher
- ```student_ensemble.py```: Training student network guided by saliency maps of two teacher
- ```teacher_[nadv/ntrades/trades/noise].py```: Different training of teacher networks
- ```student_[bbox/bbox_adv/bbox_trades].py```: specially for Tiny-imagenet and flower datasets where bounding boxes and segmentation masks are already available.
