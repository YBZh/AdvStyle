# AdvStyle on cross-dataset person re-identification

## How to install

This code is based on [Torchreid](https://arxiv.org/abs/1910.10093). Please follow the instructions at https://github.com/KaiyangZhou/deep-person-reid#installation to install `torchreid`.

## How to run

The running commands are provided in `run.sh`. You need to activate the `torchreid` environment using `conda activate torchreid` before running the code. See https://github.com/KaiyangZhou/deep-person-reid#a-unified-interface for more details on how to train and test a model.

## How to cite

If you find this code useful to your research, please cite the following papers.

```
@article{torchreid,
  title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
  author={Zhou, Kaiyang and Xiang, Tao},
  journal={arXiv preprint arXiv:1910.10093},
  year={2019}
}

@inproceedings{zhou2019osnet,
  title={Omni-Scale Feature Learning for Person Re-Identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  booktitle={ICCV},
  year={2019}
}

@article{zhang2023adversarial,
  title={Adversarial Style Augmentation for Domain Generalization},
  author={Zhang, Yabin and Deng, Bin and Li, Ruihuang and Jia, Kui and Zhang, Lei},
  journal={arXiv preprint arXiv:2301.12643},
  year={2023}
}
```