# AdvStyle on image classification across domains

## How to install

This code is based on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation to install `dassl`.

## How to run

Please follow the steps below before running the script

- modify `DATA` and `DASSL` in `*.sh` based on the paths on your computer
- activate the `dassl` environment via `conda activate dassl`
- `cd` to `scripts/`


### Domain Generalization
```bash
# PACS | MixStyle
cd .scripts
bash run.sh
```

To extract features (or feature statistics) for analysis, you can add `--vis` to the input arguments (also specify `--model-dir` and `--load-epoch`), which will run `trainer.vis()` (implemented in `trainers/vanilla2.py`). However, you need to make changes in several places to make this work, e.g., you need to modify the model's code such that the model directly outputs features. `trainer.vis()` will save the extracted features to `embed.pt`. To visualize the features, you can use `vis.py` (please see the code for more details).



## How to cite

If you find this code useful to your research, please cite the following papers.

```
@article{zhou2020domain,
  title={Domain Adaptive Ensemble Learning},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={arXiv preprint arXiv:2003.07325},
  year={2020}
}

@article{zhang2023adversarial,
  title={Adversarial Style Augmentation for Domain Generalization},
  author={Zhang, Yabin and Deng, Bin and Li, Ruihuang and Jia, Kui and Zhang, Lei},
  journal={arXiv preprint arXiv:2301.12643},
  year={2023}
}
```