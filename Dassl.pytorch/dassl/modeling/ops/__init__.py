from .mmd import MaximumMeanDiscrepancy
from .dsbn import DSBN1d, DSBN2d
from .mixup import mixup
from .mixstyle import (
    MixStyle, random_mixstyle, activate_mixstyle, run_with_mixstyle,
    deactivate_mixstyle, crossdomain_mixstyle, run_without_mixstyle
)
from .mixstyle_y import (
    MixStyleY, deactivate_training_weight_y, activate_training_weight_y, random_mixstyle_y, activate_mixstyle_y, run_with_mixstyle_y,
    deactivate_mixstyle_y, crossdomain_mixstyle_y, run_without_mixstyle_y
)
from .advstyle import AdvStyle, reset_init_flag
from .advstyle_test import AdvStyle_test
from .dsustyle import DSUStyle
from .dsustyle_test import DSUStyle_test
from .mixstyle_2w import MixStyle2w
from .mixstyle_A import MixStyleA
from .efdmix import EFDMix, quicksort_mixstyle, index_mixstyle, randomsort_mixstyle, neighbor_mixstyle
from .mixhistogram import MixHistogram
from .mixorders import MixOrders, first_order, second_order, first_second_order, all_order  #, third_order, fourth_order, first_second_three_order, first_second_three_four_order
from .randstyle import RandStyle
from .ign import IGN
from .transnorm import TransNorm1d, TransNorm2d
from .sequential2 import Sequential2
from .reverse_grad import ReverseGrad
from .cross_entropy import cross_entropy
from .optimal_transport import SinkhornDivergence, MinibatchEnergyDistance
from .cnsn import CNSN
from .snr import SNR
