#/usr/bin/env sh

export PYTHONPATH=".";

# train mclstm with sum instead of fc
python utils/create_config_files.py \
  --base_config experiments/addition/config.yml.example \
  --change model "sum_mclstm" \
  --change lr 5e-2 \
  --change num_epochs 50;
python experiments/addition/train.py --config utils/generated_configs/config_1.yml;
rm utils/generated_configs/config_1.yml;
rm -d utils/generated_configs/;

# create plot
python - <<END
from experiments.addition.test import plot_cell_states
from experiments.utils import read_config
from pathlib import Path
import torch

path, = Path("runs").glob("sum_mclstm_*")
cfg = read_config(path / "config.yml")
chkpt = torch.load(path / "model_epoch050.pt", map_location="cpu")
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 12})
fig = plot_cell_states(cfg, chkpt)
fig.gca().set_ylim(-.1, 3.1)
fig.savefig("analysis_lstm_addition.pdf")
END;
rm -r runs/sum_mclstm_*;
unset PYTHONPATH;