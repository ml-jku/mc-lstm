#/usr/bin/env sh

EXP_NAME=pe

mkdir "${EXP_NAME}_configs";

# generate configs
python utils/create_config_files.py \
  --base_config experiments/pendulum/config.yml.example \
  --change modeltype "MC-LSTM" --change modeltype "AR-LSTM" \
  --change dampening_constant 0.0 --change dampening_constant 0.1 --change dampening_constant 0.2 \
  --change dampening_constant 0.4 --change dampening_constant 0.8 \
  --change initial_amplitude 0.2 --change initial_amplitude 0.4 \
  --change pendulum_length 0.75 --change pendulum_length 1.0 \
  --change train_seq_length 100 --change train_seq_length 200 --change train_seq_length 400 \
  --change noise_std 0.0 --change noise_std 0.01 && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;

# train models
python experiments/pendulum/main.py --config-dir "${EXP_NAME}_configs";
mkdir "${EXP_NAME}_runs";
mv runs/* "${EXP_NAME}_runs";