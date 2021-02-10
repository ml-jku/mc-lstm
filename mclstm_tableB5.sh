#/usr/bin/env sh

NUM_RUNS=50
EXP_NAME=traffic4cast

export PYTHONPATH=".";
mkdir "${EXP_NAME}_configs";

# generate configs
python utils/create_config_files.py \
  --base_config experiments/traffic4cast/config.yml.example \
  --change city berlin \
  --change model "continuouslstm" \
  --change hidden_size 10 \
  --change lr 1e-2 \
  --change initial_forget_bias 0 \
  --change normalised true \
  --seeds 1 $NUM_RUNS && \
rename config_ config_berlin1_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;
python utils/create_config_files.py \
  --base_config experiments/traffic4cast/config.yml.example \
  --change city berlin \
  --change model "continuousdirectmclstm" \
  --change hidden_size 100 \
  --change lr 1e-2 \
  --change initial_state 0 \
  --change learn_initial_state true \
  --seeds 1 $NUM_RUNS && \
rename config_ config_berlin2_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;
python utils/create_config_files.py \
  --base_config experiments/traffic4cast/config.yml.example \
  --change city istanbul \
  --change model "continuouslstm" \
  --change hidden_size 100 \
  --change lr 5e-3 \
  --change initial_forget_bias 5 \
  --change normalised true \
  --seeds 1 $NUM_RUNS && \
rename config_ config_istanbul1_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;
python utils/create_config_files.py \
  --base_config experiments/traffic4cast/config.yml.example \
  --change city istanbul \
  --change model "continuousdirectmclstm" \
  --change hidden_size 50 \
  --change lr 1e-2 \
  --change initial_state 0 \
  --seeds 1 $NUM_RUNS && \
rename config_ config_istanbul2_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;

python utils/create_config_files.py \
  --base_config experiments/traffic4cast/config.yml.example \
  --change city moscow \
  --change model "continuouslstm" \
  --change hidden_size 50 \
  --change lr 1e-3 \
  --change initial_forget_bias 5 \
  --change normalised true \
  --seeds 1 $NUM_RUNS && \
rename config_ config_moscow1_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;
python utils/create_config_files.py \
  --base_config experiments/traffic4cast/config.yml.example \
  --change city moscow \
  --change model "continuousdirectmclstm" \
  --change hidden_size 10 \
  --change lr 1e-2 \
  --change initial_state 0 \
  --seeds 1 $NUM_RUNS && \
rename config_ config_moscow2_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;

# train all models
ls "${EXP_NAME}_configs"/*.yml | xargs -n1 -P10 -i -- \
python experiments/traffic4cast/train.py --config {};
mkdir "${EXP_NAME}_runs";
mv runs/* "${EXP_NAME}_runs";

# evaluate models
python experiments/traffic4cast/test.py --run_dir "${EXP_NAME}_runs" --experiment "berlin_continuouslstm*";
python experiments/traffic4cast/test.py --run_dir "${EXP_NAME}_runs" --experiment "berlin_continuousdirectmclstm*";
python experiments/traffic4cast/test.py --run_dir "${EXP_NAME}_runs" --experiment "istanbul_continuouslstm*";
python experiments/traffic4cast/test.py --run_dir "${EXP_NAME}_runs" --experiment "istanbul_continuousdirectmclstm*";
python experiments/traffic4cast/test.py --run_dir "${EXP_NAME}_runs" --experiment "moscow_continuouslstm*";
python experiments/traffic4cast/test.py --run_dir "${EXP_NAME}_runs" --experiment "moscow_continuousdirectmclstm*";
unset PYTHONPATH;