#/usr/bin/env sh

NUM_RUNS=100
EXP_NAME=lstm_addition

mkdir "${EXP_NAME}_configs";
export PYTHONPATH=".";

# generate configs
python utils/create_config_files.py \
  --base_config experiments/addition/config.yml.example \
  --seeds 1 $NUM_RUNS \
  --change model "mclstm" \
  --change lr 5e-2 && \
rename config_ config1_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;
python utils/create_config_files.py \
  --base_config experiments/addition/config.yml.example \
  --seeds 1 $NUM_RUNS \
  --change model "lstm" \
  --change lr 1e-3 && \
rename config_ config2_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;
python utils/create_config_files.py \
  --base_config experiments/addition/config.yml.example \
  --seeds 1 $NUM_RUNS \
  --change model "nalu" \
  --change lr 1e-3 && \
rename config_ config3_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;
python utils/create_config_files.py \
  --base_config experiments/addition/config.yml.example \
  --seeds 1 $NUM_RUNS \
  --change model "nau" \
  --change lr 1e-2 && \
rename config_ config4_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;

# additional baselines
python utils/create_config_files.py \
  --base_config experiments/addition/config.yml.example \
  --seeds 1 $NUM_RUNS \
  --change model "lnlstm" \
  --change lr 1e-3 && \
rename config_ config5_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;
python utils/create_config_files.py \
  --base_config experiments/addition/config.yml.example \
  --seeds 1 $NUM_RUNS \
  --change model "urnn" \
  --change lr 1e-3 && \
rename config_ config6_ utils/generated_configs/*.yml && \
mv utils/generated_configs/*.yml "${EXP_NAME}_configs" && \
rm -d utils/generated_configs;

# train all models
export PYTHONPATH=".";
ls "${EXP_NAME}_configs"/*.yml | xargs -n1 -P10 -i -- \
python experiments/addition/train.py --config {}
mkdir "${EXP_NAME}_runs";
mv runs/* "${EXP_NAME}_runs";

# evaluate models
python experiments/addition/test.py --run_dir "${EXP_NAME}_runs" --experiment "mclstm*";
python experiments/addition/test.py --run_dir "${EXP_NAME}_runs" --experiment "lstm*";
python experiments/addition/test.py --run_dir "${EXP_NAME}_runs" --experiment "nalu*";
python experiments/addition/test.py --run_dir "${EXP_NAME}_runs" --experiment "nau*";
python experiments/addition/test.py --run_dir "${EXP_NAME}_runs" --experiment "lnlstm*";
python experiments/addition/test.py --run_dir "${EXP_NAME}_runs" --experiment "urnn*";
unset PYTHONPATH;