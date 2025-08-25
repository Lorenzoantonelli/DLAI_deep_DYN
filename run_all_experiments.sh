#!/usr/bin/env bash

file_name="test.csv"

# DYN models
uv run src/tester.py --model DualStreamCAE --checkpoint ./Checkpoints/dual_waveform_1_waveform.pt --save_metrics_path ./$file_name
uv run src/tester.py --model DualStreamCAE --checkpoint ./Checkpoints/dual_waveform_1_waveform_sample_gain.pt --save_metrics_path ./$file_name
uv run src/tester.py --model DualStreamCAESharedEncoder --checkpoint ./Checkpoints/shared_encoder_2_waveform.pt --save_metrics_path ./$file_name
uv run src/tester.py --model DualStreamCAESharedEncoder --checkpoint ./Checkpoints/shared_encoder_2_waveform_sample_gain.pt --save_metrics_path ./$file_name
uv run src/tester.py --model DualStreamCAE_ShareEncDec --checkpoint ./Checkpoints/all_shared_3_waveform.pt --save_metrics_path ./$file_name
uv run src/tester.py --model DualStreamCAE_ShareEncDec --checkpoint ./Checkpoints/all_shared_3_waveform_sample_gain.pt --save_metrics_path ./$file_name
uv run src/tester.py --model DualStreamCAE_TwoEnc_DirectX --checkpoint ./Checkpoints/dual_encoder_waveform_only_4.pt --save_metrics_path ./$file_name
uv run src/tester.py --model DualStreamCAE_ShareEnc_DirectX --checkpoint ./Checkpoints/shared_encoder_waveform_only_5.pt --save_metrics_path ./$file_name

# baselines
uv run src/tester.py --model BaselineCAE --checkpoint ./Checkpoints/baseline_16bit_6.pt --save_metrics_path ./$file_name
uv run src/tester.py --model BaselineCAE --checkpoint ./Checkpoints/baseline_16bit_6_l1.pt --save_metrics_path ./$file_name
uv run src/tester.py --model BaselineCAE --checkpoint ./Checkpoints/baseline_8bit_7.pt --save_metrics_path ./$file_name --mu_law_baseline