# Q1 Novel Methodological Contributions & Post-Hoc Calibration Benchmark

## 1. Multi-Method Post-Hoc Calibration Baselines vs. FA-TS (Ours)

| task | condition | method | nll_mean | ece_15_mean | adaptive_ece_15_mean | classwise_ece_15_mean | brier_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | shift_d1_barishal_test | fa_temp_scaled_ours | 0.2367 | 0.0288 | 0.0307 | 0.0417 | 0.1314 |
| offensive | shift_d1_barishal_test | isotonic_regression | 0.222 | 0.0164 | 0.0173 | 0.0214 | 0.1285 |
| offensive | shift_d1_barishal_test | matrix_scaling_l2 | 0.2343 | 0.0226 | 0.0318 | 0.0315 | 0.1323 |
| offensive | shift_d1_barishal_test | oracle_target_temp_scaled | 0.2332 | 0.0239 | 0.0267 | 0.0323 | 0.1312 |
| offensive | shift_d1_barishal_test | source_temp_scaled | 0.2333 | 0.021 | 0.0284 | 0.0302 | 0.1319 |
| offensive | shift_d1_barishal_test | uncalibrated | 0.3138 | 0.0576 | 0.0575 | 0.0605 | 0.1437 |
| offensive | shift_d1_barishal_test | vector_scaling | 0.2328 | 0.0191 | 0.0305 | 0.0296 | 0.1316 |
| offensive | shift_d1_chittagong_test | fa_temp_scaled_ours | 0.4081 | 0.0348 | 0.043 | 0.0667 | 0.255 |
| offensive | shift_d1_chittagong_test | isotonic_regression | 0.4052 | 0.0439 | 0.0461 | 0.0563 | 0.251 |
| offensive | shift_d1_chittagong_test | matrix_scaling_l2 | 0.4386 | 0.0727 | 0.0741 | 0.0845 | 0.2661 |
| offensive | shift_d1_chittagong_test | oracle_target_temp_scaled | 0.4067 | 0.0336 | 0.0399 | 0.0645 | 0.254 |
| offensive | shift_d1_chittagong_test | source_temp_scaled | 0.43 | 0.0672 | 0.068 | 0.0826 | 0.2643 |
| offensive | shift_d1_chittagong_test | uncalibrated | 0.6289 | 0.1228 | 0.1217 | 0.1276 | 0.2935 |
| offensive | shift_d1_chittagong_test | vector_scaling | 0.4313 | 0.0681 | 0.0704 | 0.0803 | 0.2635 |
| offensive | shift_d1_noakhali_test | fa_temp_scaled_ours | 0.2514 | 0.04 | 0.0406 | 0.0488 | 0.1415 |
| offensive | shift_d1_noakhali_test | isotonic_regression | 0.2332 | 0.0136 | 0.017 | 0.0219 | 0.138 |
| offensive | shift_d1_noakhali_test | matrix_scaling_l2 | 0.2438 | 0.0256 | 0.0343 | 0.0324 | 0.142 |
| offensive | shift_d1_noakhali_test | oracle_target_temp_scaled | 0.2454 | 0.0303 | 0.0323 | 0.0376 | 0.1407 |
| offensive | shift_d1_noakhali_test | source_temp_scaled | 0.2439 | 0.0297 | 0.0316 | 0.034 | 0.1416 |
| offensive | shift_d1_noakhali_test | uncalibrated | 0.3198 | 0.0607 | 0.0601 | 0.0628 | 0.1559 |
| offensive | shift_d1_noakhali_test | vector_scaling | 0.2427 | 0.0268 | 0.0331 | 0.0316 | 0.1411 |
| offensive | shift_d2_test | fa_temp_scaled_ours | 1.0996 | 0.3525 | 0.3511 | 0.3538 | 0.7533 |
| offensive | shift_d2_test | isotonic_regression | 1.3958 | 0.3878 | 0.3858 | 0.3881 | 0.8116 |
| offensive | shift_d2_test | matrix_scaling_l2 | 1.5432 | 0.4212 | 0.4198 | 0.4218 | 0.8606 |
| offensive | shift_d2_test | oracle_target_temp_scaled | 0.719 | 0.1106 | 0.1221 | 0.1119 | 0.5244 |
| offensive | shift_d2_test | source_temp_scaled | 1.4578 | 0.4142 | 0.4125 | 0.4153 | 0.8483 |
| offensive | shift_d2_test | uncalibrated | 2.5718 | 0.4558 | 0.4541 | 0.4565 | 0.9188 |
| offensive | shift_d2_test | vector_scaling | 1.4924 | 0.4174 | 0.4149 | 0.4181 | 0.8519 |
| offensive | shift_d3_noise05 | fa_temp_scaled_ours | 0.2635 | 0.029 | 0.0318 | 0.0402 | 0.1526 |
| offensive | shift_d3_noise05 | isotonic_regression | 0.2495 | 0.0187 | 0.0171 | 0.0233 | 0.1496 |
| offensive | shift_d3_noise05 | matrix_scaling_l2 | 0.266 | 0.0276 | 0.0379 | 0.0381 | 0.1545 |
| offensive | shift_d3_noise05 | source_temp_scaled | 0.2643 | 0.0259 | 0.0345 | 0.0381 | 0.154 |
| offensive | shift_d3_noise05 | uncalibrated | 0.3587 | 0.0667 | 0.0672 | 0.0692 | 0.1683 |
| offensive | shift_d3_noise05 | vector_scaling | 0.2638 | 0.0242 | 0.0348 | 0.0364 | 0.1534 |
| offensive | shift_d3_noise10 | fa_temp_scaled_ours | 0.3026 | 0.025 | 0.0305 | 0.0507 | 0.1775 |
| offensive | shift_d3_noise10 | isotonic_regression | 0.2926 | 0.023 | 0.0226 | 0.0344 | 0.1738 |
| offensive | shift_d3_noise10 | matrix_scaling_l2 | 0.3094 | 0.0373 | 0.0412 | 0.0508 | 0.1808 |
| offensive | shift_d3_noise10 | source_temp_scaled | 0.3059 | 0.0315 | 0.037 | 0.0495 | 0.1801 |
| offensive | shift_d3_noise10 | uncalibrated | 0.4255 | 0.0796 | 0.0795 | 0.0838 | 0.1977 |
| offensive | shift_d3_noise10 | vector_scaling | 0.306 | 0.0321 | 0.0388 | 0.047 | 0.1794 |
| offensive | shift_d3_noise20 | fa_temp_scaled_ours | 0.3975 | 0.036 | 0.042 | 0.0763 | 0.2464 |
| offensive | shift_d3_noise20 | isotonic_regression | 0.3882 | 0.0468 | 0.0433 | 0.0637 | 0.2421 |
| offensive | shift_d3_noise20 | matrix_scaling_l2 | 0.426 | 0.0729 | 0.0737 | 0.0904 | 0.258 |
| offensive | shift_d3_noise20 | source_temp_scaled | 0.4173 | 0.0653 | 0.0673 | 0.0877 | 0.2559 |
| offensive | shift_d3_noise20 | uncalibrated | 0.6094 | 0.1211 | 0.1211 | 0.1269 | 0.2845 |
| offensive | shift_d3_noise20 | vector_scaling | 0.4193 | 0.0675 | 0.0681 | 0.0855 | 0.2554 |
| offensive | test | fa_temp_scaled_ours | 0.2249 | 0.0226 | 0.0324 | 0.0279 | 0.1282 |
| offensive | test | isotonic_regression | 0.2124 | 0.0151 | 0.0134 | 0.0172 | 0.1249 |
| offensive | test | matrix_scaling_l2 | 0.2255 | 0.0216 | 0.0338 | 0.0273 | 0.1287 |
| offensive | test | source_temp_scaled | 0.2249 | 0.0225 | 0.0324 | 0.0278 | 0.1282 |
| offensive | test | uncalibrated | 0.2996 | 0.0564 | 0.0566 | 0.0575 | 0.1404 |
| offensive | test | vector_scaling | 0.2244 | 0.0199 | 0.0321 | 0.0263 | 0.1279 |
| sentiment | shift_d1_test | fa_temp_scaled_ours | 0.7248 | 0.0975 | 0.1162 | 0.0981 | 0.4125 |
| sentiment | shift_d1_test | isotonic_regression | 0.7044 | 0.0623 | 0.082 | 0.0825 | 0.4008 |
| sentiment | shift_d1_test | matrix_scaling_l2 | 0.707 | 0.0723 | 0.0845 | 0.0878 | 0.406 |
| sentiment | shift_d1_test | oracle_target_temp_scaled | 0.7098 | 0.0762 | 0.0882 | 0.091 | 0.4065 |
| sentiment | shift_d1_test | source_temp_scaled | 0.7102 | 0.0766 | 0.0891 | 0.0872 | 0.4069 |
| sentiment | shift_d1_test | uncalibrated | 0.779 | 0.13 | 0.1335 | 0.1121 | 0.4336 |
| sentiment | shift_d1_test | vector_scaling | 0.7048 | 0.0724 | 0.0889 | 0.0835 | 0.4037 |
| sentiment | shift_d2_test | fa_temp_scaled_ours | 1.0518 | 0.1335 | 0.1339 | 0.1344 | 0.6253 |
| sentiment | shift_d2_test | isotonic_regression | 1.0466 | 0.1337 | 0.1439 | 0.1371 | 0.6173 |
| sentiment | shift_d2_test | matrix_scaling_l2 | 1.1396 | 0.1843 | 0.1816 | 0.17 | 0.6672 |
| sentiment | shift_d2_test | oracle_target_temp_scaled | 1.0343 | 0.129 | 0.1307 | 0.1272 | 0.6173 |
| sentiment | shift_d2_test | source_temp_scaled | 1.1334 | 0.1824 | 0.184 | 0.1694 | 0.6656 |
| sentiment | shift_d2_test | uncalibrated | 1.4317 | 0.2837 | 0.2847 | 0.2275 | 0.7599 |
| sentiment | shift_d2_test | vector_scaling | 1.1147 | 0.1653 | 0.1669 | 0.1628 | 0.6554 |
| sentiment | shift_d3_noise05 | fa_temp_scaled_ours | 0.6987 | 0.0608 | 0.0611 | 0.0571 | 0.3985 |
| sentiment | shift_d3_noise05 | isotonic_regression | 0.6856 | 0.025 | 0.0323 | 0.0344 | 0.391 |
| sentiment | shift_d3_noise05 | matrix_scaling_l2 | 0.6897 | 0.052 | 0.0551 | 0.0484 | 0.3963 |
| sentiment | shift_d3_noise05 | source_temp_scaled | 0.692 | 0.0471 | 0.0494 | 0.0507 | 0.3966 |
| sentiment | shift_d3_noise05 | uncalibrated | 0.769 | 0.1158 | 0.115 | 0.0871 | 0.4245 |
| sentiment | shift_d3_noise05 | vector_scaling | 0.6882 | 0.0442 | 0.0514 | 0.0471 | 0.3944 |
| sentiment | shift_d3_noise10 | fa_temp_scaled_ours | 0.7537 | 0.0662 | 0.0714 | 0.0745 | 0.4344 |
| sentiment | shift_d3_noise10 | isotonic_regression | 0.7437 | 0.0326 | 0.0391 | 0.0518 | 0.4245 |
| sentiment | shift_d3_noise10 | matrix_scaling_l2 | 0.7435 | 0.0533 | 0.0568 | 0.0623 | 0.4324 |
| sentiment | shift_d3_noise10 | source_temp_scaled | 0.7433 | 0.0545 | 0.0593 | 0.0623 | 0.4315 |
| sentiment | shift_d3_noise10 | uncalibrated | 0.8339 | 0.1286 | 0.1296 | 0.1006 | 0.4657 |
| sentiment | shift_d3_noise10 | vector_scaling | 0.7406 | 0.0476 | 0.0493 | 0.0589 | 0.4296 |
| sentiment | shift_d3_noise20 | fa_temp_scaled_ours | 0.8345 | 0.0838 | 0.092 | 0.1018 | 0.486 |
| sentiment | shift_d3_noise20 | isotonic_regression | 0.8178 | 0.0452 | 0.0592 | 0.0784 | 0.4751 |
| sentiment | shift_d3_noise20 | matrix_scaling_l2 | 0.823 | 0.0834 | 0.0838 | 0.088 | 0.484 |
| sentiment | shift_d3_noise20 | source_temp_scaled | 0.818 | 0.0785 | 0.0805 | 0.0879 | 0.4806 |
| sentiment | shift_d3_noise20 | uncalibrated | 0.9188 | 0.1546 | 0.1563 | 0.125 | 0.5249 |
| sentiment | shift_d3_noise20 | vector_scaling | 0.8191 | 0.0747 | 0.0777 | 0.0853 | 0.4803 |
| sentiment | test | fa_temp_scaled_ours | 0.6667 | 0.0483 | 0.0587 | 0.0453 | 0.3811 |
| sentiment | test | isotonic_regression | 0.6545 | 0.0252 | 0.0369 | 0.0266 | 0.3744 |
| sentiment | test | matrix_scaling_l2 | 0.6634 | 0.0559 | 0.0591 | 0.0431 | 0.3802 |
| sentiment | test | source_temp_scaled | 0.6667 | 0.0483 | 0.0587 | 0.0453 | 0.3811 |
| sentiment | test | uncalibrated | 0.7461 | 0.1171 | 0.1184 | 0.084 | 0.4098 |
| sentiment | test | vector_scaling | 0.6623 | 0.0478 | 0.0562 | 0.0399 | 0.3788 |


## 2. ESS-Regularized Importance-Weighted Conformal Prediction (RIW-CP — Ours)

| task | condition | conformal_variant | beta_exponent_mean | effective_sample_size_mean | empirical_coverage_mean | avg_set_size_mean | trivial_set_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | shift_d1_barishal_test | riw_cp_ours | 0.398 | 2622.64 | 0.9175 | 1.0 | 0.0 |
| offensive | shift_d1_barishal_test | standard_iw_cp | 1.0 | 125.24 | 0.9301 | 1.0308 | 0.0308 |
| offensive | shift_d1_barishal_test | unweighted_cp | 0.0 | 4718.0 | 0.9175 | 1.0 | 0.0 |
| offensive | shift_d1_chittagong_test | riw_cp_ours | 0.334 | 2673.04 | 0.8292 | 1.0 | 0.0 |
| offensive | shift_d1_chittagong_test | standard_iw_cp | 1.0 | 55.82 | 0.871 | 1.0986 | 0.0986 |
| offensive | shift_d1_chittagong_test | unweighted_cp | 0.0 | 4718.0 | 0.8292 | 1.0 | 0.0 |
| offensive | shift_d1_noakhali_test | riw_cp_ours | 0.382 | 2638.4 | 0.9092 | 1.0 | 0.0 |
| offensive | shift_d1_noakhali_test | standard_iw_cp | 1.0 | 111.46 | 0.9277 | 1.0438 | 0.0438 |
| offensive | shift_d1_noakhali_test | unweighted_cp | 0.0 | 4718.0 | 0.9092 | 1.0 | 0.0 |
| offensive | shift_d2_test | riw_cp_ours | 0.764 | 2647.04 | 0.8972 | 1.784 | 0.784 |
| offensive | shift_d2_test | standard_iw_cp | 1.0 | 817.78 | 0.997 | 1.993 | 0.993 |
| offensive | shift_d2_test | unweighted_cp | 0.0 | 4718.0 | 0.5255 | 1.0 | 0.0 |
| offensive | shift_d3_noise05 | riw_cp_ours | 0.5 | 2617.82 | 0.9019 | 1.0 | 0.0 |
| offensive | shift_d3_noise05 | standard_iw_cp | 1.0 | 343.3 | 0.9134 | 1.0274 | 0.0274 |
| offensive | shift_d3_noise05 | unweighted_cp | 0.0 | 4718.0 | 0.9019 | 1.0 | 0.0 |
| offensive | shift_d3_noise10 | riw_cp_ours | 0.366 | 2650.36 | 0.8854 | 1.0 | 0.0 |
| offensive | shift_d3_noise10 | standard_iw_cp | 1.0 | 169.48 | 0.9059 | 1.0509 | 0.0509 |
| offensive | shift_d3_noise10 | unweighted_cp | 0.0 | 4718.0 | 0.8854 | 1.0 | 0.0 |
| offensive | shift_d3_noise20 | riw_cp_ours | 0.3 | 2656.5 | 0.834 | 1.0 | 0.0 |
| offensive | shift_d3_noise20 | standard_iw_cp | 1.0 | 83.58 | 0.8762 | 1.1011 | 0.1011 |
| offensive | shift_d3_noise20 | unweighted_cp | 0.0 | 4718.0 | 0.834 | 1.0 | 0.0 |
| sentiment | shift_d1_test | riw_cp_ours | 0.336 | 759.66 | 0.9542 | 2.0667 | 0.3134 |
| sentiment | shift_d1_test | standard_iw_cp | 1.0 | 11.76 | 0.8955 | 1.7701 | 0.2338 |
| sentiment | shift_d1_test | unweighted_cp | 0.0 | 1359.0 | 0.9234 | 1.803 | 0.194 |
| sentiment | shift_d2_test | riw_cp_ours | 0.842 | 891.14 | 0.9993 | 2.9697 | 0.9719 |
| sentiment | shift_d2_test | standard_iw_cp | 1.0 | 626.96 | 1.0 | 2.9985 | 0.9993 |
| sentiment | shift_d2_test | unweighted_cp | 0.0 | 1359.0 | 0.8065 | 2.1224 | 0.3341 |
| sentiment | shift_d3_noise05 | riw_cp_ours | 0.348 | 761.96 | 0.9293 | 1.841 | 0.1938 |
| sentiment | shift_d3_noise05 | standard_iw_cp | 1.0 | 62.06 | 0.9012 | 1.6913 | 0.1459 |
| sentiment | shift_d3_noise05 | unweighted_cp | 0.0 | 1359.0 | 0.9156 | 1.7378 | 0.1494 |
| sentiment | shift_d3_noise10 | riw_cp_ours | 0.284 | 767.46 | 0.9535 | 2.1271 | 0.3281 |
| sentiment | shift_d3_noise10 | standard_iw_cp | 1.0 | 25.74 | 0.9479 | 2.1175 | 0.3663 |
| sentiment | shift_d3_noise10 | unweighted_cp | 0.0 | 1359.0 | 0.9115 | 1.8154 | 0.1731 |
| sentiment | shift_d3_noise20 | riw_cp_ours | 0.332 | 761.18 | 0.9881 | 2.6793 | 0.7269 |
| sentiment | shift_d3_noise20 | standard_iw_cp | 1.0 | 8.74 | 0.9885 | 2.846 | 0.8997 |
| sentiment | shift_d3_noise20 | unweighted_cp | 0.0 | 1359.0 | 0.9174 | 1.9646 | 0.2371 |


## 3. 2026 Multilingual LLM Zero-Shot Calibration Baseline (Qwen2.5-Instruct)

| task | model | ft_method | condition | calibration_regime | accuracy | nll | ece_15 | adaptive_ece_15 | classwise_ece_15 | overconfidence_15 | brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | test | uncalibrated | 0.44 | 1.9127 | 0.437 | 0.4315 | 0.2905 | 0.4343 | 0.9325 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | test | source_temp_scaled | 0.44 | 1.3174 | 0.2817 | 0.2761 | 0.1875 | 0.2789 | 0.7675 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d1_test | uncalibrated | 0.403 | 1.8621 | 0.4573 | 0.4573 | 0.3084 | 0.4573 | 0.9645 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d1_test | source_temp_scaled | 0.403 | 1.3001 | 0.3104 | 0.3007 | 0.2099 | 0.3055 | 0.7813 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d2_test | uncalibrated | 0.435 | 1.6202 | 0.2442 | 0.2461 | 0.1891 | 0.2433 | 0.7727 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d2_test | source_temp_scaled | 0.435 | 1.2368 | 0.139 | 0.1562 | 0.1226 | 0.1386 | 0.7027 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise05 | uncalibrated | 0.4425 | 1.8982 | 0.4308 | 0.4245 | 0.2916 | 0.4276 | 0.9281 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise05 | source_temp_scaled | 0.4425 | 1.3129 | 0.2882 | 0.2733 | 0.1894 | 0.2786 | 0.7653 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise10 | uncalibrated | 0.44 | 1.9088 | 0.4296 | 0.4259 | 0.2867 | 0.4278 | 0.9306 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise10 | source_temp_scaled | 0.44 | 1.3213 | 0.274 | 0.269 | 0.184 | 0.2715 | 0.7678 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise20 | uncalibrated | 0.44 | 1.8845 | 0.427 | 0.4229 | 0.2844 | 0.425 | 0.9246 |
| sentiment | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise20 | source_temp_scaled | 0.44 | 1.3117 | 0.2712 | 0.265 | 0.1796 | 0.2681 | 0.7632 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | test | uncalibrated | 0.4825 | 1.9214 | 0.4911 | 0.4911 | 0.4911 | 0.4911 | 0.9774 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | test | source_temp_scaled | 0.4825 | 1.1614 | 0.4 | 0.4 | 0.4 | 0.4 | 0.8109 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d1_barishal_test | uncalibrated | 0.4875 | 1.8476 | 0.4836 | 0.4836 | 0.4836 | 0.4836 | 0.9638 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d1_barishal_test | source_temp_scaled | 0.4875 | 1.1276 | 0.3883 | 0.388 | 0.3883 | 0.3881 | 0.7946 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d1_chittagong_test | uncalibrated | 0.49 | 1.8536 | 0.4806 | 0.4806 | 0.4806 | 0.4806 | 0.9603 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d1_chittagong_test | source_temp_scaled | 0.49 | 1.1319 | 0.3843 | 0.3843 | 0.3843 | 0.3843 | 0.7946 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d1_noakhali_test | uncalibrated | 0.4875 | 1.8287 | 0.4826 | 0.4826 | 0.4826 | 0.4826 | 0.9621 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d1_noakhali_test | source_temp_scaled | 0.4875 | 1.1194 | 0.3856 | 0.3856 | 0.3856 | 0.3856 | 0.7912 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d2_test | uncalibrated | 0.4775 | 1.6754 | 0.485 | 0.485 | 0.485 | 0.485 | 0.95 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d2_test | source_temp_scaled | 0.4775 | 1.0441 | 0.3851 | 0.3851 | 0.3851 | 0.3851 | 0.7587 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise05 | uncalibrated | 0.4825 | 1.9258 | 0.4914 | 0.4914 | 0.4914 | 0.4914 | 0.9782 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise05 | source_temp_scaled | 0.4825 | 1.1637 | 0.4004 | 0.4004 | 0.4004 | 0.4004 | 0.812 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise10 | uncalibrated | 0.4825 | 1.9463 | 0.4922 | 0.4922 | 0.4922 | 0.4922 | 0.9814 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise10 | source_temp_scaled | 0.4825 | 1.1737 | 0.4016 | 0.4016 | 0.4016 | 0.4016 | 0.8169 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise20 | uncalibrated | 0.4825 | 1.9705 | 0.4929 | 0.4929 | 0.4929 | 0.4929 | 0.9844 |
| offensive | qwen2.5_1.5b_instruct | zero_shot | shift_d3_noise20 | source_temp_scaled | 0.4825 | 1.1856 | 0.4032 | 0.4032 | 0.4032 | 0.4032 | 0.8225 |

