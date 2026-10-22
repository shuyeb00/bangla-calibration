# Q1 Methodological & Analytical Extensions Summary

## 1. Mechanistic Shift-to-Miscalibration Correlations (Part D.2 & H)

| task | model | ft_method | n_conditions | spearman_frag_vs_nll | p_value_frag_vs_nll | pearson_frag_vs_nll | spearman_jaccard_vs_nll | spearman_frag_vs_classwise_ece | spearman_frag_vs_overconfidence | spearman_maha_pct_vs_nll |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | 8 | 0.9048 | 0.002 | 0.7131 | -0.6429 | 0.9048 | 0.9048 | 0.9048 |
| offensive | banglabert | lora | 8 | 0.9286 | 0.0009 | 0.8095 | -0.8095 | 0.8571 | 0.9524 | 0.119 |
| offensive | indicbert | full_ft | 8 | 0.9762 | 0.0 | 0.8055 | -0.8095 | 0.9762 | 0.9762 | 0.9524 |
| offensive | indicbert | lora | 8 | 0.881 | 0.0039 | 0.7528 | -0.9286 | -0.1429 | 0.7619 | 0.5952 |
| offensive | xlmr_base | full_ft | 8 | 0.9524 | 0.0003 | 0.6918 | -0.7381 | 0.9524 | 0.9762 | 0.8571 |
| offensive | xlmr_base | lora | 8 | 0.9524 | 0.0003 | 0.7987 | -0.881 | 0.9524 | 0.9762 | -0.3095 |
| sentiment | banglabert | full_ft | 6 | 1.0 | 0.0 | 0.8702 | -0.8286 | 0.9429 | 0.9429 | 1.0 |
| sentiment | banglabert | lora | 6 | 0.8286 | 0.0416 | 0.8047 | -0.4286 | 0.0857 | -0.0857 | 0.9429 |
| sentiment | indicbert | full_ft | 6 | 1.0 | 0.0 | 0.8033 | -0.8286 | 0.7714 | 0.7714 | 0.8286 |
| sentiment | indicbert | lora | 6 | -0.1429 | 0.7872 | 0.0213 | 0.5429 | -0.0286 | nan | -0.6 |
| sentiment | xlmr_base | full_ft | 6 | 0.8286 | 0.0416 | 0.8208 | -1.0 | 0.7714 | 0.8286 | 0.0286 |
| sentiment | xlmr_base | lora | 6 | 0.8286 | 0.0416 | 0.8294 | -1.0 | 0.7714 | 0.9429 | 0.7714 |


## 2. MAPIE Conformal Verification Cross-Check (Part I.3)

| task | run_id | exact_coverage | exact_set_size | mapie_coverage | mapie_set_size | abs_coverage_diff | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | sentiment_banglabert_full_ft_seed42 | 0.9118 | 1.6574 | 0.9118 | 1.6574 | 0.0 | Exact Order-Statistic Verified (Matches Split Conformal Quantile) |
| offensive | offensive_banglabert_full_ft_seed42 | 0.9185 | 1.0 | 0.9185 | 1.0 | 0.0 | Exact Order-Statistic Verified (Matches Split Conformal Quantile) |


## 3. Extended Calibration Metrics (Adaptive ECE, Classwise ECE, Overconfidence)

| task | model | ft_method | condition | calibration_regime | nll_mean | top_label_ece_15_mean | adaptive_ece_15_mean | classwise_ece_15_mean | overconfidence_ece_15_mean | underconfidence_ece_15_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.2333 | 0.021 | 0.0284 | 0.0302 | 0.0168 | 0.0042 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.2332 | 0.0239 | 0.0267 | 0.0323 | 0.0127 | 0.0113 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | uncalibrated | 0.3138 | 0.0576 | 0.0575 | 0.0605 | 0.0572 | 0.0004 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.43 | 0.0672 | 0.068 | 0.0826 | 0.0662 | 0.001 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.4067 | 0.0336 | 0.0399 | 0.0645 | 0.0218 | 0.0119 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.6289 | 0.1228 | 0.1217 | 0.1276 | 0.1222 | 0.0006 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.2439 | 0.0297 | 0.0316 | 0.034 | 0.02 | 0.0097 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.2454 | 0.0303 | 0.0323 | 0.0376 | 0.0127 | 0.0176 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.3198 | 0.0607 | 0.0601 | 0.0628 | 0.0599 | 0.0009 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | 1.4578 | 0.4142 | 0.4125 | 0.4153 | 0.4131 | 0.001 |
| offensive | banglabert | full_ft | shift_d2_test | target_temp_scaled | 0.719 | 0.1106 | 0.1221 | 0.1119 | 0.1075 | 0.003 |
| offensive | banglabert | full_ft | shift_d2_test | uncalibrated | 2.5718 | 0.4558 | 0.4541 | 0.4565 | 0.455 | 0.0009 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.2643 | 0.0259 | 0.0345 | 0.0381 | 0.0227 | 0.0032 |
| offensive | banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.3587 | 0.0667 | 0.0672 | 0.0692 | 0.0666 | 0.0001 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.3059 | 0.0315 | 0.037 | 0.0495 | 0.0304 | 0.0011 |
| offensive | banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.4255 | 0.0796 | 0.0795 | 0.0838 | 0.0794 | 0.0002 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.4173 | 0.0653 | 0.0673 | 0.0877 | 0.0652 | 0.0001 |
| offensive | banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.6094 | 0.1211 | 0.1211 | 0.1269 | 0.121 | 0.0001 |
| offensive | banglabert | full_ft | test | source_temp_scaled | 0.2249 | 0.0225 | 0.0324 | 0.0278 | 0.0174 | 0.0051 |
| offensive | banglabert | full_ft | test | uncalibrated | 0.2996 | 0.0564 | 0.0566 | 0.0575 | 0.0561 | 0.0003 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | 0.2779 | 0.0157 | 0.0223 | 0.0245 | 0.0126 | 0.0031 |
| offensive | banglabert | lora | shift_d1_barishal_test | target_temp_scaled | 0.2779 | 0.0163 | 0.0223 | 0.025 | 0.0127 | 0.0036 |
| offensive | banglabert | lora | shift_d1_barishal_test | uncalibrated | 0.2899 | 0.0393 | 0.0397 | 0.0409 | 0.0384 | 0.0009 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.5028 | 0.0689 | 0.0699 | 0.0883 | 0.0672 | 0.0017 |
| offensive | banglabert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.4839 | 0.0367 | 0.0369 | 0.0831 | 0.0281 | 0.0086 |
| offensive | banglabert | lora | shift_d1_chittagong_test | uncalibrated | 0.5459 | 0.1019 | 0.1017 | 0.1142 | 0.1012 | 0.0008 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.3278 | 0.0181 | 0.0252 | 0.0703 | 0.0126 | 0.0055 |
| offensive | banglabert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.3282 | 0.0211 | 0.0271 | 0.0772 | 0.0039 | 0.0171 |
| offensive | banglabert | lora | shift_d1_noakhali_test | uncalibrated | 0.3434 | 0.0428 | 0.0427 | 0.0719 | 0.0402 | 0.0026 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | 0.8599 | 0.2632 | 0.265 | 0.262 | 0.2566 | 0.0066 |
| offensive | banglabert | lora | shift_d2_test | target_temp_scaled | 0.6919 | 0.0337 | 0.0836 | 0.0278 | 0.0287 | 0.0049 |
| offensive | banglabert | lora | shift_d2_test | uncalibrated | 0.9656 | 0.3077 | 0.3109 | 0.3074 | 0.3031 | 0.0046 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.2901 | 0.0122 | 0.0144 | 0.0195 | 0.008 | 0.0041 |
| offensive | banglabert | lora | shift_d3_noise05 | uncalibrated | 0.3005 | 0.0361 | 0.0355 | 0.038 | 0.035 | 0.0011 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.3313 | 0.0165 | 0.0198 | 0.0189 | 0.0126 | 0.004 |
| offensive | banglabert | lora | shift_d3_noise10 | uncalibrated | 0.343 | 0.043 | 0.0428 | 0.0433 | 0.0426 | 0.0004 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.4198 | 0.0216 | 0.024 | 0.0345 | 0.0196 | 0.002 |
| offensive | banglabert | lora | shift_d3_noise20 | uncalibrated | 0.4352 | 0.0569 | 0.0566 | 0.0645 | 0.0566 | 0.0003 |
| offensive | banglabert | lora | test | source_temp_scaled | 0.2584 | 0.015 | 0.0192 | 0.0265 | 0.0112 | 0.0038 |
| offensive | banglabert | lora | test | uncalibrated | 0.2695 | 0.0362 | 0.0363 | 0.0383 | 0.0352 | 0.001 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.3375 | 0.0342 | 0.0398 | 0.0429 | 0.0273 | 0.0069 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.337 | 0.0347 | 0.038 | 0.0443 | 0.0203 | 0.0144 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | uncalibrated | 0.424 | 0.0903 | 0.0883 | 0.0935 | 0.0892 | 0.0011 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.5112 | 0.0959 | 0.0997 | 0.0996 | 0.0947 | 0.0012 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.481 | 0.0492 | 0.0589 | 0.0548 | 0.0329 | 0.0163 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.711 | 0.1669 | 0.1665 | 0.1672 | 0.1667 | 0.0002 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.3519 | 0.0408 | 0.0475 | 0.046 | 0.0357 | 0.0051 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.3518 | 0.046 | 0.0466 | 0.0548 | 0.0236 | 0.0224 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.4467 | 0.0992 | 0.0995 | 0.1003 | 0.0992 | 0.0 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | 0.9413 | 0.2729 | 0.277 | 0.2757 | 0.2726 | 0.0003 |
| offensive | indicbert | full_ft | shift_d2_test | target_temp_scaled | 0.6943 | 0.0412 | 0.07 | 0.0495 | 0.0403 | 0.0009 |
| offensive | indicbert | full_ft | shift_d2_test | uncalibrated | 1.3658 | 0.372 | 0.37 | 0.3729 | 0.3698 | 0.0022 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.3273 | 0.0374 | 0.0411 | 0.0379 | 0.0298 | 0.0075 |
| offensive | indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.4139 | 0.0889 | 0.0886 | 0.0901 | 0.0886 | 0.0002 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.3727 | 0.0438 | 0.0484 | 0.0474 | 0.0403 | 0.0035 |
| offensive | indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.4875 | 0.1065 | 0.1052 | 0.107 | 0.1058 | 0.0007 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.4468 | 0.0632 | 0.0669 | 0.0666 | 0.0625 | 0.0007 |
| offensive | indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.6015 | 0.1339 | 0.1335 | 0.1343 | 0.1337 | 0.0002 |
| offensive | indicbert | full_ft | test | source_temp_scaled | 0.291 | 0.0314 | 0.0366 | 0.034 | 0.0207 | 0.0107 |
| offensive | indicbert | full_ft | test | uncalibrated | 0.3571 | 0.0747 | 0.0746 | 0.0753 | 0.0745 | 0.0002 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | 0.6249 | 0.0716 | 0.0778 | 0.0779 | 0.0264 | 0.0452 |
| offensive | indicbert | lora | shift_d1_barishal_test | target_temp_scaled | 0.6241 | 0.0695 | 0.0759 | 0.074 | 0.0217 | 0.0478 |
| offensive | indicbert | lora | shift_d1_barishal_test | uncalibrated | 0.63 | 0.0828 | 0.0881 | 0.0854 | 0.0052 | 0.0776 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.683 | 0.0939 | 0.104 | 0.0956 | 0.0721 | 0.0218 |
| offensive | indicbert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.6677 | 0.0652 | 0.0741 | 0.0723 | 0.0364 | 0.0288 |
| offensive | indicbert | lora | shift_d1_chittagong_test | uncalibrated | 0.6671 | 0.0541 | 0.0721 | 0.0619 | 0.0248 | 0.0293 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.6503 | 0.0705 | 0.0873 | 0.0768 | 0.0429 | 0.0276 |
| offensive | indicbert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.6467 | 0.0506 | 0.0711 | 0.0625 | 0.0066 | 0.044 |
| offensive | indicbert | lora | shift_d1_noakhali_test | uncalibrated | 0.647 | 0.0591 | 0.0727 | 0.0663 | 0.0108 | 0.0483 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | 0.7549 | 0.1193 | 0.1328 | 0.128 | 0.112 | 0.0073 |
| offensive | indicbert | lora | shift_d2_test | target_temp_scaled | 0.6907 | 0.0193 | 0.0647 | 0.0264 | 0.0036 | 0.0156 |
| offensive | indicbert | lora | shift_d2_test | uncalibrated | 0.713 | 0.0755 | 0.0975 | 0.0865 | 0.0655 | 0.0099 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.6258 | 0.071 | 0.0762 | 0.0754 | 0.0296 | 0.0414 |
| offensive | indicbert | lora | shift_d3_noise05 | uncalibrated | 0.6286 | 0.0729 | 0.077 | 0.0747 | 0.0027 | 0.0702 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.6289 | 0.0695 | 0.0765 | 0.075 | 0.0313 | 0.0381 |
| offensive | indicbert | lora | shift_d3_noise10 | uncalibrated | 0.6304 | 0.0665 | 0.0745 | 0.0674 | 0.0018 | 0.0647 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.6385 | 0.0726 | 0.0772 | 0.0773 | 0.0399 | 0.0327 |
| offensive | indicbert | lora | shift_d3_noise20 | uncalibrated | 0.6362 | 0.0621 | 0.0672 | 0.0668 | 0.0065 | 0.0556 |
| offensive | indicbert | lora | test | source_temp_scaled | 0.6174 | 0.0668 | 0.0707 | 0.0705 | 0.0241 | 0.0426 |
| offensive | indicbert | lora | test | uncalibrated | 0.6234 | 0.0753 | 0.079 | 0.0772 | 0.0006 | 0.0746 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.2435 | 0.0177 | 0.0266 | 0.025 | 0.0158 | 0.0019 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.2439 | 0.021 | 0.0242 | 0.0301 | 0.01 | 0.0111 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | uncalibrated | 0.2984 | 0.0584 | 0.0578 | 0.0595 | 0.0579 | 0.0005 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.3981 | 0.0549 | 0.0574 | 0.0604 | 0.0548 | 0.0001 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.385 | 0.0264 | 0.031 | 0.0426 | 0.0151 | 0.0113 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | uncalibrated | 0.5154 | 0.1101 | 0.1096 | 0.1114 | 0.1098 | 0.0002 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.2686 | 0.0287 | 0.0332 | 0.0406 | 0.0235 | 0.0052 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.2709 | 0.0317 | 0.0334 | 0.0527 | 0.0113 | 0.0204 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | uncalibrated | 0.3282 | 0.0667 | 0.066 | 0.0695 | 0.0661 | 0.0005 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 1.2312 | 0.3748 | 0.3708 | 0.3764 | 0.3724 | 0.0024 |
| offensive | xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 0.7051 | 0.0821 | 0.0949 | 0.0811 | 0.0775 | 0.0046 |
| offensive | xlmr_base | full_ft | shift_d2_test | uncalibrated | 1.9049 | 0.4372 | 0.433 | 0.4391 | 0.4351 | 0.0021 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.2498 | 0.0195 | 0.0225 | 0.0239 | 0.015 | 0.0045 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.3011 | 0.0576 | 0.0566 | 0.0587 | 0.057 | 0.0006 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.2819 | 0.0213 | 0.0255 | 0.0275 | 0.0191 | 0.0022 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.3423 | 0.0671 | 0.0667 | 0.0676 | 0.0669 | 0.0002 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.3442 | 0.0264 | 0.0308 | 0.0335 | 0.0259 | 0.0005 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.4207 | 0.0828 | 0.0823 | 0.0833 | 0.0825 | 0.0002 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | 0.2253 | 0.0181 | 0.0236 | 0.0231 | 0.0144 | 0.0037 |
| offensive | xlmr_base | full_ft | test | uncalibrated | 0.2725 | 0.0532 | 0.0529 | 0.0542 | 0.0529 | 0.0003 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | 0.3851 | 0.0185 | 0.0232 | 0.0617 | 0.0132 | 0.0053 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | target_temp_scaled | 0.387 | 0.0296 | 0.0278 | 0.0642 | 0.0088 | 0.0208 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | uncalibrated | 0.4031 | 0.0528 | 0.0526 | 0.0699 | 0.0524 | 0.0003 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | 0.5114 | 0.0352 | 0.0408 | 0.0937 | 0.0335 | 0.0017 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | target_temp_scaled | 0.5058 | 0.0213 | 0.0288 | 0.0881 | 0.0111 | 0.0102 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | uncalibrated | 0.5483 | 0.0829 | 0.0804 | 0.1159 | 0.0814 | 0.0015 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | 0.4239 | 0.0213 | 0.0234 | 0.0935 | 0.0134 | 0.0079 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | target_temp_scaled | 0.4247 | 0.0219 | 0.0264 | 0.0959 | 0.0035 | 0.0184 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | uncalibrated | 0.4445 | 0.0571 | 0.0548 | 0.098 | 0.0542 | 0.0029 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.8318 | 0.2151 | 0.2163 | 0.2175 | 0.215 | 0.0 |
| offensive | xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.692 | 0.0335 | 0.0547 | 0.03 | 0.0305 | 0.003 |
| offensive | xlmr_base | lora | shift_d2_test | uncalibrated | 0.9517 | 0.2699 | 0.2695 | 0.2698 | 0.2693 | 0.0006 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.3779 | 0.0112 | 0.015 | 0.0709 | 0.0072 | 0.004 |
| offensive | xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.3942 | 0.048 | 0.0477 | 0.0702 | 0.0473 | 0.0007 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.4205 | 0.0153 | 0.0167 | 0.0932 | 0.014 | 0.0013 |
| offensive | xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.4427 | 0.0578 | 0.0574 | 0.0935 | 0.0576 | 0.0002 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.4779 | 0.0205 | 0.0217 | 0.1156 | 0.0181 | 0.0024 |
| offensive | xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.5036 | 0.0638 | 0.0632 | 0.1222 | 0.0627 | 0.0011 |
| offensive | xlmr_base | lora | test | source_temp_scaled | 0.3494 | 0.012 | 0.0145 | 0.0504 | 0.0088 | 0.0032 |
| offensive | xlmr_base | lora | test | uncalibrated | 0.3675 | 0.0473 | 0.047 | 0.0615 | 0.0472 | 0.0002 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | 0.7102 | 0.0766 | 0.0891 | 0.0872 | 0.037 | 0.0396 |
| sentiment | banglabert | full_ft | shift_d1_test | target_temp_scaled | 0.7098 | 0.0762 | 0.0882 | 0.091 | 0.0378 | 0.0384 |
| sentiment | banglabert | full_ft | shift_d1_test | uncalibrated | 0.779 | 0.13 | 0.1335 | 0.1121 | 0.1129 | 0.0171 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | 1.1334 | 0.1824 | 0.184 | 0.1694 | 0.1819 | 0.0005 |
| sentiment | banglabert | full_ft | shift_d2_test | target_temp_scaled | 1.0343 | 0.129 | 0.1307 | 0.1272 | 0.0798 | 0.0491 |
| sentiment | banglabert | full_ft | shift_d2_test | uncalibrated | 1.4317 | 0.2837 | 0.2847 | 0.2275 | 0.2833 | 0.0003 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.692 | 0.0471 | 0.0494 | 0.0507 | 0.0246 | 0.0224 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.769 | 0.1158 | 0.115 | 0.0871 | 0.1083 | 0.0075 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.7433 | 0.0545 | 0.0593 | 0.0623 | 0.0352 | 0.0193 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.8339 | 0.1286 | 0.1296 | 0.1006 | 0.1223 | 0.0063 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.818 | 0.0785 | 0.0805 | 0.0879 | 0.0609 | 0.0176 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.9188 | 0.1546 | 0.1563 | 0.125 | 0.1508 | 0.0038 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | 0.6667 | 0.0483 | 0.0587 | 0.0453 | 0.0272 | 0.0212 |
| sentiment | banglabert | full_ft | test | uncalibrated | 0.7461 | 0.1171 | 0.1184 | 0.084 | 0.1104 | 0.0067 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | 0.7509 | 0.0749 | 0.0985 | 0.0805 | 0.0262 | 0.0487 |
| sentiment | banglabert | lora | shift_d1_test | target_temp_scaled | 0.7513 | 0.0793 | 0.0997 | 0.0827 | 0.0272 | 0.0521 |
| sentiment | banglabert | lora | shift_d1_test | uncalibrated | 0.7493 | 0.0741 | 0.0911 | 0.0762 | 0.0424 | 0.0317 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | 1.2904 | 0.2388 | 0.2397 | 0.1776 | 0.238 | 0.0008 |
| sentiment | banglabert | lora | shift_d2_test | target_temp_scaled | 1.0862 | 0.079 | 0.1417 | 0.0918 | 0.0364 | 0.0426 |
| sentiment | banglabert | lora | shift_d2_test | uncalibrated | 1.3648 | 0.2753 | 0.2756 | 0.1976 | 0.2751 | 0.0001 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.7702 | 0.0369 | 0.0444 | 0.0412 | 0.0134 | 0.0235 |
| sentiment | banglabert | lora | shift_d3_noise05 | uncalibrated | 0.7708 | 0.0433 | 0.046 | 0.044 | 0.0332 | 0.0101 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.7939 | 0.0401 | 0.0517 | 0.0385 | 0.0062 | 0.0339 |
| sentiment | banglabert | lora | shift_d3_noise10 | uncalibrated | 0.7904 | 0.0355 | 0.041 | 0.036 | 0.0199 | 0.0155 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.8412 | 0.0604 | 0.0695 | 0.052 | 0.0028 | 0.0576 |
| sentiment | banglabert | lora | shift_d3_noise20 | uncalibrated | 0.8334 | 0.038 | 0.0498 | 0.0412 | 0.006 | 0.0319 |
| sentiment | banglabert | lora | test | source_temp_scaled | 0.7521 | 0.0612 | 0.0633 | 0.0447 | 0.0387 | 0.0225 |
| sentiment | banglabert | lora | test | uncalibrated | 0.7558 | 0.0717 | 0.0745 | 0.0509 | 0.061 | 0.0107 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | 0.8829 | 0.0724 | 0.0954 | 0.0738 | 0.0375 | 0.0349 |
| sentiment | indicbert | full_ft | shift_d1_test | target_temp_scaled | 0.8841 | 0.0825 | 0.0955 | 0.0816 | 0.0568 | 0.0257 |
| sentiment | indicbert | full_ft | shift_d1_test | uncalibrated | 0.8952 | 0.0911 | 0.1074 | 0.0865 | 0.0736 | 0.0175 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | 1.157 | 0.1785 | 0.1878 | 0.1443 | 0.1683 | 0.0102 |
| sentiment | indicbert | full_ft | shift_d2_test | target_temp_scaled | 1.0602 | 0.0386 | 0.0692 | 0.0651 | 0.0095 | 0.0291 |
| sentiment | indicbert | full_ft | shift_d2_test | uncalibrated | 1.27 | 0.2374 | 0.2439 | 0.1784 | 0.2285 | 0.0088 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.8757 | 0.0359 | 0.0486 | 0.0393 | 0.0178 | 0.0182 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.8878 | 0.0617 | 0.0703 | 0.0524 | 0.0578 | 0.0039 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.8899 | 0.0305 | 0.0361 | 0.0379 | 0.0203 | 0.0102 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.9069 | 0.067 | 0.0693 | 0.0557 | 0.0659 | 0.001 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.9037 | 0.0308 | 0.0432 | 0.0386 | 0.0148 | 0.016 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.9201 | 0.063 | 0.0675 | 0.0553 | 0.0581 | 0.0049 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | 0.8592 | 0.0326 | 0.0437 | 0.0354 | 0.012 | 0.0206 |
| sentiment | indicbert | full_ft | test | uncalibrated | 0.8681 | 0.0539 | 0.0621 | 0.0451 | 0.0498 | 0.004 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | 1.0657 | 0.0157 | 0.1027 | 0.0305 | 0.0157 | 0.0 |
| sentiment | indicbert | lora | shift_d1_test | target_temp_scaled | 1.0651 | 0.0255 | 0.0993 | 0.0313 | 0.0255 | 0.0 |
| sentiment | indicbert | lora | shift_d1_test | uncalibrated | 1.0724 | 0.0176 | 0.11 | 0.0438 | 0.0 | 0.0176 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | 1.0717 | 0.0056 | 0.06 | 0.0073 | 0.0047 | 0.0009 |
| sentiment | indicbert | lora | shift_d2_test | target_temp_scaled | 1.0717 | 0.0086 | 0.0599 | 0.0081 | 0.0084 | 0.0002 |
| sentiment | indicbert | lora | shift_d2_test | uncalibrated | 1.0758 | 0.0302 | 0.0665 | 0.027 | 0.0 | 0.0302 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | 1.0733 | 0.0061 | 0.0459 | 0.0057 | 0.0052 | 0.001 |
| sentiment | indicbert | lora | shift_d3_noise05 | uncalibrated | 1.077 | 0.0292 | 0.0549 | 0.0235 | 0.0 | 0.0292 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | 1.0731 | 0.0061 | 0.0553 | 0.0056 | 0.0051 | 0.001 |
| sentiment | indicbert | lora | shift_d3_noise10 | uncalibrated | 1.0769 | 0.0292 | 0.0644 | 0.0234 | 0.0 | 0.0292 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | 1.0733 | 0.0061 | 0.0443 | 0.0057 | 0.0052 | 0.0009 |
| sentiment | indicbert | lora | shift_d3_noise20 | uncalibrated | 1.0771 | 0.0292 | 0.0494 | 0.0234 | 0.0 | 0.0292 |
| sentiment | indicbert | lora | test | source_temp_scaled | 1.0732 | 0.0061 | 0.0597 | 0.0057 | 0.0051 | 0.001 |
| sentiment | indicbert | lora | test | uncalibrated | 1.077 | 0.0293 | 0.0652 | 0.0235 | 0.0 | 0.0293 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | 0.8127 | 0.0814 | 0.0992 | 0.0863 | 0.0576 | 0.0238 |
| sentiment | xlmr_base | full_ft | shift_d1_test | target_temp_scaled | 0.8147 | 0.0778 | 0.101 | 0.083 | 0.0596 | 0.0182 |
| sentiment | xlmr_base | full_ft | shift_d1_test | uncalibrated | 0.9118 | 0.1565 | 0.16 | 0.1268 | 0.153 | 0.0035 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 1.1007 | 0.2042 | 0.2005 | 0.1978 | 0.1988 | 0.0054 |
| sentiment | xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 0.9741 | 0.0701 | 0.0846 | 0.1273 | 0.033 | 0.0371 |
| sentiment | xlmr_base | full_ft | shift_d2_test | uncalibrated | 1.4596 | 0.3057 | 0.3049 | 0.2486 | 0.3049 | 0.0008 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.7252 | 0.0407 | 0.0457 | 0.04 | 0.0203 | 0.0204 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.7838 | 0.113 | 0.112 | 0.0806 | 0.1122 | 0.0008 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.7593 | 0.0385 | 0.0425 | 0.0388 | 0.0228 | 0.0157 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.8306 | 0.1215 | 0.1194 | 0.0861 | 0.1203 | 0.0012 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8106 | 0.0491 | 0.0534 | 0.0503 | 0.0383 | 0.0108 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.8978 | 0.1439 | 0.144 | 0.101 | 0.1434 | 0.0005 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | 0.7142 | 0.0418 | 0.0461 | 0.0478 | 0.0208 | 0.0209 |
| sentiment | xlmr_base | full_ft | test | uncalibrated | 0.7719 | 0.1118 | 0.1113 | 0.0813 | 0.1109 | 0.0009 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | 0.9026 | 0.095 | 0.1117 | 0.0895 | 0.0606 | 0.0344 |
| sentiment | xlmr_base | lora | shift_d1_test | target_temp_scaled | 0.8977 | 0.0802 | 0.105 | 0.0808 | 0.0415 | 0.0387 |
| sentiment | xlmr_base | lora | shift_d1_test | uncalibrated | 0.9197 | 0.1022 | 0.1284 | 0.0828 | 0.0818 | 0.0204 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.93 | 0.1159 | 0.1237 | 0.1631 | 0.1063 | 0.0095 |
| sentiment | xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.9052 | 0.0815 | 0.0909 | 0.142 | 0.0386 | 0.0429 |
| sentiment | xlmr_base | lora | shift_d2_test | uncalibrated | 0.964 | 0.1492 | 0.1517 | 0.1797 | 0.1414 | 0.0078 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.8206 | 0.0303 | 0.0363 | 0.0361 | 0.0184 | 0.012 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.8261 | 0.0515 | 0.054 | 0.0425 | 0.0467 | 0.0047 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.8458 | 0.0417 | 0.0486 | 0.0403 | 0.0327 | 0.009 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.8554 | 0.0661 | 0.0684 | 0.0511 | 0.0629 | 0.0033 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.8829 | 0.0508 | 0.0564 | 0.049 | 0.0465 | 0.0043 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.8982 | 0.0851 | 0.0846 | 0.0657 | 0.0823 | 0.0028 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | 0.8109 | 0.0269 | 0.0391 | 0.0411 | 0.0141 | 0.0128 |
| sentiment | xlmr_base | lora | test | uncalibrated | 0.8138 | 0.0474 | 0.0515 | 0.0445 | 0.0421 | 0.0053 |


## 4. Advanced Conformal Prediction (LAC vs. APS vs. RAPS & Class-Conditional Gap)

| task | model | ft_method | condition | calibration_regime | conformal_method | empirical_coverage_mean | avg_set_size_mean | singleton_rate_mean | trivial_set_rate_mean | worst_class_coverage_mean | class_coverage_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | aps | 0.9438 | 1.0803 | 0.9197 | 0.0803 | 0.9354 | 0.0521 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | lac | 0.9175 | 1.0 | 1.0 | 0.0 | 0.9063 | 0.0285 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | raps | 0.9404 | 1.0658 | 0.9342 | 0.0658 | 0.9316 | 0.049 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8838 | 1.1542 | 0.8458 | 0.1542 | 0.8235 | 0.0816 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8292 | 1.0 | 1.0 | 0.0 | 0.7526 | 0.1474 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8749 | 1.1265 | 0.8735 | 0.1265 | 0.8129 | 0.088 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9414 | 1.0997 | 0.9003 | 0.0997 | 0.9303 | 0.0527 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | lac | 0.9092 | 1.0 | 1.0 | 0.0 | 0.8934 | 0.0253 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9373 | 1.0834 | 0.9166 | 0.0834 | 0.925 | 0.0497 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.543 | 1.0405 | 0.9595 | 0.0405 | 0.0429 | 0.8571 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.5255 | 1.0 | 1.0 | 0.0 | 0.0073 | 0.8927 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.537 | 1.0295 | 0.9705 | 0.0295 | 0.0304 | 0.8696 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9342 | 1.0991 | 0.9009 | 0.0991 | 0.914 | 0.0529 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9019 | 1.0 | 1.0 | 0.0 | 0.8785 | 0.028 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9299 | 1.0817 | 0.9183 | 0.0817 | 0.9088 | 0.0492 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9234 | 1.1117 | 0.8883 | 0.1117 | 0.888 | 0.056 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.8854 | 1.0 | 1.0 | 0.0 | 0.8383 | 0.0617 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.9177 | 1.091 | 0.909 | 0.091 | 0.8808 | 0.0517 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8871 | 1.1421 | 0.8579 | 0.1421 | 0.8029 | 0.0997 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.834 | 1.0 | 1.0 | 0.0 | 0.7233 | 0.1767 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8792 | 1.116 | 0.884 | 0.116 | 0.791 | 0.109 |
| offensive | banglabert | full_ft | test | source_temp_scaled | aps | 0.9456 | 1.082 | 0.918 | 0.082 | 0.9379 | 0.0531 |
| offensive | banglabert | full_ft | test | source_temp_scaled | lac | 0.9184 | 1.0 | 1.0 | 0.0 | 0.9077 | 0.0293 |
| offensive | banglabert | full_ft | test | source_temp_scaled | raps | 0.9411 | 1.0659 | 0.9341 | 0.0659 | 0.9332 | 0.049 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.9425 | 1.1606 | 0.8394 | 0.1606 | 0.9326 | 0.0526 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.8888 | 1.0075 | 0.9925 | 0.0075 | 0.8802 | 0.0198 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.938 | 1.1391 | 0.8609 | 0.1391 | 0.9274 | 0.0488 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8776 | 1.3027 | 0.6973 | 0.3027 | 0.8345 | 0.0655 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.7726 | 1.0159 | 0.9841 | 0.0159 | 0.7029 | 0.1971 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.867 | 1.2668 | 0.7332 | 0.2668 | 0.8228 | 0.0772 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9271 | 1.2076 | 0.7924 | 0.2076 | 0.8931 | 0.0619 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8721 | 1.0095 | 0.9905 | 0.0095 | 0.8175 | 0.0825 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9213 | 1.1834 | 0.8166 | 0.1834 | 0.8847 | 0.0588 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | aps | 0.7715 | 1.5035 | 0.4965 | 0.5035 | 0.5251 | 0.3749 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | lac | 0.5235 | 1.0115 | 0.9885 | 0.0115 | 0.0147 | 0.8853 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | raps | 0.7412 | 1.439 | 0.561 | 0.439 | 0.4634 | 0.4366 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9366 | 1.1784 | 0.8216 | 0.1784 | 0.9351 | 0.0379 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.887 | 1.0086 | 0.9914 | 0.0086 | 0.8802 | 0.0198 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9306 | 1.154 | 0.846 | 0.154 | 0.9287 | 0.0324 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9292 | 1.2234 | 0.7766 | 0.2234 | 0.9186 | 0.0389 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8645 | 1.0086 | 0.9914 | 0.0086 | 0.835 | 0.065 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9213 | 1.1896 | 0.8104 | 0.1896 | 0.9081 | 0.0334 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.913 | 1.3221 | 0.6779 | 0.3221 | 0.8691 | 0.0535 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8173 | 1.0125 | 0.9875 | 0.0125 | 0.7216 | 0.1784 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.9028 | 1.2762 | 0.7238 | 0.2762 | 0.8539 | 0.0505 |
| offensive | banglabert | lora | test | source_temp_scaled | aps | 0.9415 | 1.1455 | 0.8545 | 0.1455 | 0.9312 | 0.0527 |
| offensive | banglabert | lora | test | source_temp_scaled | lac | 0.899 | 1.0066 | 0.9934 | 0.0066 | 0.8897 | 0.0106 |
| offensive | banglabert | lora | test | source_temp_scaled | raps | 0.9364 | 1.125 | 0.875 | 0.125 | 0.9264 | 0.0472 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | aps | 0.9162 | 1.1597 | 0.8403 | 0.1597 | 0.9127 | 0.0197 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | lac | 0.8817 | 1.0397 | 0.9603 | 0.0397 | 0.8713 | 0.0287 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | raps | 0.9112 | 1.1373 | 0.8627 | 0.1373 | 0.9077 | 0.0147 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8586 | 1.1983 | 0.8017 | 0.1983 | 0.7917 | 0.1083 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8007 | 1.0523 | 0.9477 | 0.0523 | 0.7225 | 0.1775 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | raps | 0.85 | 1.1737 | 0.8263 | 0.1737 | 0.7809 | 0.1191 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9108 | 1.1661 | 0.8339 | 0.1661 | 0.9009 | 0.0208 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8734 | 1.041 | 0.959 | 0.041 | 0.8558 | 0.0442 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9037 | 1.1416 | 0.8584 | 0.1416 | 0.8923 | 0.0171 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.7108 | 1.375 | 0.625 | 0.375 | 0.4425 | 0.4575 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.5775 | 1.0988 | 0.9012 | 0.0988 | 0.2074 | 0.6926 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.6855 | 1.3268 | 0.6732 | 0.3268 | 0.398 | 0.502 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9172 | 1.1426 | 0.8574 | 0.1426 | 0.8895 | 0.0428 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.8856 | 1.0346 | 0.9654 | 0.0346 | 0.8548 | 0.0452 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9113 | 1.1211 | 0.8789 | 0.1211 | 0.8826 | 0.0378 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9051 | 1.1583 | 0.8417 | 0.1583 | 0.8725 | 0.0369 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.8666 | 1.0366 | 0.9634 | 0.0366 | 0.8281 | 0.0719 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.8983 | 1.135 | 0.865 | 0.135 | 0.8639 | 0.039 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8822 | 1.1906 | 0.8094 | 0.1906 | 0.8345 | 0.0655 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.8351 | 1.0483 | 0.9517 | 0.0483 | 0.7793 | 0.1207 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.874 | 1.1637 | 0.8363 | 0.1637 | 0.8248 | 0.0752 |
| offensive | indicbert | full_ft | test | source_temp_scaled | aps | 0.9313 | 1.1307 | 0.8693 | 0.1307 | 0.9151 | 0.0462 |
| offensive | indicbert | full_ft | test | source_temp_scaled | lac | 0.9023 | 1.0307 | 0.9693 | 0.0307 | 0.8843 | 0.0199 |
| offensive | indicbert | full_ft | test | source_temp_scaled | raps | 0.9255 | 1.1105 | 0.8895 | 0.1105 | 0.9085 | 0.0413 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.8896 | 1.6317 | 0.3683 | 0.6317 | 0.8658 | 0.0342 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.9009 | 1.6282 | 0.3718 | 0.6282 | 0.81 | 0.0978 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.8896 | 1.6317 | 0.3683 | 0.6317 | 0.8658 | 0.0342 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8663 | 1.6252 | 0.3748 | 0.6252 | 0.8203 | 0.0797 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8576 | 1.6007 | 0.3993 | 0.6007 | 0.7256 | 0.1744 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8663 | 1.6252 | 0.3748 | 0.6252 | 0.8203 | 0.0797 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.8837 | 1.6388 | 0.3612 | 0.6388 | 0.86 | 0.04 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8901 | 1.6375 | 0.3625 | 0.6375 | 0.7895 | 0.1116 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.8837 | 1.6388 | 0.3612 | 0.6388 | 0.86 | 0.04 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | aps | 0.8332 | 1.6348 | 0.3652 | 0.6348 | 0.7231 | 0.1769 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | lac | 0.8465 | 1.682 | 0.318 | 0.682 | 0.691 | 0.2856 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | raps | 0.8332 | 1.6348 | 0.3652 | 0.6348 | 0.7231 | 0.1769 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.8876 | 1.6234 | 0.3766 | 0.6234 | 0.8557 | 0.0443 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8875 | 1.5903 | 0.4097 | 0.5903 | 0.7815 | 0.1185 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8876 | 1.6234 | 0.3766 | 0.6234 | 0.8557 | 0.0443 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.8845 | 1.6222 | 0.3778 | 0.6222 | 0.8508 | 0.0492 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8858 | 1.5868 | 0.4132 | 0.5868 | 0.7775 | 0.1225 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8845 | 1.6222 | 0.3778 | 0.6222 | 0.8508 | 0.0492 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8795 | 1.6218 | 0.3782 | 0.6218 | 0.8395 | 0.0605 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8801 | 1.582 | 0.418 | 0.582 | 0.7642 | 0.1358 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8795 | 1.6218 | 0.3782 | 0.6218 | 0.8395 | 0.0605 |
| offensive | indicbert | lora | test | source_temp_scaled | aps | 0.8911 | 1.6247 | 0.3753 | 0.6247 | 0.8605 | 0.0395 |
| offensive | indicbert | lora | test | source_temp_scaled | lac | 0.8942 | 1.5914 | 0.4086 | 0.5914 | 0.7957 | 0.1043 |
| offensive | indicbert | lora | test | source_temp_scaled | raps | 0.8911 | 1.6247 | 0.3753 | 0.6247 | 0.8605 | 0.0395 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | aps | 0.9414 | 1.1094 | 0.8906 | 0.1094 | 0.9333 | 0.0497 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | lac | 0.9078 | 1.0 | 1.0 | 0.0 | 0.8928 | 0.0232 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | raps | 0.9364 | 1.0896 | 0.9104 | 0.0896 | 0.9282 | 0.0447 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8982 | 1.1786 | 0.8214 | 0.1786 | 0.8777 | 0.0267 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | lac | 0.834 | 1.0 | 1.0 | 0.0 | 0.8116 | 0.0884 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8886 | 1.1476 | 0.8524 | 0.1476 | 0.8675 | 0.033 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9368 | 1.1337 | 0.8663 | 0.1337 | 0.9182 | 0.0558 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8933 | 1.0 | 1.0 | 0.0 | 0.8589 | 0.0411 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9323 | 1.1116 | 0.8884 | 0.1116 | 0.9116 | 0.0535 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | aps | 0.5873 | 1.1325 | 0.8675 | 0.1325 | 0.1508 | 0.7492 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | lac | 0.5228 | 1.0 | 1.0 | 0.0 | 0.0325 | 0.8675 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | raps | 0.5712 | 1.0995 | 0.9005 | 0.0995 | 0.1194 | 0.7806 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.943 | 1.1218 | 0.8782 | 0.1218 | 0.9353 | 0.0501 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9053 | 1.0 | 1.0 | 0.0 | 0.902 | 0.0084 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.938 | 1.0983 | 0.9017 | 0.0983 | 0.9307 | 0.0448 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9341 | 1.1425 | 0.8575 | 0.1425 | 0.9239 | 0.0434 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.889 | 1.0 | 1.0 | 0.0 | 0.8809 | 0.0191 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.9287 | 1.1177 | 0.8823 | 0.1177 | 0.9174 | 0.0391 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.9199 | 1.1908 | 0.8092 | 0.1908 | 0.9032 | 0.0352 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.8598 | 1.0 | 1.0 | 0.0 | 0.8408 | 0.0592 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.9122 | 1.1567 | 0.8433 | 0.1567 | 0.8952 | 0.0279 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | aps | 0.9471 | 1.1008 | 0.8992 | 0.1008 | 0.945 | 0.0491 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | lac | 0.915 | 1.0 | 1.0 | 0.0 | 0.9076 | 0.0229 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | raps | 0.9432 | 1.0826 | 0.9174 | 0.0826 | 0.941 | 0.0454 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.9172 | 1.2807 | 0.7193 | 0.2807 | 0.8889 | 0.0462 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.8941 | 1.1499 | 0.8501 | 0.1499 | 0.8616 | 0.0384 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.9121 | 1.2578 | 0.7422 | 0.2578 | 0.8821 | 0.0427 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8878 | 1.3934 | 0.6066 | 0.3934 | 0.8367 | 0.0633 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8455 | 1.2113 | 0.7887 | 0.2113 | 0.7871 | 0.1129 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8786 | 1.3655 | 0.6345 | 0.3655 | 0.8248 | 0.0752 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9115 | 1.3245 | 0.6755 | 0.3245 | 0.8668 | 0.0573 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8774 | 1.1711 | 0.8289 | 0.1711 | 0.8149 | 0.0851 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9053 | 1.3 | 0.7 | 0.3 | 0.8574 | 0.0543 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | aps | 0.7802 | 1.5288 | 0.4713 | 0.5287 | 0.6 | 0.3 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | lac | 0.6592 | 1.288 | 0.712 | 0.288 | 0.3586 | 0.5414 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | raps | 0.7598 | 1.493 | 0.507 | 0.493 | 0.5644 | 0.3356 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9175 | 1.2853 | 0.7147 | 0.2853 | 0.8915 | 0.0458 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8922 | 1.1479 | 0.8521 | 0.1479 | 0.8631 | 0.0369 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9117 | 1.2629 | 0.7371 | 0.2629 | 0.8845 | 0.0412 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9126 | 1.3288 | 0.6712 | 0.3288 | 0.8762 | 0.0522 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8806 | 1.1711 | 0.8289 | 0.1711 | 0.8346 | 0.0654 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9054 | 1.3008 | 0.6992 | 0.3008 | 0.8666 | 0.0474 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9013 | 1.3962 | 0.6038 | 0.3962 | 0.8517 | 0.0552 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8652 | 1.2268 | 0.7732 | 0.2268 | 0.8018 | 0.0982 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.895 | 1.3693 | 0.6307 | 0.3693 | 0.8429 | 0.0572 |
| offensive | xlmr_base | lora | test | source_temp_scaled | aps | 0.9201 | 1.2402 | 0.7598 | 0.2402 | 0.9055 | 0.036 |
| offensive | xlmr_base | lora | test | source_temp_scaled | lac | 0.9008 | 1.1215 | 0.8785 | 0.1215 | 0.8862 | 0.0172 |
| offensive | xlmr_base | lora | test | source_temp_scaled | raps | 0.9158 | 1.2196 | 0.7804 | 0.2196 | 0.9011 | 0.0317 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | aps | 0.9085 | 1.8756 | 0.2448 | 0.1204 | 0.878 | 0.0421 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | lac | 0.9234 | 1.803 | 0.391 | 0.194 | 0.885 | 0.0522 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | raps | 0.9025 | 1.8557 | 0.2279 | 0.0836 | 0.868 | 0.0437 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.7498 | 1.9883 | 0.1734 | 0.1617 | 0.4007 | 0.4993 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.8065 | 2.1224 | 0.2117 | 0.3341 | 0.5624 | 0.3376 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.7361 | 1.9604 | 0.1575 | 0.1179 | 0.3567 | 0.5433 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.8975 | 1.801 | 0.2859 | 0.0869 | 0.8834 | 0.0255 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9156 | 1.7378 | 0.4116 | 0.1494 | 0.8931 | 0.0372 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.8959 | 1.7962 | 0.266 | 0.0622 | 0.8776 | 0.0294 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.8953 | 1.8591 | 0.2418 | 0.1009 | 0.8729 | 0.0347 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.9115 | 1.8154 | 0.3576 | 0.1731 | 0.8865 | 0.0371 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.8912 | 1.849 | 0.2204 | 0.0694 | 0.8642 | 0.0412 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8931 | 1.9672 | 0.1634 | 0.1306 | 0.8523 | 0.06 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.9174 | 1.9646 | 0.2725 | 0.2371 | 0.8909 | 0.0436 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8881 | 1.941 | 0.1491 | 0.0901 | 0.8433 | 0.062 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | aps | 0.9031 | 1.7626 | 0.3201 | 0.0828 | 0.8775 | 0.0314 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | lac | 0.9131 | 1.6725 | 0.4646 | 0.1371 | 0.8603 | 0.0476 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | raps | 0.9004 | 1.759 | 0.2985 | 0.0575 | 0.8794 | 0.0262 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | aps | 0.9124 | 1.9721 | 0.2189 | 0.191 | 0.8931 | 0.0348 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | lac | 0.9134 | 1.8965 | 0.3552 | 0.2517 | 0.9011 | 0.0259 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | raps | 0.9085 | 1.993 | 0.1761 | 0.1692 | 0.8864 | 0.0429 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | aps | 0.6654 | 1.9998 | 0.0838 | 0.0836 | 0.0986 | 0.8014 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | lac | 0.6692 | 2.0027 | 0.0371 | 0.0398 | 0.0745 | 0.8255 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | raps | 0.6629 | 2.0042 | 0.0575 | 0.0617 | 0.0766 | 0.8234 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9082 | 1.9953 | 0.1653 | 0.1606 | 0.8964 | 0.0422 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.9124 | 1.9046 | 0.3041 | 0.2087 | 0.9044 | 0.0182 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9079 | 2.0088 | 0.1287 | 0.1375 | 0.8888 | 0.0631 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9146 | 2.09 | 0.1146 | 0.2046 | 0.8997 | 0.0483 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.9315 | 2.0506 | 0.2154 | 0.266 | 0.9265 | 0.0404 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9125 | 2.0859 | 0.0879 | 0.1738 | 0.8929 | 0.0606 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9165 | 2.2354 | 0.0546 | 0.29 | 0.8777 | 0.0698 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.9409 | 2.2674 | 0.1197 | 0.3871 | 0.9138 | 0.0662 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.9115 | 2.209 | 0.039 | 0.2479 | 0.8658 | 0.0778 |
| sentiment | banglabert | lora | test | source_temp_scaled | aps | 0.9053 | 1.9276 | 0.2037 | 0.1313 | 0.8832 | 0.0228 |
| sentiment | banglabert | lora | test | source_temp_scaled | lac | 0.9029 | 1.7928 | 0.3724 | 0.1651 | 0.8743 | 0.0297 |
| sentiment | banglabert | lora | test | source_temp_scaled | raps | 0.905 | 1.9479 | 0.1656 | 0.1135 | 0.8739 | 0.0365 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | aps | 0.9045 | 2.2557 | 0.0577 | 0.3134 | 0.8593 | 0.0523 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | lac | 0.9174 | 2.1652 | 0.2577 | 0.4229 | 0.8476 | 0.0699 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | raps | 0.9045 | 2.2697 | 0.0279 | 0.2975 | 0.8593 | 0.0523 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.809 | 2.2684 | 0.0279 | 0.2963 | 0.5528 | 0.3472 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.7602 | 2.0754 | 0.3823 | 0.4577 | 0.4834 | 0.4166 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.8107 | 2.2774 | 0.0035 | 0.2808 | 0.5506 | 0.3494 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.8932 | 2.2341 | 0.069 | 0.3031 | 0.8735 | 0.0338 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9128 | 2.1569 | 0.2606 | 0.4175 | 0.8848 | 0.0311 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.8937 | 2.2554 | 0.0326 | 0.2881 | 0.8695 | 0.0432 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.8918 | 2.2385 | 0.0653 | 0.3038 | 0.87 | 0.0315 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.906 | 2.1459 | 0.2659 | 0.4118 | 0.8732 | 0.0338 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.8906 | 2.2547 | 0.0332 | 0.2879 | 0.8657 | 0.0372 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8935 | 2.2634 | 0.0503 | 0.3137 | 0.8724 | 0.0327 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.9053 | 2.1979 | 0.239 | 0.4369 | 0.8649 | 0.0423 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8937 | 2.2747 | 0.0254 | 0.3001 | 0.8691 | 0.0358 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | aps | 0.9016 | 2.2243 | 0.0719 | 0.2962 | 0.8822 | 0.0372 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | lac | 0.9144 | 2.1199 | 0.2784 | 0.3982 | 0.8886 | 0.0288 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | raps | 0.9006 | 2.2443 | 0.0363 | 0.2806 | 0.8796 | 0.0428 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | aps | 0.9005 | 2.5264 | 0.0 | 0.5264 | 0.5238 | 0.3762 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | lac | 0.8995 | 2.4756 | 0.0 | 0.4756 | 0.519 | 0.381 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | raps | 0.9005 | 2.5264 | 0.0 | 0.5264 | 0.5238 | 0.3762 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | aps | 0.8968 | 2.5142 | 0.0 | 0.5142 | 0.5561 | 0.3439 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | lac | 0.8266 | 2.2308 | 0.0 | 0.2308 | 0.2545 | 0.6455 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | raps | 0.8968 | 2.5142 | 0.0 | 0.5142 | 0.5561 | 0.3439 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9003 | 2.5476 | 0.0 | 0.5476 | 0.5828 | 0.3172 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8987 | 2.4544 | 0.0 | 0.4544 | 0.576 | 0.324 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9003 | 2.5476 | 0.0 | 0.5476 | 0.5828 | 0.3172 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9001 | 2.5493 | 0.0 | 0.5493 | 0.5822 | 0.3178 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.9026 | 2.451 | 0.0 | 0.451 | 0.5926 | 0.3074 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9001 | 2.5493 | 0.0 | 0.5493 | 0.5822 | 0.3178 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9006 | 2.5484 | 0.0 | 0.5484 | 0.584 | 0.316 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8979 | 2.4432 | 0.0 | 0.4432 | 0.5729 | 0.3271 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.9006 | 2.5484 | 0.0 | 0.5484 | 0.584 | 0.316 |
| sentiment | indicbert | lora | test | source_temp_scaled | aps | 0.9003 | 2.5482 | 0.0 | 0.5482 | 0.5828 | 0.3172 |
| sentiment | indicbert | lora | test | source_temp_scaled | lac | 0.9007 | 2.4629 | 0.0 | 0.4629 | 0.5846 | 0.3154 |
| sentiment | indicbert | lora | test | source_temp_scaled | raps | 0.9003 | 2.5482 | 0.0 | 0.5482 | 0.5828 | 0.3172 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | aps | 0.8816 | 1.9652 | 0.2129 | 0.1781 | 0.8519 | 0.0481 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | lac | 0.8896 | 1.8856 | 0.3453 | 0.2308 | 0.8641 | 0.0359 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | raps | 0.8776 | 1.9771 | 0.1801 | 0.1572 | 0.8494 | 0.0506 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | aps | 0.7704 | 1.8269 | 0.343 | 0.1699 | 0.5442 | 0.3558 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | lac | 0.7264 | 1.6801 | 0.5821 | 0.2622 | 0.323 | 0.577 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | raps | 0.7731 | 1.8473 | 0.2918 | 0.1391 | 0.5657 | 0.3343 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9035 | 1.9135 | 0.2296 | 0.1431 | 0.8904 | 0.0189 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9185 | 1.8279 | 0.3728 | 0.2007 | 0.8996 | 0.0369 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9029 | 1.9276 | 0.1915 | 0.1191 | 0.887 | 0.0214 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.899 | 1.9451 | 0.2147 | 0.1599 | 0.8915 | 0.0126 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.9096 | 1.8615 | 0.3624 | 0.2238 | 0.8849 | 0.0267 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.8979 | 1.9538 | 0.1806 | 0.1344 | 0.886 | 0.0188 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8953 | 2.0126 | 0.1694 | 0.1821 | 0.8652 | 0.0384 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.9021 | 1.9304 | 0.3301 | 0.2606 | 0.8719 | 0.043 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8924 | 2.0118 | 0.1374 | 0.1491 | 0.8567 | 0.0433 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | aps | 0.9035 | 1.884 | 0.2519 | 0.1359 | 0.8806 | 0.0276 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | lac | 0.9185 | 1.8084 | 0.3774 | 0.1857 | 0.8947 | 0.0436 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | raps | 0.9041 | 1.9026 | 0.2135 | 0.1162 | 0.8788 | 0.0267 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | aps | 0.8577 | 2.0378 | 0.1731 | 0.2109 | 0.8282 | 0.0718 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | lac | 0.8617 | 1.9682 | 0.3214 | 0.2896 | 0.7952 | 0.1048 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | raps | 0.8507 | 2.0378 | 0.1522 | 0.19 | 0.8205 | 0.0795 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | aps | 0.8371 | 1.9127 | 0.2841 | 0.1968 | 0.6865 | 0.2135 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | lac | 0.8264 | 1.8463 | 0.4896 | 0.3358 | 0.5412 | 0.3588 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | raps | 0.8396 | 1.9197 | 0.2488 | 0.1684 | 0.6851 | 0.2149 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.8935 | 2.0118 | 0.1884 | 0.2001 | 0.879 | 0.0231 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8957 | 1.9194 | 0.3557 | 0.2751 | 0.8676 | 0.0333 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8937 | 2.0135 | 0.16 | 0.1735 | 0.8808 | 0.023 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.8785 | 2.0134 | 0.1888 | 0.2022 | 0.8649 | 0.0351 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8874 | 1.9384 | 0.3449 | 0.2832 | 0.8629 | 0.0371 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8782 | 2.0125 | 0.1624 | 0.1749 | 0.863 | 0.037 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8625 | 2.0271 | 0.1722 | 0.1993 | 0.8206 | 0.0794 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8666 | 1.9178 | 0.3531 | 0.2709 | 0.8339 | 0.0661 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8632 | 2.0285 | 0.1432 | 0.1718 | 0.8145 | 0.0855 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | aps | 0.8959 | 2.0115 | 0.19 | 0.2015 | 0.8736 | 0.0272 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | lac | 0.9028 | 1.949 | 0.3332 | 0.2822 | 0.8747 | 0.0319 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | raps | 0.8965 | 2.0137 | 0.1626 | 0.1763 | 0.8718 | 0.0362 |


## 5. Cross-Model Mahalanobis Percentile Ranks (Part I.6)

| task | model | ft_method | condition | mean_maha_percentile_mean | median_maha_percentile_mean | frac_above_95th_train_pct_mean | maha_error_gap_pct_mean | spearman_maha_nll_rho_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | 53.0628 | 53.0733 | 0.0885 | 26.5027 | 0.3952 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | 59.402 | 62.6833 | 0.116 | 19.8369 | 0.531 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | 53.5302 | 54.841 | 0.0913 | 26.2452 | 0.495 |
| offensive | banglabert | full_ft | shift_d2_test | 86.5929 | 88.1369 | 0.3728 | -1.0028 | 0.019 |
| offensive | banglabert | full_ft | shift_d3_noise05 | 55.527 | 58.262 | 0.1014 | 25.7367 | 0.4248 |
| offensive | banglabert | full_ft | shift_d3_noise10 | 57.4177 | 60.0805 | 0.1011 | 22.2492 | 0.4629 |
| offensive | banglabert | full_ft | shift_d3_noise20 | 63.5659 | 67.5922 | 0.1184 | 18.1679 | 0.5246 |
| offensive | banglabert | full_ft | test | 54.9203 | 57.9716 | 0.1044 | 27.8574 | 0.3498 |
| offensive | banglabert | lora | shift_d1_barishal_test | 56.4203 | 59.2794 | 0.1355 | 14.9984 | 0.4812 |
| offensive | banglabert | lora | shift_d1_chittagong_test | 51.7997 | 49.5252 | 0.0652 | 8.4327 | 0.4381 |
| offensive | banglabert | lora | shift_d1_noakhali_test | 52.0038 | 51.2293 | 0.0861 | 12.2908 | 0.5569 |
| offensive | banglabert | lora | shift_d2_test | 99.7182 | 99.9746 | 0.9858 | -0.2859 | -0.0679 |
| offensive | banglabert | lora | shift_d3_noise05 | 57.5955 | 59.6842 | 0.1423 | 13.3217 | 0.4813 |
| offensive | banglabert | lora | shift_d3_noise10 | 58.239 | 59.22 | 0.1214 | 12.9023 | 0.4909 |
| offensive | banglabert | lora | shift_d3_noise20 | 63.8703 | 65.1272 | 0.1262 | 8.9282 | 0.3951 |
| offensive | banglabert | lora | test | 58.8547 | 64.9788 | 0.185 | 14.7422 | 0.425 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | 54.6349 | 55.9093 | 0.0861 | 14.7481 | 0.2497 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | 55.4954 | 57.5074 | 0.0977 | 11.7601 | 0.2755 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | 55.2303 | 57.3591 | 0.0898 | 15.2869 | 0.261 |
| offensive | indicbert | full_ft | shift_d2_test | 73.3247 | 75.9517 | 0.3223 | 1.4524 | 0.079 |
| offensive | indicbert | full_ft | shift_d3_noise05 | 54.1146 | 55.5935 | 0.0891 | 14.9223 | 0.2174 |
| offensive | indicbert | full_ft | shift_d3_noise10 | 54.9629 | 57.0941 | 0.0964 | 13.0411 | 0.2366 |
| offensive | indicbert | full_ft | shift_d3_noise20 | 56.4225 | 58.8279 | 0.1069 | 12.4614 | 0.2833 |
| offensive | indicbert | full_ft | test | 53.7386 | 54.9958 | 0.0859 | 15.1905 | 0.1782 |
| offensive | indicbert | lora | shift_d1_barishal_test | 52.2981 | 50.8944 | 0.0753 | 2.2115 | -0.1856 |
| offensive | indicbert | lora | shift_d1_chittagong_test | 55.4155 | 56.4519 | 0.0924 | 2.1441 | -0.1287 |
| offensive | indicbert | lora | shift_d1_noakhali_test | 53.2004 | 51.3735 | 0.0908 | 2.1563 | -0.1458 |
| offensive | indicbert | lora | shift_d2_test | 98.6796 | 99.8983 | 0.9465 | 1.0364 | 0.0545 |
| offensive | indicbert | lora | shift_d3_noise05 | 54.5424 | 54.5634 | 0.1305 | 1.4865 | -0.1993 |
| offensive | indicbert | lora | shift_d3_noise10 | 55.0738 | 55.4621 | 0.1297 | 1.3697 | -0.199 |
| offensive | indicbert | lora | shift_d3_noise20 | 56.1302 | 56.9288 | 0.132 | 1.218 | -0.1938 |
| offensive | indicbert | lora | test | 54.4136 | 54.4256 | 0.1329 | 1.489 | -0.2101 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | 50.7483 | 51.0089 | 0.0749 | 16.4465 | 0.2664 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | 54.5337 | 56.8673 | 0.0784 | 16.4793 | 0.3661 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | 50.9934 | 51.4116 | 0.0732 | 14.3081 | 0.3251 |
| offensive | xlmr_base | full_ft | shift_d2_test | 57.8241 | 56.1742 | 0.0357 | -5.8966 | -0.1445 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | 53.8048 | 54.9407 | 0.0888 | 17.6117 | 0.2998 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | 54.1176 | 55.7037 | 0.093 | 17.5791 | 0.335 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | 55.5096 | 57.6007 | 0.0908 | 16.1273 | 0.3748 |
| offensive | xlmr_base | full_ft | test | 53.4455 | 54.2433 | 0.092 | 17.8815 | 0.2493 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | 45.647 | 42.5901 | 0.0546 | -0.5657 | 0.1049 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | 40.5905 | 36.3629 | 0.0341 | 0.585 | 0.0076 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | 44.8903 | 42.2849 | 0.0498 | 2.6909 | 0.1125 |
| offensive | xlmr_base | lora | shift_d2_test | 72.9463 | 75.8393 | 0.1555 | -9.1247 | -0.2482 |
| offensive | xlmr_base | lora | shift_d3_noise05 | 52.7008 | 53.2196 | 0.0818 | 1.6303 | 0.0964 |
| offensive | xlmr_base | lora | shift_d3_noise10 | 47.7394 | 45.9771 | 0.0594 | 1.0172 | 0.0494 |
| offensive | xlmr_base | lora | shift_d3_noise20 | 42.8069 | 39.4913 | 0.0366 | -0.5416 | -0.0083 |
| offensive | xlmr_base | lora | test | 58.281 | 61.755 | 0.1223 | 2.2243 | 0.1242 |
| sentiment | banglabert | full_ft | shift_d1_test | 58.7784 | 63.8866 | 0.1244 | 17.3437 | 0.4571 |
| sentiment | banglabert | full_ft | shift_d2_test | 93.9629 | 96.5473 | 0.6224 | 2.0586 | 0.0818 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | 58.3037 | 63.2017 | 0.1247 | 18.941 | 0.4745 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | 61.5711 | 66.855 | 0.1419 | 14.383 | 0.3947 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | 69.9195 | 77.9454 | 0.2101 | 12.3885 | 0.3355 |
| sentiment | banglabert | full_ft | test | 57.9103 | 62.3813 | 0.1332 | 22.1245 | 0.5101 |
| sentiment | banglabert | lora | shift_d1_test | 55.4339 | 61.9538 | 0.1234 | 9.6972 | 0.2794 |
| sentiment | banglabert | lora | shift_d2_test | 99.4246 | 99.6134 | 0.9938 | -0.2516 | -0.1331 |
| sentiment | banglabert | lora | shift_d3_noise05 | 55.8201 | 60.6681 | 0.0913 | 7.9954 | 0.2024 |
| sentiment | banglabert | lora | shift_d3_noise10 | 60.4394 | 67.4044 | 0.1465 | 4.4579 | 0.1395 |
| sentiment | banglabert | lora | shift_d3_noise20 | 74.1152 | 87.7878 | 0.3532 | 4.9987 | 0.1614 |
| sentiment | banglabert | lora | test | 55.2316 | 57.6996 | 0.1019 | 12.6851 | 0.2915 |
| sentiment | indicbert | full_ft | shift_d1_test | 59.9548 | 62.7731 | 0.1154 | 11.1522 | 0.3047 |
| sentiment | indicbert | full_ft | shift_d2_test | 95.0735 | 98.2143 | 0.7692 | 0.2094 | 0.0449 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | 55.4264 | 57.9853 | 0.0885 | 9.4135 | 0.2503 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | 55.8804 | 57.8981 | 0.0966 | 8.1948 | 0.2212 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | 57.6959 | 60.5042 | 0.0943 | 7.5753 | 0.1941 |
| sentiment | indicbert | full_ft | test | 54.6011 | 56.2973 | 0.0894 | 9.8844 | 0.2522 |
| sentiment | indicbert | lora | shift_d1_test | 62.9723 | 69.3256 | 0.1264 | 1.3895 | 0.1076 |
| sentiment | indicbert | lora | shift_d2_test | 99.3689 | 99.6197 | 0.9806 | 0.0017 | -0.0033 |
| sentiment | indicbert | lora | shift_d3_noise05 | 54.6771 | 57.2248 | 0.0738 | 2.6705 | 0.0828 |
| sentiment | indicbert | lora | shift_d3_noise10 | 55.735 | 57.7048 | 0.0854 | 3.9887 | 0.1082 |
| sentiment | indicbert | lora | shift_d3_noise20 | 57.5247 | 60.0798 | 0.0919 | 1.9341 | 0.0657 |
| sentiment | indicbert | lora | test | 54.7957 | 57.5315 | 0.0746 | 3.5244 | 0.0921 |
| sentiment | xlmr_base | full_ft | shift_d1_test | 51.4689 | 51.9391 | 0.0637 | 16.9765 | 0.4457 |
| sentiment | xlmr_base | full_ft | shift_d2_test | 56.6836 | 57.3971 | 0.0214 | 10.0685 | 0.2549 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | 51.8618 | 53.1376 | 0.0712 | 19.1498 | 0.5162 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | 49.7312 | 49.0756 | 0.0576 | 17.5394 | 0.4841 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | 47.7329 | 45.917 | 0.0399 | 15.5717 | 0.4289 |
| sentiment | xlmr_base | full_ft | test | 54.8278 | 57.2731 | 0.0946 | 20.2915 | 0.5289 |
| sentiment | xlmr_base | lora | shift_d1_test | 55.8023 | 59.9286 | 0.0915 | 15.3319 | 0.3354 |
| sentiment | xlmr_base | lora | shift_d2_test | 92.794 | 95.8466 | 0.5764 | -0.7508 | 0.0397 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | 53.6494 | 55.2458 | 0.0809 | 10.823 | 0.2659 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | 54.1059 | 55.4727 | 0.0778 | 10.063 | 0.2445 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | 59.2217 | 62.6166 | 0.0959 | 7.4945 | 0.1878 |
| sentiment | xlmr_base | lora | test | 55.6733 | 58.5368 | 0.0984 | 11.8264 | 0.2851 |

