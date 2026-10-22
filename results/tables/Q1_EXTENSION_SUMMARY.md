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
| sentiment | banglabert | full_ft | 6 | 1.0 | 0.0 | 0.8697 | -0.8286 | 0.8286 | 0.8857 | 0.8286 |
| sentiment | banglabert | lora | 6 | 0.9429 | 0.0048 | 0.8046 | -0.6571 | 0.0857 | -0.1429 | 0.6571 |
| sentiment | indicbert | full_ft | 6 | 0.9429 | 0.0048 | 0.8045 | -0.6571 | 0.7714 | 0.7714 | 0.6571 |
| sentiment | indicbert | lora | 6 | -0.1429 | 0.7872 | 0.0147 | 0.5429 | -0.0286 | nan | -0.6571 |
| sentiment | xlmr_base | full_ft | 6 | 1.0 | 0.0 | 0.8361 | -0.8286 | 0.8857 | 0.8286 | -0.1429 |
| sentiment | xlmr_base | lora | 6 | 0.9429 | 0.0048 | 0.9598 | -0.9429 | 0.7714 | 0.9429 | 0.7714 |


## 2. MAPIE Conformal Verification Cross-Check (Part I.3)

| task | run_id | exact_coverage | exact_set_size | mapie_coverage | mapie_set_size | abs_coverage_diff | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | sentiment_banglabert_full_ft_seed42 | 0.9118 | 1.6574 | 0.9118 | 1.6574 | 0.0 | Fallback to Exact I.2 (TypeError) |
| offensive | offensive_banglabert_full_ft_seed42 | 0.9185 | 1.0 | 0.9185 | 1.0 | 0.0 | Fallback to Exact I.2 (TypeError) |


## 3. Extended Calibration Metrics (Adaptive ECE, Classwise ECE, Overconfidence)

| task | model | ft_method | condition | calibration_regime | nll_mean | top_label_ece_15_mean | adaptive_ece_15_mean | classwise_ece_15_mean | overconfidence_ece_15_mean | underconfidence_ece_15_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.2411 | 0.0305 | 0.0349 | 0.0364 | 0.0281 | 0.0023 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.2399 | 0.029 | 0.0337 | 0.0359 | 0.0261 | 0.0029 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | uncalibrated | 0.3138 | 0.0576 | 0.0575 | 0.0605 | 0.0572 | 0.0004 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.4597 | 0.0819 | 0.0828 | 0.093 | 0.0811 | 0.0008 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.4471 | 0.0746 | 0.0756 | 0.0876 | 0.0729 | 0.0017 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.6289 | 0.1228 | 0.1217 | 0.1276 | 0.1222 | 0.0006 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.2514 | 0.0333 | 0.038 | 0.0391 | 0.0288 | 0.0045 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.2499 | 0.0323 | 0.0367 | 0.0379 | 0.027 | 0.0053 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.3198 | 0.0607 | 0.0601 | 0.0628 | 0.0599 | 0.0009 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | 1.6467 | 0.4252 | 0.4241 | 0.4261 | 0.4242 | 0.001 |
| offensive | banglabert | full_ft | shift_d2_test | target_temp_scaled | 1.4403 | 0.4026 | 0.4015 | 0.4034 | 0.4011 | 0.0015 |
| offensive | banglabert | full_ft | shift_d2_test | uncalibrated | 2.5718 | 0.4558 | 0.4541 | 0.4565 | 0.455 | 0.0009 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.2747 | 0.0388 | 0.0414 | 0.0451 | 0.0359 | 0.0029 |
| offensive | banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.3587 | 0.0667 | 0.0672 | 0.0692 | 0.0666 | 0.0001 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.3207 | 0.0464 | 0.0482 | 0.0563 | 0.0449 | 0.0015 |
| offensive | banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.4255 | 0.0796 | 0.0795 | 0.0838 | 0.0794 | 0.0002 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.4453 | 0.0816 | 0.0821 | 0.0952 | 0.0808 | 0.0008 |
| offensive | banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.6094 | 0.1211 | 0.1211 | 0.1269 | 0.121 | 0.0001 |
| offensive | banglabert | full_ft | test | source_temp_scaled | 0.2317 | 0.0289 | 0.036 | 0.0331 | 0.0271 | 0.0018 |
| offensive | banglabert | full_ft | test | uncalibrated | 0.2996 | 0.0564 | 0.0566 | 0.0575 | 0.0561 | 0.0003 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | 0.2786 | 0.019 | 0.0216 | 0.0307 | 0.0083 | 0.0107 |
| offensive | banglabert | lora | shift_d1_barishal_test | target_temp_scaled | 0.2786 | 0.0192 | 0.0216 | 0.0305 | 0.0083 | 0.0109 |
| offensive | banglabert | lora | shift_d1_barishal_test | uncalibrated | 0.2899 | 0.0393 | 0.0397 | 0.0409 | 0.0384 | 0.0009 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.494 | 0.0556 | 0.0593 | 0.0813 | 0.0539 | 0.0017 |
| offensive | banglabert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.4867 | 0.0452 | 0.046 | 0.0791 | 0.0396 | 0.0056 |
| offensive | banglabert | lora | shift_d1_chittagong_test | uncalibrated | 0.5459 | 0.1019 | 0.1017 | 0.1142 | 0.1012 | 0.0008 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.3274 | 0.0212 | 0.0252 | 0.0734 | 0.0079 | 0.0133 |
| offensive | banglabert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.3285 | 0.0222 | 0.0277 | 0.0789 | 0.0037 | 0.0185 |
| offensive | banglabert | lora | shift_d1_noakhali_test | uncalibrated | 0.3434 | 0.0428 | 0.0427 | 0.0719 | 0.0402 | 0.0026 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | 0.8328 | 0.2458 | 0.2491 | 0.2449 | 0.2395 | 0.0063 |
| offensive | banglabert | lora | shift_d2_test | target_temp_scaled | 0.7759 | 0.2008 | 0.2067 | 0.2008 | 0.1949 | 0.006 |
| offensive | banglabert | lora | shift_d2_test | uncalibrated | 0.9656 | 0.3077 | 0.3109 | 0.3074 | 0.3031 | 0.0046 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.2913 | 0.0127 | 0.0163 | 0.0236 | 0.0021 | 0.0106 |
| offensive | banglabert | lora | shift_d3_noise05 | uncalibrated | 0.3005 | 0.0361 | 0.0355 | 0.038 | 0.035 | 0.0011 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.3322 | 0.0185 | 0.0199 | 0.022 | 0.0068 | 0.0117 |
| offensive | banglabert | lora | shift_d3_noise10 | uncalibrated | 0.343 | 0.043 | 0.0428 | 0.0433 | 0.0426 | 0.0004 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.4194 | 0.018 | 0.02 | 0.0326 | 0.0105 | 0.0075 |
| offensive | banglabert | lora | shift_d3_noise20 | uncalibrated | 0.4352 | 0.0569 | 0.0566 | 0.0645 | 0.0566 | 0.0003 |
| offensive | banglabert | lora | test | source_temp_scaled | 0.2593 | 0.016 | 0.018 | 0.0291 | 0.006 | 0.01 |
| offensive | banglabert | lora | test | uncalibrated | 0.2695 | 0.0362 | 0.0363 | 0.0383 | 0.0352 | 0.001 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.343 | 0.045 | 0.0476 | 0.0506 | 0.0413 | 0.0037 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.3416 | 0.0397 | 0.0459 | 0.0467 | 0.0369 | 0.0028 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | uncalibrated | 0.424 | 0.0903 | 0.0883 | 0.0935 | 0.0892 | 0.0011 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.5345 | 0.1116 | 0.1146 | 0.1134 | 0.1115 | 0.0001 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.5181 | 0.1012 | 0.104 | 0.1034 | 0.0999 | 0.0014 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.711 | 0.1669 | 0.1665 | 0.1672 | 0.1667 | 0.0002 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.3586 | 0.0502 | 0.0563 | 0.0535 | 0.049 | 0.0011 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.3557 | 0.0451 | 0.053 | 0.049 | 0.0433 | 0.0018 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.4467 | 0.0992 | 0.0995 | 0.1003 | 0.0992 | 0.0 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | 0.9949 | 0.2951 | 0.2971 | 0.2969 | 0.294 | 0.0011 |
| offensive | indicbert | full_ft | shift_d2_test | target_temp_scaled | 0.9241 | 0.2686 | 0.2721 | 0.2705 | 0.2681 | 0.0006 |
| offensive | indicbert | full_ft | shift_d2_test | uncalibrated | 1.3658 | 0.372 | 0.37 | 0.3729 | 0.3698 | 0.0022 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.3327 | 0.0427 | 0.0485 | 0.0454 | 0.041 | 0.0017 |
| offensive | indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.4139 | 0.0889 | 0.0886 | 0.0901 | 0.0886 | 0.0002 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.3825 | 0.0557 | 0.0599 | 0.0575 | 0.0549 | 0.0007 |
| offensive | indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.4875 | 0.1065 | 0.1052 | 0.107 | 0.1058 | 0.0007 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.4626 | 0.0803 | 0.0818 | 0.0803 | 0.08 | 0.0004 |
| offensive | indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.6015 | 0.1339 | 0.1335 | 0.1343 | 0.1337 | 0.0002 |
| offensive | indicbert | full_ft | test | source_temp_scaled | 0.2933 | 0.0352 | 0.04 | 0.0368 | 0.031 | 0.0043 |
| offensive | indicbert | full_ft | test | uncalibrated | 0.3571 | 0.0747 | 0.0746 | 0.0753 | 0.0745 | 0.0002 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | 0.6422 | 0.1086 | 0.1135 | 0.1072 | 0.0004 | 0.1081 |
| offensive | indicbert | lora | shift_d1_barishal_test | target_temp_scaled | 0.6423 | 0.1087 | 0.1135 | 0.1074 | 0.0004 | 0.1083 |
| offensive | indicbert | lora | shift_d1_barishal_test | uncalibrated | 0.63 | 0.0828 | 0.0881 | 0.0854 | 0.0052 | 0.0776 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.6666 | 0.0549 | 0.0676 | 0.0598 | 0.0071 | 0.0479 |
| offensive | indicbert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.6666 | 0.0546 | 0.0676 | 0.0593 | 0.0067 | 0.0479 |
| offensive | indicbert | lora | shift_d1_chittagong_test | uncalibrated | 0.6671 | 0.0541 | 0.0721 | 0.0619 | 0.0248 | 0.0293 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.654 | 0.076 | 0.0898 | 0.0798 | 0.002 | 0.074 |
| offensive | indicbert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.6541 | 0.077 | 0.0901 | 0.0802 | 0.0023 | 0.0747 |
| offensive | indicbert | lora | shift_d1_noakhali_test | uncalibrated | 0.647 | 0.0591 | 0.0727 | 0.0663 | 0.0108 | 0.0483 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | 0.6997 | 0.0547 | 0.0828 | 0.0601 | 0.0402 | 0.0145 |
| offensive | indicbert | lora | shift_d2_test | target_temp_scaled | 0.6969 | 0.0487 | 0.0786 | 0.0541 | 0.0341 | 0.0146 |
| offensive | indicbert | lora | shift_d2_test | uncalibrated | 0.713 | 0.0755 | 0.0975 | 0.0865 | 0.0655 | 0.0099 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.6403 | 0.1046 | 0.1056 | 0.103 | 0.0001 | 0.1045 |
| offensive | indicbert | lora | shift_d3_noise05 | uncalibrated | 0.6286 | 0.0729 | 0.077 | 0.0747 | 0.0027 | 0.0702 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.6414 | 0.1007 | 0.1015 | 0.0994 | 0.0003 | 0.1003 |
| offensive | indicbert | lora | shift_d3_noise10 | uncalibrated | 0.6304 | 0.0665 | 0.0745 | 0.0674 | 0.0018 | 0.0647 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.6452 | 0.0889 | 0.0924 | 0.0914 | 0.0013 | 0.0877 |
| offensive | indicbert | lora | shift_d3_noise20 | uncalibrated | 0.6362 | 0.0621 | 0.0672 | 0.0668 | 0.0065 | 0.0556 |
| offensive | indicbert | lora | test | source_temp_scaled | 0.6369 | 0.1115 | 0.1122 | 0.1104 | 0.0004 | 0.1112 |
| offensive | indicbert | lora | test | uncalibrated | 0.6234 | 0.0753 | 0.079 | 0.0772 | 0.0006 | 0.0746 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.246 | 0.0247 | 0.0298 | 0.0294 | 0.023 | 0.0017 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.2449 | 0.0211 | 0.0283 | 0.0267 | 0.0192 | 0.0019 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | uncalibrated | 0.2984 | 0.0584 | 0.0578 | 0.0595 | 0.0579 | 0.0005 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.4071 | 0.0634 | 0.0649 | 0.0673 | 0.0633 | 0.0001 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.3982 | 0.0525 | 0.0552 | 0.0604 | 0.0524 | 0.0002 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | uncalibrated | 0.5154 | 0.1101 | 0.1096 | 0.1114 | 0.1098 | 0.0002 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.2717 | 0.0325 | 0.0367 | 0.0439 | 0.0293 | 0.0032 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.27 | 0.0305 | 0.0344 | 0.043 | 0.025 | 0.0055 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | uncalibrated | 0.3282 | 0.0667 | 0.066 | 0.0695 | 0.0661 | 0.0005 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 1.2992 | 0.3852 | 0.3802 | 0.3873 | 0.3825 | 0.0027 |
| offensive | xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 1.1461 | 0.3551 | 0.3505 | 0.3569 | 0.3523 | 0.0028 |
| offensive | xlmr_base | full_ft | shift_d2_test | uncalibrated | 1.9049 | 0.4372 | 0.433 | 0.4391 | 0.4351 | 0.0021 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.2518 | 0.023 | 0.0261 | 0.0264 | 0.0206 | 0.0024 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.3011 | 0.0576 | 0.0566 | 0.0587 | 0.057 | 0.0006 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.2848 | 0.0276 | 0.0306 | 0.0324 | 0.0263 | 0.0013 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.3423 | 0.0671 | 0.0667 | 0.0676 | 0.0669 | 0.0002 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.3488 | 0.0353 | 0.0381 | 0.0398 | 0.0347 | 0.0005 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.4207 | 0.0828 | 0.0823 | 0.0833 | 0.0825 | 0.0002 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | 0.2269 | 0.0205 | 0.0264 | 0.0248 | 0.0192 | 0.0013 |
| offensive | xlmr_base | full_ft | test | uncalibrated | 0.2725 | 0.0532 | 0.0529 | 0.0542 | 0.0529 | 0.0003 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | 0.3853 | 0.0202 | 0.0233 | 0.0618 | 0.0105 | 0.0097 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | target_temp_scaled | 0.3862 | 0.0265 | 0.0253 | 0.0629 | 0.0097 | 0.0168 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | uncalibrated | 0.4031 | 0.0528 | 0.0526 | 0.0699 | 0.0524 | 0.0003 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | 0.5091 | 0.0297 | 0.0364 | 0.0899 | 0.0271 | 0.0026 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | target_temp_scaled | 0.5067 | 0.0231 | 0.031 | 0.0877 | 0.018 | 0.0051 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | uncalibrated | 0.5483 | 0.0829 | 0.0804 | 0.1159 | 0.0814 | 0.0015 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | 0.4237 | 0.0211 | 0.0232 | 0.0936 | 0.0096 | 0.0114 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | target_temp_scaled | 0.4241 | 0.0197 | 0.0248 | 0.0946 | 0.0049 | 0.0148 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | uncalibrated | 0.4445 | 0.0571 | 0.0548 | 0.098 | 0.0542 | 0.0029 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.8191 | 0.2075 | 0.2087 | 0.2092 | 0.2074 | 0.0001 |
| offensive | xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.7728 | 0.1736 | 0.1751 | 0.1729 | 0.1736 | 0.0 |
| offensive | xlmr_base | lora | shift_d2_test | uncalibrated | 0.9517 | 0.2699 | 0.2695 | 0.2698 | 0.2693 | 0.0006 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.3783 | 0.0133 | 0.0157 | 0.0729 | 0.0047 | 0.0085 |
| offensive | xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.3942 | 0.048 | 0.0477 | 0.0702 | 0.0473 | 0.0007 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.42 | 0.0126 | 0.0144 | 0.093 | 0.0091 | 0.0035 |
| offensive | xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.4427 | 0.0578 | 0.0574 | 0.0935 | 0.0576 | 0.0002 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.4769 | 0.0143 | 0.0174 | 0.1145 | 0.0115 | 0.0028 |
| offensive | xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.5036 | 0.0638 | 0.0632 | 0.1222 | 0.0627 | 0.0011 |
| offensive | xlmr_base | lora | test | source_temp_scaled | 0.3496 | 0.0112 | 0.014 | 0.0513 | 0.005 | 0.0062 |
| offensive | xlmr_base | lora | test | uncalibrated | 0.3675 | 0.0473 | 0.047 | 0.0615 | 0.0472 | 0.0002 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | 0.7651 | 0.0978 | 0.1168 | 0.13 | 0.0591 | 0.0386 |
| sentiment | banglabert | full_ft | shift_d1_test | target_temp_scaled | 0.772 | 0.1026 | 0.1214 | 0.1345 | 0.0583 | 0.0443 |
| sentiment | banglabert | full_ft | shift_d1_test | uncalibrated | 0.8104 | 0.142 | 0.1515 | 0.1375 | 0.1247 | 0.0173 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | 1.1708 | 0.2009 | 0.1999 | 0.1781 | 0.1989 | 0.0019 |
| sentiment | banglabert | full_ft | shift_d2_test | target_temp_scaled | 1.1358 | 0.1804 | 0.1805 | 0.1666 | 0.1735 | 0.0069 |
| sentiment | banglabert | full_ft | shift_d2_test | uncalibrated | 1.4317 | 0.2837 | 0.2847 | 0.2275 | 0.2833 | 0.0003 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.6966 | 0.055 | 0.0602 | 0.0527 | 0.0361 | 0.0189 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.769 | 0.1158 | 0.115 | 0.0871 | 0.1083 | 0.0075 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.7503 | 0.0619 | 0.0662 | 0.0664 | 0.0462 | 0.0156 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.8339 | 0.1286 | 0.1296 | 0.1006 | 0.1223 | 0.0063 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8264 | 0.0874 | 0.0915 | 0.0912 | 0.0728 | 0.0146 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.9188 | 0.1546 | 0.1563 | 0.125 | 0.1508 | 0.0038 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | 0.6715 | 0.0603 | 0.0675 | 0.0511 | 0.0407 | 0.0196 |
| sentiment | banglabert | full_ft | test | uncalibrated | 0.7461 | 0.1171 | 0.1184 | 0.084 | 0.1104 | 0.0067 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | 0.7971 | 0.1278 | 0.1392 | 0.1012 | 0.0128 | 0.115 |
| sentiment | banglabert | lora | shift_d1_test | target_temp_scaled | 0.814 | 0.1479 | 0.1571 | 0.1146 | 0.0126 | 0.1353 |
| sentiment | banglabert | lora | shift_d1_test | uncalibrated | 0.7641 | 0.096 | 0.1116 | 0.082 | 0.023 | 0.073 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | 1.2403 | 0.2103 | 0.2137 | 0.1642 | 0.2082 | 0.002 |
| sentiment | banglabert | lora | shift_d2_test | target_temp_scaled | 1.1596 | 0.1596 | 0.168 | 0.1408 | 0.1493 | 0.0103 |
| sentiment | banglabert | lora | shift_d2_test | uncalibrated | 1.3648 | 0.2753 | 0.2756 | 0.1976 | 0.2751 | 0.0001 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.7759 | 0.0472 | 0.056 | 0.0469 | 0.0044 | 0.0428 |
| sentiment | banglabert | lora | shift_d3_noise05 | uncalibrated | 0.7708 | 0.0433 | 0.046 | 0.044 | 0.0332 | 0.0101 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.8021 | 0.0562 | 0.0662 | 0.0495 | 0.0008 | 0.0554 |
| sentiment | banglabert | lora | shift_d3_noise10 | uncalibrated | 0.7904 | 0.0355 | 0.041 | 0.036 | 0.0199 | 0.0155 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.8515 | 0.0807 | 0.0869 | 0.0624 | 0.0012 | 0.0795 |
| sentiment | banglabert | lora | shift_d3_noise20 | uncalibrated | 0.8334 | 0.038 | 0.0498 | 0.0412 | 0.006 | 0.0319 |
| sentiment | banglabert | lora | test | source_temp_scaled | 0.756 | 0.0574 | 0.0634 | 0.0477 | 0.0221 | 0.0353 |
| sentiment | banglabert | lora | test | uncalibrated | 0.7558 | 0.0717 | 0.0745 | 0.0509 | 0.061 | 0.0107 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | 0.8793 | 0.0772 | 0.1087 | 0.0806 | 0.0154 | 0.0618 |
| sentiment | indicbert | full_ft | shift_d1_test | target_temp_scaled | 0.8739 | 0.0708 | 0.0989 | 0.0768 | 0.0214 | 0.0494 |
| sentiment | indicbert | full_ft | shift_d1_test | uncalibrated | 0.8797 | 0.0721 | 0.0939 | 0.0843 | 0.0466 | 0.0255 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | 1.143 | 0.1677 | 0.1776 | 0.1363 | 0.1556 | 0.0121 |
| sentiment | indicbert | full_ft | shift_d2_test | target_temp_scaled | 1.1055 | 0.1383 | 0.1437 | 0.115 | 0.1191 | 0.0192 |
| sentiment | indicbert | full_ft | shift_d2_test | uncalibrated | 1.27 | 0.2374 | 0.2439 | 0.1784 | 0.2285 | 0.0088 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.8777 | 0.0402 | 0.0519 | 0.04 | 0.0126 | 0.0276 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.8878 | 0.0617 | 0.0703 | 0.0524 | 0.0578 | 0.0039 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.8911 | 0.0346 | 0.0387 | 0.04 | 0.015 | 0.0196 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.9069 | 0.067 | 0.0693 | 0.0557 | 0.0659 | 0.001 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.9046 | 0.0349 | 0.0472 | 0.0394 | 0.0096 | 0.0253 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.9201 | 0.063 | 0.0675 | 0.0553 | 0.0581 | 0.0049 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | 0.8616 | 0.0419 | 0.0499 | 0.0388 | 0.0091 | 0.0328 |
| sentiment | indicbert | full_ft | test | uncalibrated | 0.8681 | 0.0539 | 0.0621 | 0.0451 | 0.0498 | 0.004 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | 1.079 | 0.0349 | 0.1172 | 0.0569 | 0.0 | 0.0349 |
| sentiment | indicbert | lora | shift_d1_test | target_temp_scaled | 1.079 | 0.0349 | 0.1172 | 0.0569 | 0.0 | 0.0349 |
| sentiment | indicbert | lora | shift_d1_test | uncalibrated | 1.0726 | 0.0176 | 0.1029 | 0.0444 | 0.0 | 0.0176 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | 1.0811 | 0.0479 | 0.074 | 0.0396 | 0.0 | 0.0479 |
| sentiment | indicbert | lora | shift_d2_test | target_temp_scaled | 1.0811 | 0.0478 | 0.074 | 0.0396 | 0.0 | 0.0478 |
| sentiment | indicbert | lora | shift_d2_test | uncalibrated | 1.0758 | 0.0302 | 0.0665 | 0.027 | 0.0 | 0.0302 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | 1.082 | 0.0466 | 0.064 | 0.0359 | 0.0 | 0.0466 |
| sentiment | indicbert | lora | shift_d3_noise05 | uncalibrated | 1.077 | 0.0292 | 0.0549 | 0.0235 | 0.0 | 0.0292 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | 1.082 | 0.0466 | 0.07 | 0.0359 | 0.0 | 0.0466 |
| sentiment | indicbert | lora | shift_d3_noise10 | uncalibrated | 1.0769 | 0.0292 | 0.0644 | 0.0234 | 0.0 | 0.0292 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | 1.082 | 0.0465 | 0.0583 | 0.0359 | 0.0 | 0.0465 |
| sentiment | indicbert | lora | shift_d3_noise20 | uncalibrated | 1.0771 | 0.0292 | 0.0494 | 0.0234 | 0.0 | 0.0292 |
| sentiment | indicbert | lora | test | source_temp_scaled | 1.082 | 0.0466 | 0.0724 | 0.0359 | 0.0 | 0.0466 |
| sentiment | indicbert | lora | test | uncalibrated | 1.077 | 0.0293 | 0.0652 | 0.0235 | 0.0 | 0.0293 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | 0.7636 | 0.0951 | 0.1077 | 0.0806 | 0.0385 | 0.0566 |
| sentiment | xlmr_base | full_ft | shift_d1_test | target_temp_scaled | 0.7619 | 0.0801 | 0.1011 | 0.0793 | 0.0395 | 0.0406 |
| sentiment | xlmr_base | full_ft | shift_d1_test | uncalibrated | 0.7925 | 0.11 | 0.1279 | 0.0944 | 0.0946 | 0.0153 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 1.1302 | 0.2202 | 0.2164 | 0.2061 | 0.2155 | 0.0047 |
| sentiment | xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 1.0772 | 0.1926 | 0.1877 | 0.1909 | 0.1864 | 0.0062 |
| sentiment | xlmr_base | full_ft | shift_d2_test | uncalibrated | 1.4596 | 0.3057 | 0.3049 | 0.2486 | 0.3049 | 0.0008 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.7252 | 0.0443 | 0.0474 | 0.0408 | 0.0304 | 0.0139 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.7838 | 0.113 | 0.112 | 0.0806 | 0.1122 | 0.0008 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.7606 | 0.0423 | 0.0449 | 0.0388 | 0.033 | 0.0093 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.8306 | 0.1215 | 0.1194 | 0.0861 | 0.1203 | 0.0012 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.814 | 0.0558 | 0.0618 | 0.0512 | 0.05 | 0.0058 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.8978 | 0.1439 | 0.144 | 0.101 | 0.1434 | 0.0005 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | 0.7141 | 0.0464 | 0.0495 | 0.0466 | 0.0313 | 0.015 |
| sentiment | xlmr_base | full_ft | test | uncalibrated | 0.7719 | 0.1118 | 0.1113 | 0.0813 | 0.1109 | 0.0009 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | 0.8645 | 0.0775 | 0.1099 | 0.0765 | 0.0302 | 0.0473 |
| sentiment | xlmr_base | lora | shift_d1_test | target_temp_scaled | 0.8663 | 0.0965 | 0.1121 | 0.0803 | 0.0346 | 0.0619 |
| sentiment | xlmr_base | lora | shift_d1_test | uncalibrated | 0.869 | 0.0918 | 0.1068 | 0.0818 | 0.0673 | 0.0245 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.9154 | 0.0992 | 0.1064 | 0.1516 | 0.0847 | 0.0145 |
| sentiment | xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.9047 | 0.0757 | 0.0911 | 0.1425 | 0.0514 | 0.0243 |
| sentiment | xlmr_base | lora | shift_d2_test | uncalibrated | 0.964 | 0.1492 | 0.1517 | 0.1797 | 0.1414 | 0.0078 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.8228 | 0.0308 | 0.0377 | 0.0379 | 0.0061 | 0.0248 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.8261 | 0.0515 | 0.054 | 0.0425 | 0.0467 | 0.0047 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.8456 | 0.0348 | 0.0472 | 0.0374 | 0.0166 | 0.0182 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.8554 | 0.0661 | 0.0684 | 0.0511 | 0.0629 | 0.0033 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.8791 | 0.0377 | 0.0467 | 0.0455 | 0.0269 | 0.0108 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.8982 | 0.0851 | 0.0846 | 0.0657 | 0.0823 | 0.0028 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | 0.8148 | 0.0355 | 0.045 | 0.0447 | 0.006 | 0.0295 |
| sentiment | xlmr_base | lora | test | uncalibrated | 0.8138 | 0.0474 | 0.0515 | 0.0445 | 0.0421 | 0.0053 |


## 4. Advanced Conformal Prediction (LAC vs. APS vs. RAPS & Class-Conditional Gap)

| task | model | ft_method | condition | calibration_regime | conformal_method | empirical_coverage_mean | avg_set_size_mean | singleton_rate_mean | trivial_set_rate_mean | worst_class_coverage_mean | class_coverage_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | aps | 0.9425 | 1.0752 | 0.9248 | 0.0752 | 0.9337 | 0.0511 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | lac | 0.9175 | 1.0 | 1.0 | 0.0 | 0.9063 | 0.0285 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | raps | 0.939 | 1.0615 | 0.9385 | 0.0615 | 0.9298 | 0.048 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8817 | 1.1484 | 0.8516 | 0.1484 | 0.8209 | 0.0838 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8292 | 1.0 | 1.0 | 0.0 | 0.7526 | 0.1474 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8739 | 1.1204 | 0.8796 | 0.1204 | 0.8109 | 0.0898 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9406 | 1.0956 | 0.9044 | 0.0956 | 0.9286 | 0.0527 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | lac | 0.9092 | 1.0 | 1.0 | 0.0 | 0.8934 | 0.0253 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9358 | 1.0776 | 0.9224 | 0.0776 | 0.9224 | 0.0493 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.5445 | 1.0425 | 0.9575 | 0.0425 | 0.0461 | 0.8539 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.5255 | 1.0 | 1.0 | 0.0 | 0.0073 | 0.8927 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.5382 | 1.031 | 0.969 | 0.031 | 0.033 | 0.867 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9334 | 1.0944 | 0.9056 | 0.0944 | 0.9133 | 0.0518 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9019 | 1.0 | 1.0 | 0.0 | 0.8785 | 0.028 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9294 | 1.0788 | 0.9212 | 0.0788 | 0.9087 | 0.0484 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9223 | 1.1061 | 0.8939 | 0.1061 | 0.8863 | 0.0555 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.8854 | 1.0 | 1.0 | 0.0 | 0.8383 | 0.0617 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.9167 | 1.0861 | 0.9139 | 0.0861 | 0.8792 | 0.0513 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8858 | 1.1364 | 0.8636 | 0.1364 | 0.8006 | 0.1021 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.834 | 1.0 | 1.0 | 0.0 | 0.7233 | 0.1767 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8778 | 1.1114 | 0.8886 | 0.1114 | 0.7892 | 0.1108 |
| offensive | banglabert | full_ft | test | source_temp_scaled | aps | 0.9443 | 1.0771 | 0.9229 | 0.0771 | 0.9364 | 0.0522 |
| offensive | banglabert | full_ft | test | source_temp_scaled | lac | 0.9184 | 1.0 | 1.0 | 0.0 | 0.9077 | 0.0293 |
| offensive | banglabert | full_ft | test | source_temp_scaled | raps | 0.9405 | 1.0621 | 0.9379 | 0.0621 | 0.9322 | 0.0488 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.9418 | 1.1618 | 0.8382 | 0.1618 | 0.9319 | 0.0518 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.8888 | 1.0075 | 0.9925 | 0.0075 | 0.8802 | 0.0198 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.9377 | 1.1413 | 0.8587 | 0.1413 | 0.9271 | 0.0485 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8765 | 1.3021 | 0.6979 | 0.3021 | 0.8345 | 0.0655 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.7726 | 1.0159 | 0.9841 | 0.0159 | 0.7029 | 0.1971 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8667 | 1.2681 | 0.7319 | 0.2681 | 0.8227 | 0.0773 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9261 | 1.2083 | 0.7917 | 0.2083 | 0.8914 | 0.0616 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8721 | 1.0095 | 0.9905 | 0.0095 | 0.8175 | 0.0825 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9207 | 1.1858 | 0.8142 | 0.1858 | 0.8828 | 0.0594 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | aps | 0.7697 | 1.4985 | 0.5015 | 0.4985 | 0.5225 | 0.3775 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | lac | 0.5235 | 1.0115 | 0.9885 | 0.0115 | 0.0147 | 0.8853 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | raps | 0.7423 | 1.443 | 0.557 | 0.443 | 0.4654 | 0.4346 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9356 | 1.18 | 0.82 | 0.18 | 0.9339 | 0.0372 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.887 | 1.0086 | 0.9914 | 0.0086 | 0.8802 | 0.0198 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9299 | 1.1563 | 0.8437 | 0.1563 | 0.9283 | 0.0314 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9281 | 1.2266 | 0.7734 | 0.2266 | 0.918 | 0.0374 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8645 | 1.0086 | 0.9914 | 0.0086 | 0.835 | 0.065 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.921 | 1.1937 | 0.8063 | 0.1937 | 0.9083 | 0.0327 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9123 | 1.3259 | 0.6741 | 0.3259 | 0.8686 | 0.0525 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8173 | 1.0125 | 0.9875 | 0.0125 | 0.7216 | 0.1784 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.9033 | 1.2814 | 0.7186 | 0.2814 | 0.8552 | 0.0497 |
| offensive | banglabert | lora | test | source_temp_scaled | aps | 0.9409 | 1.1462 | 0.8538 | 0.1462 | 0.9308 | 0.0518 |
| offensive | banglabert | lora | test | source_temp_scaled | lac | 0.899 | 1.0066 | 0.9934 | 0.0066 | 0.8897 | 0.0106 |
| offensive | banglabert | lora | test | source_temp_scaled | raps | 0.9361 | 1.1273 | 0.8727 | 0.1273 | 0.9258 | 0.0473 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | aps | 0.9175 | 1.1587 | 0.8413 | 0.1587 | 0.9142 | 0.0209 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | lac | 0.8817 | 1.0397 | 0.9603 | 0.0397 | 0.8713 | 0.0287 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | raps | 0.9111 | 1.1342 | 0.8658 | 0.1342 | 0.9078 | 0.0144 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8593 | 1.1974 | 0.8026 | 0.1974 | 0.7924 | 0.1076 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8007 | 1.0523 | 0.9477 | 0.0523 | 0.7225 | 0.1775 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8507 | 1.1713 | 0.8287 | 0.1713 | 0.7818 | 0.1182 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9115 | 1.1645 | 0.8355 | 0.1645 | 0.9017 | 0.0214 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8734 | 1.041 | 0.959 | 0.041 | 0.8558 | 0.0442 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9046 | 1.1395 | 0.8605 | 0.1395 | 0.8934 | 0.0171 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.7133 | 1.3765 | 0.6235 | 0.3765 | 0.4457 | 0.4543 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.5775 | 1.0988 | 0.9012 | 0.0988 | 0.2074 | 0.6926 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.6868 | 1.3252 | 0.6748 | 0.3252 | 0.3958 | 0.5042 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9177 | 1.1407 | 0.8593 | 0.1407 | 0.8898 | 0.0434 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.8856 | 1.0346 | 0.9654 | 0.0346 | 0.8548 | 0.0452 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9114 | 1.119 | 0.881 | 0.119 | 0.8826 | 0.0378 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9052 | 1.1559 | 0.8441 | 0.1559 | 0.8721 | 0.0375 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.8666 | 1.0366 | 0.9634 | 0.0366 | 0.8281 | 0.0719 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.8986 | 1.1332 | 0.8668 | 0.1332 | 0.8639 | 0.0395 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8825 | 1.1883 | 0.8117 | 0.1883 | 0.8338 | 0.0662 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.8351 | 1.0483 | 0.9517 | 0.0483 | 0.7793 | 0.1207 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.874 | 1.1612 | 0.8388 | 0.1612 | 0.8239 | 0.0761 |
| offensive | indicbert | full_ft | test | source_temp_scaled | aps | 0.9312 | 1.1285 | 0.8715 | 0.1285 | 0.9144 | 0.0466 |
| offensive | indicbert | full_ft | test | source_temp_scaled | lac | 0.9023 | 1.0307 | 0.9693 | 0.0307 | 0.8843 | 0.0199 |
| offensive | indicbert | full_ft | test | source_temp_scaled | raps | 0.9259 | 1.1084 | 0.8916 | 0.1084 | 0.9087 | 0.0418 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.8872 | 1.632 | 0.368 | 0.632 | 0.8701 | 0.0299 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.9009 | 1.6282 | 0.3718 | 0.6282 | 0.81 | 0.0978 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.8872 | 1.632 | 0.368 | 0.632 | 0.8701 | 0.0299 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8652 | 1.6284 | 0.3716 | 0.6284 | 0.8335 | 0.0665 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8576 | 1.6007 | 0.3993 | 0.6007 | 0.7256 | 0.1744 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8652 | 1.6284 | 0.3716 | 0.6284 | 0.8335 | 0.0665 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.877 | 1.6332 | 0.3668 | 0.6332 | 0.8623 | 0.0377 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8901 | 1.6375 | 0.3625 | 0.6375 | 0.7895 | 0.1116 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.877 | 1.6332 | 0.3668 | 0.6332 | 0.8623 | 0.0377 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | aps | 0.8345 | 1.6248 | 0.3752 | 0.6248 | 0.7348 | 0.1652 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | lac | 0.8465 | 1.682 | 0.318 | 0.682 | 0.691 | 0.2856 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | raps | 0.8345 | 1.6248 | 0.3752 | 0.6248 | 0.7348 | 0.1652 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.8878 | 1.636 | 0.364 | 0.636 | 0.8712 | 0.0288 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8875 | 1.5903 | 0.4097 | 0.5903 | 0.7815 | 0.1185 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8878 | 1.636 | 0.364 | 0.636 | 0.8712 | 0.0288 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.8851 | 1.6367 | 0.3633 | 0.6367 | 0.8704 | 0.0296 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8858 | 1.5868 | 0.4132 | 0.5868 | 0.7775 | 0.1225 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8851 | 1.6367 | 0.3633 | 0.6367 | 0.8704 | 0.0296 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8806 | 1.6352 | 0.3648 | 0.6352 | 0.8611 | 0.0389 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8801 | 1.582 | 0.418 | 0.582 | 0.7642 | 0.1358 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8806 | 1.6352 | 0.3648 | 0.6352 | 0.8611 | 0.0389 |
| offensive | indicbert | lora | test | source_temp_scaled | aps | 0.8903 | 1.6369 | 0.3631 | 0.6369 | 0.8735 | 0.0265 |
| offensive | indicbert | lora | test | source_temp_scaled | lac | 0.8942 | 1.5914 | 0.4086 | 0.5914 | 0.7957 | 0.1043 |
| offensive | indicbert | lora | test | source_temp_scaled | raps | 0.8903 | 1.6369 | 0.3631 | 0.6369 | 0.8735 | 0.0265 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | aps | 0.9412 | 1.1067 | 0.8933 | 0.1067 | 0.9331 | 0.0494 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | lac | 0.9078 | 1.0 | 1.0 | 0.0 | 0.8928 | 0.0232 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | raps | 0.9361 | 1.0873 | 0.9127 | 0.0873 | 0.9277 | 0.0447 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8973 | 1.1744 | 0.8256 | 0.1744 | 0.8766 | 0.0277 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | lac | 0.834 | 1.0 | 1.0 | 0.0 | 0.8116 | 0.0884 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8881 | 1.1447 | 0.8553 | 0.1447 | 0.8672 | 0.0333 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9367 | 1.1293 | 0.8707 | 0.1293 | 0.918 | 0.0558 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8933 | 1.0 | 1.0 | 0.0 | 0.8589 | 0.0411 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9316 | 1.1085 | 0.8915 | 0.1085 | 0.9111 | 0.0526 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | aps | 0.585 | 1.1285 | 0.8715 | 0.1285 | 0.1461 | 0.7539 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | lac | 0.5228 | 1.0 | 1.0 | 0.0 | 0.0325 | 0.8675 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | raps | 0.5692 | 1.0958 | 0.9042 | 0.0958 | 0.1152 | 0.7848 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9427 | 1.1177 | 0.8823 | 0.1177 | 0.9351 | 0.0497 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9053 | 1.0 | 1.0 | 0.0 | 0.902 | 0.0084 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9379 | 1.0962 | 0.9038 | 0.0962 | 0.9307 | 0.0444 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9332 | 1.138 | 0.862 | 0.138 | 0.9225 | 0.0431 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.889 | 1.0 | 1.0 | 0.0 | 0.8809 | 0.0191 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.9285 | 1.1155 | 0.8845 | 0.1155 | 0.9173 | 0.0387 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.9194 | 1.1841 | 0.8159 | 0.1841 | 0.9023 | 0.0352 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.8598 | 1.0 | 1.0 | 0.0 | 0.8408 | 0.0592 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.9121 | 1.1534 | 0.8466 | 0.1534 | 0.895 | 0.0278 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | aps | 0.9469 | 1.0974 | 0.9026 | 0.0974 | 0.9451 | 0.0485 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | lac | 0.915 | 1.0 | 1.0 | 0.0 | 0.9076 | 0.0229 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | raps | 0.9431 | 1.0808 | 0.9192 | 0.0808 | 0.9409 | 0.0452 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.917 | 1.282 | 0.718 | 0.282 | 0.8888 | 0.0459 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.8941 | 1.1499 | 0.8501 | 0.1499 | 0.8616 | 0.0384 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.9117 | 1.26 | 0.74 | 0.26 | 0.8815 | 0.0427 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8871 | 1.3937 | 0.6063 | 0.3937 | 0.8359 | 0.0641 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8455 | 1.2113 | 0.7887 | 0.2113 | 0.7871 | 0.1129 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8784 | 1.3667 | 0.6333 | 0.3667 | 0.8249 | 0.0751 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9108 | 1.3252 | 0.6748 | 0.3252 | 0.8656 | 0.0571 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8774 | 1.1711 | 0.8289 | 0.1711 | 0.8149 | 0.0851 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9048 | 1.2992 | 0.7008 | 0.2992 | 0.8563 | 0.0545 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | aps | 0.7772 | 1.5268 | 0.4732 | 0.5268 | 0.5958 | 0.3042 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | lac | 0.6592 | 1.288 | 0.712 | 0.288 | 0.3586 | 0.5414 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | raps | 0.758 | 1.4922 | 0.5078 | 0.4922 | 0.5634 | 0.3366 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9167 | 1.2869 | 0.7131 | 0.2869 | 0.8906 | 0.0451 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8922 | 1.1479 | 0.8521 | 0.1479 | 0.8631 | 0.0369 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9114 | 1.2641 | 0.7359 | 0.2641 | 0.8843 | 0.0409 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9116 | 1.3293 | 0.6707 | 0.3293 | 0.875 | 0.0513 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8806 | 1.1711 | 0.8289 | 0.1711 | 0.8346 | 0.0654 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9048 | 1.3021 | 0.6979 | 0.3021 | 0.8665 | 0.0465 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.901 | 1.3959 | 0.6041 | 0.3959 | 0.8517 | 0.0545 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8652 | 1.2268 | 0.7732 | 0.2268 | 0.8018 | 0.0982 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8932 | 1.3685 | 0.6315 | 0.3685 | 0.8406 | 0.0594 |
| offensive | xlmr_base | lora | test | source_temp_scaled | aps | 0.9195 | 1.2427 | 0.7573 | 0.2427 | 0.905 | 0.0352 |
| offensive | xlmr_base | lora | test | source_temp_scaled | lac | 0.9008 | 1.1215 | 0.8785 | 0.1215 | 0.8862 | 0.0172 |
| offensive | xlmr_base | lora | test | source_temp_scaled | raps | 0.9156 | 1.2218 | 0.7782 | 0.2218 | 0.9011 | 0.0313 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | aps | 0.9413 | 2.0219 | 0.1612 | 0.1831 | 0.9121 | 0.081 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | lac | 0.9582 | 2.0438 | 0.2289 | 0.2726 | 0.9368 | 0.088 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | raps | 0.9343 | 1.9652 | 0.1473 | 0.1124 | 0.8971 | 0.081 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.7485 | 1.9856 | 0.1823 | 0.1679 | 0.4 | 0.5 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.8085 | 2.1281 | 0.2139 | 0.342 | 0.5667 | 0.3333 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.7321 | 1.9485 | 0.1667 | 0.1152 | 0.3518 | 0.5482 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.899 | 1.7969 | 0.2965 | 0.0934 | 0.8848 | 0.0265 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9156 | 1.7406 | 0.4138 | 0.1544 | 0.8917 | 0.0375 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.8951 | 1.7831 | 0.28 | 0.0631 | 0.8793 | 0.0252 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.8981 | 1.8575 | 0.2528 | 0.1103 | 0.8756 | 0.0332 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.9118 | 1.8163 | 0.3607 | 0.1771 | 0.8847 | 0.0381 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.8904 | 1.8354 | 0.2341 | 0.0696 | 0.8636 | 0.0395 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8953 | 1.9675 | 0.1718 | 0.1393 | 0.853 | 0.0612 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.9187 | 1.9715 | 0.2741 | 0.2456 | 0.8928 | 0.0432 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8884 | 1.9346 | 0.1568 | 0.0913 | 0.8433 | 0.0623 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | aps | 0.9031 | 1.7526 | 0.3362 | 0.0888 | 0.8713 | 0.0363 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | lac | 0.9129 | 1.6724 | 0.4668 | 0.1391 | 0.8591 | 0.0481 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | raps | 0.9004 | 1.7413 | 0.3176 | 0.059 | 0.876 | 0.0278 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | aps | 0.9393 | 2.2537 | 0.0557 | 0.3095 | 0.9051 | 0.0952 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | lac | 0.9692 | 2.2587 | 0.1124 | 0.3711 | 0.9513 | 0.0952 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | raps | 0.9363 | 2.2209 | 0.0408 | 0.2617 | 0.9051 | 0.0952 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | aps | 0.6664 | 2.01 | 0.0709 | 0.0808 | 0.095 | 0.805 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | lac | 0.6689 | 2.0077 | 0.0284 | 0.0361 | 0.0695 | 0.8305 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | raps | 0.6619 | 2.0057 | 0.0502 | 0.056 | 0.0667 | 0.8333 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9084 | 2.0141 | 0.1371 | 0.1512 | 0.8923 | 0.0557 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.9124 | 1.9084 | 0.2906 | 0.199 | 0.9076 | 0.0193 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9054 | 2.0103 | 0.1149 | 0.1251 | 0.8841 | 0.0674 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9129 | 2.0976 | 0.096 | 0.1937 | 0.8966 | 0.0514 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.9312 | 2.0518 | 0.2056 | 0.2574 | 0.9238 | 0.0431 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.91 | 2.0809 | 0.0785 | 0.1594 | 0.8898 | 0.0631 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9119 | 2.2229 | 0.0457 | 0.2687 | 0.8705 | 0.0711 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.9391 | 2.26 | 0.114 | 0.374 | 0.9108 | 0.0674 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.9074 | 2.1935 | 0.0343 | 0.2278 | 0.8573 | 0.0797 |
| sentiment | banglabert | lora | test | source_temp_scaled | aps | 0.9057 | 1.9494 | 0.175 | 0.1244 | 0.8803 | 0.0293 |
| sentiment | banglabert | lora | test | source_temp_scaled | lac | 0.9044 | 1.8001 | 0.3606 | 0.1607 | 0.8799 | 0.0256 |
| sentiment | banglabert | lora | test | source_temp_scaled | raps | 0.9034 | 1.96 | 0.1453 | 0.1053 | 0.8688 | 0.0404 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | aps | 0.9114 | 2.2816 | 0.0328 | 0.3144 | 0.8656 | 0.0581 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | lac | 0.9343 | 2.2736 | 0.2 | 0.4736 | 0.8696 | 0.0859 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | raps | 0.9114 | 2.2935 | 0.0119 | 0.3055 | 0.8704 | 0.0533 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.8107 | 2.2739 | 0.0231 | 0.297 | 0.5568 | 0.3432 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.7602 | 2.0756 | 0.3794 | 0.455 | 0.4824 | 0.4176 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.8129 | 2.2828 | 0.0027 | 0.2856 | 0.5558 | 0.3442 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.8932 | 2.2469 | 0.0532 | 0.3001 | 0.8721 | 0.0367 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9119 | 2.154 | 0.2581 | 0.4121 | 0.8869 | 0.0292 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.894 | 2.2685 | 0.0231 | 0.2916 | 0.8698 | 0.0453 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.8913 | 2.2474 | 0.0526 | 0.3 | 0.8678 | 0.0354 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.9053 | 2.1435 | 0.2635 | 0.4071 | 0.8738 | 0.0328 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.8906 | 2.2669 | 0.0249 | 0.2918 | 0.8643 | 0.0386 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8941 | 2.271 | 0.0407 | 0.3118 | 0.8734 | 0.0327 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.9049 | 2.1959 | 0.2368 | 0.4326 | 0.8663 | 0.0423 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8956 | 2.2871 | 0.0181 | 0.3051 | 0.8716 | 0.0369 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | aps | 0.8999 | 2.2346 | 0.0575 | 0.2921 | 0.8804 | 0.037 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | lac | 0.9138 | 2.1181 | 0.2754 | 0.3935 | 0.8888 | 0.0275 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | raps | 0.9009 | 2.2603 | 0.0243 | 0.2846 | 0.8796 | 0.0438 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | aps | 0.9005 | 2.5274 | 0.0 | 0.5274 | 0.5238 | 0.3762 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | lac | 0.8806 | 2.4537 | 0.0 | 0.4537 | 0.4286 | 0.4714 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | raps | 0.9005 | 2.5274 | 0.0 | 0.5274 | 0.5238 | 0.3762 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | aps | 0.8965 | 2.5134 | 0.0 | 0.5134 | 0.5551 | 0.3449 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | lac | 0.8261 | 2.2296 | 0.0 | 0.2296 | 0.2524 | 0.648 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | raps | 0.8965 | 2.5134 | 0.0 | 0.5134 | 0.5551 | 0.3449 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.8988 | 2.544 | 0.0 | 0.544 | 0.5766 | 0.3234 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8984 | 2.4522 | 0.0 | 0.4522 | 0.5748 | 0.3252 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8988 | 2.544 | 0.0 | 0.544 | 0.5766 | 0.3234 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.8987 | 2.5443 | 0.0 | 0.5443 | 0.576 | 0.324 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.9019 | 2.4482 | 0.0 | 0.4482 | 0.5895 | 0.3105 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8987 | 2.5443 | 0.0 | 0.5443 | 0.576 | 0.324 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8984 | 2.5438 | 0.0 | 0.5438 | 0.5748 | 0.3252 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8974 | 2.4419 | 0.0 | 0.4419 | 0.5705 | 0.3295 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8984 | 2.5438 | 0.0 | 0.5438 | 0.5748 | 0.3252 |
| sentiment | indicbert | lora | test | source_temp_scaled | aps | 0.8988 | 2.544 | 0.0 | 0.544 | 0.5766 | 0.3234 |
| sentiment | indicbert | lora | test | source_temp_scaled | lac | 0.9004 | 2.4612 | 0.0 | 0.4612 | 0.5834 | 0.3166 |
| sentiment | indicbert | lora | test | source_temp_scaled | raps | 0.8988 | 2.544 | 0.0 | 0.544 | 0.5766 | 0.3234 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | aps | 0.9224 | 2.0836 | 0.1731 | 0.2567 | 0.8914 | 0.0495 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | lac | 0.9363 | 2.0378 | 0.2746 | 0.3124 | 0.918 | 0.0503 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | raps | 0.9164 | 2.0617 | 0.1473 | 0.209 | 0.8864 | 0.0421 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | aps | 0.7667 | 1.8167 | 0.3624 | 0.1791 | 0.5314 | 0.3686 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | lac | 0.7269 | 1.6803 | 0.5836 | 0.2639 | 0.3219 | 0.5781 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | raps | 0.7677 | 1.8244 | 0.3192 | 0.1435 | 0.5464 | 0.3536 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9051 | 1.911 | 0.241 | 0.1521 | 0.8921 | 0.0189 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9188 | 1.8291 | 0.3754 | 0.2046 | 0.8998 | 0.0372 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9031 | 1.9132 | 0.2079 | 0.1212 | 0.8875 | 0.0205 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9009 | 1.9466 | 0.2234 | 0.17 | 0.8953 | 0.0114 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.9109 | 1.8622 | 0.3665 | 0.2287 | 0.8838 | 0.0282 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.899 | 1.9406 | 0.1965 | 0.1371 | 0.8877 | 0.0194 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8974 | 2.0159 | 0.179 | 0.1949 | 0.8709 | 0.0327 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.9019 | 1.9318 | 0.3337 | 0.2654 | 0.8713 | 0.042 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8919 | 2.0018 | 0.1513 | 0.1531 | 0.8567 | 0.0433 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | aps | 0.9059 | 1.8822 | 0.2625 | 0.1447 | 0.8837 | 0.0261 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | lac | 0.9188 | 1.8097 | 0.3803 | 0.19 | 0.8945 | 0.0447 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | raps | 0.9041 | 1.8897 | 0.2282 | 0.1179 | 0.8801 | 0.0267 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | aps | 0.8746 | 2.0617 | 0.1403 | 0.202 | 0.8296 | 0.0704 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | lac | 0.8766 | 2.0129 | 0.2925 | 0.3055 | 0.8284 | 0.0749 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | raps | 0.8667 | 2.0458 | 0.1264 | 0.1721 | 0.8148 | 0.0852 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | aps | 0.84 | 1.9341 | 0.2475 | 0.1816 | 0.6894 | 0.2106 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | lac | 0.8251 | 1.844 | 0.4873 | 0.3313 | 0.5422 | 0.3578 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | raps | 0.8396 | 1.9306 | 0.2209 | 0.1515 | 0.673 | 0.227 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.8947 | 2.0274 | 0.1643 | 0.1916 | 0.8823 | 0.0205 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.895 | 1.9168 | 0.3504 | 0.2672 | 0.8695 | 0.0311 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8912 | 2.0091 | 0.1466 | 0.1557 | 0.8765 | 0.0261 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.8793 | 2.0253 | 0.1665 | 0.1918 | 0.8654 | 0.0346 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8868 | 1.9353 | 0.3401 | 0.2754 | 0.8631 | 0.0369 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8768 | 2.0093 | 0.1503 | 0.1596 | 0.8615 | 0.0385 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8644 | 2.0401 | 0.149 | 0.1891 | 0.8209 | 0.0791 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8649 | 1.9126 | 0.3482 | 0.2609 | 0.8333 | 0.0667 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8626 | 2.0299 | 0.1271 | 0.1569 | 0.8096 | 0.0904 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | aps | 0.8966 | 2.0268 | 0.166 | 0.1928 | 0.8735 | 0.0316 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | lac | 0.9022 | 1.9446 | 0.3299 | 0.2744 | 0.8739 | 0.0311 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | raps | 0.8931 | 2.0138 | 0.1465 | 0.1603 | 0.8671 | 0.0372 |


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
| sentiment | banglabert | full_ft | shift_d1_test | 74.3613 | 79.1492 | 0.1313 | 9.4642 | 0.3778 |
| sentiment | banglabert | full_ft | shift_d2_test | 93.9629 | 96.5473 | 0.6224 | 2.0586 | 0.0818 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | 58.3037 | 63.2017 | 0.1247 | 18.941 | 0.4745 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | 61.5711 | 66.855 | 0.1419 | 14.383 | 0.3947 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | 69.9195 | 77.9454 | 0.2101 | 12.3885 | 0.3355 |
| sentiment | banglabert | full_ft | test | 57.9103 | 62.3813 | 0.1332 | 22.1245 | 0.5101 |
| sentiment | banglabert | lora | shift_d1_test | 74.7429 | 82.3803 | 0.2448 | 7.8048 | 0.2309 |
| sentiment | banglabert | lora | shift_d2_test | 99.4246 | 99.6134 | 0.9938 | -0.2516 | -0.1331 |
| sentiment | banglabert | lora | shift_d3_noise05 | 55.8201 | 60.6681 | 0.0913 | 7.9954 | 0.2024 |
| sentiment | banglabert | lora | shift_d3_noise10 | 60.4394 | 67.4044 | 0.1465 | 4.4579 | 0.1395 |
| sentiment | banglabert | lora | shift_d3_noise20 | 74.1152 | 87.7878 | 0.3532 | 4.9987 | 0.1614 |
| sentiment | banglabert | lora | test | 55.2316 | 57.6996 | 0.1019 | 12.6851 | 0.2915 |
| sentiment | indicbert | full_ft | shift_d1_test | 77.8676 | 78.7962 | 0.1313 | 4.7273 | 0.2339 |
| sentiment | indicbert | full_ft | shift_d2_test | 95.0735 | 98.2143 | 0.7692 | 0.2094 | 0.0449 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | 55.4264 | 57.9853 | 0.0885 | 9.4135 | 0.2503 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | 55.8804 | 57.8981 | 0.0966 | 8.1948 | 0.2212 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | 57.6959 | 60.5042 | 0.0943 | 7.5753 | 0.1941 |
| sentiment | indicbert | full_ft | test | 54.6011 | 56.2973 | 0.0894 | 9.8844 | 0.2522 |
| sentiment | indicbert | lora | shift_d1_test | 100.0 | 100.0 | 1.0 | 0.0 | 0.0 |
| sentiment | indicbert | lora | shift_d2_test | 99.3689 | 99.6197 | 0.9806 | 0.0017 | -0.0033 |
| sentiment | indicbert | lora | shift_d3_noise05 | 54.6771 | 57.2248 | 0.0738 | 2.6705 | 0.0828 |
| sentiment | indicbert | lora | shift_d3_noise10 | 55.735 | 57.7048 | 0.0854 | 3.9887 | 0.1082 |
| sentiment | indicbert | lora | shift_d3_noise20 | 57.5247 | 60.0798 | 0.0919 | 1.9341 | 0.0657 |
| sentiment | indicbert | lora | test | 54.7957 | 57.5315 | 0.0746 | 3.5244 | 0.0921 |
| sentiment | xlmr_base | full_ft | shift_d1_test | 87.8865 | 88.6954 | 0.1632 | 3.2513 | 0.4303 |
| sentiment | xlmr_base | full_ft | shift_d2_test | 56.6836 | 57.3971 | 0.0214 | 10.0685 | 0.2549 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | 51.8618 | 53.1376 | 0.0712 | 19.1498 | 0.5162 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | 49.7312 | 49.0756 | 0.0576 | 17.5394 | 0.4841 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | 47.7329 | 45.917 | 0.0399 | 15.5717 | 0.4289 |
| sentiment | xlmr_base | full_ft | test | 54.8278 | 57.2731 | 0.0946 | 20.2915 | 0.5289 |
| sentiment | xlmr_base | lora | shift_d1_test | 77.7421 | 79.2164 | 0.1144 | 4.2223 | 0.2368 |
| sentiment | xlmr_base | lora | shift_d2_test | 92.794 | 95.8466 | 0.5764 | -0.7508 | 0.0397 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | 53.6494 | 55.2458 | 0.0809 | 10.823 | 0.2659 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | 54.1059 | 55.4727 | 0.0778 | 10.063 | 0.2445 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | 59.2217 | 62.6166 | 0.0959 | 7.4945 | 0.1878 |
| sentiment | xlmr_base | lora | test | 55.6733 | 58.5368 | 0.0984 | 11.8264 | 0.2851 |

