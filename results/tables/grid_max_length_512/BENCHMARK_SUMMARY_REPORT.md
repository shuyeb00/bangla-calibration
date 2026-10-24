# Benchmark Results (3 Encoders x 2 Fine-Tuning Regimes x 2 Tasks x 5 Seeds)

## 1. Shift Quantification

| task | condition | n_samples | vocab_jaccard_overlap | subword_fragmentation | non_bangla_char_ratio | median_char_length | std_char_length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | D0_source_test | 1360 | 1.0 | 1.1851 | 0.0014 | 65.0 | 52.67 |
| sentiment | shift_d1_test | 201 | 0.1423 | 1.4575 | 0.0126 | 62.0 | 54.8 |
| sentiment | shift_d2_test | 804 | 0.058 | 2.1616 | 0.7084 | 496.5 | 108.2 |
| sentiment | shift_d3_noise05 | 1360 | 0.5745 | 1.3743 | 0.0014 | 65.0 | 52.7 |
| sentiment | shift_d3_noise10 | 1360 | 0.3846 | 1.5597 | 0.0013 | 65.0 | 52.74 |
| sentiment | shift_d3_noise20 | 1360 | 0.1993 | 1.9363 | 0.0013 | 65.0 | 52.89 |
| offensive | D0_source_test | 4722 | 1.0 | 1.3541 | 0.0018 | 44.0 | 119.11 |
| offensive | shift_d1_barishal_test | 2445 | 0.1698 | 1.6264 | 0.0 | 47.0 | 44.96 |
| offensive | shift_d1_chittagong_test | 2443 | 0.1021 | 1.9065 | 0.0 | 46.0 | 43.1 |
| offensive | shift_d1_noakhali_test | 2445 | 0.1515 | 1.7016 | 0.0 | 46.0 | 45.74 |
| offensive | shift_d2_test | 800 | 0.0596 | 2.1673 | 0.704 | 492.5 | 131.43 |
| offensive | shift_d3_noise05 | 4722 | 0.5359 | 1.5423 | 0.0018 | 44.0 | 119.18 |
| offensive | shift_d3_noise10 | 4722 | 0.3512 | 1.7068 | 0.0018 | 44.0 | 119.15 |
| offensive | shift_d3_noise20 | 4722 | 0.1808 | 2.0704 | 0.0018 | 44.0 | 119.32 |


## 2. Sentiment Task — Seed-Aggregated Metrics (5 Seeds)

| model | ft_method | condition | calibration_regime | accuracy_mean | macro_f1_mean | nll_mean | nll_std | ece_15_mean | brier_mean | aurc_mean | conformal_coverage_mean | conformal_set_size_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| banglabert | full_ft | shift_d1_test | source_temp_scaled | 0.7124 | 0.6858 | 0.6977 | 0.0229 | 0.0666 | 0.3987 | 0.1353 | 0.9234 | 1.7851 |
| banglabert | full_ft | shift_d1_test | target_temp_scaled | 0.7124 | 0.6858 | 0.6977 | 0.0244 | 0.0642 | 0.3986 | 0.1353 | 0.9224 | 1.797 |
| banglabert | full_ft | shift_d1_test | uncalibrated | 0.7124 | 0.6858 | 0.7741 | 0.0847 | 0.1257 | 0.4271 | 0.1358 | 0.9234 | 1.803 |
| banglabert | full_ft | shift_d2_test | source_temp_scaled | 0.4898 | 0.3807 | 1.0372 | 0.0364 | 0.1797 | 0.6122 | 0.2904 | 0.8552 | 2.0706 |
| banglabert | full_ft | shift_d2_test | target_temp_scaled | 0.4898 | 0.3807 | 0.9864 | 0.0166 | 0.1539 | 0.5844 | 0.2901 | 0.9015 | 2.2779 |
| banglabert | full_ft | shift_d2_test | uncalibrated | 0.4898 | 0.3807 | 1.2582 | 0.1469 | 0.2714 | 0.6971 | 0.2919 | 0.8749 | 2.1383 |
| banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.7131 | 0.6949 | 0.6954 | 0.0078 | 0.045 | 0.3967 | 0.1398 | 0.9074 | 1.7268 |
| banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.7131 | 0.6949 | 0.7828 | 0.0654 | 0.1216 | 0.4257 | 0.1395 | 0.9135 | 1.7479 |
| banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.6768 | 0.6632 | 0.7488 | 0.0194 | 0.0592 | 0.4337 | 0.1646 | 0.9056 | 1.7963 |
| banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.6768 | 0.6632 | 0.8568 | 0.0856 | 0.1438 | 0.4741 | 0.1653 | 0.9124 | 1.8301 |
| banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.6203 | 0.617 | 0.8249 | 0.0334 | 0.0925 | 0.4844 | 0.197 | 0.9109 | 1.9393 |
| banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.6203 | 0.617 | 0.9513 | 0.1067 | 0.1783 | 0.5392 | 0.1987 | 0.9194 | 1.9906 |
| banglabert | full_ft | test | source_temp_scaled | 0.7251 | 0.6981 | 0.6683 | 0.0071 | 0.0509 | 0.3803 | 0.1285 | 0.9085 | 1.6576 |
| banglabert | full_ft | test | uncalibrated | 0.7251 | 0.6981 | 0.7555 | 0.0624 | 0.1224 | 0.4102 | 0.1281 | 0.9128 | 1.6731 |
| banglabert | lora | shift_d1_test | source_temp_scaled | 0.7085 | 0.6759 | 0.699 | 0.0371 | 0.0813 | 0.3998 | 0.1309 | 0.9164 | 1.7831 |
| banglabert | lora | shift_d1_test | target_temp_scaled | 0.7085 | 0.6759 | 0.6988 | 0.0394 | 0.0816 | 0.3999 | 0.1309 | 0.9244 | 1.8279 |
| banglabert | lora | shift_d1_test | uncalibrated | 0.7085 | 0.6759 | 0.7645 | 0.0666 | 0.1061 | 0.4157 | 0.1303 | 0.9174 | 1.7881 |
| banglabert | lora | shift_d2_test | source_temp_scaled | 0.5234 | 0.4388 | 1.009 | 0.0952 | 0.1129 | 0.5865 | 0.2828 | 0.8704 | 2.295 |
| banglabert | lora | shift_d2_test | target_temp_scaled | 0.5234 | 0.4388 | 0.9723 | 0.06 | 0.0974 | 0.5729 | 0.2828 | 0.8973 | 2.3159 |
| banglabert | lora | shift_d2_test | uncalibrated | 0.5234 | 0.4388 | 1.1633 | 0.2612 | 0.1673 | 0.626 | 0.2828 | 0.8913 | 2.35 |
| banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.7024 | 0.6743 | 0.6899 | 0.0109 | 0.0396 | 0.3991 | 0.137 | 0.9076 | 1.7241 |
| banglabert | lora | shift_d3_noise05 | uncalibrated | 0.7024 | 0.6743 | 0.7724 | 0.0369 | 0.1062 | 0.4226 | 0.1372 | 0.9101 | 1.7303 |
| banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.6768 | 0.6531 | 0.739 | 0.0088 | 0.0402 | 0.4295 | 0.1616 | 0.9062 | 1.8232 |
| banglabert | lora | shift_d3_noise10 | uncalibrated | 0.6768 | 0.6531 | 0.8234 | 0.0487 | 0.1111 | 0.4541 | 0.1621 | 0.9107 | 1.8319 |
| banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.6351 | 0.6225 | 0.8043 | 0.0118 | 0.0411 | 0.47 | 0.1949 | 0.9168 | 2.0031 |
| banglabert | lora | shift_d3_noise20 | uncalibrated | 0.6351 | 0.6225 | 0.8626 | 0.0251 | 0.0981 | 0.4868 | 0.1952 | 0.9216 | 2.0325 |
| banglabert | lora | test | source_temp_scaled | 0.7191 | 0.6892 | 0.6636 | 0.0052 | 0.035 | 0.384 | 0.1271 | 0.9071 | 1.6681 |
| banglabert | lora | test | uncalibrated | 0.7191 | 0.6892 | 0.7528 | 0.0415 | 0.1081 | 0.4101 | 0.1269 | 0.9071 | 1.6672 |
| indicbert | full_ft | shift_d1_test | source_temp_scaled | 0.6259 | 0.5859 | 0.8608 | 0.0173 | 0.088 | 0.4997 | 0.2207 | 0.8706 | 1.9433 |
| indicbert | full_ft | shift_d1_test | target_temp_scaled | 0.6259 | 0.5859 | 0.8642 | 0.0168 | 0.0957 | 0.5014 | 0.2206 | 0.8647 | 1.9075 |
| indicbert | full_ft | shift_d1_test | uncalibrated | 0.6259 | 0.5859 | 0.9282 | 0.0448 | 0.1598 | 0.5283 | 0.2202 | 0.8716 | 1.9264 |
| indicbert | full_ft | shift_d2_test | source_temp_scaled | 0.4488 | 0.26 | 1.3188 | 0.2023 | 0.2866 | 0.7848 | 0.4949 | 0.6582 | 1.7435 |
| indicbert | full_ft | shift_d2_test | target_temp_scaled | 0.4488 | 0.26 | 1.0659 | 0.0261 | 0.0588 | 0.6432 | 0.496 | 0.909 | 2.5943 |
| indicbert | full_ft | shift_d2_test | uncalibrated | 0.4488 | 0.26 | 1.6648 | 0.4191 | 0.3772 | 0.8814 | 0.494 | 0.6575 | 1.7331 |
| indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.6541 | 0.621 | 0.7918 | 0.0103 | 0.0395 | 0.4593 | 0.1883 | 0.9066 | 1.911 |
| indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.6541 | 0.621 | 0.8362 | 0.0406 | 0.1056 | 0.4797 | 0.1878 | 0.9054 | 1.9047 |
| indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.6325 | 0.5931 | 0.8346 | 0.0071 | 0.0515 | 0.4869 | 0.2118 | 0.8853 | 1.9054 |
| indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.6325 | 0.5931 | 0.8977 | 0.0409 | 0.131 | 0.5146 | 0.2113 | 0.8871 | 1.9035 |
| indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.5899 | 0.5426 | 0.9169 | 0.0185 | 0.0957 | 0.5394 | 0.2515 | 0.8446 | 1.8668 |
| indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.5899 | 0.5426 | 1.0204 | 0.0722 | 0.1794 | 0.5832 | 0.2513 | 0.8419 | 1.8554 |
| indicbert | full_ft | test | source_temp_scaled | 0.6632 | 0.6351 | 0.7669 | 0.0102 | 0.046 | 0.4435 | 0.1748 | 0.9193 | 1.9203 |
| indicbert | full_ft | test | uncalibrated | 0.6632 | 0.6351 | 0.7986 | 0.0337 | 0.0965 | 0.4589 | 0.1745 | 0.9178 | 1.9176 |
| indicbert | lora | shift_d1_test | source_temp_scaled | 0.6139 | 0.5495 | 0.8958 | 0.033 | 0.0797 | 0.5189 | 0.2446 | 0.8627 | 2.0826 |
| indicbert | lora | shift_d1_test | target_temp_scaled | 0.6139 | 0.5495 | 0.8966 | 0.0334 | 0.0766 | 0.518 | 0.2447 | 0.8796 | 2.1592 |
| indicbert | lora | shift_d1_test | uncalibrated | 0.6139 | 0.5495 | 0.9015 | 0.0274 | 0.0953 | 0.5203 | 0.2444 | 0.8647 | 2.0876 |
| indicbert | lora | shift_d2_test | source_temp_scaled | 0.44 | 0.2543 | 1.4614 | 0.1892 | 0.3011 | 0.8163 | 0.4646 | 0.5823 | 1.5468 |
| indicbert | lora | shift_d2_test | target_temp_scaled | 0.44 | 0.2543 | 1.0737 | 0.0273 | 0.0569 | 0.6483 | 0.4631 | 0.8963 | 2.5649 |
| indicbert | lora | shift_d2_test | uncalibrated | 0.44 | 0.2543 | 1.5488 | 0.2699 | 0.3145 | 0.832 | 0.4645 | 0.5823 | 1.5475 |
| indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.6096 | 0.5496 | 0.8507 | 0.028 | 0.0469 | 0.4965 | 0.2208 | 0.8956 | 2.0409 |
| indicbert | lora | shift_d3_noise05 | uncalibrated | 0.6096 | 0.5496 | 0.854 | 0.0254 | 0.0541 | 0.4976 | 0.2207 | 0.8953 | 2.0437 |
| indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.5882 | 0.5273 | 0.8873 | 0.022 | 0.0544 | 0.5193 | 0.2426 | 0.8787 | 2.0469 |
| indicbert | lora | shift_d3_noise10 | uncalibrated | 0.5882 | 0.5273 | 0.8945 | 0.0156 | 0.0622 | 0.5223 | 0.2426 | 0.8791 | 2.05 |
| indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.5582 | 0.4988 | 0.9547 | 0.0173 | 0.069 | 0.5628 | 0.2854 | 0.85 | 2.0604 |
| indicbert | lora | shift_d3_noise20 | uncalibrated | 0.5582 | 0.4988 | 0.9689 | 0.019 | 0.0799 | 0.5691 | 0.2853 | 0.8501 | 2.064 |
| indicbert | lora | test | source_temp_scaled | 0.62 | 0.5612 | 0.8339 | 0.0331 | 0.0488 | 0.4851 | 0.2106 | 0.9024 | 2.0222 |
| indicbert | lora | test | uncalibrated | 0.62 | 0.5612 | 0.8354 | 0.0318 | 0.0579 | 0.4852 | 0.2104 | 0.9031 | 2.0231 |
| xlmr_base | full_ft | shift_d1_test | source_temp_scaled | 0.6279 | 0.5902 | 0.8069 | 0.023 | 0.0924 | 0.4691 | 0.1838 | 0.8975 | 1.8796 |
| xlmr_base | full_ft | shift_d1_test | target_temp_scaled | 0.6279 | 0.5902 | 0.8103 | 0.0243 | 0.1032 | 0.4714 | 0.1838 | 0.8766 | 1.7751 |
| xlmr_base | full_ft | shift_d1_test | uncalibrated | 0.6279 | 0.5902 | 0.9002 | 0.0276 | 0.1706 | 0.5088 | 0.1842 | 0.8995 | 1.8577 |
| xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 0.5657 | 0.4238 | 1.0055 | 0.0669 | 0.1561 | 0.5953 | 0.2822 | 0.7806 | 1.7813 |
| xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 0.5657 | 0.4238 | 0.9403 | 0.0336 | 0.0878 | 0.556 | 0.2839 | 0.9174 | 2.2281 |
| xlmr_base | full_ft | shift_d2_test | uncalibrated | 0.5657 | 0.4238 | 1.2675 | 0.144 | 0.2645 | 0.6759 | 0.2802 | 0.7706 | 1.7577 |
| xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.6794 | 0.6483 | 0.731 | 0.0107 | 0.0416 | 0.4243 | 0.1579 | 0.9156 | 1.8228 |
| xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.6794 | 0.6483 | 0.786 | 0.0117 | 0.1141 | 0.4477 | 0.1581 | 0.911 | 1.8038 |
| xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.6535 | 0.6228 | 0.7672 | 0.0076 | 0.0521 | 0.4484 | 0.1767 | 0.9063 | 1.8515 |
| xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.6535 | 0.6228 | 0.8387 | 0.0135 | 0.135 | 0.48 | 0.1769 | 0.9032 | 1.8275 |
| xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.6321 | 0.5941 | 0.8166 | 0.0121 | 0.0575 | 0.4778 | 0.2007 | 0.8943 | 1.9003 |
| xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.6321 | 0.5941 | 0.9052 | 0.0097 | 0.151 | 0.5147 | 0.2016 | 0.891 | 1.8791 |
| xlmr_base | full_ft | test | source_temp_scaled | 0.6868 | 0.6568 | 0.7169 | 0.0077 | 0.0411 | 0.4153 | 0.1513 | 0.9162 | 1.8143 |
| xlmr_base | full_ft | test | uncalibrated | 0.6868 | 0.6568 | 0.7671 | 0.0107 | 0.1096 | 0.4363 | 0.1518 | 0.9143 | 1.7947 |
| xlmr_base | lora | shift_d1_test | source_temp_scaled | 0.6368 | 0.5947 | 0.8053 | 0.0139 | 0.0837 | 0.4708 | 0.1891 | 0.8846 | 1.8896 |
| xlmr_base | lora | shift_d1_test | target_temp_scaled | 0.6368 | 0.5947 | 0.806 | 0.0146 | 0.0829 | 0.4713 | 0.1891 | 0.8905 | 1.8955 |
| xlmr_base | lora | shift_d1_test | uncalibrated | 0.6368 | 0.5947 | 0.8286 | 0.0173 | 0.1047 | 0.4819 | 0.1896 | 0.8846 | 1.8896 |
| xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.5485 | 0.397 | 1.0369 | 0.0674 | 0.2013 | 0.6107 | 0.2675 | 0.7545 | 1.6522 |
| xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.5485 | 0.397 | 0.9225 | 0.022 | 0.105 | 0.5454 | 0.2695 | 0.9095 | 2.1386 |
| xlmr_base | lora | shift_d2_test | uncalibrated | 0.5485 | 0.397 | 1.1904 | 0.0998 | 0.2559 | 0.6613 | 0.2665 | 0.7515 | 1.6512 |
| xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.6697 | 0.6319 | 0.7392 | 0.0113 | 0.0353 | 0.4329 | 0.1647 | 0.9137 | 1.8563 |
| xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.6697 | 0.6319 | 0.75 | 0.0129 | 0.0632 | 0.4386 | 0.1647 | 0.9138 | 1.8534 |
| xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.6506 | 0.6096 | 0.7667 | 0.004 | 0.0323 | 0.4515 | 0.1814 | 0.9068 | 1.8819 |
| xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.6506 | 0.6096 | 0.7818 | 0.0118 | 0.0668 | 0.4597 | 0.1813 | 0.9071 | 1.8834 |
| xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.6319 | 0.5821 | 0.813 | 0.0099 | 0.0399 | 0.4775 | 0.2025 | 0.8959 | 1.9141 |
| xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.6319 | 0.5821 | 0.8357 | 0.0218 | 0.0799 | 0.489 | 0.2028 | 0.8962 | 1.9144 |
| xlmr_base | lora | test | source_temp_scaled | 0.6753 | 0.6425 | 0.7266 | 0.014 | 0.032 | 0.4248 | 0.1576 | 0.9156 | 1.8456 |
| xlmr_base | lora | test | uncalibrated | 0.6753 | 0.6425 | 0.7362 | 0.0139 | 0.051 | 0.4292 | 0.1578 | 0.9157 | 1.8415 |


## 2. Offensive Task — Seed-Aggregated Metrics (5 Seeds)

| model | ft_method | condition | calibration_regime | accuracy_mean | macro_f1_mean | nll_mean | nll_std | ece_15_mean | brier_mean | aurc_mean | conformal_coverage_mean | conformal_set_size_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.9177 | 0.9177 | 0.2328 | 0.0068 | 0.0183 | 0.1311 | 0.0237 | 0.9177 | 1.0 |
| banglabert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.9177 | 0.9177 | 0.2325 | 0.0065 | 0.0212 | 0.1305 | 0.0237 | 0.9177 | 1.0 |
| banglabert | full_ft | shift_d1_barishal_test | uncalibrated | 0.9177 | 0.9177 | 0.3106 | 0.0399 | 0.06 | 0.1429 | 0.0237 | 0.9177 | 1.0 |
| banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.8302 | 0.8297 | 0.4366 | 0.0265 | 0.0704 | 0.2651 | 0.0814 | 0.8302 | 1.0 |
| banglabert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.8302 | 0.8297 | 0.4081 | 0.0109 | 0.0266 | 0.2545 | 0.0814 | 0.901 | 1.1989 |
| banglabert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.8302 | 0.8297 | 0.6342 | 0.1228 | 0.1275 | 0.294 | 0.0814 | 0.8302 | 1.0 |
| banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.9025 | 0.9024 | 0.2516 | 0.0193 | 0.0271 | 0.1485 | 0.0243 | 0.9025 | 1.0 |
| banglabert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.9025 | 0.9024 | 0.2535 | 0.019 | 0.0334 | 0.147 | 0.0243 | 0.9227 | 1.0443 |
| banglabert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.9025 | 0.9024 | 0.3266 | 0.0332 | 0.0679 | 0.1649 | 0.0243 | 0.9025 | 1.0 |
| banglabert | full_ft | shift_d2_test | source_temp_scaled | 0.5312 | 0.3639 | 1.4134 | 0.1957 | 0.4058 | 0.8302 | 0.5208 | 0.5312 | 1.0 |
| banglabert | full_ft | shift_d2_test | target_temp_scaled | 0.5312 | 0.3639 | 0.7129 | 0.0096 | 0.0968 | 0.5186 | 0.5208 | 0.9048 | 1.838 |
| banglabert | full_ft | shift_d2_test | uncalibrated | 0.5312 | 0.3639 | 2.4545 | 0.5198 | 0.4553 | 0.9119 | 0.5208 | 0.5312 | 1.0 |
| banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.9027 | 0.9025 | 0.2613 | 0.0107 | 0.025 | 0.1523 | 0.0276 | 0.9027 | 1.0 |
| banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.9027 | 0.9025 | 0.3511 | 0.0548 | 0.0692 | 0.1676 | 0.0276 | 0.9027 | 1.0 |
| banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.8893 | 0.8889 | 0.2993 | 0.0178 | 0.0301 | 0.1742 | 0.0411 | 0.8893 | 1.0 |
| banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.8893 | 0.8889 | 0.4118 | 0.0755 | 0.0793 | 0.1908 | 0.0411 | 0.8893 | 1.0 |
| banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8448 | 0.8428 | 0.4107 | 0.0417 | 0.0591 | 0.2459 | 0.0735 | 0.8448 | 1.0 |
| banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.8448 | 0.8428 | 0.5972 | 0.1469 | 0.1158 | 0.2712 | 0.0735 | 0.8448 | 1.0 |
| banglabert | full_ft | test | source_temp_scaled | 0.9185 | 0.9184 | 0.2194 | 0.0055 | 0.0224 | 0.1264 | 0.0182 | 0.9185 | 1.0 |
| banglabert | full_ft | test | uncalibrated | 0.9185 | 0.9184 | 0.2856 | 0.0331 | 0.0574 | 0.1392 | 0.0182 | 0.9185 | 1.0 |
| banglabert | lora | shift_d1_barishal_test | source_temp_scaled | 0.9162 | 0.9162 | 0.2213 | 0.0034 | 0.0128 | 0.1283 | 0.0199 | 0.9162 | 1.0 |
| banglabert | lora | shift_d1_barishal_test | target_temp_scaled | 0.9162 | 0.9162 | 0.2211 | 0.0031 | 0.0124 | 0.1278 | 0.0199 | 0.9162 | 1.0 |
| banglabert | lora | shift_d1_barishal_test | uncalibrated | 0.9162 | 0.9162 | 0.2793 | 0.0059 | 0.0525 | 0.1383 | 0.0199 | 0.9162 | 1.0 |
| banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.8269 | 0.8262 | 0.412 | 0.0125 | 0.0548 | 0.2563 | 0.0684 | 0.8269 | 1.0 |
| banglabert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.8269 | 0.8262 | 0.3948 | 0.0071 | 0.0249 | 0.2493 | 0.0684 | 0.8945 | 1.1663 |
| banglabert | lora | shift_d1_chittagong_test | uncalibrated | 0.8269 | 0.8262 | 0.5505 | 0.0284 | 0.1118 | 0.2821 | 0.0684 | 0.8269 | 1.0 |
| banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.9007 | 0.9006 | 0.2456 | 0.004 | 0.0161 | 0.1471 | 0.0228 | 0.9007 | 1.0 |
| banglabert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.9007 | 0.9006 | 0.2461 | 0.0038 | 0.018 | 0.1467 | 0.0228 | 0.9193 | 1.0403 |
| banglabert | lora | shift_d1_noakhali_test | uncalibrated | 0.9007 | 0.9006 | 0.2996 | 0.0114 | 0.0594 | 0.1596 | 0.0228 | 0.9007 | 1.0 |
| banglabert | lora | shift_d2_test | source_temp_scaled | 0.525 | 0.349 | 1.261 | 0.0966 | 0.3815 | 0.7957 | 0.5055 | 0.525 | 1.0 |
| banglabert | lora | shift_d2_test | target_temp_scaled | 0.525 | 0.349 | 0.7079 | 0.0047 | 0.0795 | 0.5141 | 0.5055 | 0.9032 | 1.82 |
| banglabert | lora | shift_d2_test | uncalibrated | 0.525 | 0.349 | 2.0596 | 0.1775 | 0.4497 | 0.9038 | 0.5055 | 0.525 | 1.0 |
| banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.9003 | 0.9001 | 0.2476 | 0.0039 | 0.0168 | 0.1483 | 0.0229 | 0.9003 | 1.0 |
| banglabert | lora | shift_d3_noise05 | uncalibrated | 0.9003 | 0.9001 | 0.3067 | 0.0116 | 0.0602 | 0.1611 | 0.0229 | 0.9003 | 1.0 |
| banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.8818 | 0.8813 | 0.2941 | 0.0057 | 0.0227 | 0.1764 | 0.0355 | 0.8818 | 1.0 |
| banglabert | lora | shift_d3_noise10 | uncalibrated | 0.8818 | 0.8813 | 0.3749 | 0.0139 | 0.0731 | 0.1914 | 0.0355 | 0.8818 | 1.0 |
| banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.8398 | 0.8375 | 0.39 | 0.0165 | 0.0472 | 0.2417 | 0.0618 | 0.8398 | 1.0 |
| banglabert | lora | shift_d3_noise20 | uncalibrated | 0.8398 | 0.8375 | 0.5151 | 0.0375 | 0.1057 | 0.2663 | 0.0618 | 0.8398 | 1.0 |
| banglabert | lora | test | source_temp_scaled | 0.9163 | 0.9162 | 0.2116 | 0.0014 | 0.0134 | 0.1247 | 0.016 | 0.9163 | 1.0 |
| banglabert | lora | test | uncalibrated | 0.9163 | 0.9162 | 0.2622 | 0.0056 | 0.0512 | 0.1352 | 0.016 | 0.9163 | 1.0 |
| indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.8953 | 0.8953 | 0.2791 | 0.0144 | 0.0341 | 0.1644 | 0.0303 | 0.8953 | 1.0 |
| indicbert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.8953 | 0.8953 | 0.2785 | 0.0139 | 0.036 | 0.1635 | 0.0303 | 0.9021 | 1.0148 |
| indicbert | full_ft | shift_d1_barishal_test | uncalibrated | 0.8953 | 0.8953 | 0.3826 | 0.0627 | 0.0782 | 0.1839 | 0.0303 | 0.8953 | 1.0 |
| indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.805 | 0.8046 | 0.4734 | 0.0186 | 0.0832 | 0.2989 | 0.0934 | 0.805 | 1.0 |
| indicbert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.805 | 0.8046 | 0.445 | 0.0117 | 0.0416 | 0.2841 | 0.0934 | 0.9009 | 1.2525 |
| indicbert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.805 | 0.8046 | 0.705 | 0.1122 | 0.1529 | 0.3413 | 0.0934 | 0.805 | 1.0 |
| indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.886 | 0.886 | 0.2947 | 0.0121 | 0.0392 | 0.176 | 0.0319 | 0.886 | 1.0 |
| indicbert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.886 | 0.886 | 0.2948 | 0.0113 | 0.0347 | 0.1744 | 0.0319 | 0.9133 | 1.0623 |
| indicbert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.886 | 0.886 | 0.3994 | 0.0585 | 0.0842 | 0.1974 | 0.0319 | 0.886 | 1.0 |
| indicbert | full_ft | shift_d2_test | source_temp_scaled | 0.5195 | 0.3566 | 1.3165 | 0.0765 | 0.3983 | 0.8232 | 0.5021 | 0.5195 | 1.0 |
| indicbert | full_ft | shift_d2_test | target_temp_scaled | 0.5195 | 0.3566 | 0.7154 | 0.0071 | 0.1049 | 0.5213 | 0.5021 | 0.9185 | 1.8402 |
| indicbert | full_ft | shift_d2_test | uncalibrated | 0.5195 | 0.3566 | 2.3356 | 0.3676 | 0.4609 | 0.9279 | 0.5021 | 0.5195 | 1.0 |
| indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.8888 | 0.8885 | 0.2894 | 0.0075 | 0.0369 | 0.1721 | 0.0324 | 0.8888 | 1.0 |
| indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.8888 | 0.8885 | 0.3931 | 0.0498 | 0.082 | 0.1928 | 0.0324 | 0.8888 | 1.0 |
| indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.8714 | 0.871 | 0.3271 | 0.0125 | 0.039 | 0.198 | 0.0447 | 0.8714 | 1.0 |
| indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.8714 | 0.871 | 0.4495 | 0.068 | 0.0937 | 0.2224 | 0.0447 | 0.8714 | 1.0 |
| indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8215 | 0.821 | 0.4221 | 0.0156 | 0.0611 | 0.2669 | 0.0725 | 0.8215 | 1.0 |
| indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.8215 | 0.821 | 0.6009 | 0.0944 | 0.133 | 0.3055 | 0.0725 | 0.8215 | 1.0 |
| indicbert | full_ft | test | source_temp_scaled | 0.9078 | 0.9076 | 0.2508 | 0.0068 | 0.0325 | 0.145 | 0.0244 | 0.9078 | 1.0 |
| indicbert | full_ft | test | uncalibrated | 0.9078 | 0.9076 | 0.3352 | 0.0385 | 0.0681 | 0.1609 | 0.0244 | 0.9078 | 1.0 |
| indicbert | lora | shift_d1_barishal_test | source_temp_scaled | 0.883 | 0.883 | 0.2917 | 0.0107 | 0.0218 | 0.1758 | 0.034 | 0.8834 | 1.0012 |
| indicbert | lora | shift_d1_barishal_test | target_temp_scaled | 0.883 | 0.883 | 0.2917 | 0.011 | 0.0202 | 0.1757 | 0.034 | 0.9079 | 1.0558 |
| indicbert | lora | shift_d1_barishal_test | uncalibrated | 0.883 | 0.883 | 0.3076 | 0.0099 | 0.0476 | 0.1817 | 0.034 | 0.8834 | 1.0012 |
| indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.8012 | 0.8009 | 0.4583 | 0.0146 | 0.0575 | 0.2917 | 0.0915 | 0.8019 | 1.0016 |
| indicbert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.8012 | 0.8009 | 0.4428 | 0.0101 | 0.023 | 0.2843 | 0.0915 | 0.8966 | 1.2539 |
| indicbert | lora | shift_d1_chittagong_test | uncalibrated | 0.8012 | 0.8009 | 0.5053 | 0.021 | 0.0958 | 0.3055 | 0.0915 | 0.8019 | 1.0016 |
| indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.8732 | 0.8732 | 0.3038 | 0.0141 | 0.0311 | 0.1863 | 0.0354 | 0.8737 | 1.0006 |
| indicbert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.8732 | 0.8732 | 0.304 | 0.0139 | 0.0272 | 0.1857 | 0.0354 | 0.9116 | 1.0909 |
| indicbert | lora | shift_d1_noakhali_test | uncalibrated | 0.8732 | 0.8732 | 0.3185 | 0.0164 | 0.0483 | 0.193 | 0.0354 | 0.8737 | 1.0006 |
| indicbert | lora | shift_d2_test | source_temp_scaled | 0.52 | 0.3864 | 1.1702 | 0.1572 | 0.3355 | 0.7544 | 0.5297 | 0.5203 | 1.0006 |
| indicbert | lora | shift_d2_test | target_temp_scaled | 0.52 | 0.3864 | 0.702 | 0.0028 | 0.0562 | 0.5086 | 0.5297 | 0.8803 | 1.7884 |
| indicbert | lora | shift_d2_test | uncalibrated | 0.52 | 0.3864 | 1.4329 | 0.237 | 0.3782 | 0.8135 | 0.5297 | 0.5203 | 1.0006 |
| indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.8806 | 0.8803 | 0.297 | 0.0108 | 0.0206 | 0.1792 | 0.0363 | 0.8813 | 1.001 |
| indicbert | lora | shift_d3_noise05 | uncalibrated | 0.8806 | 0.8803 | 0.3144 | 0.0106 | 0.0479 | 0.1852 | 0.0363 | 0.8813 | 1.001 |
| indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.8596 | 0.8593 | 0.3386 | 0.0103 | 0.0226 | 0.2077 | 0.0493 | 0.86 | 1.0008 |
| indicbert | lora | shift_d3_noise10 | uncalibrated | 0.8596 | 0.8593 | 0.36 | 0.009 | 0.0567 | 0.2149 | 0.0493 | 0.86 | 1.0008 |
| indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.8143 | 0.814 | 0.4259 | 0.0113 | 0.0389 | 0.2719 | 0.0806 | 0.8153 | 1.0022 |
| indicbert | lora | shift_d3_noise20 | uncalibrated | 0.8143 | 0.814 | 0.4585 | 0.0124 | 0.0771 | 0.2829 | 0.0806 | 0.8153 | 1.0022 |
| indicbert | lora | test | source_temp_scaled | 0.9014 | 0.9011 | 0.258 | 0.0089 | 0.0189 | 0.1518 | 0.0272 | 0.9015 | 1.0005 |
| indicbert | lora | test | uncalibrated | 0.9014 | 0.9011 | 0.2722 | 0.0079 | 0.0399 | 0.1567 | 0.0272 | 0.9015 | 1.0005 |
| xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.9091 | 0.9091 | 0.2424 | 0.0038 | 0.0166 | 0.1405 | 0.0223 | 0.9091 | 1.0 |
| xlmr_base | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.9091 | 0.9091 | 0.243 | 0.0036 | 0.0218 | 0.1399 | 0.0223 | 0.9138 | 1.0084 |
| xlmr_base | full_ft | shift_d1_barishal_test | uncalibrated | 0.9091 | 0.9091 | 0.2992 | 0.0339 | 0.0557 | 0.1512 | 0.0223 | 0.9091 | 1.0 |
| xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.835 | 0.8348 | 0.3915 | 0.0071 | 0.0501 | 0.2439 | 0.0612 | 0.835 | 1.0 |
| xlmr_base | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.835 | 0.8348 | 0.3814 | 0.0039 | 0.0283 | 0.2378 | 0.0612 | 0.9117 | 1.1932 |
| xlmr_base | full_ft | shift_d1_chittagong_test | uncalibrated | 0.835 | 0.8348 | 0.508 | 0.0712 | 0.105 | 0.2682 | 0.0612 | 0.835 | 1.0 |
| xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.8919 | 0.8919 | 0.2667 | 0.0076 | 0.0282 | 0.1607 | 0.0259 | 0.8919 | 1.0 |
| xlmr_base | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.8919 | 0.8919 | 0.2682 | 0.0066 | 0.0305 | 0.1594 | 0.0259 | 0.9131 | 1.0478 |
| xlmr_base | full_ft | shift_d1_noakhali_test | uncalibrated | 0.8919 | 0.8919 | 0.3253 | 0.0317 | 0.0655 | 0.1752 | 0.0259 | 0.8919 | 1.0 |
| xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 0.528 | 0.3734 | 1.2865 | 0.1609 | 0.38 | 0.8019 | 0.5103 | 0.528 | 1.0 |
| xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 0.528 | 0.3734 | 0.7092 | 0.0101 | 0.0851 | 0.5153 | 0.5103 | 0.911 | 1.836 |
| xlmr_base | full_ft | shift_d2_test | uncalibrated | 0.528 | 0.3734 | 2.0676 | 0.5776 | 0.4332 | 0.8862 | 0.5103 | 0.528 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.9041 | 0.9039 | 0.2478 | 0.0059 | 0.0201 | 0.1458 | 0.0242 | 0.9041 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.9041 | 0.9039 | 0.3023 | 0.0389 | 0.0567 | 0.1572 | 0.0242 | 0.9041 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.891 | 0.8907 | 0.2788 | 0.0049 | 0.0213 | 0.166 | 0.0318 | 0.891 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.891 | 0.8907 | 0.3435 | 0.0471 | 0.065 | 0.1798 | 0.0318 | 0.891 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8599 | 0.8593 | 0.3447 | 0.0082 | 0.0268 | 0.2104 | 0.0513 | 0.8599 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.8599 | 0.8593 | 0.4295 | 0.0604 | 0.0808 | 0.2277 | 0.0513 | 0.8599 | 1.0 |
| xlmr_base | full_ft | test | source_temp_scaled | 0.9149 | 0.9148 | 0.2232 | 0.0047 | 0.0188 | 0.1292 | 0.0192 | 0.9149 | 1.0 |
| xlmr_base | full_ft | test | uncalibrated | 0.9149 | 0.9148 | 0.2729 | 0.0325 | 0.0521 | 0.1395 | 0.0192 | 0.9149 | 1.0 |
| xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | 0.8929 | 0.8929 | 0.2623 | 0.0041 | 0.0177 | 0.1576 | 0.0271 | 0.8929 | 1.0 |
| xlmr_base | lora | shift_d1_barishal_test | target_temp_scaled | 0.8929 | 0.8929 | 0.2634 | 0.0038 | 0.0189 | 0.1575 | 0.0271 | 0.9126 | 1.0411 |
| xlmr_base | lora | shift_d1_barishal_test | uncalibrated | 0.8929 | 0.8929 | 0.2804 | 0.0076 | 0.0422 | 0.1629 | 0.0271 | 0.8929 | 1.0 |
| xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | 0.818 | 0.8178 | 0.4002 | 0.0112 | 0.0344 | 0.2559 | 0.0676 | 0.818 | 1.0 |
| xlmr_base | lora | shift_d1_chittagong_test | target_temp_scaled | 0.818 | 0.8178 | 0.3956 | 0.0091 | 0.0217 | 0.2533 | 0.0676 | 0.8984 | 1.1862 |
| xlmr_base | lora | shift_d1_chittagong_test | uncalibrated | 0.818 | 0.8178 | 0.4418 | 0.0196 | 0.0783 | 0.2688 | 0.0676 | 0.818 | 1.0 |
| xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | 0.8782 | 0.8781 | 0.2905 | 0.0127 | 0.0207 | 0.1793 | 0.0323 | 0.8782 | 1.0 |
| xlmr_base | lora | shift_d1_noakhali_test | target_temp_scaled | 0.8782 | 0.8781 | 0.2915 | 0.0128 | 0.0198 | 0.179 | 0.0323 | 0.9171 | 1.0973 |
| xlmr_base | lora | shift_d1_noakhali_test | uncalibrated | 0.8782 | 0.8781 | 0.3087 | 0.0195 | 0.0455 | 0.1856 | 0.0323 | 0.8782 | 1.0 |
| xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.5247 | 0.3634 | 1.2663 | 0.1065 | 0.3793 | 0.7918 | 0.4677 | 0.5247 | 1.0 |
| xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.5247 | 0.3634 | 0.6997 | 0.0049 | 0.065 | 0.5064 | 0.4677 | 0.9062 | 1.7913 |
| xlmr_base | lora | shift_d2_test | uncalibrated | 0.5247 | 0.3634 | 1.7016 | 0.2063 | 0.4245 | 0.8642 | 0.4677 | 0.5247 | 1.0 |
| xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.8977 | 0.8974 | 0.2571 | 0.0028 | 0.0128 | 0.1541 | 0.0257 | 0.8977 | 1.0 |
| xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.8977 | 0.8974 | 0.2729 | 0.0071 | 0.0377 | 0.1586 | 0.0257 | 0.8977 | 1.0 |
| xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.8801 | 0.8798 | 0.2929 | 0.0033 | 0.0098 | 0.1775 | 0.035 | 0.8801 | 1.0 |
| xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.8801 | 0.8798 | 0.3126 | 0.0046 | 0.0433 | 0.1826 | 0.035 | 0.8801 | 1.0 |
| xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.8504 | 0.8498 | 0.351 | 0.0026 | 0.0124 | 0.2177 | 0.0533 | 0.8504 | 1.0 |
| xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.8504 | 0.8498 | 0.3727 | 0.0072 | 0.0501 | 0.2235 | 0.0533 | 0.8504 | 1.0 |
| xlmr_base | lora | test | source_temp_scaled | 0.9061 | 0.9059 | 0.2372 | 0.002 | 0.0128 | 0.141 | 0.0215 | 0.9061 | 1.0 |
| xlmr_base | lora | test | uncalibrated | 0.9061 | 0.9059 | 0.2557 | 0.0062 | 0.0395 | 0.1461 | 0.0215 | 0.9061 | 1.0 |


## 5. Statistical Significance (Holm-Corrected)

| task | model | ft_method | comparison | metric | endpoint_family | mean_diff | median_raw_p | median_holm_p | seeds_sig_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0035 | 0.8029 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0027 | 0.5473 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0248 | 0.2992 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1548 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0702 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.3489 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0255 | 0.0666 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.011 | 0.1172 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0407 | 0.1514 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.7728 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.3982 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d2_test | nll | secondary | -2.1695 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0284 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.012 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0656 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0516 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0218 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.1262 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.132 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0585 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.3119 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0029 | 0.8349 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.002 | 0.7113 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0169 | 0.5207 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1469 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0613 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.2887 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0242 | 0.0282 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0086 | 0.1548 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0372 | 0.1168 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d2_test | brier | secondary | -0.7688 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.3989 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d2_test | nll | secondary | -1.7978 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0259 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0093 | 0.0006 | 0.2616 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0447 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0562 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.022 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.1126 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.1313 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0549 | 0.0002 | 0.2068 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.2533 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0228 | 0.047 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0109 | 0.0606 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_barishal_test | nll | secondary | -0.047 | 0.0442 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1805 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0854 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.3702 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0362 | 0.0054 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0166 | 0.0396 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0638 | 0.021 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.7674 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.394 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d2_test | nll | secondary | -2.0012 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0319 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0142 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0581 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0615 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0261 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.1143 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.1447 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0652 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.2659 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0249 | 0.0323 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0081 | 0.3048 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0353 | 0.0675 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1488 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0563 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.2332 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0361 | 0.0017 | 0.6034 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0092 | 0.2503 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0459 | 0.0087 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d2_test | brier | secondary | -0.6572 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.3402 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d2_test | nll | secondary | -1.1615 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0286 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0078 | 0.0293 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0423 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0582 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0163 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0878 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.1263 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0372 | 0.0002 | 0.2068 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.1865 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0116 | 0.3164 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0044 | 0.4194 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0261 | 0.2526 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1287 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0538 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.2352 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0356 | 0.0032 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0141 | 0.0306 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0522 | 0.0314 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.747 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.3831 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d2_test | nll | secondary | -1.795 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0177 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0047 | 0.0232 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0294 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0403 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0132 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.0706 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.0883 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.029 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.1567 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0166 | 0.1176 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0035 | 0.5353 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0245 | 0.1578 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1227 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0393 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.1862 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0394 | 0.0004 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0077 | 0.1724 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0528 | 0.0024 | 0.9239 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d2_test | brier | secondary | -0.7183 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.3862 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d2_test | nll | secondary | -1.4464 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0126 | 0.0006 | 0.2616 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0013 | 0.3638 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0172 | 0.0018 | 0.7073 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0365 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0041 | 0.3128 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0569 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0775 | 0.0002 | 0.2068 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0109 | 0.0098 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.1171 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d1_test | brier | secondary | -0.0593 | 0.0668 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d1_test | ece_15 | secondary | -0.017 | 0.4822 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d1_test | nll | primary | -0.1067 | 0.0478 | 0.0478 | 3 |
| sentiment | banglabert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.2867 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.1473 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d2_test | nll | primary | -0.5024 | 0.0002 | 0.0002 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0154 | 0.057 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0017 | 0.385 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise05 | nll | primary | -0.0272 | 0.134 | 0.134 | 2 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0638 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.021 | 0.134 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise10 | nll | primary | -0.101 | 0.0002 | 0.0002 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.1289 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0542 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise20 | nll | primary | -0.1955 | 0.0002 | 0.0002 | 5 |
| sentiment | banglabert | lora | D0_vs_shift_d1_test | brier | secondary | -0.0545 | 0.1118 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d1_test | ece_15 | secondary | -0.0164 | 0.5919 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d1_test | nll | secondary | -0.1151 | 0.1022 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d2_test | brier | secondary | -0.2157 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.0578 | 0.0174 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d2_test | nll | secondary | -0.4102 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0124 | 0.1176 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0025 | 0.4842 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0195 | 0.1946 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0438 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0025 | 0.6899 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0703 | 0.0008 | 0.3376 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0765 | 0.0002 | 0.2068 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | 0.0111 | 0.3458 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.1096 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d1_test | brier | secondary | -0.1068 | 0.0028 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d1_test | ece_15 | secondary | -0.0617 | 0.055 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d1_test | nll | secondary | -0.1806 | 0.0038 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.4226 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.2797 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d2_test | nll | secondary | -0.8666 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0208 | 0.0012 | 0.4944 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0099 | 0.185 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0376 | 0.0012 | 0.4944 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0556 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.034 | 0.0016 | 0.6335 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.099 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.1242 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0821 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.2218 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d1_test | brier | secondary | -0.0703 | 0.0024 | 0.9239 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d1_test | ece_15 | secondary | 0.0103 | 0.9095 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d1_test | nll | secondary | -0.112 | 0.0046 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d2_test | brier | secondary | -0.3471 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.2544 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d2_test | nll | secondary | -0.7141 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0124 | 0.0054 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0029 | 0.4034 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0186 | 0.0086 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0371 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0042 | 0.146 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0591 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0839 | 0.0002 | 0.2068 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0221 | 0.2144 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.1335 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d1_test | brier | secondary | -0.0759 | 0.0116 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d1_test | ece_15 | secondary | -0.0418 | 0.1608 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d1_test | nll | secondary | -0.1506 | 0.0048 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.2393 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.1554 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d2_test | nll | secondary | -0.5 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0114 | 0.0646 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0047 | 0.447 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0188 | 0.0522 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0435 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0249 | 0.012 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.0713 | 0.0004 | 0.2068 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.0783 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0401 | 0.0026 | 0.9957 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.1381 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d1_test | brier | secondary | -0.0633 | 0.0084 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d1_test | ece_15 | secondary | -0.0267 | 0.5869 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d1_test | nll | secondary | -0.1179 | 0.0026 | 0.9957 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d2_test | brier | secondary | -0.232 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.2027 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d2_test | nll | secondary | -0.454 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0093 | 0.1028 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0101 | 0.3578 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0137 | 0.2466 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0303 | 0.0006 | 0.2616 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0149 | 0.0428 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0454 | 0.0014 | 0.5669 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0597 | 0.0002 | 0.2068 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0265 | 0.0018 | 0.7073 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.0994 | 0.0002 | 0.2068 | 0 |

