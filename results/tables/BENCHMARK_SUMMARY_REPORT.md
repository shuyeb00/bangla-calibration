# Thesis Master Results Summary (60-Run Full Grid)

## 1. Shift Quantification (Part D.2)

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
| banglabert | full_ft | shift_d1_test | source_temp_scaled | 0.6985 | 0.6745 | 0.7102 | 0.0187 | 0.0766 | 0.4069 | 0.137 | 0.9234 | 1.803 |
| banglabert | full_ft | shift_d1_test | target_temp_scaled | 0.6985 | 0.6745 | 0.7098 | 0.0185 | 0.0762 | 0.4065 | 0.1371 | 0.9274 | 1.8229 |
| banglabert | full_ft | shift_d1_test | uncalibrated | 0.6985 | 0.6745 | 0.779 | 0.0668 | 0.13 | 0.4336 | 0.1367 | 0.9254 | 1.81 |
| banglabert | full_ft | shift_d2_test | source_temp_scaled | 0.4547 | 0.3759 | 1.1334 | 0.0734 | 0.1824 | 0.6656 | 0.354 | 0.8065 | 2.1224 |
| banglabert | full_ft | shift_d2_test | target_temp_scaled | 0.4547 | 0.3759 | 1.0343 | 0.0218 | 0.129 | 0.6173 | 0.3544 | 0.901 | 2.4371 |
| banglabert | full_ft | shift_d2_test | uncalibrated | 0.4547 | 0.3759 | 1.4317 | 0.3117 | 0.2837 | 0.7599 | 0.3546 | 0.8206 | 2.1473 |
| banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.709 | 0.6917 | 0.692 | 0.0114 | 0.0471 | 0.3966 | 0.1373 | 0.9156 | 1.7378 |
| banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.709 | 0.6917 | 0.769 | 0.0713 | 0.1158 | 0.4245 | 0.137 | 0.9165 | 1.7381 |
| banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.6781 | 0.6659 | 0.7433 | 0.0179 | 0.0545 | 0.4315 | 0.1641 | 0.9115 | 1.8154 |
| banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.6781 | 0.6659 | 0.8339 | 0.0907 | 0.1286 | 0.4657 | 0.1645 | 0.9143 | 1.8187 |
| banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.62 | 0.6162 | 0.818 | 0.0172 | 0.0785 | 0.4806 | 0.1966 | 0.9174 | 1.9646 |
| banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.62 | 0.6162 | 0.9188 | 0.091 | 0.1546 | 0.5249 | 0.1971 | 0.9215 | 1.9897 |
| banglabert | full_ft | test | source_temp_scaled | 0.7216 | 0.6959 | 0.6667 | 0.0116 | 0.0483 | 0.3811 | 0.1265 | 0.9131 | 1.6725 |
| banglabert | full_ft | test | uncalibrated | 0.7216 | 0.6959 | 0.7461 | 0.0768 | 0.1171 | 0.4098 | 0.1258 | 0.9104 | 1.6609 |
| banglabert | lora | shift_d1_test | source_temp_scaled | 0.6826 | 0.6063 | 0.7509 | 0.0099 | 0.0749 | 0.4262 | 0.1479 | 0.9134 | 1.8965 |
| banglabert | lora | shift_d1_test | target_temp_scaled | 0.6826 | 0.6063 | 0.7513 | 0.01 | 0.0793 | 0.4265 | 0.148 | 0.9214 | 1.9443 |
| banglabert | lora | shift_d1_test | uncalibrated | 0.6826 | 0.6063 | 0.7493 | 0.0104 | 0.0741 | 0.4248 | 0.1477 | 0.9144 | 1.9124 |
| banglabert | lora | shift_d2_test | source_temp_scaled | 0.4172 | 0.1971 | 1.2904 | 0.0636 | 0.2388 | 0.7485 | 0.4211 | 0.6692 | 2.0027 |
| banglabert | lora | shift_d2_test | target_temp_scaled | 0.4172 | 0.1971 | 1.0862 | 0.0028 | 0.079 | 0.6564 | 0.4268 | 0.8883 | 2.498 |
| banglabert | lora | shift_d2_test | uncalibrated | 0.4172 | 0.1971 | 1.3648 | 0.0808 | 0.2753 | 0.7777 | 0.4204 | 0.6769 | 2.0137 |
| banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.6665 | 0.6088 | 0.7702 | 0.0034 | 0.0369 | 0.4457 | 0.1725 | 0.9124 | 1.9046 |
| banglabert | lora | shift_d3_noise05 | uncalibrated | 0.6665 | 0.6088 | 0.7708 | 0.0036 | 0.0433 | 0.4463 | 0.1725 | 0.9144 | 1.9074 |
| banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.6524 | 0.5929 | 0.7939 | 0.0028 | 0.0401 | 0.4631 | 0.1945 | 0.9315 | 2.0506 |
| banglabert | lora | shift_d3_noise10 | uncalibrated | 0.6524 | 0.5929 | 0.7904 | 0.0027 | 0.0355 | 0.4618 | 0.1944 | 0.9335 | 2.0565 |
| banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.6269 | 0.5658 | 0.8412 | 0.0032 | 0.0604 | 0.4902 | 0.2147 | 0.9409 | 2.2674 |
| banglabert | lora | shift_d3_noise20 | uncalibrated | 0.6269 | 0.5658 | 0.8334 | 0.0034 | 0.038 | 0.4853 | 0.2143 | 0.9421 | 2.281 |
| banglabert | lora | test | source_temp_scaled | 0.6643 | 0.5941 | 0.7521 | 0.0043 | 0.0612 | 0.4366 | 0.1608 | 0.9029 | 1.7928 |
| banglabert | lora | test | uncalibrated | 0.6643 | 0.5941 | 0.7558 | 0.0042 | 0.0717 | 0.4397 | 0.1607 | 0.9044 | 1.7972 |
| indicbert | full_ft | shift_d1_test | source_temp_scaled | 0.5861 | 0.5259 | 0.8829 | 0.0265 | 0.0724 | 0.5152 | 0.2366 | 0.9174 | 2.1652 |
| indicbert | full_ft | shift_d1_test | target_temp_scaled | 0.5861 | 0.5259 | 0.8841 | 0.0298 | 0.0825 | 0.5161 | 0.2364 | 0.8776 | 2.0179 |
| indicbert | full_ft | shift_d1_test | uncalibrated | 0.5861 | 0.5259 | 0.8952 | 0.0341 | 0.0911 | 0.5214 | 0.2364 | 0.9184 | 2.1721 |
| indicbert | full_ft | shift_d2_test | source_temp_scaled | 0.453 | 0.3202 | 1.157 | 0.0987 | 0.1785 | 0.7028 | 0.4842 | 0.7602 | 2.0754 |
| indicbert | full_ft | shift_d2_test | target_temp_scaled | 0.453 | 0.3202 | 1.0602 | 0.0212 | 0.0386 | 0.6399 | 0.484 | 0.9159 | 2.6169 |
| indicbert | full_ft | shift_d2_test | uncalibrated | 0.453 | 0.3202 | 1.27 | 0.1706 | 0.2374 | 0.754 | 0.4842 | 0.759 | 2.0697 |
| indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.5957 | 0.5542 | 0.8757 | 0.0164 | 0.0359 | 0.5126 | 0.2421 | 0.9128 | 2.1569 |
| indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.5957 | 0.5542 | 0.8878 | 0.0091 | 0.0617 | 0.5188 | 0.242 | 0.9134 | 2.1576 |
| indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.5862 | 0.5418 | 0.8899 | 0.0131 | 0.0305 | 0.5218 | 0.2545 | 0.906 | 2.1459 |
| indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.5862 | 0.5418 | 0.9069 | 0.0133 | 0.067 | 0.5301 | 0.2544 | 0.9074 | 2.1485 |
| indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.5894 | 0.5462 | 0.9037 | 0.0161 | 0.0308 | 0.5309 | 0.2712 | 0.9053 | 2.1979 |
| indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.5894 | 0.5462 | 0.9201 | 0.0072 | 0.063 | 0.5378 | 0.271 | 0.9062 | 2.1984 |
| indicbert | full_ft | test | source_temp_scaled | 0.6097 | 0.5667 | 0.8592 | 0.0196 | 0.0326 | 0.5013 | 0.2289 | 0.9144 | 2.1199 |
| indicbert | full_ft | test | uncalibrated | 0.6097 | 0.5667 | 0.8681 | 0.0133 | 0.0539 | 0.5059 | 0.2289 | 0.9163 | 2.1253 |
| indicbert | lora | shift_d1_test | source_temp_scaled | 0.403 | 0.1915 | 1.0657 | 0.0017 | 0.0157 | 0.6465 | 0.5887 | 0.8995 | 2.4756 |
| indicbert | lora | shift_d1_test | target_temp_scaled | 0.403 | 0.1915 | 1.0651 | 0.002 | 0.0255 | 0.6464 | 0.5881 | 0.8925 | 2.4438 |
| indicbert | lora | shift_d1_test | uncalibrated | 0.403 | 0.1915 | 1.0724 | 0.0016 | 0.0176 | 0.65 | 0.5891 | 0.8995 | 2.4756 |
| indicbert | lora | shift_d2_test | source_temp_scaled | 0.4167 | 0.1961 | 1.0717 | 0.0005 | 0.0056 | 0.6494 | 0.5858 | 0.8266 | 2.2308 |
| indicbert | lora | shift_d2_test | target_temp_scaled | 0.4167 | 0.1961 | 1.0717 | 0.0004 | 0.0086 | 0.6494 | 0.5858 | 0.8948 | 2.5087 |
| indicbert | lora | shift_d2_test | uncalibrated | 0.4167 | 0.1961 | 1.0758 | 0.0009 | 0.0302 | 0.6519 | 0.586 | 0.8264 | 2.2306 |
| indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.4147 | 0.1954 | 1.0733 | 0.0004 | 0.0061 | 0.6504 | 0.5566 | 0.8987 | 2.4544 |
| indicbert | lora | shift_d3_noise05 | uncalibrated | 0.4147 | 0.1954 | 1.077 | 0.0007 | 0.0292 | 0.6526 | 0.5567 | 0.8985 | 2.4531 |
| indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.4147 | 0.1954 | 1.0731 | 0.0004 | 0.0061 | 0.6503 | 0.5495 | 0.9026 | 2.451 |
| indicbert | lora | shift_d3_noise10 | uncalibrated | 0.4147 | 0.1954 | 1.0769 | 0.0006 | 0.0292 | 0.6526 | 0.5494 | 0.9025 | 2.4499 |
| indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.4147 | 0.1954 | 1.0733 | 0.0004 | 0.0061 | 0.6504 | 0.5649 | 0.8979 | 2.4432 |
| indicbert | lora | shift_d3_noise20 | uncalibrated | 0.4147 | 0.1954 | 1.0771 | 0.0007 | 0.0292 | 0.6526 | 0.5648 | 0.8975 | 2.4426 |
| indicbert | lora | test | source_temp_scaled | 0.4147 | 0.1954 | 1.0732 | 0.0004 | 0.0061 | 0.6504 | 0.5534 | 0.9007 | 2.4629 |
| indicbert | lora | test | uncalibrated | 0.4147 | 0.1954 | 1.077 | 0.0007 | 0.0293 | 0.6526 | 0.5535 | 0.9006 | 2.4621 |
| xlmr_base | full_ft | shift_d1_test | source_temp_scaled | 0.6368 | 0.5984 | 0.8127 | 0.0198 | 0.0814 | 0.4707 | 0.1886 | 0.8896 | 1.8856 |
| xlmr_base | full_ft | shift_d1_test | target_temp_scaled | 0.6368 | 0.5984 | 0.8147 | 0.022 | 0.0778 | 0.4717 | 0.1887 | 0.8766 | 1.806 |
| xlmr_base | full_ft | shift_d1_test | uncalibrated | 0.6368 | 0.5984 | 0.9118 | 0.0326 | 0.1565 | 0.5084 | 0.1889 | 0.8876 | 1.8697 |
| xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 0.541 | 0.3961 | 1.1007 | 0.0722 | 0.2042 | 0.6474 | 0.3259 | 0.7264 | 1.6801 |
| xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 0.541 | 0.3961 | 0.9741 | 0.0166 | 0.0701 | 0.5792 | 0.3293 | 0.9231 | 2.357 |
| xlmr_base | full_ft | shift_d2_test | uncalibrated | 0.541 | 0.3961 | 1.4596 | 0.1268 | 0.3057 | 0.74 | 0.3227 | 0.7221 | 1.6689 |
| xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.6822 | 0.6506 | 0.7252 | 0.008 | 0.0407 | 0.4207 | 0.1541 | 0.9185 | 1.8278 |
| xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.6822 | 0.6506 | 0.7838 | 0.0294 | 0.113 | 0.4447 | 0.1541 | 0.9172 | 1.8137 |
| xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.6675 | 0.6365 | 0.7593 | 0.0025 | 0.0385 | 0.4417 | 0.1725 | 0.9096 | 1.8615 |
| xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.6675 | 0.6365 | 0.8306 | 0.0226 | 0.1215 | 0.4698 | 0.1726 | 0.9088 | 1.8476 |
| xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.6337 | 0.5974 | 0.8106 | 0.0101 | 0.0491 | 0.4737 | 0.1995 | 0.9021 | 1.9304 |
| xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.6337 | 0.5974 | 0.8978 | 0.0391 | 0.1439 | 0.5084 | 0.1999 | 0.8991 | 1.92 |
| xlmr_base | full_ft | test | source_temp_scaled | 0.6866 | 0.6567 | 0.7142 | 0.007 | 0.0418 | 0.4134 | 0.148 | 0.9185 | 1.8084 |
| xlmr_base | full_ft | test | uncalibrated | 0.6866 | 0.6567 | 0.7719 | 0.0275 | 0.1118 | 0.4363 | 0.1484 | 0.9175 | 1.7959 |
| xlmr_base | lora | shift_d1_test | source_temp_scaled | 0.601 | 0.5432 | 0.9026 | 0.0126 | 0.095 | 0.5275 | 0.2557 | 0.8617 | 1.9682 |
| xlmr_base | lora | shift_d1_test | target_temp_scaled | 0.601 | 0.5432 | 0.8977 | 0.0105 | 0.0802 | 0.5254 | 0.2553 | 0.8836 | 2.0925 |
| xlmr_base | lora | shift_d1_test | uncalibrated | 0.601 | 0.5432 | 0.9197 | 0.0176 | 0.1022 | 0.5345 | 0.2559 | 0.8637 | 1.9801 |
| xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.5754 | 0.4444 | 0.93 | 0.0237 | 0.1159 | 0.5573 | 0.2666 | 0.8264 | 1.8463 |
| xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.5754 | 0.4444 | 0.9052 | 0.0133 | 0.0815 | 0.5366 | 0.2684 | 0.909 | 2.1037 |
| xlmr_base | lora | shift_d2_test | uncalibrated | 0.5754 | 0.4444 | 0.964 | 0.0329 | 0.1492 | 0.5747 | 0.2658 | 0.8279 | 1.852 |
| xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.6309 | 0.5703 | 0.8206 | 0.0049 | 0.0303 | 0.4802 | 0.2116 | 0.8957 | 1.9194 |
| xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.6309 | 0.5703 | 0.8261 | 0.0057 | 0.0515 | 0.4836 | 0.2117 | 0.8978 | 1.9282 |
| xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.6138 | 0.5564 | 0.8458 | 0.0039 | 0.0417 | 0.4972 | 0.2266 | 0.8874 | 1.9384 |
| xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.6138 | 0.5564 | 0.8554 | 0.0047 | 0.0661 | 0.5027 | 0.2267 | 0.8903 | 1.9465 |
| xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.5975 | 0.5334 | 0.8829 | 0.0014 | 0.0508 | 0.5189 | 0.2471 | 0.8666 | 1.9178 |
| xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.5975 | 0.5334 | 0.8982 | 0.0043 | 0.0851 | 0.5268 | 0.2471 | 0.8681 | 1.9247 |
| xlmr_base | lora | test | source_temp_scaled | 0.6322 | 0.5758 | 0.8109 | 0.0067 | 0.0269 | 0.4741 | 0.2032 | 0.9028 | 1.949 |
| xlmr_base | lora | test | uncalibrated | 0.6322 | 0.5758 | 0.8138 | 0.0081 | 0.0474 | 0.4763 | 0.2033 | 0.905 | 1.956 |


## 2. Offensive Task — Seed-Aggregated Metrics (5 Seeds)

| model | ft_method | condition | calibration_regime | accuracy_mean | macro_f1_mean | nll_mean | nll_std | ece_15_mean | brier_mean | aurc_mean | conformal_coverage_mean | conformal_set_size_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.9175 | 0.9174 | 0.2333 | 0.0065 | 0.021 | 0.1319 | 0.0227 | 0.9175 | 1.0 |
| banglabert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.9175 | 0.9174 | 0.2332 | 0.0066 | 0.0239 | 0.1312 | 0.0227 | 0.9175 | 1.0 |
| banglabert | full_ft | shift_d1_barishal_test | uncalibrated | 0.9175 | 0.9174 | 0.3138 | 0.0562 | 0.0576 | 0.1437 | 0.0227 | 0.9175 | 1.0 |
| banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.8292 | 0.8278 | 0.43 | 0.019 | 0.0672 | 0.2643 | 0.0786 | 0.8292 | 1.0 |
| banglabert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.8292 | 0.8278 | 0.4067 | 0.0097 | 0.0336 | 0.254 | 0.0786 | 0.9005 | 1.1886 |
| banglabert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.8292 | 0.8278 | 0.6289 | 0.139 | 0.1228 | 0.2935 | 0.0786 | 0.8292 | 1.0 |
| banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.9092 | 0.9092 | 0.2439 | 0.0101 | 0.0297 | 0.1416 | 0.0229 | 0.9092 | 1.0 |
| banglabert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.9092 | 0.9092 | 0.2454 | 0.0109 | 0.0303 | 0.1407 | 0.0229 | 0.9252 | 1.0356 |
| banglabert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.9092 | 0.9092 | 0.3198 | 0.0639 | 0.0607 | 0.1559 | 0.0229 | 0.9092 | 1.0 |
| banglabert | full_ft | shift_d2_test | source_temp_scaled | 0.5255 | 0.351 | 1.4578 | 0.1651 | 0.4142 | 0.8483 | 0.4554 | 0.5255 | 1.0 |
| banglabert | full_ft | shift_d2_test | target_temp_scaled | 0.5255 | 0.351 | 0.719 | 0.0125 | 0.1106 | 0.5244 | 0.4554 | 0.913 | 1.8058 |
| banglabert | full_ft | shift_d2_test | uncalibrated | 0.5255 | 0.351 | 2.5718 | 0.6953 | 0.4558 | 0.9188 | 0.4554 | 0.5255 | 1.0 |
| banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.9019 | 0.9015 | 0.2643 | 0.006 | 0.0259 | 0.154 | 0.0277 | 0.9019 | 1.0 |
| banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.9019 | 0.9015 | 0.3587 | 0.0637 | 0.0667 | 0.1683 | 0.0277 | 0.9019 | 1.0 |
| banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.8854 | 0.8846 | 0.3059 | 0.0076 | 0.0315 | 0.1801 | 0.041 | 0.8854 | 1.0 |
| banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.8854 | 0.8846 | 0.4255 | 0.0761 | 0.0796 | 0.1977 | 0.041 | 0.8854 | 1.0 |
| banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.834 | 0.8305 | 0.4173 | 0.0183 | 0.0653 | 0.2559 | 0.0727 | 0.834 | 1.0 |
| banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.834 | 0.8305 | 0.6094 | 0.1271 | 0.1211 | 0.2845 | 0.0727 | 0.834 | 1.0 |
| banglabert | full_ft | test | source_temp_scaled | 0.9184 | 0.9183 | 0.2249 | 0.0048 | 0.0225 | 0.1282 | 0.0195 | 0.9184 | 1.0 |
| banglabert | full_ft | test | uncalibrated | 0.9184 | 0.9183 | 0.2996 | 0.0534 | 0.0564 | 0.1404 | 0.0195 | 0.9184 | 1.0 |
| banglabert | lora | shift_d1_barishal_test | source_temp_scaled | 0.8854 | 0.8854 | 0.2779 | 0.0006 | 0.0157 | 0.168 | 0.0297 | 0.8888 | 1.0075 |
| banglabert | lora | shift_d1_barishal_test | target_temp_scaled | 0.8854 | 0.8854 | 0.2779 | 0.0007 | 0.0163 | 0.168 | 0.0297 | 0.8953 | 1.0223 |
| banglabert | lora | shift_d1_barishal_test | uncalibrated | 0.8854 | 0.8854 | 0.2899 | 0.001 | 0.0393 | 0.1724 | 0.0297 | 0.8888 | 1.0075 |
| banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.7648 | 0.7638 | 0.5028 | 0.0064 | 0.0689 | 0.3307 | 0.1096 | 0.7726 | 1.0159 |
| banglabert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.7648 | 0.7638 | 0.4839 | 0.0048 | 0.0367 | 0.3207 | 0.1096 | 0.8946 | 1.3244 |
| banglabert | lora | shift_d1_chittagong_test | uncalibrated | 0.7648 | 0.7638 | 0.5459 | 0.0104 | 0.1019 | 0.3454 | 0.1096 | 0.7726 | 1.0159 |
| banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.8674 | 0.8671 | 0.3278 | 0.0056 | 0.0181 | 0.2006 | 0.0442 | 0.8721 | 1.0095 |
| banglabert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.8674 | 0.8671 | 0.3282 | 0.0054 | 0.0211 | 0.2004 | 0.0442 | 0.9109 | 1.1124 |
| banglabert | lora | shift_d1_noakhali_test | uncalibrated | 0.8674 | 0.8671 | 0.3434 | 0.0078 | 0.0428 | 0.2045 | 0.0442 | 0.8721 | 1.0095 |
| banglabert | lora | shift_d2_test | source_temp_scaled | 0.5162 | 0.3467 | 0.8599 | 0.0448 | 0.2632 | 0.6301 | 0.423 | 0.5235 | 1.0115 |
| banglabert | lora | shift_d2_test | target_temp_scaled | 0.5162 | 0.3467 | 0.6919 | 0.0007 | 0.0337 | 0.4987 | 0.423 | 0.9215 | 1.8042 |
| banglabert | lora | shift_d2_test | uncalibrated | 0.5162 | 0.3467 | 0.9656 | 0.0723 | 0.3077 | 0.6873 | 0.423 | 0.5235 | 1.0115 |
| banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.883 | 0.8828 | 0.2901 | 0.0009 | 0.0122 | 0.1751 | 0.0334 | 0.887 | 1.0086 |
| banglabert | lora | shift_d3_noise05 | uncalibrated | 0.883 | 0.8828 | 0.3005 | 0.0017 | 0.0361 | 0.1786 | 0.0334 | 0.887 | 1.0086 |
| banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.8603 | 0.8597 | 0.3313 | 0.0018 | 0.0165 | 0.204 | 0.0455 | 0.8645 | 1.0086 |
| banglabert | lora | shift_d3_noise10 | uncalibrated | 0.8603 | 0.8597 | 0.343 | 0.0034 | 0.043 | 0.2086 | 0.0455 | 0.8645 | 1.0086 |
| banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.8106 | 0.8076 | 0.4198 | 0.0025 | 0.0216 | 0.2688 | 0.0796 | 0.8173 | 1.0125 |
| banglabert | lora | shift_d3_noise20 | uncalibrated | 0.8106 | 0.8076 | 0.4352 | 0.0048 | 0.0569 | 0.2756 | 0.0796 | 0.8173 | 1.0125 |
| banglabert | lora | test | source_temp_scaled | 0.8962 | 0.8962 | 0.2584 | 0.0012 | 0.015 | 0.1549 | 0.025 | 0.899 | 1.0066 |
| banglabert | lora | test | uncalibrated | 0.8962 | 0.8962 | 0.2695 | 0.0022 | 0.0362 | 0.1589 | 0.025 | 0.899 | 1.0066 |
| indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.8651 | 0.8651 | 0.3375 | 0.0035 | 0.0342 | 0.2052 | 0.0478 | 0.8817 | 1.0397 |
| indicbert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.8651 | 0.8651 | 0.337 | 0.0034 | 0.0347 | 0.2041 | 0.0478 | 0.9056 | 1.1 |
| indicbert | full_ft | shift_d1_barishal_test | uncalibrated | 0.8651 | 0.8651 | 0.424 | 0.0193 | 0.0903 | 0.2266 | 0.0478 | 0.8817 | 1.0397 |
| indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.7746 | 0.7728 | 0.5112 | 0.0115 | 0.0959 | 0.3318 | 0.1033 | 0.8007 | 1.0523 |
| indicbert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.7746 | 0.7728 | 0.481 | 0.0076 | 0.0492 | 0.3136 | 0.1033 | 0.9086 | 1.3335 |
| indicbert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.7746 | 0.7728 | 0.711 | 0.0638 | 0.1669 | 0.3789 | 0.1033 | 0.8007 | 1.0523 |
| indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.8529 | 0.8529 | 0.3519 | 0.0042 | 0.0408 | 0.2173 | 0.0489 | 0.8734 | 1.041 |
| indicbert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.8529 | 0.8529 | 0.3518 | 0.0039 | 0.046 | 0.215 | 0.0489 | 0.9153 | 1.1495 |
| indicbert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.8529 | 0.8529 | 0.4467 | 0.0214 | 0.0992 | 0.2429 | 0.0489 | 0.8734 | 1.041 |
| indicbert | full_ft | shift_d2_test | source_temp_scaled | 0.527 | 0.4182 | 0.9413 | 0.1181 | 0.2729 | 0.664 | 0.4453 | 0.5775 | 1.0987 |
| indicbert | full_ft | shift_d2_test | target_temp_scaled | 0.527 | 0.4182 | 0.6943 | 0.0018 | 0.0412 | 0.5011 | 0.4452 | 0.907 | 1.7812 |
| indicbert | full_ft | shift_d2_test | uncalibrated | 0.527 | 0.4182 | 1.3658 | 0.2513 | 0.372 | 0.789 | 0.4452 | 0.5775 | 1.0987 |
| indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.8693 | 0.8687 | 0.3273 | 0.0047 | 0.0374 | 0.1982 | 0.0425 | 0.8856 | 1.0346 |
| indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.8693 | 0.8687 | 0.4139 | 0.0201 | 0.0889 | 0.2197 | 0.0425 | 0.8856 | 1.0346 |
| indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.8496 | 0.8488 | 0.3727 | 0.0076 | 0.0438 | 0.2298 | 0.0562 | 0.8666 | 1.0366 |
| indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.8496 | 0.8488 | 0.4875 | 0.0283 | 0.1065 | 0.2562 | 0.0562 | 0.8666 | 1.0366 |
| indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8117 | 0.8102 | 0.4468 | 0.0124 | 0.0632 | 0.2828 | 0.0839 | 0.8351 | 1.0483 |
| indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.8117 | 0.8102 | 0.6015 | 0.0328 | 0.1339 | 0.3176 | 0.0839 | 0.8351 | 1.0483 |
| indicbert | full_ft | test | source_temp_scaled | 0.8873 | 0.887 | 0.291 | 0.0044 | 0.0314 | 0.172 | 0.0334 | 0.9023 | 1.0307 |
| indicbert | full_ft | test | uncalibrated | 0.8873 | 0.887 | 0.3571 | 0.0153 | 0.0747 | 0.1888 | 0.0334 | 0.9023 | 1.0307 |
| indicbert | lora | shift_d1_barishal_test | source_temp_scaled | 0.686 | 0.6846 | 0.6249 | 0.0139 | 0.0716 | 0.4333 | 0.262 | 0.9009 | 1.6282 |
| indicbert | lora | shift_d1_barishal_test | target_temp_scaled | 0.686 | 0.6846 | 0.6241 | 0.0137 | 0.0695 | 0.433 | 0.262 | 0.8998 | 1.6283 |
| indicbert | lora | shift_d1_barishal_test | uncalibrated | 0.686 | 0.6846 | 0.63 | 0.0145 | 0.0828 | 0.4388 | 0.262 | 0.9009 | 1.6281 |
| indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.6218 | 0.6193 | 0.683 | 0.0133 | 0.0939 | 0.4849 | 0.341 | 0.8576 | 1.6007 |
| indicbert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.6218 | 0.6193 | 0.6677 | 0.0058 | 0.0652 | 0.4741 | 0.341 | 0.8834 | 1.6634 |
| indicbert | lora | shift_d1_chittagong_test | uncalibrated | 0.6218 | 0.6193 | 0.6671 | 0.0062 | 0.0541 | 0.4738 | 0.341 | 0.8576 | 1.6007 |
| indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.6485 | 0.6474 | 0.6503 | 0.0133 | 0.0705 | 0.4565 | 0.2961 | 0.8901 | 1.6375 |
| indicbert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.6485 | 0.6474 | 0.6467 | 0.0115 | 0.0506 | 0.4547 | 0.2961 | 0.9113 | 1.6982 |
| indicbert | lora | shift_d1_noakhali_test | uncalibrated | 0.6485 | 0.6474 | 0.647 | 0.0121 | 0.0591 | 0.455 | 0.2961 | 0.89 | 1.6375 |
| indicbert | lora | shift_d2_test | source_temp_scaled | 0.54 | 0.4837 | 0.7549 | 0.0807 | 0.1193 | 0.5483 | 0.4529 | 0.8465 | 1.682 |
| indicbert | lora | shift_d2_test | target_temp_scaled | 0.54 | 0.4837 | 0.6907 | 0.0038 | 0.0193 | 0.4975 | 0.4529 | 0.8985 | 1.7802 |
| indicbert | lora | shift_d2_test | uncalibrated | 0.54 | 0.4837 | 0.713 | 0.0321 | 0.0755 | 0.5176 | 0.4529 | 0.8465 | 1.682 |
| indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.687 | 0.6856 | 0.6258 | 0.0135 | 0.071 | 0.4334 | 0.2688 | 0.8875 | 1.5903 |
| indicbert | lora | shift_d3_noise05 | uncalibrated | 0.687 | 0.6856 | 0.6286 | 0.0147 | 0.0729 | 0.4373 | 0.2688 | 0.8875 | 1.5903 |
| indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.6831 | 0.6817 | 0.6289 | 0.0127 | 0.0695 | 0.4361 | 0.2726 | 0.8858 | 1.5868 |
| indicbert | lora | shift_d3_noise10 | uncalibrated | 0.6831 | 0.6817 | 0.6304 | 0.0138 | 0.0665 | 0.439 | 0.2726 | 0.8858 | 1.5867 |
| indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.6699 | 0.6681 | 0.6385 | 0.0118 | 0.0726 | 0.4445 | 0.2835 | 0.8801 | 1.582 |
| indicbert | lora | shift_d3_noise20 | uncalibrated | 0.6699 | 0.6681 | 0.6362 | 0.0114 | 0.0621 | 0.4445 | 0.2835 | 0.8801 | 1.5819 |
| indicbert | lora | test | source_temp_scaled | 0.6931 | 0.6918 | 0.6174 | 0.0142 | 0.0668 | 0.4262 | 0.255 | 0.8942 | 1.5914 |
| indicbert | lora | test | uncalibrated | 0.6931 | 0.6918 | 0.6234 | 0.0162 | 0.0753 | 0.4325 | 0.255 | 0.8942 | 1.5913 |
| xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.9078 | 0.9078 | 0.2435 | 0.0062 | 0.0177 | 0.1416 | 0.0234 | 0.9078 | 1.0 |
| xlmr_base | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.9078 | 0.9078 | 0.2439 | 0.0061 | 0.021 | 0.1408 | 0.0234 | 0.9104 | 1.0061 |
| xlmr_base | full_ft | shift_d1_barishal_test | uncalibrated | 0.9078 | 0.9078 | 0.2984 | 0.0274 | 0.0584 | 0.1528 | 0.0234 | 0.9078 | 1.0 |
| xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.834 | 0.8338 | 0.3981 | 0.0158 | 0.0549 | 0.2468 | 0.0657 | 0.834 | 1.0 |
| xlmr_base | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.834 | 0.8338 | 0.385 | 0.0105 | 0.0264 | 0.2398 | 0.0657 | 0.9124 | 1.1957 |
| xlmr_base | full_ft | shift_d1_chittagong_test | uncalibrated | 0.834 | 0.8338 | 0.5154 | 0.0558 | 0.1101 | 0.2716 | 0.0657 | 0.834 | 1.0 |
| xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.8933 | 0.8933 | 0.2686 | 0.0075 | 0.0287 | 0.1612 | 0.0268 | 0.8933 | 1.0 |
| xlmr_base | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.8933 | 0.8933 | 0.2709 | 0.0076 | 0.0317 | 0.1597 | 0.0268 | 0.9206 | 1.0646 |
| xlmr_base | full_ft | shift_d1_noakhali_test | uncalibrated | 0.8933 | 0.8933 | 0.3282 | 0.0346 | 0.0667 | 0.1755 | 0.0268 | 0.8933 | 1.0 |
| xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 0.5228 | 0.3699 | 1.2312 | 0.0949 | 0.3748 | 0.7883 | 0.4639 | 0.5228 | 1.0 |
| xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 0.5228 | 0.3699 | 0.7051 | 0.0045 | 0.0821 | 0.5115 | 0.4639 | 0.9262 | 1.8298 |
| xlmr_base | full_ft | shift_d2_test | uncalibrated | 0.5228 | 0.3699 | 1.9049 | 0.3257 | 0.4372 | 0.8873 | 0.4639 | 0.5228 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.9053 | 0.9051 | 0.2498 | 0.0051 | 0.0195 | 0.1464 | 0.0255 | 0.9053 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.9053 | 0.9051 | 0.3011 | 0.0226 | 0.0576 | 0.1575 | 0.0255 | 0.9053 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.889 | 0.8888 | 0.2819 | 0.0053 | 0.0213 | 0.1677 | 0.0343 | 0.889 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.889 | 0.8888 | 0.3423 | 0.0259 | 0.0671 | 0.1812 | 0.0343 | 0.889 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8598 | 0.8594 | 0.3442 | 0.0088 | 0.0264 | 0.2096 | 0.0527 | 0.8598 | 1.0 |
| xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.8598 | 0.8594 | 0.4207 | 0.034 | 0.0828 | 0.2268 | 0.0527 | 0.8598 | 1.0 |
| xlmr_base | full_ft | test | source_temp_scaled | 0.915 | 0.9149 | 0.2253 | 0.0058 | 0.0181 | 0.1302 | 0.0201 | 0.915 | 1.0 |
| xlmr_base | full_ft | test | uncalibrated | 0.915 | 0.9149 | 0.2725 | 0.0193 | 0.0532 | 0.1405 | 0.0201 | 0.915 | 1.0 |
| xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | 0.8243 | 0.824 | 0.3851 | 0.005 | 0.0185 | 0.2433 | 0.0664 | 0.8941 | 1.1499 |
| xlmr_base | lora | shift_d1_barishal_test | target_temp_scaled | 0.8243 | 0.824 | 0.387 | 0.0045 | 0.0296 | 0.2437 | 0.0664 | 0.9145 | 1.2154 |
| xlmr_base | lora | shift_d1_barishal_test | uncalibrated | 0.8243 | 0.824 | 0.4031 | 0.0076 | 0.0528 | 0.2493 | 0.0664 | 0.8941 | 1.1499 |
| xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | 0.7542 | 0.7526 | 0.5114 | 0.0081 | 0.0352 | 0.3395 | 0.1297 | 0.8455 | 1.2113 |
| xlmr_base | lora | shift_d1_chittagong_test | target_temp_scaled | 0.7542 | 0.7526 | 0.5058 | 0.0061 | 0.0213 | 0.3365 | 0.1297 | 0.8955 | 1.3727 |
| xlmr_base | lora | shift_d1_chittagong_test | uncalibrated | 0.7542 | 0.7526 | 0.5483 | 0.0125 | 0.0829 | 0.3526 | 0.1297 | 0.8455 | 1.2113 |
| xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | 0.8077 | 0.8065 | 0.4239 | 0.0036 | 0.0213 | 0.2733 | 0.0819 | 0.8774 | 1.1711 |
| xlmr_base | lora | shift_d1_noakhali_test | target_temp_scaled | 0.8077 | 0.8065 | 0.4247 | 0.0033 | 0.0219 | 0.2734 | 0.0819 | 0.9107 | 1.2721 |
| xlmr_base | lora | shift_d1_noakhali_test | uncalibrated | 0.8077 | 0.8065 | 0.4445 | 0.0054 | 0.0571 | 0.2803 | 0.0819 | 0.8774 | 1.171 |
| xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.509 | 0.4291 | 0.8318 | 0.04 | 0.2151 | 0.6029 | 0.4501 | 0.6592 | 1.288 |
| xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.509 | 0.4291 | 0.692 | 0.0026 | 0.0335 | 0.4989 | 0.4501 | 0.9158 | 1.8062 |
| xlmr_base | lora | shift_d2_test | uncalibrated | 0.509 | 0.4291 | 0.9517 | 0.0667 | 0.2699 | 0.6613 | 0.4501 | 0.6592 | 1.288 |
| xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.8299 | 0.8298 | 0.3779 | 0.0028 | 0.0112 | 0.2398 | 0.0619 | 0.8922 | 1.1479 |
| xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.8299 | 0.8298 | 0.3942 | 0.0043 | 0.048 | 0.2457 | 0.0619 | 0.8922 | 1.1479 |
| xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.8041 | 0.8038 | 0.4205 | 0.0049 | 0.0153 | 0.2702 | 0.0806 | 0.8806 | 1.1711 |
| xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.8041 | 0.8038 | 0.4427 | 0.0076 | 0.0578 | 0.2776 | 0.0806 | 0.8806 | 1.1711 |
| xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.7696 | 0.7683 | 0.4779 | 0.0098 | 0.0205 | 0.3148 | 0.1102 | 0.8652 | 1.2268 |
| xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.7696 | 0.7683 | 0.5036 | 0.0134 | 0.0638 | 0.3239 | 0.1102 | 0.8652 | 1.2268 |
| xlmr_base | lora | test | source_temp_scaled | 0.8472 | 0.8472 | 0.3494 | 0.0021 | 0.012 | 0.2181 | 0.0521 | 0.9008 | 1.1215 |
| xlmr_base | lora | test | uncalibrated | 0.8472 | 0.8472 | 0.3675 | 0.004 | 0.0473 | 0.224 | 0.0521 | 0.9008 | 1.1215 |


## 3. Importance-Weighted Conformal Prediction (Part I.7)

| task | model | ft_method | condition | weighted_coverage | weighted_avg_set_size | effective_sample_size |
| --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | 0.9453 | 1.195 | 27.0447 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | 0.9276 | 1.3965 | 6.3953 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | 0.9365 | 1.1449 | 42.332 |
| offensive | banglabert | full_ft | shift_d2_test | 1.0 | 2.0 | 1893.3097 |
| offensive | banglabert | full_ft | shift_d3_noise05 | 0.9091 | 1.0194 | 335.9058 |
| offensive | banglabert | full_ft | shift_d3_noise10 | 0.9144 | 1.1088 | 59.3095 |
| offensive | banglabert | full_ft | shift_d3_noise20 | 0.9456 | 1.477 | 23.716 |
| offensive | banglabert | lora | shift_d1_barishal_test | 0.9301 | 1.2193 | 21.3302 |
| offensive | banglabert | lora | shift_d1_chittagong_test | 0.9398 | 1.6337 | 17.5505 |
| offensive | banglabert | lora | shift_d1_noakhali_test | 0.9189 | 1.2766 | 11.1435 |
| offensive | banglabert | lora | shift_d2_test | 1.0 | 2.0 | 3279.7825 |
| offensive | banglabert | lora | shift_d3_noise05 | 0.8887 | 1.019 | 122.2603 |
| offensive | banglabert | lora | shift_d3_noise10 | 0.8972 | 1.1778 | 25.8001 |
| offensive | banglabert | lora | shift_d3_noise20 | 0.9467 | 1.6452 | 8.0394 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | 0.8944 | 1.0737 | 1381.2492 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | 0.8274 | 1.1126 | 1297.4551 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | 0.9056 | 1.1248 | 990.8196 |
| offensive | indicbert | full_ft | shift_d2_test | 0.8872 | 1.7608 | 319.8006 |
| offensive | indicbert | full_ft | shift_d3_noise05 | 0.8951 | 1.0559 | 3770.1353 |
| offensive | indicbert | full_ft | shift_d3_noise10 | 0.885 | 1.0819 | 3351.6381 |
| offensive | indicbert | full_ft | shift_d3_noise20 | 0.8677 | 1.1278 | 2310.7054 |
| offensive | indicbert | lora | shift_d1_barishal_test | 0.9176 | 1.6798 | 135.5366 |
| offensive | indicbert | lora | shift_d1_chittagong_test | 0.8567 | 1.6011 | 79.0213 |
| offensive | indicbert | lora | shift_d1_noakhali_test | 0.8916 | 1.6321 | 239.0071 |
| offensive | indicbert | lora | shift_d2_test | 1.0 | 2.0 | 634.4144 |
| offensive | indicbert | lora | shift_d3_noise05 | 0.8925 | 1.6056 | 3565.5033 |
| offensive | indicbert | lora | shift_d3_noise10 | 0.8989 | 1.6231 | 2649.4684 |
| offensive | indicbert | lora | shift_d3_noise20 | 0.9006 | 1.6438 | 815.9189 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | 0.9171 | 1.0353 | 403.7726 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | 0.917 | 1.2503 | 130.3259 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | 0.9106 | 1.0596 | 279.876 |
| offensive | xlmr_base | full_ft | shift_d2_test | 0.999 | 1.9982 | 169.6691 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | 0.9089 | 1.0079 | 1737.1533 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | 0.9014 | 1.0291 | 309.1843 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | 0.9081 | 1.148 | 49.517 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | 0.9349 | 1.3844 | 27.2227 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | 0.9601 | 1.7537 | 19.1615 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | 0.9484 | 1.5148 | 43.2271 |
| offensive | xlmr_base | lora | shift_d2_test | 1.0 | 2.0 | 2163.0938 |
| offensive | xlmr_base | lora | shift_d3_noise05 | 0.8975 | 1.1631 | 746.142 |
| offensive | xlmr_base | lora | shift_d3_noise10 | 0.8746 | 1.1742 | 75.7214 |
| offensive | xlmr_base | lora | shift_d3_noise20 | 0.8813 | 1.3929 | 11.5422 |
| sentiment | banglabert | full_ft | shift_d1_test | 1.0 | 3.0 | 133.9015 |
| sentiment | banglabert | full_ft | shift_d2_test | 1.0 | 3.0 | 987.6945 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | 0.9541 | 2.2765 | 78.1525 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | 0.9974 | 2.9171 | 90.35 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | 1.0 | 2.9999 | 77.1346 |
| sentiment | banglabert | lora | shift_d1_test | 1.0 | 3.0 | 157.8043 |
| sentiment | banglabert | lora | shift_d2_test | 1.0 | 3.0 | 1032.6403 |
| sentiment | banglabert | lora | shift_d3_noise05 | 0.9632 | 2.5028 | 38.2887 |
| sentiment | banglabert | lora | shift_d3_noise10 | 0.9996 | 2.9749 | 88.0339 |
| sentiment | banglabert | lora | shift_d3_noise20 | 1.0 | 3.0 | 82.5351 |
| sentiment | indicbert | full_ft | shift_d1_test | 0.9652 | 2.4945 | 355.919 |
| sentiment | indicbert | full_ft | shift_d2_test | 1.0 | 3.0 | 417.3387 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | 0.9143 | 2.1694 | 482.4312 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | 0.9184 | 2.2303 | 454.918 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | 0.9194 | 2.291 | 260.4155 |
| sentiment | indicbert | lora | shift_d1_test | 1.0 | 3.0 | 278.7373 |
| sentiment | indicbert | lora | shift_d2_test | 1.0 | 3.0 | 898.7095 |
| sentiment | indicbert | lora | shift_d3_noise05 | 0.9018 | 2.4635 | 437.1908 |
| sentiment | indicbert | lora | shift_d3_noise10 | 0.9072 | 2.4641 | 265.7282 |
| sentiment | indicbert | lora | shift_d3_noise20 | 0.9118 | 2.5338 | 75.2199 |
| sentiment | xlmr_base | full_ft | shift_d1_test | 0.994 | 2.8945 | 156.1281 |
| sentiment | xlmr_base | full_ft | shift_d2_test | 1.0 | 3.0 | 169.1401 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | 0.9468 | 2.0325 | 183.5071 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | 0.9157 | 2.0126 | 48.8234 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | 0.9782 | 2.7413 | 42.2931 |
| sentiment | xlmr_base | lora | shift_d1_test | 1.0 | 3.0 | 132.2681 |
| sentiment | xlmr_base | lora | shift_d2_test | 1.0 | 3.0 | 859.4521 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | 0.9649 | 2.4328 | 17.1273 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | 0.8776 | 2.1975 | 5.2404 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | 0.9991 | 2.9951 | 77.4972 |


## 4. Representation Distance Misclassification Detection (Part I.6)

| task | model | ft_method | condition | predictor | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | confidence_only | 0.7866 | 0.2996 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | entropy_only | 0.8082 | 0.3017 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | maha_distance_only | 0.7598 | 0.2486 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | maha_plus_confidence | 0.8011 | 0.294 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | confidence_only | 0.7281 | 0.3583 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | entropy_only | 0.7381 | 0.3582 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | maha_distance_only | 0.707 | 0.3141 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | maha_plus_confidence | 0.7524 | 0.3639 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | confidence_only | 0.8283 | 0.3379 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | entropy_only | 0.8363 | 0.3361 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | maha_distance_only | 0.7592 | 0.2496 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | maha_plus_confidence | 0.8105 | 0.3064 |
| offensive | banglabert | full_ft | shift_d2_test | confidence_only | 0.504 | 0.4715 |
| offensive | banglabert | full_ft | shift_d2_test | entropy_only | 0.5086 | 0.4714 |
| offensive | banglabert | full_ft | shift_d2_test | maha_distance_only | 0.493 | 0.478 |
| offensive | banglabert | full_ft | shift_d2_test | maha_plus_confidence | 0.5072 | 0.4866 |
| offensive | banglabert | full_ft | shift_d3_noise05 | confidence_only | 0.812 | 0.3271 |
| offensive | banglabert | full_ft | shift_d3_noise05 | entropy_only | 0.8209 | 0.3286 |
| offensive | banglabert | full_ft | shift_d3_noise05 | maha_distance_only | 0.7586 | 0.2617 |
| offensive | banglabert | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.8085 | 0.3192 |
| offensive | banglabert | full_ft | shift_d3_noise10 | confidence_only | 0.781 | 0.326 |
| offensive | banglabert | full_ft | shift_d3_noise10 | entropy_only | 0.7894 | 0.3255 |
| offensive | banglabert | full_ft | shift_d3_noise10 | maha_distance_only | 0.7332 | 0.2713 |
| offensive | banglabert | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.7833 | 0.3235 |
| offensive | banglabert | full_ft | shift_d3_noise20 | confidence_only | 0.7473 | 0.3657 |
| offensive | banglabert | full_ft | shift_d3_noise20 | entropy_only | 0.7528 | 0.3667 |
| offensive | banglabert | full_ft | shift_d3_noise20 | maha_distance_only | 0.7101 | 0.3181 |
| offensive | banglabert | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7597 | 0.3707 |
| offensive | banglabert | full_ft | test | confidence_only | 0.8312 | 0.3287 |
| offensive | banglabert | full_ft | test | entropy_only | 0.8436 | 0.3303 |
| offensive | banglabert | full_ft | test | maha_distance_only | 0.776 | 0.2528 |
| offensive | banglabert | full_ft | test | maha_plus_confidence | 0.8251 | 0.3118 |
| offensive | banglabert | lora | shift_d1_barishal_test | confidence_only | 0.8328 | 0.3614 |
| offensive | banglabert | lora | shift_d1_barishal_test | entropy_only | 0.835 | 0.3599 |
| offensive | banglabert | lora | shift_d1_barishal_test | maha_distance_only | 0.6341 | 0.1522 |
| offensive | banglabert | lora | shift_d1_barishal_test | maha_plus_confidence | 0.8125 | 0.3497 |
| offensive | banglabert | lora | shift_d1_chittagong_test | confidence_only | 0.7279 | 0.4011 |
| offensive | banglabert | lora | shift_d1_chittagong_test | entropy_only | 0.7295 | 0.3985 |
| offensive | banglabert | lora | shift_d1_chittagong_test | maha_distance_only | 0.6023 | 0.2681 |
| offensive | banglabert | lora | shift_d1_chittagong_test | maha_plus_confidence | 0.7205 | 0.396 |
| offensive | banglabert | lora | shift_d1_noakhali_test | confidence_only | 0.7873 | 0.3365 |
| offensive | banglabert | lora | shift_d1_noakhali_test | entropy_only | 0.79 | 0.3317 |
| offensive | banglabert | lora | shift_d1_noakhali_test | maha_distance_only | 0.6224 | 0.1627 |
| offensive | banglabert | lora | shift_d1_noakhali_test | maha_plus_confidence | 0.7755 | 0.3266 |
| offensive | banglabert | lora | shift_d2_test | confidence_only | 0.5131 | 0.4846 |
| offensive | banglabert | lora | shift_d2_test | entropy_only | 0.5446 | 0.4997 |
| offensive | banglabert | lora | shift_d2_test | maha_distance_only | 0.5539 | 0.5598 |
| offensive | banglabert | lora | shift_d2_test | maha_plus_confidence | 0.5763 | 0.5793 |
| offensive | banglabert | lora | shift_d3_noise05 | confidence_only | 0.8172 | 0.3452 |
| offensive | banglabert | lora | shift_d3_noise05 | entropy_only | 0.8187 | 0.344 |
| offensive | banglabert | lora | shift_d3_noise05 | maha_distance_only | 0.6241 | 0.1511 |
| offensive | banglabert | lora | shift_d3_noise05 | maha_plus_confidence | 0.8051 | 0.3423 |
| offensive | banglabert | lora | shift_d3_noise10 | confidence_only | 0.8018 | 0.3701 |
| offensive | banglabert | lora | shift_d3_noise10 | entropy_only | 0.8023 | 0.3693 |
| offensive | banglabert | lora | shift_d3_noise10 | maha_distance_only | 0.6325 | 0.183 |
| offensive | banglabert | lora | shift_d3_noise10 | maha_plus_confidence | 0.7948 | 0.3624 |
| offensive | banglabert | lora | shift_d3_noise20 | confidence_only | 0.7576 | 0.3924 |
| offensive | banglabert | lora | shift_d3_noise20 | entropy_only | 0.7576 | 0.3881 |
| offensive | banglabert | lora | shift_d3_noise20 | maha_distance_only | 0.6007 | 0.2313 |
| offensive | banglabert | lora | shift_d3_noise20 | maha_plus_confidence | 0.7547 | 0.3863 |
| offensive | banglabert | lora | test | confidence_only | 0.8428 | 0.3455 |
| offensive | banglabert | lora | test | entropy_only | 0.8441 | 0.3451 |
| offensive | banglabert | lora | test | maha_distance_only | 0.6202 | 0.1298 |
| offensive | banglabert | lora | test | maha_plus_confidence | 0.8265 | 0.3396 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | confidence_only | 0.7836 | 0.3439 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | entropy_only | 0.7877 | 0.3419 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | maha_distance_only | 0.6481 | 0.2398 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | maha_plus_confidence | 0.7689 | 0.3335 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | confidence_only | 0.7316 | 0.4112 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | entropy_only | 0.7367 | 0.4113 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | maha_distance_only | 0.6166 | 0.3222 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | maha_plus_confidence | 0.7187 | 0.4029 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | confidence_only | 0.7959 | 0.3903 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | entropy_only | 0.8007 | 0.3879 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | maha_distance_only | 0.6411 | 0.2443 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | maha_plus_confidence | 0.7834 | 0.3758 |
| offensive | indicbert | full_ft | shift_d2_test | confidence_only | 0.5218 | 0.4916 |
| offensive | indicbert | full_ft | shift_d2_test | entropy_only | 0.5255 | 0.4947 |
| offensive | indicbert | full_ft | shift_d2_test | maha_distance_only | 0.5131 | 0.4787 |
| offensive | indicbert | full_ft | shift_d2_test | maha_plus_confidence | 0.5284 | 0.4973 |
| offensive | indicbert | full_ft | shift_d3_noise05 | confidence_only | 0.7962 | 0.3579 |
| offensive | indicbert | full_ft | shift_d3_noise05 | entropy_only | 0.8013 | 0.3572 |
| offensive | indicbert | full_ft | shift_d3_noise05 | maha_distance_only | 0.6517 | 0.2396 |
| offensive | indicbert | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.7829 | 0.3486 |
| offensive | indicbert | full_ft | shift_d3_noise10 | confidence_only | 0.77 | 0.3509 |
| offensive | indicbert | full_ft | shift_d3_noise10 | entropy_only | 0.7743 | 0.3511 |
| offensive | indicbert | full_ft | shift_d3_noise10 | maha_distance_only | 0.6349 | 0.2533 |
| offensive | indicbert | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.7527 | 0.3427 |
| offensive | indicbert | full_ft | shift_d3_noise20 | confidence_only | 0.74 | 0.3735 |
| offensive | indicbert | full_ft | shift_d3_noise20 | entropy_only | 0.742 | 0.374 |
| offensive | indicbert | full_ft | shift_d3_noise20 | maha_distance_only | 0.6274 | 0.2907 |
| offensive | indicbert | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7292 | 0.3708 |
| offensive | indicbert | full_ft | test | confidence_only | 0.8152 | 0.353 |
| offensive | indicbert | full_ft | test | entropy_only | 0.8173 | 0.354 |
| offensive | indicbert | full_ft | test | maha_distance_only | 0.6502 | 0.219 |
| offensive | indicbert | full_ft | test | maha_plus_confidence | 0.7987 | 0.3426 |
| offensive | indicbert | lora | shift_d1_barishal_test | confidence_only | 0.5624 | 0.3738 |
| offensive | indicbert | lora | shift_d1_barishal_test | entropy_only | 0.5579 | 0.3646 |
| offensive | indicbert | lora | shift_d1_barishal_test | maha_distance_only | 0.5108 | 0.3151 |
| offensive | indicbert | lora | shift_d1_barishal_test | maha_plus_confidence | 0.5807 | 0.3777 |
| offensive | indicbert | lora | shift_d1_chittagong_test | confidence_only | 0.5337 | 0.4159 |
| offensive | indicbert | lora | shift_d1_chittagong_test | entropy_only | 0.5289 | 0.4067 |
| offensive | indicbert | lora | shift_d1_chittagong_test | maha_distance_only | 0.4978 | 0.3727 |
| offensive | indicbert | lora | shift_d1_chittagong_test | maha_plus_confidence | 0.542 | 0.4179 |
| offensive | indicbert | lora | shift_d1_noakhali_test | confidence_only | 0.5677 | 0.4158 |
| offensive | indicbert | lora | shift_d1_noakhali_test | entropy_only | 0.5625 | 0.4065 |
| offensive | indicbert | lora | shift_d1_noakhali_test | maha_distance_only | 0.4995 | 0.3467 |
| offensive | indicbert | lora | shift_d1_noakhali_test | maha_plus_confidence | 0.5756 | 0.4147 |
| offensive | indicbert | lora | shift_d2_test | confidence_only | 0.5232 | 0.4768 |
| offensive | indicbert | lora | shift_d2_test | entropy_only | 0.5172 | 0.4721 |
| offensive | indicbert | lora | shift_d2_test | maha_distance_only | 0.5484 | 0.4932 |
| offensive | indicbert | lora | shift_d2_test | maha_plus_confidence | 0.5494 | 0.4931 |
| offensive | indicbert | lora | shift_d3_noise05 | confidence_only | 0.5596 | 0.3728 |
| offensive | indicbert | lora | shift_d3_noise05 | entropy_only | 0.5549 | 0.3636 |
| offensive | indicbert | lora | shift_d3_noise05 | maha_distance_only | 0.503 | 0.3099 |
| offensive | indicbert | lora | shift_d3_noise05 | maha_plus_confidence | 0.5669 | 0.3732 |
| offensive | indicbert | lora | shift_d3_noise10 | confidence_only | 0.5637 | 0.3758 |
| offensive | indicbert | lora | shift_d3_noise10 | entropy_only | 0.558 | 0.3642 |
| offensive | indicbert | lora | shift_d3_noise10 | maha_distance_only | 0.5041 | 0.3122 |
| offensive | indicbert | lora | shift_d3_noise10 | maha_plus_confidence | 0.5705 | 0.3769 |
| offensive | indicbert | lora | shift_d3_noise20 | confidence_only | 0.5673 | 0.3973 |
| offensive | indicbert | lora | shift_d3_noise20 | entropy_only | 0.5619 | 0.387 |
| offensive | indicbert | lora | shift_d3_noise20 | maha_distance_only | 0.5054 | 0.3245 |
| offensive | indicbert | lora | shift_d3_noise20 | maha_plus_confidence | 0.5707 | 0.3953 |
| offensive | indicbert | lora | test | confidence_only | 0.5757 | 0.3805 |
| offensive | indicbert | lora | test | entropy_only | 0.5699 | 0.3699 |
| offensive | indicbert | lora | test | maha_distance_only | 0.5051 | 0.3047 |
| offensive | indicbert | lora | test | maha_plus_confidence | 0.5804 | 0.3801 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | confidence_only | 0.8257 | 0.3471 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | entropy_only | 0.8318 | 0.3443 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | maha_distance_only | 0.6796 | 0.1903 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | maha_plus_confidence | 0.798 | 0.3164 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | confidence_only | 0.7668 | 0.3845 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | entropy_only | 0.7707 | 0.3857 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | maha_distance_only | 0.679 | 0.2825 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | maha_plus_confidence | 0.7695 | 0.3794 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | confidence_only | 0.8303 | 0.3575 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | entropy_only | 0.8376 | 0.3571 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | maha_distance_only | 0.6779 | 0.2002 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | maha_plus_confidence | 0.8048 | 0.3295 |
| offensive | xlmr_base | full_ft | shift_d2_test | confidence_only | 0.4863 | 0.4699 |
| offensive | xlmr_base | full_ft | shift_d2_test | entropy_only | 0.4831 | 0.4656 |
| offensive | xlmr_base | full_ft | shift_d2_test | maha_distance_only | 0.5902 | 0.5654 |
| offensive | xlmr_base | full_ft | shift_d2_test | maha_plus_confidence | 0.5931 | 0.565 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | confidence_only | 0.8286 | 0.3287 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | entropy_only | 0.8323 | 0.328 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | maha_distance_only | 0.6814 | 0.1905 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.8177 | 0.313 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | confidence_only | 0.8151 | 0.354 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | entropy_only | 0.8181 | 0.3549 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | maha_distance_only | 0.6853 | 0.213 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.8071 | 0.3436 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | confidence_only | 0.7809 | 0.3604 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | entropy_only | 0.7825 | 0.3599 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | maha_distance_only | 0.6759 | 0.2433 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7827 | 0.3546 |
| offensive | xlmr_base | full_ft | test | confidence_only | 0.8417 | 0.3443 |
| offensive | xlmr_base | full_ft | test | entropy_only | 0.8481 | 0.3438 |
| offensive | xlmr_base | full_ft | test | maha_distance_only | 0.682 | 0.1788 |
| offensive | xlmr_base | full_ft | test | maha_plus_confidence | 0.8291 | 0.3274 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | confidence_only | 0.7874 | 0.4138 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | entropy_only | 0.788 | 0.41 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | maha_distance_only | 0.5067 | 0.1756 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | maha_plus_confidence | 0.7835 | 0.4137 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | confidence_only | 0.7027 | 0.3981 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | entropy_only | 0.7031 | 0.396 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | maha_distance_only | 0.4982 | 0.245 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | maha_plus_confidence | 0.7013 | 0.3958 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | confidence_only | 0.7513 | 0.3741 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | entropy_only | 0.752 | 0.3729 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | maha_distance_only | 0.51 | 0.1959 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | maha_plus_confidence | 0.7503 | 0.3745 |
| offensive | xlmr_base | lora | shift_d2_test | confidence_only | 0.5355 | 0.513 |
| offensive | xlmr_base | lora | shift_d2_test | entropy_only | 0.5366 | 0.5134 |
| offensive | xlmr_base | lora | shift_d2_test | maha_distance_only | 0.6322 | 0.6058 |
| offensive | xlmr_base | lora | shift_d2_test | maha_plus_confidence | 0.6271 | 0.5927 |
| offensive | xlmr_base | lora | shift_d3_noise05 | confidence_only | 0.7873 | 0.3928 |
| offensive | xlmr_base | lora | shift_d3_noise05 | entropy_only | 0.7875 | 0.386 |
| offensive | xlmr_base | lora | shift_d3_noise05 | maha_distance_only | 0.5066 | 0.1706 |
| offensive | xlmr_base | lora | shift_d3_noise05 | maha_plus_confidence | 0.7846 | 0.3923 |
| offensive | xlmr_base | lora | shift_d3_noise10 | confidence_only | 0.7681 | 0.4083 |
| offensive | xlmr_base | lora | shift_d3_noise10 | entropy_only | 0.7684 | 0.4054 |
| offensive | xlmr_base | lora | shift_d3_noise10 | maha_distance_only | 0.5012 | 0.1936 |
| offensive | xlmr_base | lora | shift_d3_noise10 | maha_plus_confidence | 0.7674 | 0.407 |
| offensive | xlmr_base | lora | shift_d3_noise20 | confidence_only | 0.7306 | 0.4019 |
| offensive | xlmr_base | lora | shift_d3_noise20 | entropy_only | 0.7304 | 0.3981 |
| offensive | xlmr_base | lora | shift_d3_noise20 | maha_distance_only | 0.509 | 0.2396 |
| offensive | xlmr_base | lora | shift_d3_noise20 | maha_plus_confidence | 0.7299 | 0.4007 |
| offensive | xlmr_base | lora | test | confidence_only | 0.7991 | 0.3823 |
| offensive | xlmr_base | lora | test | entropy_only | 0.7996 | 0.3783 |
| offensive | xlmr_base | lora | test | maha_distance_only | 0.5048 | 0.1487 |
| offensive | xlmr_base | lora | test | maha_plus_confidence | 0.797 | 0.3817 |
| sentiment | banglabert | full_ft | shift_d1_test | confidence_only | 0.7312 | 0.5367 |
| sentiment | banglabert | full_ft | shift_d1_test | entropy_only | 0.7367 | 0.5403 |
| sentiment | banglabert | full_ft | shift_d1_test | maha_distance_only | 0.6162 | 0.4373 |
| sentiment | banglabert | full_ft | shift_d1_test | maha_plus_confidence | 0.7161 | 0.5025 |
| sentiment | banglabert | full_ft | shift_d2_test | confidence_only | 0.7147 | 0.7027 |
| sentiment | banglabert | full_ft | shift_d2_test | entropy_only | 0.7165 | 0.6992 |
| sentiment | banglabert | full_ft | shift_d2_test | maha_distance_only | 0.5906 | 0.5988 |
| sentiment | banglabert | full_ft | shift_d2_test | maha_plus_confidence | 0.706 | 0.6931 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | confidence_only | 0.7552 | 0.5146 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | entropy_only | 0.7558 | 0.5165 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | maha_distance_only | 0.6782 | 0.4236 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.7552 | 0.5083 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | confidence_only | 0.7365 | 0.5147 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | entropy_only | 0.7392 | 0.5187 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | maha_distance_only | 0.6371 | 0.4195 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.7296 | 0.5038 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | confidence_only | 0.7412 | 0.5844 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | entropy_only | 0.7451 | 0.593 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | maha_distance_only | 0.6243 | 0.4664 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7325 | 0.5771 |
| sentiment | banglabert | full_ft | test | confidence_only | 0.7638 | 0.5189 |
| sentiment | banglabert | full_ft | test | entropy_only | 0.7644 | 0.5209 |
| sentiment | banglabert | full_ft | test | maha_distance_only | 0.7032 | 0.4411 |
| sentiment | banglabert | full_ft | test | maha_plus_confidence | 0.7643 | 0.5142 |
| sentiment | banglabert | lora | shift_d1_test | confidence_only | 0.769 | 0.6008 |
| sentiment | banglabert | lora | shift_d1_test | entropy_only | 0.7545 | 0.5838 |
| sentiment | banglabert | lora | shift_d1_test | maha_distance_only | 0.5687 | 0.3953 |
| sentiment | banglabert | lora | shift_d1_test | maha_plus_confidence | 0.7419 | 0.5668 |
| sentiment | banglabert | lora | shift_d2_test | confidence_only | 0.689 | 0.7114 |
| sentiment | banglabert | lora | shift_d2_test | entropy_only | 0.6846 | 0.7113 |
| sentiment | banglabert | lora | shift_d2_test | maha_distance_only | 0.6232 | 0.6997 |
| sentiment | banglabert | lora | shift_d2_test | maha_plus_confidence | 0.7701 | 0.8215 |
| sentiment | banglabert | lora | shift_d3_noise05 | confidence_only | 0.7363 | 0.5357 |
| sentiment | banglabert | lora | shift_d3_noise05 | entropy_only | 0.7315 | 0.5258 |
| sentiment | banglabert | lora | shift_d3_noise05 | maha_distance_only | 0.5717 | 0.3806 |
| sentiment | banglabert | lora | shift_d3_noise05 | maha_plus_confidence | 0.7331 | 0.5369 |
| sentiment | banglabert | lora | shift_d3_noise10 | confidence_only | 0.7166 | 0.5345 |
| sentiment | banglabert | lora | shift_d3_noise10 | entropy_only | 0.7092 | 0.5231 |
| sentiment | banglabert | lora | shift_d3_noise10 | maha_distance_only | 0.5192 | 0.3483 |
| sentiment | banglabert | lora | shift_d3_noise10 | maha_plus_confidence | 0.7157 | 0.5337 |
| sentiment | banglabert | lora | shift_d3_noise20 | confidence_only | 0.7166 | 0.5508 |
| sentiment | banglabert | lora | shift_d3_noise20 | entropy_only | 0.7057 | 0.5388 |
| sentiment | banglabert | lora | shift_d3_noise20 | maha_distance_only | 0.5053 | 0.3685 |
| sentiment | banglabert | lora | shift_d3_noise20 | maha_plus_confidence | 0.7199 | 0.5553 |
| sentiment | banglabert | lora | test | confidence_only | 0.7595 | 0.5621 |
| sentiment | banglabert | lora | test | entropy_only | 0.7534 | 0.549 |
| sentiment | banglabert | lora | test | maha_distance_only | 0.6177 | 0.4299 |
| sentiment | banglabert | lora | test | maha_plus_confidence | 0.7556 | 0.5615 |
| sentiment | indicbert | full_ft | shift_d1_test | confidence_only | 0.7061 | 0.5657 |
| sentiment | indicbert | full_ft | shift_d1_test | entropy_only | 0.7022 | 0.559 |
| sentiment | indicbert | full_ft | shift_d1_test | maha_distance_only | 0.5742 | 0.4602 |
| sentiment | indicbert | full_ft | shift_d1_test | maha_plus_confidence | 0.6983 | 0.5473 |
| sentiment | indicbert | full_ft | shift_d2_test | confidence_only | 0.5629 | 0.5895 |
| sentiment | indicbert | full_ft | shift_d2_test | entropy_only | 0.56 | 0.5829 |
| sentiment | indicbert | full_ft | shift_d2_test | maha_distance_only | 0.5256 | 0.5724 |
| sentiment | indicbert | full_ft | shift_d2_test | maha_plus_confidence | 0.5562 | 0.5935 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | confidence_only | 0.7073 | 0.5744 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | entropy_only | 0.7065 | 0.5721 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | maha_distance_only | 0.5875 | 0.4666 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.7081 | 0.573 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | confidence_only | 0.7033 | 0.583 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | entropy_only | 0.7021 | 0.582 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | maha_distance_only | 0.5753 | 0.4619 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.7043 | 0.5828 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | confidence_only | 0.6741 | 0.559 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | entropy_only | 0.6731 | 0.5527 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | maha_distance_only | 0.5644 | 0.4584 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.6752 | 0.5521 |
| sentiment | indicbert | full_ft | test | confidence_only | 0.7115 | 0.5703 |
| sentiment | indicbert | full_ft | test | entropy_only | 0.7124 | 0.5672 |
| sentiment | indicbert | full_ft | test | maha_distance_only | 0.5919 | 0.4534 |
| sentiment | indicbert | full_ft | test | maha_plus_confidence | 0.7137 | 0.5655 |
| sentiment | indicbert | lora | shift_d1_test | confidence_only | 0.4856 | 0.5934 |
| sentiment | indicbert | lora | shift_d1_test | entropy_only | 0.483 | 0.5905 |
| sentiment | indicbert | lora | shift_d1_test | maha_distance_only | 0.5273 | 0.616 |
| sentiment | indicbert | lora | shift_d1_test | maha_plus_confidence | 0.5249 | 0.6142 |
| sentiment | indicbert | lora | shift_d2_test | confidence_only | 0.5029 | 0.5939 |
| sentiment | indicbert | lora | shift_d2_test | entropy_only | 0.5 | 0.5942 |
| sentiment | indicbert | lora | shift_d2_test | maha_distance_only | 0.5184 | 0.594 |
| sentiment | indicbert | lora | shift_d2_test | maha_plus_confidence | 0.5183 | 0.5939 |
| sentiment | indicbert | lora | shift_d3_noise05 | confidence_only | 0.5348 | 0.6219 |
| sentiment | indicbert | lora | shift_d3_noise05 | entropy_only | 0.5353 | 0.6248 |
| sentiment | indicbert | lora | shift_d3_noise05 | maha_distance_only | 0.5079 | 0.5894 |
| sentiment | indicbert | lora | shift_d3_noise05 | maha_plus_confidence | 0.5096 | 0.5906 |
| sentiment | indicbert | lora | shift_d3_noise10 | confidence_only | 0.5414 | 0.6271 |
| sentiment | indicbert | lora | shift_d3_noise10 | entropy_only | 0.5429 | 0.6327 |
| sentiment | indicbert | lora | shift_d3_noise10 | maha_distance_only | 0.5335 | 0.6098 |
| sentiment | indicbert | lora | shift_d3_noise10 | maha_plus_confidence | 0.5337 | 0.6099 |
| sentiment | indicbert | lora | shift_d3_noise20 | confidence_only | 0.5238 | 0.6159 |
| sentiment | indicbert | lora | shift_d3_noise20 | entropy_only | 0.5291 | 0.6208 |
| sentiment | indicbert | lora | shift_d3_noise20 | maha_distance_only | 0.499 | 0.5838 |
| sentiment | indicbert | lora | shift_d3_noise20 | maha_plus_confidence | 0.4992 | 0.5839 |
| sentiment | indicbert | lora | test | confidence_only | 0.5379 | 0.6241 |
| sentiment | indicbert | lora | test | entropy_only | 0.5401 | 0.6293 |
| sentiment | indicbert | lora | test | maha_distance_only | 0.5263 | 0.6045 |
| sentiment | indicbert | lora | test | maha_plus_confidence | 0.5266 | 0.6047 |
| sentiment | xlmr_base | full_ft | shift_d1_test | confidence_only | 0.7369 | 0.509 |
| sentiment | xlmr_base | full_ft | shift_d1_test | entropy_only | 0.7414 | 0.5112 |
| sentiment | xlmr_base | full_ft | shift_d1_test | maha_distance_only | 0.6364 | 0.4209 |
| sentiment | xlmr_base | full_ft | shift_d1_test | maha_plus_confidence | 0.724 | 0.4846 |
| sentiment | xlmr_base | full_ft | shift_d2_test | confidence_only | 0.6429 | 0.5658 |
| sentiment | xlmr_base | full_ft | shift_d2_test | entropy_only | 0.6479 | 0.562 |
| sentiment | xlmr_base | full_ft | shift_d2_test | maha_distance_only | 0.6282 | 0.5653 |
| sentiment | xlmr_base | full_ft | shift_d2_test | maha_plus_confidence | 0.6439 | 0.5825 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | confidence_only | 0.7483 | 0.5237 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | entropy_only | 0.7484 | 0.5216 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | maha_distance_only | 0.6841 | 0.4482 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.7504 | 0.5206 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | confidence_only | 0.7296 | 0.5157 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | entropy_only | 0.7292 | 0.5186 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | maha_distance_only | 0.6753 | 0.4506 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.731 | 0.5104 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | confidence_only | 0.7228 | 0.5575 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | entropy_only | 0.7243 | 0.5561 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | maha_distance_only | 0.6665 | 0.4867 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7235 | 0.5508 |
| sentiment | xlmr_base | full_ft | test | confidence_only | 0.7564 | 0.531 |
| sentiment | xlmr_base | full_ft | test | entropy_only | 0.7562 | 0.5297 |
| sentiment | xlmr_base | full_ft | test | maha_distance_only | 0.6898 | 0.4488 |
| sentiment | xlmr_base | full_ft | test | maha_plus_confidence | 0.7574 | 0.5299 |
| sentiment | xlmr_base | lora | shift_d1_test | confidence_only | 0.6945 | 0.5306 |
| sentiment | xlmr_base | lora | shift_d1_test | entropy_only | 0.6898 | 0.5202 |
| sentiment | xlmr_base | lora | shift_d1_test | maha_distance_only | 0.5785 | 0.48 |
| sentiment | xlmr_base | lora | shift_d1_test | maha_plus_confidence | 0.6915 | 0.5385 |
| sentiment | xlmr_base | lora | shift_d2_test | confidence_only | 0.6798 | 0.5568 |
| sentiment | xlmr_base | lora | shift_d2_test | entropy_only | 0.6742 | 0.544 |
| sentiment | xlmr_base | lora | shift_d2_test | maha_distance_only | 0.5334 | 0.4556 |
| sentiment | xlmr_base | lora | shift_d2_test | maha_plus_confidence | 0.6829 | 0.5684 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | confidence_only | 0.7054 | 0.5338 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | entropy_only | 0.7048 | 0.5319 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | maha_distance_only | 0.6029 | 0.4541 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | maha_plus_confidence | 0.7063 | 0.5299 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | confidence_only | 0.7016 | 0.5506 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | entropy_only | 0.7016 | 0.5493 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | maha_distance_only | 0.5992 | 0.4703 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | maha_plus_confidence | 0.7026 | 0.5462 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | confidence_only | 0.6891 | 0.557 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | entropy_only | 0.6889 | 0.5511 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | maha_distance_only | 0.5782 | 0.4746 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | maha_plus_confidence | 0.6906 | 0.5619 |
| sentiment | xlmr_base | lora | test | confidence_only | 0.7201 | 0.5522 |
| sentiment | xlmr_base | lora | test | entropy_only | 0.719 | 0.5535 |
| sentiment | xlmr_base | lora | test | maha_distance_only | 0.6122 | 0.4575 |
| sentiment | xlmr_base | lora | test | maha_plus_confidence | 0.7206 | 0.5467 |


## 5. Statistical Significance Summary (Part J — Holm-Corrected)

| task | model | ft_method | comparison | metric | endpoint_family | mean_diff | median_raw_p | median_holm_p | seeds_sig_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | D0_vs_shift_d1_barishal_test | brier | secondary | -0.003 | 0.82 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0016 | 0.854 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0133 | 0.62 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1528 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0666 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.3286 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0151 | 0.294 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0048 | 0.376 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0193 | 0.412 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.7787 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.3996 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d2_test | nll | secondary | -2.2733 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0281 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0106 | 0.0 | 0.0 | 4 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0594 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0575 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0234 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.1263 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.1443 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0646 | 0.0 | 0.0 | 5 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.3102 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0127 | 0.23 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0033 | 0.598 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0187 | 0.286 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1865 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0664 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.2762 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0449 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.008 | 0.3 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0727 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d2_test | brier | secondary | -0.5288 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.2728 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d2_test | nll | secondary | -0.6964 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0199 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0003 | 0.88 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0311 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.05 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0069 | 0.152 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0738 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.1168 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0207 | 0.0 | 0.0 | 5 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.1656 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0378 | 0.002 | 0.964 | 1 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0163 | 0.04 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0668 | 0.01 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1897 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0921 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.3533 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.054 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0247 | 0.004 | 1.0 | 1 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0891 | 0.0 | 0.0 | 3 |
| offensive | indicbert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.6001 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.2981 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d2_test | nll | secondary | -1.0085 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0311 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0144 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0572 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0675 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0317 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.1306 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.1289 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0593 | 0.0 | 0.0 | 5 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.2445 | 0.0 | 0.0 | 5 |
| offensive | indicbert | lora | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0062 | 0.342 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0082 | 0.452 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0065 | 0.358 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.0414 | 0.0 | 0.0 | 5 |
| offensive | indicbert | lora | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | 0.0198 | 0.002 | 0.964 | 1 |
| offensive | indicbert | lora | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.0437 | 0.0 | 0.0 | 5 |
| offensive | indicbert | lora | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0223 | 0.0 | 0.0 | 4 |
| offensive | indicbert | lora | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | 0.0156 | 0.206 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0233 | 0.0 | 0.0 | 4 |
| offensive | indicbert | lora | D0_vs_shift_d2_test | brier | secondary | -0.085 | 0.0 | 0.0 | 5 |
| offensive | indicbert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.0066 | 0.162 | 1.0 | 2 |
| offensive | indicbert | lora | D0_vs_shift_d2_test | nll | secondary | -0.0894 | 0.0 | 0.0 | 5 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0049 | 0.0 | 0.0 | 3 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0029 | 0.46 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0052 | 0.0 | 0.0 | 4 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0066 | 0.002 | 0.964 | 2 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | 0.0087 | 0.1 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.007 | 0.002 | 0.964 | 2 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0121 | 0.0 | 0.0 | 5 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | 0.0141 | 0.046 | 1.0 | 1 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.0128 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0118 | 0.412 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0053 | 0.272 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0246 | 0.244 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1312 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0572 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.243 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0344 | 0.002 | 0.964 | 1 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.014 | 0.04 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0546 | 0.012 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.7472 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.3856 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d2_test | nll | secondary | -1.6332 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0172 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0048 | 0.052 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0288 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.041 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0142 | 0.0 | 0.0 | 4 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.0703 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.0862 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0297 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.1482 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0247 | 0.034 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0053 | 0.364 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0344 | 0.052 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1288 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0362 | 0.0 | 0.0 | 4 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.1813 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0558 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0103 | 0.224 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0761 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d2_test | brier | secondary | -0.4369 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.2224 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d2_test | nll | secondary | -0.5838 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0217 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0007 | 0.864 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0265 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0536 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0097 | 0.032 | 1.0 | 1 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0752 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0997 | 0.0 | 0.0 | 5 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0166 | 0.0 | 0.0 | 3 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.1357 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d1_test | brier | secondary | -0.1145 | 0.0 | 0.0 | 3 |
| sentiment | banglabert | full_ft | D0_vs_shift_d1_test | ece_15 | secondary | -0.0309 | 0.332 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d1_test | nll | primary | -0.1469 | 0.0 | 0.0 | 4 |
| sentiment | banglabert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.3491 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.1655 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d2_test | nll | primary | -0.6831 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0148 | 0.032 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0013 | 0.708 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise05 | nll | primary | -0.0231 | 0.038 | 0.038 | 3 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.056 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0121 | 0.414 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise10 | nll | primary | -0.0878 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.1147 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0371 | 0.002 | 0.964 | 2 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise20 | nll | primary | -0.172 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | lora | D0_vs_shift_d1_test | brier | secondary | -0.0569 | 0.004 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d1_test | ece_15 | secondary | -0.0221 | 0.486 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d1_test | nll | secondary | -0.0951 | 0.004 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d2_test | brier | secondary | -0.3377 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.2019 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | lora | D0_vs_shift_d2_test | nll | secondary | -0.6085 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0066 | 0.164 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0231 | 0.024 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0148 | 0.078 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.022 | 0.002 | 0.964 | 1 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | 0.0297 | 0.02 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0343 | 0.004 | 1.0 | 1 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0451 | 0.0 | 0.0 | 5 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | 0.0266 | 0.124 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.0768 | 0.0 | 0.0 | 5 |
| sentiment | indicbert | full_ft | D0_vs_shift_d1_test | brier | secondary | -0.0181 | 0.344 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d1_test | ece_15 | secondary | 0.0128 | 0.52 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d1_test | nll | secondary | -0.0321 | 0.22 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.2481 | 0.0 | 0.0 | 5 |
| sentiment | indicbert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.1789 | 0.0 | 0.0 | 4 |
| sentiment | indicbert | full_ft | D0_vs_shift_d2_test | nll | secondary | -0.4018 | 0.0 | 0.0 | 5 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.013 | 0.014 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0069 | 0.808 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0199 | 0.016 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0241 | 0.002 | 0.964 | 2 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0108 | 0.082 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.0385 | 0.002 | 0.964 | 2 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.0321 | 0.0 | 0.0 | 4 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0085 | 0.678 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.0523 | 0.0 | 0.0 | 3 |
| sentiment | indicbert | lora | D0_vs_shift_d1_test | brier | secondary | -0.0002 | 0.678 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d1_test | ece_15 | secondary | -0.0 | 0.778 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d1_test | nll | secondary | -0.0003 | 0.766 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d2_test | brier | secondary | 0.0007 | 0.872 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.0009 | 0.956 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d2_test | nll | secondary | 0.0011 | 0.858 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0 | 0.624 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0 | 0.052 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0 | 0.66 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise10 | brier | secondary | 0.0 | 0.242 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | 0.0 | 0.04 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise10 | nll | secondary | 0.0001 | 0.162 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0 | 0.326 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | 0.0001 | 0.038 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.0 | 0.272 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d1_test | brier | secondary | -0.021 | 0.588 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d1_test | ece_15 | secondary | 0.0047 | 0.642 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d1_test | nll | secondary | -0.0338 | 0.546 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.3039 | 0.0 | 0.0 | 5 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.1967 | 0.0 | 0.0 | 5 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d2_test | nll | secondary | -0.6877 | 0.0 | 0.0 | 5 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0084 | 0.196 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0013 | 0.454 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0119 | 0.386 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0338 | 0.008 | 1.0 | 2 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0107 | 0.6 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.0593 | 0.004 | 1.0 | 2 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.072 | 0.0 | 0.0 | 5 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0312 | 0.006 | 1.0 | 1 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.1259 | 0.0 | 0.0 | 5 |
| sentiment | xlmr_base | lora | D0_vs_shift_d1_test | brier | secondary | -0.0417 | 0.06 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d1_test | ece_15 | secondary | -0.0035 | 0.918 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d1_test | nll | secondary | -0.0641 | 0.072 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d2_test | brier | secondary | -0.0982 | 0.0 | 0.0 | 4 |
| sentiment | xlmr_base | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.1007 | 0.0 | 0.0 | 5 |
| sentiment | xlmr_base | lora | D0_vs_shift_d2_test | nll | secondary | -0.1496 | 0.0 | 0.0 | 4 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0072 | 0.1 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0049 | 0.63 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0122 | 0.096 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0266 | 0.0 | 0.0 | 4 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0187 | 0.178 | 1.0 | 1 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0421 | 0.0 | 0.0 | 5 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0504 | 0.0 | 0.0 | 5 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0355 | 0.0 | 0.0 | 3 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.0843 | 0.0 | 0.0 | 5 |

