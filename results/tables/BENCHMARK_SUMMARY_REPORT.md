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
| banglabert | lora | shift_d1_test | source_temp_scaled | 0.6876 | 0.6574 | 0.7076 | 0.0182 | 0.0746 | 0.4106 | 0.1417 | 0.9134 | 1.7692 |
| banglabert | lora | shift_d1_test | target_temp_scaled | 0.6876 | 0.6574 | 0.7083 | 0.0168 | 0.0893 | 0.4109 | 0.1417 | 0.9353 | 1.8716 |
| banglabert | lora | shift_d1_test | uncalibrated | 0.6876 | 0.6574 | 0.7963 | 0.0326 | 0.1293 | 0.4368 | 0.1421 | 0.9114 | 1.7682 |
| banglabert | lora | shift_d2_test | source_temp_scaled | 0.4642 | 0.411 | 1.0932 | 0.1022 | 0.1327 | 0.6384 | 0.3523 | 0.8095 | 2.1219 |
| banglabert | lora | shift_d2_test | target_temp_scaled | 0.4642 | 0.411 | 1.026 | 0.0218 | 0.0914 | 0.6106 | 0.3536 | 0.9017 | 2.4719 |
| banglabert | lora | shift_d2_test | uncalibrated | 0.4642 | 0.411 | 1.3878 | 0.3124 | 0.2498 | 0.7229 | 0.3507 | 0.8259 | 2.1674 |
| banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.7047 | 0.6784 | 0.6905 | 0.0058 | 0.0298 | 0.4 | 0.1386 | 0.9046 | 1.7143 |
| banglabert | lora | shift_d3_noise05 | uncalibrated | 0.7047 | 0.6784 | 0.7886 | 0.0394 | 0.1165 | 0.4267 | 0.1386 | 0.9084 | 1.7232 |
| banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.6687 | 0.6464 | 0.7482 | 0.0141 | 0.0394 | 0.4364 | 0.167 | 0.8954 | 1.7972 |
| banglabert | lora | shift_d3_noise10 | uncalibrated | 0.6687 | 0.6464 | 0.8572 | 0.0476 | 0.1308 | 0.4676 | 0.168 | 0.9015 | 1.8097 |
| banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.6235 | 0.6135 | 0.8171 | 0.0195 | 0.051 | 0.4788 | 0.2028 | 0.9021 | 1.9541 |
| banglabert | lora | shift_d3_noise20 | uncalibrated | 0.6235 | 0.6135 | 0.9088 | 0.0548 | 0.128 | 0.5069 | 0.2046 | 0.9121 | 1.9903 |
| banglabert | lora | test | source_temp_scaled | 0.7166 | 0.6891 | 0.6664 | 0.0044 | 0.0412 | 0.3853 | 0.1288 | 0.9015 | 1.6491 |
| banglabert | lora | test | uncalibrated | 0.7166 | 0.6891 | 0.7741 | 0.0391 | 0.1222 | 0.4155 | 0.1283 | 0.9029 | 1.6524 |
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
| indicbert | lora | shift_d1_test | source_temp_scaled | 0.4786 | 0.3274 | 1.0288 | 0.0361 | 0.0637 | 0.6189 | 0.4571 | 0.8836 | 2.3264 |
| indicbert | lora | shift_d1_test | target_temp_scaled | 0.4786 | 0.3274 | 1.0299 | 0.0365 | 0.0696 | 0.6194 | 0.4568 | 0.9035 | 2.392 |
| indicbert | lora | shift_d1_test | uncalibrated | 0.4786 | 0.3274 | 1.0305 | 0.0366 | 0.0642 | 0.6197 | 0.4567 | 0.8836 | 2.3303 |
| indicbert | lora | shift_d2_test | source_temp_scaled | 0.4264 | 0.2224 | 1.1097 | 0.0482 | 0.1027 | 0.6722 | 0.5515 | 0.7266 | 1.8781 |
| indicbert | lora | shift_d2_test | target_temp_scaled | 0.4264 | 0.2224 | 1.0698 | 0.0055 | 0.0215 | 0.6474 | 0.5518 | 0.9187 | 2.6478 |
| indicbert | lora | shift_d2_test | uncalibrated | 0.4264 | 0.2224 | 1.1174 | 0.056 | 0.1108 | 0.6761 | 0.5514 | 0.7281 | 1.8913 |
| indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.466 | 0.3172 | 1.0324 | 0.0215 | 0.0372 | 0.6208 | 0.4343 | 0.9006 | 2.4129 |
| indicbert | lora | shift_d3_noise05 | uncalibrated | 0.466 | 0.3172 | 1.0327 | 0.0215 | 0.0392 | 0.6207 | 0.4341 | 0.9006 | 2.4144 |
| indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.464 | 0.3153 | 1.0349 | 0.0188 | 0.0367 | 0.6231 | 0.442 | 0.8938 | 2.3916 |
| indicbert | lora | shift_d3_noise10 | uncalibrated | 0.464 | 0.3153 | 1.0356 | 0.0186 | 0.038 | 0.6233 | 0.442 | 0.8944 | 2.3949 |
| indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.4581 | 0.3092 | 1.0451 | 0.0143 | 0.0393 | 0.6298 | 0.4632 | 0.8912 | 2.3972 |
| indicbert | lora | shift_d3_noise20 | uncalibrated | 0.4581 | 0.3092 | 1.0467 | 0.0141 | 0.0409 | 0.6304 | 0.4631 | 0.8913 | 2.4006 |
| indicbert | lora | test | source_temp_scaled | 0.4743 | 0.3231 | 1.0287 | 0.0239 | 0.0391 | 0.6183 | 0.4251 | 0.9028 | 2.411 |
| indicbert | lora | test | uncalibrated | 0.4743 | 0.3231 | 1.0287 | 0.0239 | 0.0408 | 0.618 | 0.4251 | 0.9029 | 2.4128 |
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
| xlmr_base | lora | shift_d1_test | source_temp_scaled | 0.6328 | 0.5989 | 0.8226 | 0.0385 | 0.0781 | 0.4766 | 0.1991 | 0.8925 | 1.9035 |
| xlmr_base | lora | shift_d1_test | target_temp_scaled | 0.6328 | 0.5989 | 0.8251 | 0.0418 | 0.0806 | 0.4773 | 0.1993 | 0.8896 | 1.8796 |
| xlmr_base | lora | shift_d1_test | uncalibrated | 0.6328 | 0.5989 | 0.8459 | 0.0457 | 0.0953 | 0.484 | 0.1994 | 0.8955 | 1.9114 |
| xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.5326 | 0.3759 | 1.0564 | 0.0861 | 0.2099 | 0.6257 | 0.2858 | 0.741 | 1.6463 |
| xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.5326 | 0.3759 | 0.9391 | 0.0251 | 0.0988 | 0.5575 | 0.2903 | 0.9147 | 2.2052 |
| xlmr_base | lora | shift_d2_test | uncalibrated | 0.5326 | 0.3759 | 1.2067 | 0.1586 | 0.2625 | 0.6742 | 0.284 | 0.7398 | 1.647 |
| xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.6719 | 0.6376 | 0.7485 | 0.0211 | 0.0333 | 0.4378 | 0.169 | 0.911 | 1.8653 |
| xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.6719 | 0.6376 | 0.7589 | 0.0184 | 0.054 | 0.4425 | 0.169 | 0.9116 | 1.8649 |
| xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.6469 | 0.61 | 0.7738 | 0.012 | 0.035 | 0.4556 | 0.1857 | 0.9097 | 1.9082 |
| xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.6469 | 0.61 | 0.7866 | 0.0118 | 0.0646 | 0.4628 | 0.1854 | 0.909 | 1.909 |
| xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.6276 | 0.5788 | 0.8192 | 0.0226 | 0.0362 | 0.4805 | 0.2095 | 0.8969 | 1.9356 |
| xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.6276 | 0.5788 | 0.835 | 0.0254 | 0.0677 | 0.4888 | 0.2094 | 0.8966 | 1.9401 |
| xlmr_base | lora | test | source_temp_scaled | 0.6701 | 0.6409 | 0.7399 | 0.0241 | 0.0281 | 0.4325 | 0.1628 | 0.9124 | 1.8541 |
| xlmr_base | lora | test | uncalibrated | 0.6701 | 0.6409 | 0.7508 | 0.0205 | 0.0529 | 0.4369 | 0.1629 | 0.9124 | 1.8534 |


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
| banglabert | full_ft | shift_d2_test | source_temp_scaled | 0.5255 | 0.351 | 1.4578 | 0.1651 | 0.4142 | 0.8483 | 0.4555 | 0.5255 | 1.0 |
| banglabert | full_ft | shift_d2_test | target_temp_scaled | 0.5255 | 0.351 | 0.719 | 0.0125 | 0.1106 | 0.5244 | 0.4555 | 0.913 | 1.8058 |
| banglabert | full_ft | shift_d2_test | uncalibrated | 0.5255 | 0.351 | 2.5718 | 0.6953 | 0.4558 | 0.9188 | 0.4555 | 0.5255 | 1.0 |
| banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.9019 | 0.9015 | 0.2643 | 0.006 | 0.0259 | 0.154 | 0.0277 | 0.9019 | 1.0 |
| banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.9019 | 0.9015 | 0.3587 | 0.0637 | 0.0667 | 0.1683 | 0.0277 | 0.9019 | 1.0 |
| banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.8854 | 0.8846 | 0.3059 | 0.0076 | 0.0315 | 0.1801 | 0.041 | 0.8854 | 1.0 |
| banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.8854 | 0.8846 | 0.4255 | 0.0761 | 0.0796 | 0.1977 | 0.041 | 0.8854 | 1.0 |
| banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.834 | 0.8305 | 0.4173 | 0.0183 | 0.0653 | 0.2559 | 0.0727 | 0.834 | 1.0 |
| banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.834 | 0.8305 | 0.6094 | 0.1271 | 0.1211 | 0.2845 | 0.0727 | 0.834 | 1.0 |
| banglabert | full_ft | test | source_temp_scaled | 0.9184 | 0.9183 | 0.2249 | 0.0048 | 0.0225 | 0.1282 | 0.0195 | 0.9184 | 1.0 |
| banglabert | full_ft | test | uncalibrated | 0.9184 | 0.9183 | 0.2996 | 0.0534 | 0.0564 | 0.1404 | 0.0195 | 0.9184 | 1.0 |
| banglabert | lora | shift_d1_barishal_test | source_temp_scaled | 0.9136 | 0.9136 | 0.2253 | 0.0039 | 0.0155 | 0.1307 | 0.0209 | 0.9136 | 1.0 |
| banglabert | lora | shift_d1_barishal_test | target_temp_scaled | 0.9136 | 0.9136 | 0.2248 | 0.0037 | 0.0162 | 0.1301 | 0.0209 | 0.9136 | 1.0 |
| banglabert | lora | shift_d1_barishal_test | uncalibrated | 0.9136 | 0.9136 | 0.2789 | 0.032 | 0.0484 | 0.1395 | 0.0209 | 0.9136 | 1.0 |
| banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.8219 | 0.8213 | 0.4194 | 0.0146 | 0.0525 | 0.2626 | 0.0725 | 0.8219 | 1.0 |
| banglabert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.8219 | 0.8213 | 0.4034 | 0.0101 | 0.0239 | 0.2561 | 0.0725 | 0.8993 | 1.1956 |
| banglabert | lora | shift_d1_chittagong_test | uncalibrated | 0.8219 | 0.8213 | 0.5391 | 0.0743 | 0.1007 | 0.2844 | 0.0725 | 0.8219 | 1.0 |
| banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.9013 | 0.9013 | 0.2471 | 0.0063 | 0.0153 | 0.1479 | 0.023 | 0.9013 | 1.0 |
| banglabert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.9013 | 0.9013 | 0.2477 | 0.0057 | 0.0158 | 0.1476 | 0.023 | 0.9226 | 1.0488 |
| banglabert | lora | shift_d1_noakhali_test | uncalibrated | 0.9013 | 0.9013 | 0.2938 | 0.03 | 0.0504 | 0.1576 | 0.023 | 0.9013 | 1.0 |
| banglabert | lora | shift_d2_test | source_temp_scaled | 0.5222 | 0.3559 | 1.1387 | 0.2156 | 0.337 | 0.7392 | 0.452 | 0.5222 | 1.0 |
| banglabert | lora | shift_d2_test | target_temp_scaled | 0.5222 | 0.3559 | 0.7017 | 0.0103 | 0.0691 | 0.5081 | 0.452 | 0.925 | 1.8222 |
| banglabert | lora | shift_d2_test | uncalibrated | 0.5222 | 0.3559 | 1.7133 | 0.5682 | 0.3994 | 0.8324 | 0.452 | 0.5222 | 1.0 |
| banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.8997 | 0.8995 | 0.2491 | 0.0055 | 0.0136 | 0.1492 | 0.0235 | 0.8997 | 1.0 |
| banglabert | lora | shift_d3_noise05 | uncalibrated | 0.8997 | 0.8995 | 0.2987 | 0.0336 | 0.0513 | 0.1593 | 0.0235 | 0.8997 | 1.0 |
| banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.8847 | 0.8842 | 0.2919 | 0.0055 | 0.0175 | 0.175 | 0.0354 | 0.8847 | 1.0 |
| banglabert | lora | shift_d3_noise10 | uncalibrated | 0.8847 | 0.8842 | 0.3564 | 0.0415 | 0.0596 | 0.1866 | 0.0354 | 0.8847 | 1.0 |
| banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.839 | 0.8368 | 0.3815 | 0.0137 | 0.0379 | 0.2382 | 0.0604 | 0.839 | 1.0 |
| banglabert | lora | shift_d3_noise20 | uncalibrated | 0.839 | 0.8368 | 0.4794 | 0.0696 | 0.0879 | 0.259 | 0.0604 | 0.839 | 1.0 |
| banglabert | lora | test | source_temp_scaled | 0.916 | 0.9159 | 0.2135 | 0.0027 | 0.013 | 0.1258 | 0.0168 | 0.916 | 1.0 |
| banglabert | lora | test | uncalibrated | 0.916 | 0.9159 | 0.2571 | 0.0255 | 0.0441 | 0.1342 | 0.0168 | 0.916 | 1.0 |
| indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.8651 | 0.8651 | 0.3375 | 0.0035 | 0.0342 | 0.2052 | 0.0478 | 0.8817 | 1.0397 |
| indicbert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.8651 | 0.8651 | 0.337 | 0.0034 | 0.0347 | 0.2041 | 0.0478 | 0.9056 | 1.1 |
| indicbert | full_ft | shift_d1_barishal_test | uncalibrated | 0.8651 | 0.8651 | 0.424 | 0.0193 | 0.0903 | 0.2266 | 0.0478 | 0.8817 | 1.0397 |
| indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.7746 | 0.7728 | 0.5112 | 0.0115 | 0.0959 | 0.3318 | 0.1032 | 0.8007 | 1.0523 |
| indicbert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.7746 | 0.7728 | 0.481 | 0.0076 | 0.0492 | 0.3136 | 0.1032 | 0.9086 | 1.3335 |
| indicbert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.7746 | 0.7728 | 0.711 | 0.0638 | 0.1669 | 0.3789 | 0.1032 | 0.8007 | 1.0523 |
| indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.8529 | 0.8529 | 0.3519 | 0.0042 | 0.0408 | 0.2173 | 0.0489 | 0.8734 | 1.041 |
| indicbert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.8529 | 0.8529 | 0.3518 | 0.0039 | 0.046 | 0.215 | 0.0489 | 0.9153 | 1.1495 |
| indicbert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.8529 | 0.8529 | 0.4467 | 0.0214 | 0.0992 | 0.2429 | 0.0489 | 0.8734 | 1.041 |
| indicbert | full_ft | shift_d2_test | source_temp_scaled | 0.527 | 0.4182 | 0.9413 | 0.1181 | 0.2729 | 0.664 | 0.4452 | 0.5775 | 1.0987 |
| indicbert | full_ft | shift_d2_test | target_temp_scaled | 0.527 | 0.4182 | 0.6943 | 0.0018 | 0.0412 | 0.5011 | 0.4452 | 0.907 | 1.7812 |
| indicbert | full_ft | shift_d2_test | uncalibrated | 0.527 | 0.4182 | 1.3658 | 0.2513 | 0.372 | 0.789 | 0.4452 | 0.5775 | 1.0987 |
| indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.8693 | 0.8687 | 0.3273 | 0.0047 | 0.0374 | 0.1982 | 0.0424 | 0.8856 | 1.0346 |
| indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.8693 | 0.8687 | 0.4139 | 0.0201 | 0.0889 | 0.2197 | 0.0424 | 0.8856 | 1.0346 |
| indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.8496 | 0.8488 | 0.3727 | 0.0076 | 0.0438 | 0.2298 | 0.0562 | 0.8666 | 1.0366 |
| indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.8496 | 0.8488 | 0.4875 | 0.0283 | 0.1065 | 0.2562 | 0.0562 | 0.8666 | 1.0366 |
| indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8117 | 0.8102 | 0.4468 | 0.0124 | 0.0632 | 0.2828 | 0.0839 | 0.8351 | 1.0483 |
| indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.8117 | 0.8102 | 0.6015 | 0.0328 | 0.1339 | 0.3176 | 0.0839 | 0.8351 | 1.0483 |
| indicbert | full_ft | test | source_temp_scaled | 0.8873 | 0.887 | 0.291 | 0.0044 | 0.0314 | 0.172 | 0.0333 | 0.9023 | 1.0307 |
| indicbert | full_ft | test | uncalibrated | 0.8873 | 0.887 | 0.3571 | 0.0153 | 0.0747 | 0.1888 | 0.0333 | 0.9023 | 1.0307 |
| indicbert | lora | shift_d1_barishal_test | source_temp_scaled | 0.7969 | 0.7968 | 0.448 | 0.0076 | 0.0296 | 0.2875 | 0.0989 | 0.8879 | 1.2196 |
| indicbert | lora | shift_d1_barishal_test | target_temp_scaled | 0.7969 | 0.7968 | 0.4467 | 0.0068 | 0.0247 | 0.2866 | 0.0989 | 0.8963 | 1.2433 |
| indicbert | lora | shift_d1_barishal_test | uncalibrated | 0.7969 | 0.7968 | 0.4504 | 0.0063 | 0.036 | 0.2888 | 0.0989 | 0.8879 | 1.2196 |
| indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.7122 | 0.7074 | 0.6003 | 0.0282 | 0.1045 | 0.4011 | 0.1847 | 0.8157 | 1.2233 |
| indicbert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.7122 | 0.7074 | 0.5618 | 0.0149 | 0.0445 | 0.3803 | 0.1847 | 0.8825 | 1.4008 |
| indicbert | lora | shift_d1_chittagong_test | uncalibrated | 0.7122 | 0.7074 | 0.612 | 0.0229 | 0.1155 | 0.4062 | 0.1847 | 0.8157 | 1.2233 |
| indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.7912 | 0.7907 | 0.4544 | 0.0169 | 0.0305 | 0.2932 | 0.1 | 0.8855 | 1.2243 |
| indicbert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.7912 | 0.7907 | 0.4543 | 0.0135 | 0.0322 | 0.2921 | 0.1 | 0.9124 | 1.3096 |
| indicbert | lora | shift_d1_noakhali_test | uncalibrated | 0.7912 | 0.7907 | 0.4568 | 0.0165 | 0.0375 | 0.2944 | 0.1 | 0.8855 | 1.2243 |
| indicbert | lora | shift_d2_test | source_temp_scaled | 0.4895 | 0.4388 | 0.8907 | 0.1163 | 0.2337 | 0.6397 | 0.5235 | 0.7245 | 1.4658 |
| indicbert | lora | shift_d2_test | target_temp_scaled | 0.4895 | 0.4388 | 0.697 | 0.0022 | 0.0399 | 0.5038 | 0.5235 | 0.9108 | 1.8282 |
| indicbert | lora | shift_d2_test | uncalibrated | 0.4895 | 0.4388 | 0.9134 | 0.1286 | 0.2436 | 0.6509 | 0.5235 | 0.7248 | 1.466 |
| indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.8093 | 0.8079 | 0.4319 | 0.0143 | 0.0239 | 0.2742 | 0.0939 | 0.8888 | 1.1939 |
| indicbert | lora | shift_d3_noise05 | uncalibrated | 0.8093 | 0.8079 | 0.4343 | 0.0133 | 0.0314 | 0.2754 | 0.0939 | 0.8888 | 1.1939 |
| indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.7945 | 0.7927 | 0.4582 | 0.0123 | 0.0308 | 0.2936 | 0.1074 | 0.8817 | 1.2065 |
| indicbert | lora | shift_d3_noise10 | uncalibrated | 0.7945 | 0.7927 | 0.4619 | 0.0106 | 0.0415 | 0.2952 | 0.1074 | 0.8817 | 1.2065 |
| indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.7664 | 0.7643 | 0.5082 | 0.01 | 0.0485 | 0.3315 | 0.1343 | 0.8635 | 1.2318 |
| indicbert | lora | shift_d3_noise20 | uncalibrated | 0.7664 | 0.7643 | 0.5143 | 0.0077 | 0.0591 | 0.3342 | 0.1343 | 0.8635 | 1.2318 |
| indicbert | lora | test | source_temp_scaled | 0.8276 | 0.8264 | 0.4048 | 0.0149 | 0.0161 | 0.2532 | 0.0814 | 0.8996 | 1.1812 |
| indicbert | lora | test | uncalibrated | 0.8276 | 0.8264 | 0.4059 | 0.0144 | 0.0222 | 0.2537 | 0.0814 | 0.8996 | 1.1812 |
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
| xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | 0.8919 | 0.8918 | 0.2642 | 0.0039 | 0.0175 | 0.1589 | 0.0273 | 0.8919 | 1.0 |
| xlmr_base | lora | shift_d1_barishal_test | target_temp_scaled | 0.8919 | 0.8918 | 0.2654 | 0.0043 | 0.0182 | 0.1588 | 0.0273 | 0.9122 | 1.0436 |
| xlmr_base | lora | shift_d1_barishal_test | uncalibrated | 0.8919 | 0.8918 | 0.2849 | 0.0047 | 0.0449 | 0.165 | 0.0273 | 0.8919 | 1.0 |
| xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | 0.8174 | 0.8173 | 0.4001 | 0.0042 | 0.0353 | 0.2557 | 0.0678 | 0.8174 | 1.0 |
| xlmr_base | lora | shift_d1_chittagong_test | target_temp_scaled | 0.8174 | 0.8173 | 0.3963 | 0.0034 | 0.0206 | 0.2534 | 0.0678 | 0.8983 | 1.1854 |
| xlmr_base | lora | shift_d1_chittagong_test | uncalibrated | 0.8174 | 0.8173 | 0.444 | 0.0125 | 0.0793 | 0.2692 | 0.0678 | 0.8174 | 1.0 |
| xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | 0.8757 | 0.8757 | 0.2925 | 0.007 | 0.0188 | 0.1806 | 0.0327 | 0.8757 | 1.0 |
| xlmr_base | lora | shift_d1_noakhali_test | target_temp_scaled | 0.8757 | 0.8757 | 0.2946 | 0.0073 | 0.0202 | 0.1806 | 0.0327 | 0.9216 | 1.1124 |
| xlmr_base | lora | shift_d1_noakhali_test | uncalibrated | 0.8757 | 0.8757 | 0.312 | 0.0064 | 0.0467 | 0.1873 | 0.0327 | 0.8757 | 1.0 |
| xlmr_base | lora | shift_d2_test | source_temp_scaled | 0.5232 | 0.3662 | 1.2441 | 0.0728 | 0.3717 | 0.7801 | 0.4639 | 0.5232 | 1.0 |
| xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.5232 | 0.3662 | 0.6996 | 0.0037 | 0.0622 | 0.5062 | 0.4639 | 0.926 | 1.8378 |
| xlmr_base | lora | shift_d2_test | uncalibrated | 0.5232 | 0.3662 | 1.6927 | 0.1516 | 0.4203 | 0.8553 | 0.4639 | 0.5232 | 1.0 |
| xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.8962 | 0.896 | 0.2612 | 0.0036 | 0.0129 | 0.1567 | 0.0267 | 0.8962 | 1.0 |
| xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.8962 | 0.896 | 0.2802 | 0.0078 | 0.0403 | 0.1618 | 0.0267 | 0.8962 | 1.0 |
| xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.8799 | 0.8796 | 0.2973 | 0.0057 | 0.0097 | 0.1804 | 0.0361 | 0.8799 | 1.0 |
| xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.8799 | 0.8796 | 0.3207 | 0.0101 | 0.0448 | 0.1862 | 0.0361 | 0.8799 | 1.0 |
| xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.8491 | 0.8489 | 0.3515 | 0.0046 | 0.0095 | 0.2182 | 0.0534 | 0.8491 | 1.0 |
| xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.8491 | 0.8489 | 0.3745 | 0.0102 | 0.0501 | 0.2242 | 0.0534 | 0.8491 | 1.0 |
| xlmr_base | lora | test | source_temp_scaled | 0.9057 | 0.9056 | 0.2393 | 0.0023 | 0.0127 | 0.1421 | 0.022 | 0.9057 | 1.0 |
| xlmr_base | lora | test | uncalibrated | 0.9057 | 0.9056 | 0.2611 | 0.0067 | 0.0418 | 0.148 | 0.022 | 0.9057 | 1.0 |


## 3. Importance-Weighted Conformal Prediction

| task | model | ft_method | condition | weighted_coverage | weighted_avg_set_size | effective_sample_size |
| --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | 0.9453 | 1.195 | 27.0447 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | 0.9276 | 1.3965 | 6.3953 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | 0.9365 | 1.1449 | 42.332 |
| offensive | banglabert | full_ft | shift_d2_test | 1.0 | 2.0 | 1893.3097 |
| offensive | banglabert | full_ft | shift_d3_noise05 | 0.9091 | 1.0194 | 335.9058 |
| offensive | banglabert | full_ft | shift_d3_noise10 | 0.9144 | 1.1088 | 59.3095 |
| offensive | banglabert | full_ft | shift_d3_noise20 | 0.9456 | 1.477 | 23.716 |
| offensive | banglabert | lora | shift_d1_barishal_test | 0.9309 | 1.1036 | 30.6351 |
| offensive | banglabert | lora | shift_d1_chittagong_test | 0.9375 | 1.4787 | 13.8568 |
| offensive | banglabert | lora | shift_d1_noakhali_test | 0.9315 | 1.1796 | 25.9692 |
| offensive | banglabert | lora | shift_d2_test | 1.0 | 2.0 | 3377.008 |
| offensive | banglabert | lora | shift_d3_noise05 | 0.9036 | 1.0123 | 216.0173 |
| offensive | banglabert | lora | shift_d3_noise10 | 0.9109 | 1.1126 | 37.0119 |
| offensive | banglabert | lora | shift_d3_noise20 | 0.9457 | 1.5071 | 8.8487 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | 0.8944 | 1.0739 | 1386.4753 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | 0.8277 | 1.113 | 1290.5032 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | 0.9054 | 1.1234 | 999.7849 |
| offensive | indicbert | full_ft | shift_d2_test | 0.8882 | 1.763 | 314.5761 |
| offensive | indicbert | full_ft | shift_d3_noise05 | 0.895 | 1.0558 | 3768.2416 |
| offensive | indicbert | full_ft | shift_d3_noise10 | 0.885 | 1.082 | 3351.2366 |
| offensive | indicbert | full_ft | shift_d3_noise20 | 0.8676 | 1.1276 | 2305.9478 |
| offensive | indicbert | lora | shift_d1_barishal_test | 0.9017 | 1.2813 | 231.4463 |
| offensive | indicbert | lora | shift_d1_chittagong_test | 0.8283 | 1.2652 | 164.2775 |
| offensive | indicbert | lora | shift_d1_noakhali_test | 0.8853 | 1.2411 | 267.1639 |
| offensive | indicbert | lora | shift_d2_test | 0.999 | 1.997 | 429.8825 |
| offensive | indicbert | lora | shift_d3_noise05 | 0.8914 | 1.2021 | 3677.099 |
| offensive | indicbert | lora | shift_d3_noise10 | 0.8885 | 1.2261 | 2981.4109 |
| offensive | indicbert | lora | shift_d3_noise20 | 0.8731 | 1.2597 | 992.8688 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | 0.917 | 1.0351 | 403.6952 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | 0.9163 | 1.2485 | 129.9068 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | 0.9106 | 1.0595 | 279.8509 |
| offensive | xlmr_base | full_ft | shift_d2_test | 0.999 | 1.9982 | 169.6887 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | 0.9089 | 1.0079 | 1731.0084 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | 0.9014 | 1.029 | 310.836 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | 0.9081 | 1.1481 | 49.3565 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | 0.9144 | 1.1479 | 55.1169 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | 0.94 | 1.5263 | 42.942 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | 0.9212 | 1.234 | 44.8556 |
| offensive | xlmr_base | lora | shift_d2_test | 1.0 | 2.0 | 650.2161 |
| offensive | xlmr_base | lora | shift_d3_noise05 | 0.9048 | 1.0194 | 1186.8616 |
| offensive | xlmr_base | lora | shift_d3_noise10 | 0.8892 | 1.0274 | 163.8905 |
| offensive | xlmr_base | lora | shift_d3_noise20 | 0.8833 | 1.1654 | 30.1474 |
| sentiment | banglabert | full_ft | shift_d1_test | 0.999 | 2.8617 | 115.0613 |
| sentiment | banglabert | full_ft | shift_d2_test | 1.0 | 3.0 | 987.6945 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | 0.9541 | 2.2765 | 78.1525 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | 0.9974 | 2.9171 | 90.35 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | 1.0 | 2.9999 | 77.1346 |
| sentiment | banglabert | lora | shift_d1_test | 1.0 | 2.997 | 165.0108 |
| sentiment | banglabert | lora | shift_d2_test | 1.0 | 3.0 | 1076.1352 |
| sentiment | banglabert | lora | shift_d3_noise05 | 0.9491 | 2.2363 | 46.1476 |
| sentiment | banglabert | lora | shift_d3_noise10 | 0.9991 | 2.9409 | 129.6316 |
| sentiment | banglabert | lora | shift_d3_noise20 | 1.0 | 3.0 | 123.636 |
| sentiment | indicbert | full_ft | shift_d1_test | 0.9632 | 2.5095 | 335.5675 |
| sentiment | indicbert | full_ft | shift_d2_test | 1.0 | 3.0 | 417.3387 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | 0.9143 | 2.1694 | 482.4312 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | 0.9184 | 2.2303 | 454.918 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | 0.9194 | 2.291 | 260.4155 |
| sentiment | indicbert | lora | shift_d1_test | 0.8955 | 2.4239 | 520.9018 |
| sentiment | indicbert | lora | shift_d2_test | 1.0 | 3.0 | 336.5857 |
| sentiment | indicbert | lora | shift_d3_noise05 | 0.8875 | 2.3569 | 797.8627 |
| sentiment | indicbert | lora | shift_d3_noise10 | 0.8916 | 2.37 | 597.1822 |
| sentiment | indicbert | lora | shift_d3_noise20 | 0.8941 | 2.4341 | 269.3125 |
| sentiment | xlmr_base | full_ft | shift_d1_test | 0.9731 | 2.4945 | 177.5932 |
| sentiment | xlmr_base | full_ft | shift_d2_test | 1.0 | 3.0 | 169.1401 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | 0.9468 | 2.0325 | 183.5071 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | 0.9157 | 2.0126 | 48.8234 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | 0.9782 | 2.7413 | 42.2931 |
| sentiment | xlmr_base | lora | shift_d1_test | 0.991 | 2.8378 | 160.7945 |
| sentiment | xlmr_base | lora | shift_d2_test | 1.0 | 3.0 | 300.9159 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | 0.895 | 1.8247 | 54.5044 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | 0.89 | 2.0816 | 20.48 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | 0.9957 | 2.9315 | 43.4973 |


## 4. Misclassification Detection from Representation Distance

| task | model | ft_method | condition | predictor | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | confidence_only | 0.7866 | 0.2996 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | entropy_only | 0.8082 | 0.3017 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | maha_distance_only | 0.7556 | 0.2449 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | maha_plus_confidence | 0.8005 | 0.2926 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | confidence_only | 0.7281 | 0.3583 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | entropy_only | 0.7381 | 0.3582 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | maha_distance_only | 0.7045 | 0.3102 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | maha_plus_confidence | 0.7515 | 0.3605 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | confidence_only | 0.8283 | 0.3379 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | entropy_only | 0.8363 | 0.3361 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | maha_distance_only | 0.7534 | 0.2441 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | maha_plus_confidence | 0.8086 | 0.3043 |
| offensive | banglabert | full_ft | shift_d2_test | confidence_only | 0.504 | 0.4715 |
| offensive | banglabert | full_ft | shift_d2_test | entropy_only | 0.5086 | 0.4714 |
| offensive | banglabert | full_ft | shift_d2_test | maha_distance_only | 0.4991 | 0.4829 |
| offensive | banglabert | full_ft | shift_d2_test | maha_plus_confidence | 0.5093 | 0.4898 |
| offensive | banglabert | full_ft | shift_d3_noise05 | confidence_only | 0.812 | 0.3271 |
| offensive | banglabert | full_ft | shift_d3_noise05 | entropy_only | 0.8209 | 0.3286 |
| offensive | banglabert | full_ft | shift_d3_noise05 | maha_distance_only | 0.754 | 0.2567 |
| offensive | banglabert | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.8068 | 0.3171 |
| offensive | banglabert | full_ft | shift_d3_noise10 | confidence_only | 0.781 | 0.326 |
| offensive | banglabert | full_ft | shift_d3_noise10 | entropy_only | 0.7894 | 0.3255 |
| offensive | banglabert | full_ft | shift_d3_noise10 | maha_distance_only | 0.7303 | 0.2665 |
| offensive | banglabert | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.7827 | 0.321 |
| offensive | banglabert | full_ft | shift_d3_noise20 | confidence_only | 0.7473 | 0.3657 |
| offensive | banglabert | full_ft | shift_d3_noise20 | entropy_only | 0.7528 | 0.3667 |
| offensive | banglabert | full_ft | shift_d3_noise20 | maha_distance_only | 0.7094 | 0.3168 |
| offensive | banglabert | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7601 | 0.3705 |
| offensive | banglabert | full_ft | test | confidence_only | 0.8312 | 0.3287 |
| offensive | banglabert | full_ft | test | entropy_only | 0.8436 | 0.3303 |
| offensive | banglabert | full_ft | test | maha_distance_only | 0.7706 | 0.2487 |
| offensive | banglabert | full_ft | test | maha_plus_confidence | 0.8237 | 0.3114 |
| offensive | banglabert | lora | shift_d1_barishal_test | confidence_only | 0.8384 | 0.3586 |
| offensive | banglabert | lora | shift_d1_barishal_test | entropy_only | 0.8448 | 0.3566 |
| offensive | banglabert | lora | shift_d1_barishal_test | maha_distance_only | 0.7683 | 0.2457 |
| offensive | banglabert | lora | shift_d1_barishal_test | maha_plus_confidence | 0.8138 | 0.3302 |
| offensive | banglabert | lora | shift_d1_chittagong_test | confidence_only | 0.7496 | 0.3704 |
| offensive | banglabert | lora | shift_d1_chittagong_test | entropy_only | 0.7586 | 0.3699 |
| offensive | banglabert | lora | shift_d1_chittagong_test | maha_distance_only | 0.7058 | 0.2981 |
| offensive | banglabert | lora | shift_d1_chittagong_test | maha_plus_confidence | 0.7503 | 0.3602 |
| offensive | banglabert | lora | shift_d1_noakhali_test | confidence_only | 0.838 | 0.3476 |
| offensive | banglabert | lora | shift_d1_noakhali_test | entropy_only | 0.8457 | 0.3473 |
| offensive | banglabert | lora | shift_d1_noakhali_test | maha_distance_only | 0.7242 | 0.2057 |
| offensive | banglabert | lora | shift_d1_noakhali_test | maha_plus_confidence | 0.8043 | 0.3142 |
| offensive | banglabert | lora | shift_d2_test | confidence_only | 0.5118 | 0.4867 |
| offensive | banglabert | lora | shift_d2_test | entropy_only | 0.5126 | 0.4849 |
| offensive | banglabert | lora | shift_d2_test | maha_distance_only | 0.5064 | 0.4972 |
| offensive | banglabert | lora | shift_d2_test | maha_plus_confidence | 0.5154 | 0.5038 |
| offensive | banglabert | lora | shift_d3_noise05 | confidence_only | 0.8429 | 0.3638 |
| offensive | banglabert | lora | shift_d3_noise05 | entropy_only | 0.848 | 0.3607 |
| offensive | banglabert | lora | shift_d3_noise05 | maha_distance_only | 0.7547 | 0.2276 |
| offensive | banglabert | lora | shift_d3_noise05 | maha_plus_confidence | 0.8222 | 0.3375 |
| offensive | banglabert | lora | shift_d3_noise10 | confidence_only | 0.8018 | 0.3358 |
| offensive | banglabert | lora | shift_d3_noise10 | entropy_only | 0.809 | 0.3381 |
| offensive | banglabert | lora | shift_d3_noise10 | maha_distance_only | 0.7403 | 0.2426 |
| offensive | banglabert | lora | shift_d3_noise10 | maha_plus_confidence | 0.7938 | 0.3272 |
| offensive | banglabert | lora | shift_d3_noise20 | confidence_only | 0.7745 | 0.3697 |
| offensive | banglabert | lora | shift_d3_noise20 | entropy_only | 0.7765 | 0.3695 |
| offensive | banglabert | lora | shift_d3_noise20 | maha_distance_only | 0.7168 | 0.2844 |
| offensive | banglabert | lora | shift_d3_noise20 | maha_plus_confidence | 0.77 | 0.3599 |
| offensive | banglabert | lora | test | confidence_only | 0.8557 | 0.3595 |
| offensive | banglabert | lora | test | entropy_only | 0.8647 | 0.3603 |
| offensive | banglabert | lora | test | maha_distance_only | 0.7657 | 0.2139 |
| offensive | banglabert | lora | test | maha_plus_confidence | 0.8294 | 0.3268 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | confidence_only | 0.7836 | 0.3439 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | entropy_only | 0.7877 | 0.3419 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | maha_distance_only | 0.6423 | 0.2324 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | maha_plus_confidence | 0.7704 | 0.3341 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | confidence_only | 0.7316 | 0.4112 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | entropy_only | 0.7367 | 0.4113 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | maha_distance_only | 0.6085 | 0.3133 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | maha_plus_confidence | 0.7184 | 0.4034 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | confidence_only | 0.7959 | 0.3903 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | entropy_only | 0.8007 | 0.3879 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | maha_distance_only | 0.6358 | 0.2381 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | maha_plus_confidence | 0.7832 | 0.3786 |
| offensive | indicbert | full_ft | shift_d2_test | confidence_only | 0.5218 | 0.4916 |
| offensive | indicbert | full_ft | shift_d2_test | entropy_only | 0.5255 | 0.4947 |
| offensive | indicbert | full_ft | shift_d2_test | maha_distance_only | 0.5153 | 0.4801 |
| offensive | indicbert | full_ft | shift_d2_test | maha_plus_confidence | 0.5309 | 0.5006 |
| offensive | indicbert | full_ft | shift_d3_noise05 | confidence_only | 0.7962 | 0.3579 |
| offensive | indicbert | full_ft | shift_d3_noise05 | entropy_only | 0.8013 | 0.3572 |
| offensive | indicbert | full_ft | shift_d3_noise05 | maha_distance_only | 0.6403 | 0.2313 |
| offensive | indicbert | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.7821 | 0.3495 |
| offensive | indicbert | full_ft | shift_d3_noise10 | confidence_only | 0.77 | 0.3509 |
| offensive | indicbert | full_ft | shift_d3_noise10 | entropy_only | 0.7743 | 0.3511 |
| offensive | indicbert | full_ft | shift_d3_noise10 | maha_distance_only | 0.6268 | 0.2445 |
| offensive | indicbert | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.7554 | 0.3435 |
| offensive | indicbert | full_ft | shift_d3_noise20 | confidence_only | 0.74 | 0.3735 |
| offensive | indicbert | full_ft | shift_d3_noise20 | entropy_only | 0.742 | 0.374 |
| offensive | indicbert | full_ft | shift_d3_noise20 | maha_distance_only | 0.6196 | 0.2835 |
| offensive | indicbert | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7292 | 0.3707 |
| offensive | indicbert | full_ft | test | confidence_only | 0.8152 | 0.353 |
| offensive | indicbert | full_ft | test | entropy_only | 0.8173 | 0.354 |
| offensive | indicbert | full_ft | test | maha_distance_only | 0.6406 | 0.2104 |
| offensive | indicbert | full_ft | test | maha_plus_confidence | 0.8013 | 0.3449 |
| offensive | indicbert | lora | shift_d1_barishal_test | confidence_only | 0.7386 | 0.3881 |
| offensive | indicbert | lora | shift_d1_barishal_test | entropy_only | 0.7383 | 0.3824 |
| offensive | indicbert | lora | shift_d1_barishal_test | maha_distance_only | 0.5556 | 0.2169 |
| offensive | indicbert | lora | shift_d1_barishal_test | maha_plus_confidence | 0.736 | 0.3863 |
| offensive | indicbert | lora | shift_d1_chittagong_test | confidence_only | 0.6748 | 0.43 |
| offensive | indicbert | lora | shift_d1_chittagong_test | entropy_only | 0.6752 | 0.4278 |
| offensive | indicbert | lora | shift_d1_chittagong_test | maha_distance_only | 0.5458 | 0.2999 |
| offensive | indicbert | lora | shift_d1_chittagong_test | maha_plus_confidence | 0.6714 | 0.429 |
| offensive | indicbert | lora | shift_d1_noakhali_test | confidence_only | 0.7411 | 0.3987 |
| offensive | indicbert | lora | shift_d1_noakhali_test | entropy_only | 0.7416 | 0.3949 |
| offensive | indicbert | lora | shift_d1_noakhali_test | maha_distance_only | 0.5533 | 0.2221 |
| offensive | indicbert | lora | shift_d1_noakhali_test | maha_plus_confidence | 0.7376 | 0.3977 |
| offensive | indicbert | lora | shift_d2_test | confidence_only | 0.4922 | 0.505 |
| offensive | indicbert | lora | shift_d2_test | entropy_only | 0.4902 | 0.5038 |
| offensive | indicbert | lora | shift_d2_test | maha_distance_only | 0.493 | 0.5107 |
| offensive | indicbert | lora | shift_d2_test | maha_plus_confidence | 0.4935 | 0.5114 |
| offensive | indicbert | lora | shift_d3_noise05 | confidence_only | 0.7388 | 0.3794 |
| offensive | indicbert | lora | shift_d3_noise05 | entropy_only | 0.7387 | 0.3772 |
| offensive | indicbert | lora | shift_d3_noise05 | maha_distance_only | 0.5656 | 0.2176 |
| offensive | indicbert | lora | shift_d3_noise05 | maha_plus_confidence | 0.7383 | 0.379 |
| offensive | indicbert | lora | shift_d3_noise10 | confidence_only | 0.7253 | 0.3829 |
| offensive | indicbert | lora | shift_d3_noise10 | entropy_only | 0.7253 | 0.3805 |
| offensive | indicbert | lora | shift_d3_noise10 | maha_distance_only | 0.564 | 0.2303 |
| offensive | indicbert | lora | shift_d3_noise10 | maha_plus_confidence | 0.7271 | 0.3808 |
| offensive | indicbert | lora | shift_d3_noise20 | confidence_only | 0.6988 | 0.3879 |
| offensive | indicbert | lora | shift_d3_noise20 | entropy_only | 0.6989 | 0.3856 |
| offensive | indicbert | lora | shift_d3_noise20 | maha_distance_only | 0.5494 | 0.2525 |
| offensive | indicbert | lora | shift_d3_noise20 | maha_plus_confidence | 0.7006 | 0.3887 |
| offensive | indicbert | lora | test | confidence_only | 0.7435 | 0.3603 |
| offensive | indicbert | lora | test | entropy_only | 0.7439 | 0.3571 |
| offensive | indicbert | lora | test | maha_distance_only | 0.5799 | 0.2022 |
| offensive | indicbert | lora | test | maha_plus_confidence | 0.7451 | 0.3598 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | confidence_only | 0.8257 | 0.3471 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | entropy_only | 0.8318 | 0.3443 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | maha_distance_only | 0.676 | 0.1805 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | maha_plus_confidence | 0.7945 | 0.3162 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | confidence_only | 0.7668 | 0.3845 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | entropy_only | 0.7707 | 0.3857 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | maha_distance_only | 0.6622 | 0.2699 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | maha_plus_confidence | 0.7684 | 0.3782 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | confidence_only | 0.8303 | 0.3575 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | entropy_only | 0.8376 | 0.3571 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | maha_distance_only | 0.67 | 0.1901 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | maha_plus_confidence | 0.8032 | 0.3295 |
| offensive | xlmr_base | full_ft | shift_d2_test | confidence_only | 0.4863 | 0.4699 |
| offensive | xlmr_base | full_ft | shift_d2_test | entropy_only | 0.4831 | 0.4656 |
| offensive | xlmr_base | full_ft | shift_d2_test | maha_distance_only | 0.596 | 0.5691 |
| offensive | xlmr_base | full_ft | shift_d2_test | maha_plus_confidence | 0.5994 | 0.57 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | confidence_only | 0.8286 | 0.3287 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | entropy_only | 0.8323 | 0.328 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | maha_distance_only | 0.6702 | 0.1818 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.8151 | 0.3119 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | confidence_only | 0.8151 | 0.354 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | entropy_only | 0.8181 | 0.3549 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | maha_distance_only | 0.6711 | 0.2027 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.8058 | 0.3436 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | confidence_only | 0.7809 | 0.3604 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | entropy_only | 0.7825 | 0.3599 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | maha_distance_only | 0.6641 | 0.2338 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7819 | 0.3546 |
| offensive | xlmr_base | full_ft | test | confidence_only | 0.8417 | 0.3443 |
| offensive | xlmr_base | full_ft | test | entropy_only | 0.8481 | 0.3438 |
| offensive | xlmr_base | full_ft | test | maha_distance_only | 0.6764 | 0.1695 |
| offensive | xlmr_base | full_ft | test | maha_plus_confidence | 0.827 | 0.3265 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | confidence_only | 0.8396 | 0.3671 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | entropy_only | 0.8412 | 0.366 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | maha_distance_only | 0.5586 | 0.1331 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | maha_plus_confidence | 0.8207 | 0.3561 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | confidence_only | 0.7799 | 0.3966 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | entropy_only | 0.782 | 0.3949 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | maha_distance_only | 0.5593 | 0.2105 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | maha_plus_confidence | 0.7722 | 0.3936 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | confidence_only | 0.8319 | 0.3705 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | entropy_only | 0.8327 | 0.3666 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | maha_distance_only | 0.5566 | 0.1505 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | maha_plus_confidence | 0.8158 | 0.3606 |
| offensive | xlmr_base | lora | shift_d2_test | confidence_only | 0.5017 | 0.4839 |
| offensive | xlmr_base | lora | shift_d2_test | entropy_only | 0.5066 | 0.4843 |
| offensive | xlmr_base | lora | shift_d2_test | maha_distance_only | 0.5545 | 0.5257 |
| offensive | xlmr_base | lora | shift_d2_test | maha_plus_confidence | 0.5511 | 0.5183 |
| offensive | xlmr_base | lora | shift_d3_noise05 | confidence_only | 0.8311 | 0.3414 |
| offensive | xlmr_base | lora | shift_d3_noise05 | entropy_only | 0.8338 | 0.3392 |
| offensive | xlmr_base | lora | shift_d3_noise05 | maha_distance_only | 0.5727 | 0.1313 |
| offensive | xlmr_base | lora | shift_d3_noise05 | maha_plus_confidence | 0.8131 | 0.3386 |
| offensive | xlmr_base | lora | shift_d3_noise10 | confidence_only | 0.8099 | 0.3435 |
| offensive | xlmr_base | lora | shift_d3_noise10 | entropy_only | 0.8112 | 0.3411 |
| offensive | xlmr_base | lora | shift_d3_noise10 | maha_distance_only | 0.558 | 0.1421 |
| offensive | xlmr_base | lora | shift_d3_noise10 | maha_plus_confidence | 0.7937 | 0.337 |
| offensive | xlmr_base | lora | shift_d3_noise20 | confidence_only | 0.7897 | 0.374 |
| offensive | xlmr_base | lora | shift_d3_noise20 | entropy_only | 0.7903 | 0.3709 |
| offensive | xlmr_base | lora | shift_d3_noise20 | maha_distance_only | 0.5506 | 0.1745 |
| offensive | xlmr_base | lora | shift_d3_noise20 | maha_plus_confidence | 0.7828 | 0.3712 |
| offensive | xlmr_base | lora | test | confidence_only | 0.8458 | 0.3429 |
| offensive | xlmr_base | lora | test | entropy_only | 0.8482 | 0.3421 |
| offensive | xlmr_base | lora | test | maha_distance_only | 0.5726 | 0.1179 |
| offensive | xlmr_base | lora | test | maha_plus_confidence | 0.8222 | 0.3351 |
| sentiment | banglabert | full_ft | shift_d1_test | confidence_only | 0.7513 | 0.5224 |
| sentiment | banglabert | full_ft | shift_d1_test | entropy_only | 0.7617 | 0.5387 |
| sentiment | banglabert | full_ft | shift_d1_test | maha_distance_only | 0.658 | 0.4252 |
| sentiment | banglabert | full_ft | shift_d1_test | maha_plus_confidence | 0.7357 | 0.5017 |
| sentiment | banglabert | full_ft | shift_d2_test | confidence_only | 0.7147 | 0.7027 |
| sentiment | banglabert | full_ft | shift_d2_test | entropy_only | 0.7165 | 0.6992 |
| sentiment | banglabert | full_ft | shift_d2_test | maha_distance_only | 0.605 | 0.6175 |
| sentiment | banglabert | full_ft | shift_d2_test | maha_plus_confidence | 0.7075 | 0.6942 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | confidence_only | 0.7552 | 0.5146 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | entropy_only | 0.7558 | 0.5165 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | maha_distance_only | 0.6743 | 0.4197 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.755 | 0.5085 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | confidence_only | 0.7365 | 0.5147 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | entropy_only | 0.7392 | 0.5187 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | maha_distance_only | 0.6375 | 0.4209 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.7295 | 0.5042 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | confidence_only | 0.7412 | 0.5844 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | entropy_only | 0.7451 | 0.593 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | maha_distance_only | 0.6291 | 0.4705 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7331 | 0.5782 |
| sentiment | banglabert | full_ft | test | confidence_only | 0.7638 | 0.5189 |
| sentiment | banglabert | full_ft | test | entropy_only | 0.7644 | 0.5209 |
| sentiment | banglabert | full_ft | test | maha_distance_only | 0.6967 | 0.4318 |
| sentiment | banglabert | full_ft | test | maha_plus_confidence | 0.7639 | 0.5128 |
| sentiment | banglabert | lora | shift_d1_test | confidence_only | 0.7602 | 0.5571 |
| sentiment | banglabert | lora | shift_d1_test | entropy_only | 0.7632 | 0.5571 |
| sentiment | banglabert | lora | shift_d1_test | maha_distance_only | 0.6751 | 0.4375 |
| sentiment | banglabert | lora | shift_d1_test | maha_plus_confidence | 0.7426 | 0.5126 |
| sentiment | banglabert | lora | shift_d2_test | confidence_only | 0.6997 | 0.671 |
| sentiment | banglabert | lora | shift_d2_test | entropy_only | 0.6948 | 0.6698 |
| sentiment | banglabert | lora | shift_d2_test | maha_distance_only | 0.6307 | 0.6482 |
| sentiment | banglabert | lora | shift_d2_test | maha_plus_confidence | 0.7077 | 0.6862 |
| sentiment | banglabert | lora | shift_d3_noise05 | confidence_only | 0.75 | 0.5034 |
| sentiment | banglabert | lora | shift_d3_noise05 | entropy_only | 0.7522 | 0.5052 |
| sentiment | banglabert | lora | shift_d3_noise05 | maha_distance_only | 0.6712 | 0.4015 |
| sentiment | banglabert | lora | shift_d3_noise05 | maha_plus_confidence | 0.7458 | 0.4935 |
| sentiment | banglabert | lora | shift_d3_noise10 | confidence_only | 0.7433 | 0.5551 |
| sentiment | banglabert | lora | shift_d3_noise10 | entropy_only | 0.7471 | 0.5577 |
| sentiment | banglabert | lora | shift_d3_noise10 | maha_distance_only | 0.6511 | 0.4409 |
| sentiment | banglabert | lora | shift_d3_noise10 | maha_plus_confidence | 0.7416 | 0.5434 |
| sentiment | banglabert | lora | shift_d3_noise20 | confidence_only | 0.7264 | 0.5599 |
| sentiment | banglabert | lora | shift_d3_noise20 | entropy_only | 0.7305 | 0.5681 |
| sentiment | banglabert | lora | shift_d3_noise20 | maha_distance_only | 0.6217 | 0.4562 |
| sentiment | banglabert | lora | shift_d3_noise20 | maha_plus_confidence | 0.7243 | 0.5579 |
| sentiment | banglabert | lora | test | confidence_only | 0.7601 | 0.5082 |
| sentiment | banglabert | lora | test | entropy_only | 0.7593 | 0.5033 |
| sentiment | banglabert | lora | test | maha_distance_only | 0.6874 | 0.4154 |
| sentiment | banglabert | lora | test | maha_plus_confidence | 0.7572 | 0.499 |
| sentiment | indicbert | full_ft | shift_d1_test | confidence_only | 0.7253 | 0.6037 |
| sentiment | indicbert | full_ft | shift_d1_test | entropy_only | 0.7278 | 0.5892 |
| sentiment | indicbert | full_ft | shift_d1_test | maha_distance_only | 0.5896 | 0.4854 |
| sentiment | indicbert | full_ft | shift_d1_test | maha_plus_confidence | 0.7131 | 0.5697 |
| sentiment | indicbert | full_ft | shift_d2_test | confidence_only | 0.5629 | 0.5895 |
| sentiment | indicbert | full_ft | shift_d2_test | entropy_only | 0.56 | 0.5829 |
| sentiment | indicbert | full_ft | shift_d2_test | maha_distance_only | 0.5214 | 0.5693 |
| sentiment | indicbert | full_ft | shift_d2_test | maha_plus_confidence | 0.5561 | 0.5942 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | confidence_only | 0.7073 | 0.5744 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | entropy_only | 0.7065 | 0.5721 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | maha_distance_only | 0.5807 | 0.4626 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.708 | 0.5729 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | confidence_only | 0.7033 | 0.583 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | entropy_only | 0.7021 | 0.582 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | maha_distance_only | 0.569 | 0.4584 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.7044 | 0.5824 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | confidence_only | 0.6741 | 0.559 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | entropy_only | 0.6731 | 0.5527 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | maha_distance_only | 0.561 | 0.4556 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.6749 | 0.5522 |
| sentiment | indicbert | full_ft | test | confidence_only | 0.7115 | 0.5703 |
| sentiment | indicbert | full_ft | test | entropy_only | 0.7124 | 0.5672 |
| sentiment | indicbert | full_ft | test | maha_distance_only | 0.5857 | 0.4491 |
| sentiment | indicbert | full_ft | test | maha_plus_confidence | 0.714 | 0.566 |
| sentiment | indicbert | lora | shift_d1_test | confidence_only | 0.5347 | 0.5357 |
| sentiment | indicbert | lora | shift_d1_test | entropy_only | 0.5312 | 0.5359 |
| sentiment | indicbert | lora | shift_d1_test | maha_distance_only | 0.5214 | 0.5306 |
| sentiment | indicbert | lora | shift_d1_test | maha_plus_confidence | 0.5185 | 0.5295 |
| sentiment | indicbert | lora | shift_d2_test | confidence_only | 0.5181 | 0.5851 |
| sentiment | indicbert | lora | shift_d2_test | entropy_only | 0.5181 | 0.5831 |
| sentiment | indicbert | lora | shift_d2_test | maha_distance_only | 0.5388 | 0.5995 |
| sentiment | indicbert | lora | shift_d2_test | maha_plus_confidence | 0.5381 | 0.5982 |
| sentiment | indicbert | lora | shift_d3_noise05 | confidence_only | 0.6007 | 0.6097 |
| sentiment | indicbert | lora | shift_d3_noise05 | entropy_only | 0.5894 | 0.5884 |
| sentiment | indicbert | lora | shift_d3_noise05 | maha_distance_only | 0.5767 | 0.5799 |
| sentiment | indicbert | lora | shift_d3_noise05 | maha_plus_confidence | 0.5945 | 0.5975 |
| sentiment | indicbert | lora | shift_d3_noise10 | confidence_only | 0.5873 | 0.5987 |
| sentiment | indicbert | lora | shift_d3_noise10 | entropy_only | 0.5821 | 0.5856 |
| sentiment | indicbert | lora | shift_d3_noise10 | maha_distance_only | 0.567 | 0.5779 |
| sentiment | indicbert | lora | shift_d3_noise10 | maha_plus_confidence | 0.5812 | 0.5926 |
| sentiment | indicbert | lora | shift_d3_noise20 | confidence_only | 0.58 | 0.5994 |
| sentiment | indicbert | lora | shift_d3_noise20 | entropy_only | 0.5745 | 0.5926 |
| sentiment | indicbert | lora | shift_d3_noise20 | maha_distance_only | 0.5609 | 0.5778 |
| sentiment | indicbert | lora | shift_d3_noise20 | maha_plus_confidence | 0.5766 | 0.592 |
| sentiment | indicbert | lora | test | confidence_only | 0.6005 | 0.5971 |
| sentiment | indicbert | lora | test | entropy_only | 0.5904 | 0.5843 |
| sentiment | indicbert | lora | test | maha_distance_only | 0.5715 | 0.5653 |
| sentiment | indicbert | lora | test | maha_plus_confidence | 0.5898 | 0.5813 |
| sentiment | xlmr_base | full_ft | shift_d1_test | confidence_only | 0.7355 | 0.5534 |
| sentiment | xlmr_base | full_ft | shift_d1_test | entropy_only | 0.7382 | 0.5506 |
| sentiment | xlmr_base | full_ft | shift_d1_test | maha_distance_only | 0.6619 | 0.4729 |
| sentiment | xlmr_base | full_ft | shift_d1_test | maha_plus_confidence | 0.7144 | 0.5159 |
| sentiment | xlmr_base | full_ft | shift_d2_test | confidence_only | 0.6429 | 0.5658 |
| sentiment | xlmr_base | full_ft | shift_d2_test | entropy_only | 0.6479 | 0.562 |
| sentiment | xlmr_base | full_ft | shift_d2_test | maha_distance_only | 0.6365 | 0.5751 |
| sentiment | xlmr_base | full_ft | shift_d2_test | maha_plus_confidence | 0.6492 | 0.5882 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | confidence_only | 0.7483 | 0.5237 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | entropy_only | 0.7484 | 0.5216 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | maha_distance_only | 0.6824 | 0.4461 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | maha_plus_confidence | 0.7501 | 0.5216 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | confidence_only | 0.7296 | 0.5157 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | entropy_only | 0.7292 | 0.5186 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | maha_distance_only | 0.6739 | 0.4476 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | maha_plus_confidence | 0.7309 | 0.5105 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | confidence_only | 0.7228 | 0.5575 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | entropy_only | 0.7243 | 0.5561 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | maha_distance_only | 0.6644 | 0.4839 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | maha_plus_confidence | 0.7235 | 0.5511 |
| sentiment | xlmr_base | full_ft | test | confidence_only | 0.7564 | 0.531 |
| sentiment | xlmr_base | full_ft | test | entropy_only | 0.7562 | 0.5297 |
| sentiment | xlmr_base | full_ft | test | maha_distance_only | 0.6884 | 0.4452 |
| sentiment | xlmr_base | full_ft | test | maha_plus_confidence | 0.757 | 0.5296 |
| sentiment | xlmr_base | lora | shift_d1_test | confidence_only | 0.7257 | 0.5538 |
| sentiment | xlmr_base | lora | shift_d1_test | entropy_only | 0.7208 | 0.545 |
| sentiment | xlmr_base | lora | shift_d1_test | maha_distance_only | 0.4881 | 0.3875 |
| sentiment | xlmr_base | lora | shift_d1_test | maha_plus_confidence | 0.7227 | 0.5451 |
| sentiment | xlmr_base | lora | shift_d2_test | confidence_only | 0.7091 | 0.6059 |
| sentiment | xlmr_base | lora | shift_d2_test | entropy_only | 0.7095 | 0.61 |
| sentiment | xlmr_base | lora | shift_d2_test | maha_distance_only | 0.6045 | 0.5384 |
| sentiment | xlmr_base | lora | shift_d2_test | maha_plus_confidence | 0.6911 | 0.5963 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | confidence_only | 0.7265 | 0.5057 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | entropy_only | 0.7246 | 0.5038 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | maha_distance_only | 0.5183 | 0.3433 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | maha_plus_confidence | 0.725 | 0.503 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | confidence_only | 0.7293 | 0.5382 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | entropy_only | 0.7238 | 0.5327 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | maha_distance_only | 0.4991 | 0.3577 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | maha_plus_confidence | 0.7282 | 0.536 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | confidence_only | 0.7158 | 0.5559 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | entropy_only | 0.7115 | 0.5493 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | maha_distance_only | 0.5063 | 0.3814 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | maha_plus_confidence | 0.7146 | 0.5529 |
| sentiment | xlmr_base | lora | test | confidence_only | 0.7452 | 0.5393 |
| sentiment | xlmr_base | lora | test | entropy_only | 0.7428 | 0.5374 |
| sentiment | xlmr_base | lora | test | maha_distance_only | 0.516 | 0.3414 |
| sentiment | xlmr_base | lora | test | maha_plus_confidence | 0.7441 | 0.5378 |


## 5. Statistical Significance (Holm-Corrected)

| task | model | ft_method | comparison | metric | endpoint_family | mean_diff | median_raw_p | median_holm_p | seeds_sig_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0033 | 0.7853 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0017 | 0.7977 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0141 | 0.6215 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1532 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0668 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.3297 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0154 | 0.2784 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.005 | 0.3896 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0199 | 0.4094 | 1.0 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.7787 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.3996 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d2_test | nll | secondary | -2.273 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.028 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0105 | 0.0004 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0593 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0573 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0232 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.1259 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.1442 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0645 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.3102 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0052 | 0.6377 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0048 | 0.5081 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0217 | 0.3026 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1503 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0572 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.2824 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0233 | 0.0304 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0071 | 0.2182 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0365 | 0.0776 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d2_test | brier | secondary | -0.6984 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.3569 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d2_test | nll | secondary | -1.4568 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0252 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0073 | 0.0048 | 1.0 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0418 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0525 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0157 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0993 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.1249 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0443 | 0.0002 | 0.211 | 0 |
| offensive | banglabert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.2226 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0378 | 0.0056 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0163 | 0.0386 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0669 | 0.0084 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1902 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0924 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.3543 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0541 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0247 | 0.0032 | 1.0 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0895 | 0.0006 | 0.2898 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.6002 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.2981 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d2_test | nll | secondary | -1.0088 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0311 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0144 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.057 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0675 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0317 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.1306 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.1289 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0593 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.2446 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_barishal_test | brier | secondary | -0.035 | 0.004 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0141 | 0.114 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0445 | 0.005 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1525 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0915 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.2062 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0405 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0158 | 0.0192 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0506 | 0.0006 | 0.2898 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d2_test | brier | secondary | -0.3972 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.2228 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d2_test | nll | secondary | -0.5076 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0216 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0086 | 0.0826 | 1.0 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0284 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0415 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.018 | 0.0012 | 0.5459 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.056 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0804 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0356 | 0.0002 | 0.211 | 0 |
| offensive | indicbert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.1084 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0122 | 0.3804 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0056 | 0.2626 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0257 | 0.2328 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1311 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0572 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.243 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0348 | 0.003 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0141 | 0.0228 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0555 | 0.0112 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.7471 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.3856 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d2_test | nll | secondary | -1.6327 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0171 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0048 | 0.0568 | 1.0 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0286 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0408 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.014 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.0698 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.0864 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0298 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.1483 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_barishal_test | brier | secondary | -0.0168 | 0.1546 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_barishal_test | ece_15 | secondary | -0.0039 | 0.5773 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_barishal_test | nll | secondary | -0.0236 | 0.255 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_chittagong_test | brier | secondary | -0.1212 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_chittagong_test | ece_15 | secondary | -0.0382 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_chittagong_test | nll | secondary | -0.183 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_noakhali_test | brier | secondary | -0.0392 | 0.0012 | 0.5459 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_noakhali_test | ece_15 | secondary | -0.0058 | 0.4522 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d1_noakhali_test | nll | secondary | -0.0506 | 0.0152 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d2_test | brier | secondary | -0.7075 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.3791 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d2_test | nll | secondary | -1.4321 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0138 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.001 | 0.7097 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0191 | 0.0006 | 0.2898 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0382 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0038 | 0.2614 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0596 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0763 | 0.0002 | 0.211 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0087 | 0.0626 | 1.0 | 0 |
| offensive | xlmr_base | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.1135 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d1_test | brier | secondary | -0.0638 | 0.0676 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d1_test | ece_15 | secondary | -0.015 | 0.446 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d1_test | nll | primary | -0.1151 | 0.0256 | 0.0256 | 3 |
| sentiment | banglabert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.3501 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.1659 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d2_test | nll | primary | -0.6854 | 0.0002 | 0.0002 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0147 | 0.0392 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0014 | 0.7349 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise05 | nll | primary | -0.0228 | 0.0294 | 0.0294 | 3 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0557 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0118 | 0.4332 | 1.0 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise10 | nll | primary | -0.0875 | 0.0002 | 0.0002 | 5 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.115 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0373 | 0.0012 | 0.5459 | 0 |
| sentiment | banglabert | full_ft | D0_vs_shift_d3_noise20 | nll | primary | -0.1725 | 0.0002 | 0.0002 | 5 |
| sentiment | banglabert | lora | D0_vs_shift_d1_test | brier | secondary | -0.0684 | 0.1102 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d1_test | ece_15 | secondary | -0.0238 | 0.4804 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d1_test | nll | secondary | -0.1283 | 0.0774 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d2_test | brier | secondary | -0.3073 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.1266 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d2_test | nll | secondary | -0.6135 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0111 | 0.1692 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0056 | 0.3844 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0144 | 0.4556 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0519 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0078 | 0.7585 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0827 | 0.0004 | 0.211 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0913 | 0.0002 | 0.211 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0055 | 0.1594 | 1.0 | 0 |
| sentiment | banglabert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.1344 | 0.0002 | 0.211 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d1_test | brier | secondary | -0.0302 | 0.142 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d1_test | ece_15 | secondary | -0.0019 | 0.7023 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d1_test | nll | secondary | -0.0482 | 0.0988 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.2482 | 0.0002 | 0.211 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.1792 | 0.0002 | 0.211 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d2_test | nll | secondary | -0.4019 | 0.0002 | 0.211 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.013 | 0.014 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0068 | 0.7789 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0197 | 0.0178 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0242 | 0.001 | 0.459 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.011 | 0.0914 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.0387 | 0.0014 | 0.6299 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.0319 | 0.0002 | 0.211 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0084 | 0.6539 | 1.0 | 0 |
| sentiment | indicbert | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.052 | 0.0002 | 0.211 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d1_test | brier | secondary | -0.0055 | 0.3874 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d1_test | ece_15 | secondary | 0.0072 | 0.4338 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d1_test | nll | secondary | -0.0076 | 0.4736 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d2_test | brier | secondary | -0.0583 | 0.0004 | 0.211 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.0684 | 0.0004 | 0.211 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d2_test | nll | secondary | -0.0889 | 0.0004 | 0.211 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0027 | 0.3744 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | 0.0023 | 0.8967 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.004 | 0.3796 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0053 | 0.0534 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | 0.003 | 0.7745 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0069 | 0.0642 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0124 | 0.0002 | 0.211 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0002 | 0.7555 | 1.0 | 0 |
| sentiment | indicbert | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.018 | 0.0006 | 0.2898 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d1_test | brier | secondary | -0.0734 | 0.0386 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d1_test | ece_15 | secondary | -0.0372 | 0.1058 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d1_test | nll | secondary | -0.1507 | 0.015 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d2_test | brier | secondary | -0.3035 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d2_test | ece_15 | secondary | -0.1966 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d2_test | nll | secondary | -0.6873 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | brier | secondary | -0.0084 | 0.2212 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0014 | 0.4828 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise05 | nll | secondary | -0.0118 | 0.3752 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | brier | secondary | -0.0333 | 0.0072 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0104 | 0.6499 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise10 | nll | secondary | -0.0584 | 0.004 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | brier | secondary | -0.0719 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0313 | 0.0092 | 1.0 | 0 |
| sentiment | xlmr_base | full_ft | D0_vs_shift_d3_noise20 | nll | secondary | -0.1257 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d1_test | brier | secondary | -0.0596 | 0.0094 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d1_test | ece_15 | secondary | -0.0181 | 0.433 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d1_test | nll | secondary | -0.1199 | 0.0042 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d2_test | brier | secondary | -0.2373 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d2_test | ece_15 | secondary | -0.2062 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d2_test | nll | secondary | -0.4559 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise05 | brier | secondary | -0.0056 | 0.3698 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise05 | ece_15 | secondary | -0.0036 | 0.2268 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise05 | nll | secondary | -0.0081 | 0.341 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise10 | brier | secondary | -0.0259 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise10 | ece_15 | secondary | -0.0116 | 0.1482 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise10 | nll | secondary | -0.0356 | 0.0036 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise20 | brier | secondary | -0.0519 | 0.0002 | 0.211 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise20 | ece_15 | secondary | -0.0155 | 0.115 | 1.0 | 0 |
| sentiment | xlmr_base | lora | D0_vs_shift_d3_noise20 | nll | secondary | -0.0841 | 0.0002 | 0.211 | 0 |

