# Extended Calibration, Conformal, and Representation Diagnostics

## 1. Correlation of Shift Statistics with Miscalibration

| task | model | ft_method | n_conditions | spearman_frag_vs_nll | p_value_frag_vs_nll_exact | pearson_frag_vs_nll | spearman_jaccard_vs_nll | spearman_frag_vs_classwise_ece | spearman_frag_vs_overconfidence | spearman_maha_pct_vs_nll |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | 8 | 0.9048 | 0.0046 | 0.7202 | -0.6429 | 0.9524 | 0.9048 | nan |
| offensive | banglabert | lora | 8 | 0.9048 | 0.0046 | 0.72 | -0.6429 | 0.9524 | 0.9048 | nan |
| offensive | indicbert | full_ft | 8 | 0.9524 | 0.0011 | 0.7204 | -0.7381 | 0.9524 | 0.9524 | nan |
| offensive | indicbert | lora | 8 | 0.9524 | 0.0011 | 0.7377 | -0.7381 | 0.9762 | 0.9524 | nan |
| offensive | xlmr_base | full_ft | 8 | 0.9524 | 0.0011 | 0.6855 | -0.7381 | 0.9524 | 0.9286 | nan |
| offensive | xlmr_base | lora | 8 | 0.9762 | 0.0004 | 0.6823 | -0.8095 | 0.8095 | 0.9286 | nan |
| sentiment | banglabert | full_ft | 6 | 0.9429 | 0.0167 | 0.9252 | -0.6571 | 0.9429 | 0.7714 | nan |
| sentiment | banglabert | lora | 6 | 0.9429 | 0.0167 | 0.8771 | -0.6571 | 0.7714 | 0.3143 | nan |
| sentiment | indicbert | full_ft | 6 | 0.9429 | 0.0167 | 0.8664 | -0.9429 | 0.9429 | 0.9429 | nan |
| sentiment | indicbert | lora | 6 | 0.9429 | 0.0167 | 0.8348 | -0.9429 | 0.9429 | 1.0 | nan |
| sentiment | xlmr_base | full_ft | 6 | 0.9429 | 0.0167 | 0.8626 | -0.9429 | 0.8286 | 0.8286 | nan |
| sentiment | xlmr_base | lora | 6 | 0.9429 | 0.0167 | 0.8406 | -0.9429 | 0.7714 | 0.8286 | nan |


## 2. Split-Conformal Cross-Check Against MAPIE

| run_id | task | model | ft_method | seed | ours_coverage | ours_set_size | ours_mapie_rule_coverage | ours_mapie_rule_set_size | mapie_version | mapie_coverage | mapie_set_size | status | abs_coverage_diff | abs_set_size_diff | abs_coverage_diff_same_rule | abs_set_size_diff_same_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment_banglabert_full_ft_seed42 | sentiment | banglabert | full_ft | 42 | 0.913235294117647 | 1.6852941176470588 | 0.913970588235294 | 1.6875 | 1.1.0 | 0.913970588235294 | 1.6875 | ok | 0.000735294117647 | 0.0022058823529411 | 0.0 | 0.0 |
| sentiment_banglabert_full_ft_seed123 | sentiment | banglabert | full_ft | 123 | 0.9125 | 1.7102941176470587 | 0.9125 | 1.7110294117647058 | 1.1.0 | 0.9125 | 1.7110294117647058 | ok | 0.0 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_banglabert_full_ft_seed2026 | sentiment | banglabert | full_ft | 2026 | 0.9147058823529411 | 1.675 | 0.9147058823529411 | 1.675735294117647 | 1.1.0 | 0.9147058823529411 | 1.675735294117647 | ok | 0.0 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_banglabert_full_ft_seed7 | sentiment | banglabert | full_ft | 7 | 0.9198529411764704 | 1.6801470588235294 | 0.9198529411764704 | 1.6808823529411765 | 1.1.0 | 0.9198529411764704 | 1.6808823529411765 | ok | 0.0 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_banglabert_full_ft_seed99 | sentiment | banglabert | full_ft | 99 | 0.9036764705882352 | 1.6147058823529412 | 0.9044117647058824 | 1.6213235294117647 | 1.1.0 | 0.9044117647058824 | 1.6213235294117647 | ok | 0.000735294117647 | 0.0066176470588235 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed42 | sentiment | banglabert | lora | 42 | 0.9088235294117648 | 1.6970588235294115 | 0.9088235294117648 | 1.701470588235294 | 1.1.0 | 0.9088235294117648 | 1.701470588235294 | ok | 0.0 | 0.0044117647058823 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed123 | sentiment | banglabert | lora | 123 | 0.9 | 1.6492647058823529 | 0.9007352941176472 | 1.6536764705882352 | 1.1.0 | 0.9007352941176472 | 1.6536764705882352 | ok | 0.000735294117647 | 0.0044117647058823 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed2026 | sentiment | banglabert | lora | 2026 | 0.9022058823529412 | 1.661764705882353 | 0.9022058823529412 | 1.6625 | 1.1.0 | 0.9022058823529412 | 1.6625 | ok | 0.0 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed7 | sentiment | banglabert | lora | 7 | 0.9088235294117648 | 1.6639705882352942 | 0.9088235294117648 | 1.6647058823529413 | 1.1.0 | 0.9088235294117648 | 1.6647058823529413 | ok | 0.0 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed99 | sentiment | banglabert | lora | 99 | 0.9154411764705882 | 1.6639705882352942 | 0.9161764705882351 | 1.6661764705882354 | 1.1.0 | 0.9161764705882351 | 1.6661764705882354 | ok | 0.000735294117647 | 0.0022058823529411 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed42 | sentiment | indicbert | full_ft | 42 | 0.923529411764706 | 1.9264705882352944 | 0.925 | 1.9294117647058824 | 1.1.0 | 0.925 | 1.9294117647058824 | ok | 0.0014705882352941 | 0.0029411764705882 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed123 | sentiment | indicbert | full_ft | 123 | 0.924264705882353 | 1.938235294117647 | 0.9257352941176472 | 1.9419117647058823 | 1.1.0 | 0.9257352941176472 | 1.9419117647058823 | ok | 0.0014705882352941 | 0.0036764705882352 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed2026 | sentiment | indicbert | full_ft | 2026 | 0.913970588235294 | 1.922058823529412 | 0.913970588235294 | 1.924264705882353 | 1.1.0 | 0.913970588235294 | 1.924264705882353 | ok | 0.0 | 0.0022058823529411 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed7 | sentiment | indicbert | full_ft | 7 | 0.9088235294117648 | 1.8507352941176471 | 0.9088235294117648 | 1.8529411764705883 | 1.1.0 | 0.9088235294117648 | 1.8529411764705883 | ok | 0.0 | 0.0022058823529411 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed99 | sentiment | indicbert | full_ft | 99 | 0.9183823529411764 | 1.950735294117647 | 0.9205882352941176 | 1.9580882352941176 | 1.1.0 | 0.9205882352941176 | 1.9580882352941176 | ok | 0.0022058823529411 | 0.0073529411764705 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed42 | sentiment | indicbert | lora | 42 | 0.913970588235294 | 2.0426470588235293 | 0.9147058823529411 | 2.0536764705882358 | 1.1.0 | 0.9147058823529411 | 2.0536764705882358 | ok | 0.000735294117647 | 0.011029411764706 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed123 | sentiment | indicbert | lora | 123 | 0.9044117647058824 | 2.0669117647058823 | 0.9044117647058824 | 2.069117647058824 | 1.1.0 | 0.9044117647058824 | 2.069117647058824 | ok | 0.0 | 0.0022058823529413 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed2026 | sentiment | indicbert | lora | 2026 | 0.8977941176470589 | 1.9286764705882355 | 0.899264705882353 | 1.931617647058824 | 1.1.0 | 0.899264705882353 | 1.931617647058824 | ok | 0.0014705882352941 | 0.0029411764705882 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed7 | sentiment | indicbert | lora | 7 | 0.9073529411764706 | 2.0205882352941176 | 0.9095588235294118 | 2.030147058823529 | 1.1.0 | 0.9095588235294118 | 2.030147058823529 | ok | 0.0022058823529411 | 0.0095588235294119 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed99 | sentiment | indicbert | lora | 99 | 0.8919117647058824 | 2.056617647058824 | 0.8919117647058824 | 2.0573529411764704 | 1.1.0 | 0.8919117647058824 | 2.0573529411764704 | ok | 0.0 | 0.0007352941176468 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed42 | sentiment | xlmr_base | full_ft | 42 | 0.9080882352941176 | 1.7610294117647058 | 0.911029411764706 | 1.776470588235294 | 1.1.0 | 0.911029411764706 | 1.776470588235294 | ok | 0.0029411764705882 | 0.0154411764705881 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed123 | sentiment | xlmr_base | full_ft | 123 | 0.913235294117647 | 1.7691176470588237 | 0.9147058823529411 | 1.777941176470588 | 1.1.0 | 0.9147058823529411 | 1.777941176470588 | ok | 0.0014705882352941 | 0.0088235294117646 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed2026 | sentiment | xlmr_base | full_ft | 2026 | 0.913235294117647 | 1.7661764705882352 | 0.913970588235294 | 1.7742647058823529 | 1.1.0 | 0.913970588235294 | 1.7742647058823529 | ok | 0.000735294117647 | 0.0080882352941176 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed7 | sentiment | xlmr_base | full_ft | 7 | 0.9183823529411764 | 1.8080882352941177 | 0.9198529411764704 | 1.8110294117647057 | 1.1.0 | 0.9198529411764704 | 1.8110294117647057 | ok | 0.0014705882352941 | 0.0029411764705882 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed99 | sentiment | xlmr_base | full_ft | 99 | 0.9183823529411764 | 1.869117647058824 | 0.9183823529411764 | 1.869117647058824 | 1.1.0 | 0.9183823529411764 | 1.869117647058824 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed42 | sentiment | xlmr_base | lora | 42 | 0.9102941176470588 | 1.8294117647058823 | 0.9102941176470588 | 1.8294117647058823 | 1.1.0 | 0.9102941176470588 | 1.8294117647058823 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed123 | sentiment | xlmr_base | lora | 123 | 0.9161764705882351 | 1.8235294117647056 | 0.9161764705882351 | 1.825735294117647 | 1.1.0 | 0.9161764705882351 | 1.825735294117647 | ok | 0.0 | 0.0022058823529411 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed2026 | sentiment | xlmr_base | lora | 2026 | 0.9095588235294118 | 1.8698529411764704 | 0.9095588235294118 | 1.8705882352941177 | 1.1.0 | 0.9095588235294118 | 1.8705882352941177 | ok | 0.0 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed7 | sentiment | xlmr_base | lora | 7 | 0.9176470588235294 | 1.7904411764705883 | 0.9176470588235294 | 1.7977941176470589 | 1.1.0 | 0.9176470588235294 | 1.7977941176470589 | ok | 0.0 | 0.0073529411764705 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed99 | sentiment | xlmr_base | lora | 99 | 0.925 | 1.8941176470588237 | 0.925 | 1.8955882352941176 | 1.1.0 | 0.925 | 1.8955882352941176 | ok | 0.0 | 0.0014705882352941 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed42 | offensive | banglabert | full_ft | 42 | 0.9201609487505296 | 1.0 | 0.9201609487505296 | 1.0 | 1.1.0 | 0.9201609487505296 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed123 | offensive | banglabert | full_ft | 123 | 0.9197373994070308 | 1.0 | 0.9197373994070308 | 1.0 | 1.1.0 | 0.9197373994070308 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed2026 | offensive | banglabert | full_ft | 2026 | 0.9186785260482846 | 1.0 | 0.9186785260482846 | 1.0 | 1.1.0 | 0.9186785260482846 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed7 | offensive | banglabert | full_ft | 7 | 0.9163490046590428 | 1.0 | 0.9163490046590428 | 1.0 | 1.1.0 | 0.9163490046590428 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed99 | offensive | banglabert | full_ft | 99 | 0.9174078780177892 | 1.0 | 0.9174078780177892 | 1.0 | 1.1.0 | 0.9174078780177892 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed42 | offensive | banglabert | lora | 42 | 0.914866581956798 | 1.0 | 0.914866581956798 | 1.0 | 1.1.0 | 0.914866581956798 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed123 | offensive | banglabert | lora | 123 | 0.9203727234222788 | 1.0 | 0.9203727234222788 | 1.0 | 1.1.0 | 0.9203727234222788 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed2026 | offensive | banglabert | lora | 2026 | 0.9133841592545532 | 1.0 | 0.9133841592545532 | 1.0 | 1.1.0 | 0.9133841592545532 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed7 | offensive | banglabert | lora | 7 | 0.9161372299872936 | 1.0 | 0.9161372299872936 | 1.0 | 1.1.0 | 0.9161372299872936 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed99 | offensive | banglabert | lora | 99 | 0.9167725540025412 | 1.0 | 0.9167725540025412 | 1.0 | 1.1.0 | 0.9167725540025412 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed42 | offensive | indicbert | full_ft | 42 | 0.9063955950868277 | 1.0 | 0.9063955950868277 | 1.0 | 1.1.0 | 0.9063955950868277 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed123 | offensive | indicbert | full_ft | 123 | 0.9066073697585768 | 1.0 | 0.9066073697585768 | 1.0 | 1.1.0 | 0.9066073697585768 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed2026 | offensive | indicbert | full_ft | 2026 | 0.9078780177890724 | 1.0 | 0.9078780177890724 | 1.0 | 1.1.0 | 0.9078780177890724 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed7 | offensive | indicbert | full_ft | 7 | 0.9072426937738246 | 1.0 | 0.9072426937738246 | 1.0 | 1.1.0 | 0.9072426937738246 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed99 | offensive | indicbert | full_ft | 99 | 0.910842863193562 | 1.0 | 0.910842863193562 | 1.0 | 1.1.0 | 0.910842863193562 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed42 | offensive | indicbert | lora | 42 | 0.9030072003388396 | 1.0 | 0.9030072003388396 | 1.0 | 1.1.0 | 0.9030072003388396 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed123 | offensive | indicbert | lora | 123 | 0.901736552308344 | 1.0 | 0.901736552308344 | 1.0 | 1.1.0 | 0.901736552308344 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed2026 | offensive | indicbert | lora | 2026 | 0.8983481575603558 | 1.0019059720457433 | 0.8983481575603558 | 1.0019059720457433 | 1.1.0 | 0.8983481575603558 | 1.0019059720457433 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed7 | offensive | indicbert | lora | 7 | 0.8907242693773825 | 1.711139347734011 | 0.8907242693773825 | 1.711139347734011 | 1.1.0 | 0.8907242693773825 | 1.711139347734011 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed99 | offensive | indicbert | lora | 99 | 0.9030072003388396 | 1.0 | 0.9030072003388396 | 1.0 | 1.1.0 | 0.9030072003388396 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed42 | offensive | xlmr_base | full_ft | 42 | 0.9142312579415502 | 1.0 | 0.9142312579415502 | 1.0 | 1.1.0 | 0.9142312579415502 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed123 | offensive | xlmr_base | full_ft | 123 | 0.9171961033460398 | 1.0 | 0.9171961033460398 | 1.0 | 1.1.0 | 0.9171961033460398 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed2026 | offensive | xlmr_base | full_ft | 2026 | 0.913172384582804 | 1.0 | 0.913172384582804 | 1.0 | 1.1.0 | 0.913172384582804 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed7 | offensive | xlmr_base | full_ft | 7 | 0.9176196526895384 | 1.0 | 0.9176196526895384 | 1.0 | 1.1.0 | 0.9176196526895384 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed99 | offensive | xlmr_base | full_ft | 99 | 0.9123252858958067 | 1.0 | 0.9123252858958067 | 1.0 | 1.1.0 | 0.9123252858958067 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed42 | offensive | xlmr_base | lora | 42 | 0.9106310885218128 | 1.0 | 0.9106310885218128 | 1.0 | 1.1.0 | 0.9106310885218128 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed123 | offensive | xlmr_base | lora | 123 | 0.9038542990258364 | 1.0 | 0.9038542990258364 | 1.0 | 1.1.0 | 0.9038542990258364 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed2026 | offensive | xlmr_base | lora | 2026 | 0.9055484963998306 | 1.0 | 0.9055484963998306 | 1.0 | 1.1.0 | 0.9055484963998306 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed7 | offensive | xlmr_base | lora | 7 | 0.9059720457433292 | 1.0 | 0.9059720457433292 | 1.0 | 1.1.0 | 0.9059720457433292 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed99 | offensive | xlmr_base | lora | 99 | 0.904277848369335 | 1.0 | 0.904277848369335 | 1.0 | 1.1.0 | 0.904277848369335 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |


## 3. Extended Calibration Metrics (Adaptive ECE, Classwise ECE, Over/Under-Confidence)

| task | model | ft_method | condition | calibration_regime | nll_mean | top_label_ece_15_mean | adaptive_ece_15_mean | classwise_ece_15_mean | overconfidence_ece_15_mean | underconfidence_ece_15_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.2328 | 0.0183 | 0.0273 | 0.0312 | 0.0158 | 0.0025 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.2325 | 0.0212 | 0.0266 | 0.0313 | 0.011 | 0.0103 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | uncalibrated | 0.3106 | 0.06 | 0.0588 | 0.0609 | 0.0593 | 0.0007 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.4366 | 0.0704 | 0.069 | 0.0812 | 0.0684 | 0.002 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.4081 | 0.0266 | 0.0371 | 0.0663 | 0.0149 | 0.0117 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.6342 | 0.1275 | 0.1243 | 0.129 | 0.1259 | 0.0016 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.2516 | 0.0271 | 0.0317 | 0.0435 | 0.0221 | 0.005 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.2535 | 0.0334 | 0.0333 | 0.0516 | 0.014 | 0.0193 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.3266 | 0.0679 | 0.0674 | 0.0706 | 0.0674 | 0.0005 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | 1.4134 | 0.4058 | 0.4042 | 0.4098 | 0.405 | 0.0008 |
| offensive | banglabert | full_ft | shift_d2_test | target_temp_scaled | 0.7129 | 0.0968 | 0.1047 | 0.1032 | 0.0959 | 0.0008 |
| offensive | banglabert | full_ft | shift_d2_test | uncalibrated | 2.4545 | 0.4553 | 0.4534 | 0.4578 | 0.4544 | 0.0009 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.2613 | 0.025 | 0.032 | 0.0374 | 0.0225 | 0.0025 |
| offensive | banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.3511 | 0.0692 | 0.0688 | 0.0706 | 0.0688 | 0.0004 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.2993 | 0.0301 | 0.0336 | 0.0463 | 0.0287 | 0.0013 |
| offensive | banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.4118 | 0.0793 | 0.0789 | 0.0802 | 0.0791 | 0.0002 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.4107 | 0.0591 | 0.0606 | 0.0771 | 0.0585 | 0.0006 |
| offensive | banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.5972 | 0.1158 | 0.1151 | 0.1172 | 0.1154 | 0.0004 |
| offensive | banglabert | full_ft | test | source_temp_scaled | 0.2194 | 0.0224 | 0.0291 | 0.0285 | 0.0172 | 0.0051 |
| offensive | banglabert | full_ft | test | uncalibrated | 0.2856 | 0.0574 | 0.0572 | 0.0582 | 0.0572 | 0.0002 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | 0.2213 | 0.0128 | 0.0174 | 0.0175 | 0.0117 | 0.0012 |
| offensive | banglabert | lora | shift_d1_barishal_test | target_temp_scaled | 0.2211 | 0.0124 | 0.0151 | 0.017 | 0.0068 | 0.0056 |
| offensive | banglabert | lora | shift_d1_barishal_test | uncalibrated | 0.2793 | 0.0525 | 0.0513 | 0.0532 | 0.0518 | 0.0007 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.412 | 0.0548 | 0.056 | 0.0597 | 0.0544 | 0.0005 |
| offensive | banglabert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.3948 | 0.0249 | 0.0286 | 0.0472 | 0.0172 | 0.0077 |
| offensive | banglabert | lora | shift_d1_chittagong_test | uncalibrated | 0.5505 | 0.1118 | 0.1119 | 0.1134 | 0.1118 | 0.0 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.2456 | 0.0161 | 0.0199 | 0.0294 | 0.0131 | 0.0029 |
| offensive | banglabert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.2461 | 0.018 | 0.0182 | 0.0334 | 0.0096 | 0.0085 |
| offensive | banglabert | lora | shift_d1_noakhali_test | uncalibrated | 0.2996 | 0.0594 | 0.0581 | 0.0623 | 0.0586 | 0.0007 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | 1.261 | 0.3815 | 0.3807 | 0.3829 | 0.3811 | 0.0004 |
| offensive | banglabert | lora | shift_d2_test | target_temp_scaled | 0.7079 | 0.0795 | 0.0968 | 0.0817 | 0.0794 | 0.0001 |
| offensive | banglabert | lora | shift_d2_test | uncalibrated | 2.0596 | 0.4497 | 0.4492 | 0.4512 | 0.4495 | 0.0002 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.2476 | 0.0168 | 0.0217 | 0.0233 | 0.015 | 0.0019 |
| offensive | banglabert | lora | shift_d3_noise05 | uncalibrated | 0.3067 | 0.0602 | 0.0594 | 0.0611 | 0.0597 | 0.0004 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.2941 | 0.0227 | 0.0262 | 0.0334 | 0.0224 | 0.0003 |
| offensive | banglabert | lora | shift_d3_noise10 | uncalibrated | 0.3749 | 0.0731 | 0.0728 | 0.074 | 0.0729 | 0.0002 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.39 | 0.0472 | 0.0476 | 0.0659 | 0.046 | 0.0012 |
| offensive | banglabert | lora | shift_d3_noise20 | uncalibrated | 0.5151 | 0.1057 | 0.1046 | 0.11 | 0.1051 | 0.0006 |
| offensive | banglabert | lora | test | source_temp_scaled | 0.2116 | 0.0134 | 0.0209 | 0.0165 | 0.0119 | 0.0015 |
| offensive | banglabert | lora | test | uncalibrated | 0.2622 | 0.0512 | 0.0512 | 0.0517 | 0.0511 | 0.0001 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.2791 | 0.0341 | 0.0386 | 0.0392 | 0.027 | 0.0071 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.2785 | 0.036 | 0.0372 | 0.041 | 0.0236 | 0.0124 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | uncalibrated | 0.3826 | 0.0782 | 0.078 | 0.0807 | 0.078 | 0.0002 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.4734 | 0.0832 | 0.0829 | 0.0862 | 0.0823 | 0.0008 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.445 | 0.0416 | 0.0474 | 0.0487 | 0.0275 | 0.0141 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.705 | 0.1529 | 0.1514 | 0.1536 | 0.1521 | 0.0007 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.2947 | 0.0392 | 0.0452 | 0.0428 | 0.0304 | 0.0088 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.2948 | 0.0347 | 0.0427 | 0.0446 | 0.0179 | 0.0168 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.3994 | 0.0842 | 0.0835 | 0.0855 | 0.0836 | 0.0005 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | 1.3165 | 0.3983 | 0.3962 | 0.4001 | 0.3972 | 0.001 |
| offensive | indicbert | full_ft | shift_d2_test | target_temp_scaled | 0.7154 | 0.1049 | 0.1132 | 0.11 | 0.1012 | 0.0037 |
| offensive | indicbert | full_ft | shift_d2_test | uncalibrated | 2.3356 | 0.4609 | 0.4602 | 0.4624 | 0.4605 | 0.0004 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.2894 | 0.0369 | 0.039 | 0.0385 | 0.0292 | 0.0077 |
| offensive | indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.3931 | 0.082 | 0.0813 | 0.0825 | 0.0816 | 0.0004 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.3271 | 0.039 | 0.041 | 0.042 | 0.0339 | 0.0051 |
| offensive | indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.4495 | 0.0937 | 0.0931 | 0.0944 | 0.0934 | 0.0003 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.4221 | 0.0611 | 0.0663 | 0.0676 | 0.0608 | 0.0003 |
| offensive | indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.6009 | 0.133 | 0.1327 | 0.134 | 0.1328 | 0.0002 |
| offensive | indicbert | full_ft | test | source_temp_scaled | 0.2508 | 0.0325 | 0.0341 | 0.0345 | 0.0224 | 0.0102 |
| offensive | indicbert | full_ft | test | uncalibrated | 0.3352 | 0.0681 | 0.0675 | 0.0684 | 0.0677 | 0.0004 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | 0.2917 | 0.0218 | 0.0267 | 0.0289 | 0.0169 | 0.0049 |
| offensive | indicbert | lora | shift_d1_barishal_test | target_temp_scaled | 0.2917 | 0.0202 | 0.0263 | 0.0276 | 0.0157 | 0.0046 |
| offensive | indicbert | lora | shift_d1_barishal_test | uncalibrated | 0.3076 | 0.0476 | 0.0476 | 0.05 | 0.0464 | 0.0013 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.4583 | 0.0575 | 0.0594 | 0.0626 | 0.0573 | 0.0003 |
| offensive | indicbert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.4428 | 0.023 | 0.0287 | 0.0373 | 0.0162 | 0.0067 |
| offensive | indicbert | lora | shift_d1_chittagong_test | uncalibrated | 0.5053 | 0.0958 | 0.0954 | 0.0973 | 0.0954 | 0.0004 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.3038 | 0.0311 | 0.0314 | 0.0418 | 0.0223 | 0.0088 |
| offensive | indicbert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.304 | 0.0272 | 0.031 | 0.0393 | 0.0159 | 0.0113 |
| offensive | indicbert | lora | shift_d1_noakhali_test | uncalibrated | 0.3185 | 0.0483 | 0.0501 | 0.0532 | 0.0481 | 0.0002 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | 1.1702 | 0.3355 | 0.3336 | 0.3398 | 0.3346 | 0.0009 |
| offensive | indicbert | lora | shift_d2_test | target_temp_scaled | 0.702 | 0.0562 | 0.0798 | 0.0652 | 0.0511 | 0.0051 |
| offensive | indicbert | lora | shift_d2_test | uncalibrated | 1.4329 | 0.3782 | 0.377 | 0.3797 | 0.3776 | 0.0006 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.297 | 0.0206 | 0.0256 | 0.0267 | 0.0174 | 0.0032 |
| offensive | indicbert | lora | shift_d3_noise05 | uncalibrated | 0.3144 | 0.0479 | 0.0491 | 0.0482 | 0.0475 | 0.0003 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.3386 | 0.0226 | 0.0279 | 0.0285 | 0.0215 | 0.0011 |
| offensive | indicbert | lora | shift_d3_noise10 | uncalibrated | 0.36 | 0.0567 | 0.0574 | 0.0575 | 0.0564 | 0.0003 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.4259 | 0.0389 | 0.0438 | 0.0567 | 0.0375 | 0.0014 |
| offensive | indicbert | lora | shift_d3_noise20 | uncalibrated | 0.4585 | 0.0771 | 0.0767 | 0.0833 | 0.076 | 0.0011 |
| offensive | indicbert | lora | test | source_temp_scaled | 0.258 | 0.0189 | 0.0219 | 0.0227 | 0.0137 | 0.0052 |
| offensive | indicbert | lora | test | uncalibrated | 0.2722 | 0.0399 | 0.041 | 0.0411 | 0.0395 | 0.0004 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.2424 | 0.0166 | 0.025 | 0.0225 | 0.0141 | 0.0024 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.243 | 0.0218 | 0.0231 | 0.0283 | 0.0097 | 0.0121 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | uncalibrated | 0.2992 | 0.0557 | 0.0556 | 0.058 | 0.0552 | 0.0004 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.3915 | 0.0501 | 0.0544 | 0.0546 | 0.05 | 0.0001 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.3814 | 0.0283 | 0.0325 | 0.0412 | 0.0144 | 0.0138 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | uncalibrated | 0.508 | 0.105 | 0.1048 | 0.1072 | 0.1047 | 0.0003 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.2667 | 0.0282 | 0.0325 | 0.0405 | 0.0228 | 0.0054 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.2682 | 0.0305 | 0.0307 | 0.0499 | 0.0138 | 0.0167 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | uncalibrated | 0.3253 | 0.0655 | 0.0651 | 0.0692 | 0.0648 | 0.0007 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 1.2865 | 0.38 | 0.3766 | 0.3825 | 0.3782 | 0.0017 |
| offensive | xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 0.7092 | 0.0851 | 0.0961 | 0.0861 | 0.0802 | 0.0049 |
| offensive | xlmr_base | full_ft | shift_d2_test | uncalibrated | 2.0676 | 0.4332 | 0.4302 | 0.4354 | 0.4317 | 0.0015 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.2478 | 0.0201 | 0.0231 | 0.0227 | 0.0156 | 0.0045 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.3023 | 0.0567 | 0.0565 | 0.057 | 0.0563 | 0.0003 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.2788 | 0.0213 | 0.0262 | 0.0258 | 0.018 | 0.0033 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.3435 | 0.065 | 0.0635 | 0.0661 | 0.0642 | 0.0009 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.3447 | 0.0268 | 0.0309 | 0.0361 | 0.0257 | 0.0012 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.4295 | 0.0808 | 0.0802 | 0.0822 | 0.0804 | 0.0004 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | 0.2232 | 0.0188 | 0.0242 | 0.021 | 0.0145 | 0.0042 |
| offensive | xlmr_base | full_ft | test | uncalibrated | 0.2729 | 0.0521 | 0.052 | 0.0525 | 0.0517 | 0.0004 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | 0.2623 | 0.0177 | 0.0185 | 0.0271 | 0.011 | 0.0067 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | target_temp_scaled | 0.2634 | 0.0189 | 0.0209 | 0.0296 | 0.0066 | 0.0123 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | uncalibrated | 0.2804 | 0.0422 | 0.0414 | 0.0457 | 0.0413 | 0.0009 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | 0.4002 | 0.0344 | 0.0388 | 0.0413 | 0.0339 | 0.0006 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | target_temp_scaled | 0.3956 | 0.0217 | 0.0261 | 0.0338 | 0.0152 | 0.0065 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | uncalibrated | 0.4418 | 0.0783 | 0.0779 | 0.08 | 0.0777 | 0.0006 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | 0.2905 | 0.0207 | 0.0228 | 0.043 | 0.0121 | 0.0087 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | target_temp_scaled | 0.2915 | 0.0198 | 0.0222 | 0.0489 | 0.0061 | 0.0137 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | uncalibrated | 0.3087 | 0.0455 | 0.0455 | 0.0548 | 0.044 | 0.0015 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | 1.2663 | 0.3793 | 0.3763 | 0.3806 | 0.3778 | 0.0015 |
| offensive | xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.6997 | 0.065 | 0.0829 | 0.0656 | 0.0632 | 0.0018 |
| offensive | xlmr_base | lora | shift_d2_test | uncalibrated | 1.7016 | 0.4245 | 0.4226 | 0.4259 | 0.4235 | 0.0009 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.2571 | 0.0128 | 0.014 | 0.0163 | 0.006 | 0.0068 |
| offensive | xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.2729 | 0.0377 | 0.0366 | 0.0387 | 0.0367 | 0.001 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.2929 | 0.0098 | 0.0125 | 0.0171 | 0.0063 | 0.0035 |
| offensive | xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.3126 | 0.0433 | 0.0425 | 0.0454 | 0.0427 | 0.0006 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.351 | 0.0124 | 0.0155 | 0.0216 | 0.0076 | 0.0048 |
| offensive | xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.3727 | 0.0501 | 0.0479 | 0.0526 | 0.0487 | 0.0014 |
| offensive | xlmr_base | lora | test | source_temp_scaled | 0.2372 | 0.0128 | 0.0138 | 0.0174 | 0.0086 | 0.0042 |
| offensive | xlmr_base | lora | test | uncalibrated | 0.2557 | 0.0395 | 0.0378 | 0.0406 | 0.0385 | 0.001 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | 0.6977 | 0.0666 | 0.1041 | 0.0838 | 0.0285 | 0.0381 |
| sentiment | banglabert | full_ft | shift_d1_test | target_temp_scaled | 0.6977 | 0.0642 | 0.1052 | 0.0821 | 0.0272 | 0.037 |
| sentiment | banglabert | full_ft | shift_d1_test | uncalibrated | 0.7741 | 0.1257 | 0.1372 | 0.1091 | 0.1161 | 0.0095 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | 1.0372 | 0.1797 | 0.1828 | 0.1752 | 0.1666 | 0.0131 |
| sentiment | banglabert | full_ft | shift_d2_test | target_temp_scaled | 0.9864 | 0.1539 | 0.1608 | 0.1591 | 0.0939 | 0.06 |
| sentiment | banglabert | full_ft | shift_d2_test | uncalibrated | 1.2582 | 0.2714 | 0.2714 | 0.2252 | 0.271 | 0.0005 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.6954 | 0.045 | 0.0477 | 0.0462 | 0.0241 | 0.0209 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.7828 | 0.1216 | 0.1214 | 0.0892 | 0.1204 | 0.0012 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.7488 | 0.0592 | 0.0636 | 0.0621 | 0.0422 | 0.017 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.8568 | 0.1438 | 0.1443 | 0.106 | 0.1435 | 0.0003 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8249 | 0.0925 | 0.0941 | 0.0891 | 0.0737 | 0.0188 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.9513 | 0.1783 | 0.1799 | 0.1334 | 0.1775 | 0.0008 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | 0.6683 | 0.0509 | 0.053 | 0.0425 | 0.0282 | 0.0227 |
| sentiment | banglabert | full_ft | test | uncalibrated | 0.7555 | 0.1224 | 0.1227 | 0.0855 | 0.1217 | 0.0007 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | 0.699 | 0.0813 | 0.0945 | 0.0826 | 0.0257 | 0.0555 |
| sentiment | banglabert | lora | shift_d1_test | target_temp_scaled | 0.6988 | 0.0816 | 0.0945 | 0.082 | 0.0254 | 0.0562 |
| sentiment | banglabert | lora | shift_d1_test | uncalibrated | 0.7645 | 0.1061 | 0.1124 | 0.1003 | 0.0972 | 0.0089 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | 1.009 | 0.1129 | 0.1239 | 0.149 | 0.0681 | 0.0448 |
| sentiment | banglabert | lora | shift_d2_test | target_temp_scaled | 0.9723 | 0.0974 | 0.1108 | 0.1405 | 0.0442 | 0.0532 |
| sentiment | banglabert | lora | shift_d2_test | uncalibrated | 1.1633 | 0.1673 | 0.1701 | 0.1785 | 0.1505 | 0.0168 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.6899 | 0.0396 | 0.0462 | 0.042 | 0.015 | 0.0246 |
| sentiment | banglabert | lora | shift_d3_noise05 | uncalibrated | 0.7724 | 0.1062 | 0.1074 | 0.0804 | 0.1062 | 0.0 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.739 | 0.0402 | 0.0418 | 0.0508 | 0.0147 | 0.0255 |
| sentiment | banglabert | lora | shift_d3_noise10 | uncalibrated | 0.8234 | 0.1111 | 0.1113 | 0.0875 | 0.1089 | 0.0022 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.8043 | 0.0411 | 0.0514 | 0.066 | 0.0101 | 0.031 |
| sentiment | banglabert | lora | shift_d3_noise20 | uncalibrated | 0.8626 | 0.0981 | 0.0995 | 0.0898 | 0.0976 | 0.0005 |
| sentiment | banglabert | lora | test | source_temp_scaled | 0.6636 | 0.035 | 0.0437 | 0.0395 | 0.0135 | 0.0216 |
| sentiment | banglabert | lora | test | uncalibrated | 0.7528 | 0.1081 | 0.1071 | 0.0809 | 0.107 | 0.001 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | 0.8608 | 0.088 | 0.1252 | 0.0846 | 0.0619 | 0.0261 |
| sentiment | indicbert | full_ft | shift_d1_test | target_temp_scaled | 0.8642 | 0.0957 | 0.1256 | 0.0863 | 0.0713 | 0.0244 |
| sentiment | indicbert | full_ft | shift_d1_test | uncalibrated | 0.9282 | 0.1598 | 0.1654 | 0.1169 | 0.1434 | 0.0164 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | 1.3188 | 0.2866 | 0.2904 | 0.2131 | 0.2752 | 0.0115 |
| sentiment | indicbert | full_ft | shift_d2_test | target_temp_scaled | 1.0659 | 0.0588 | 0.0923 | 0.0703 | 0.0279 | 0.0309 |
| sentiment | indicbert | full_ft | shift_d2_test | uncalibrated | 1.6648 | 0.3772 | 0.3757 | 0.2688 | 0.3654 | 0.0118 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.7918 | 0.0395 | 0.0446 | 0.0414 | 0.0272 | 0.0122 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.8362 | 0.1056 | 0.1068 | 0.0778 | 0.1052 | 0.0004 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.8346 | 0.0515 | 0.0575 | 0.0464 | 0.0443 | 0.0073 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.8977 | 0.131 | 0.1299 | 0.0936 | 0.1292 | 0.0018 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.9169 | 0.0957 | 0.0973 | 0.0858 | 0.0903 | 0.0054 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | uncalibrated | 1.0204 | 0.1794 | 0.1795 | 0.1318 | 0.1788 | 0.0006 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | 0.7669 | 0.046 | 0.0512 | 0.0439 | 0.0244 | 0.0216 |
| sentiment | indicbert | full_ft | test | uncalibrated | 0.7986 | 0.0965 | 0.0958 | 0.0694 | 0.094 | 0.0025 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | 0.8958 | 0.0797 | 0.1149 | 0.0801 | 0.0334 | 0.0464 |
| sentiment | indicbert | lora | shift_d1_test | target_temp_scaled | 0.8966 | 0.0766 | 0.1163 | 0.0827 | 0.0376 | 0.039 |
| sentiment | indicbert | lora | shift_d1_test | uncalibrated | 0.9015 | 0.0953 | 0.1107 | 0.0855 | 0.0503 | 0.045 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | 1.4614 | 0.3011 | 0.3011 | 0.2279 | 0.2975 | 0.0036 |
| sentiment | indicbert | lora | shift_d2_test | target_temp_scaled | 1.0737 | 0.0569 | 0.1262 | 0.0781 | 0.0153 | 0.0416 |
| sentiment | indicbert | lora | shift_d2_test | uncalibrated | 1.5488 | 0.3145 | 0.3147 | 0.2352 | 0.3117 | 0.0028 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.8507 | 0.0469 | 0.0555 | 0.0422 | 0.0254 | 0.0215 |
| sentiment | indicbert | lora | shift_d3_noise05 | uncalibrated | 0.854 | 0.0541 | 0.0583 | 0.0442 | 0.0377 | 0.0164 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.8873 | 0.0544 | 0.0665 | 0.0496 | 0.0386 | 0.0158 |
| sentiment | indicbert | lora | shift_d3_noise10 | uncalibrated | 0.8945 | 0.0622 | 0.0745 | 0.0527 | 0.0517 | 0.0105 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.9547 | 0.069 | 0.0808 | 0.0838 | 0.0586 | 0.0104 |
| sentiment | indicbert | lora | shift_d3_noise20 | uncalibrated | 0.9689 | 0.0799 | 0.095 | 0.0862 | 0.0738 | 0.0061 |
| sentiment | indicbert | lora | test | source_temp_scaled | 0.8339 | 0.0488 | 0.0556 | 0.043 | 0.0232 | 0.0255 |
| sentiment | indicbert | lora | test | uncalibrated | 0.8354 | 0.0579 | 0.0568 | 0.0434 | 0.0363 | 0.0216 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | 0.8069 | 0.0924 | 0.1112 | 0.0848 | 0.0694 | 0.023 |
| sentiment | xlmr_base | full_ft | shift_d1_test | target_temp_scaled | 0.8103 | 0.1032 | 0.1151 | 0.0864 | 0.0834 | 0.0198 |
| sentiment | xlmr_base | full_ft | shift_d1_test | uncalibrated | 0.9002 | 0.1706 | 0.1728 | 0.133 | 0.167 | 0.0036 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 1.0055 | 0.1561 | 0.1598 | 0.173 | 0.1484 | 0.0077 |
| sentiment | xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 0.9403 | 0.0878 | 0.1056 | 0.1381 | 0.0425 | 0.0453 |
| sentiment | xlmr_base | full_ft | shift_d2_test | uncalibrated | 1.2675 | 0.2645 | 0.262 | 0.2254 | 0.2592 | 0.0054 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.731 | 0.0416 | 0.0411 | 0.0398 | 0.0213 | 0.0203 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.786 | 0.1141 | 0.1142 | 0.081 | 0.1138 | 0.0002 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.7672 | 0.0521 | 0.0551 | 0.0442 | 0.0366 | 0.0155 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.8387 | 0.135 | 0.134 | 0.0932 | 0.1345 | 0.0005 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.8166 | 0.0575 | 0.0638 | 0.0542 | 0.0448 | 0.0127 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.9052 | 0.151 | 0.1503 | 0.1059 | 0.1496 | 0.0014 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | 0.7169 | 0.0411 | 0.0451 | 0.0472 | 0.0187 | 0.0224 |
| sentiment | xlmr_base | full_ft | test | uncalibrated | 0.7671 | 0.1096 | 0.108 | 0.0804 | 0.1087 | 0.0009 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | 0.8053 | 0.0837 | 0.105 | 0.0777 | 0.0483 | 0.0354 |
| sentiment | xlmr_base | lora | shift_d1_test | target_temp_scaled | 0.806 | 0.0829 | 0.0997 | 0.0813 | 0.0506 | 0.0323 |
| sentiment | xlmr_base | lora | shift_d1_test | uncalibrated | 0.8286 | 0.1047 | 0.1133 | 0.0911 | 0.0899 | 0.0148 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | 1.0369 | 0.2013 | 0.204 | 0.1945 | 0.1966 | 0.0046 |
| sentiment | xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.9225 | 0.105 | 0.1168 | 0.141 | 0.0617 | 0.0433 |
| sentiment | xlmr_base | lora | shift_d2_test | uncalibrated | 1.1904 | 0.2559 | 0.2582 | 0.2242 | 0.2531 | 0.0028 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.7392 | 0.0353 | 0.044 | 0.0379 | 0.0135 | 0.0218 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.75 | 0.0632 | 0.0641 | 0.0492 | 0.0574 | 0.0058 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.7667 | 0.0323 | 0.041 | 0.0386 | 0.0181 | 0.0142 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.7818 | 0.0668 | 0.0677 | 0.0535 | 0.0657 | 0.0011 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.813 | 0.0399 | 0.0489 | 0.0495 | 0.026 | 0.0139 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.8357 | 0.0799 | 0.0847 | 0.0677 | 0.0773 | 0.0025 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | 0.7266 | 0.032 | 0.0342 | 0.0431 | 0.0103 | 0.0217 |
| sentiment | xlmr_base | lora | test | uncalibrated | 0.7362 | 0.051 | 0.0524 | 0.0517 | 0.0489 | 0.0021 |


## 4. Conformal Score Functions (LAC, APS, RAPS) and Class-Conditional Coverage

| task | model | ft_method | condition | calibration_regime | conformal_method | empirical_coverage_mean | avg_set_size_mean | singleton_rate_mean | trivial_set_rate_mean | worst_class_coverage_mean | class_coverage_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | aps | 0.9438 | 1.0845 | 0.9155 | 0.0845 | 0.9334 | 0.0542 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | lac | 0.9177 | 1.0 | 1.0 | 0.0 | 0.9059 | 0.0297 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | raps | 0.9408 | 1.069 | 0.931 | 0.069 | 0.93 | 0.0516 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8836 | 1.1567 | 0.8433 | 0.1567 | 0.841 | 0.059 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8302 | 1.0 | 1.0 | 0.0 | 0.7834 | 0.1166 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8746 | 1.1275 | 0.8725 | 0.1275 | 0.8318 | 0.0682 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9407 | 1.1067 | 0.8933 | 0.1067 | 0.9196 | 0.0622 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | lac | 0.9025 | 1.0 | 1.0 | 0.0 | 0.8689 | 0.0431 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9337 | 1.0861 | 0.9139 | 0.0861 | 0.9114 | 0.0565 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.5495 | 1.0365 | 0.9635 | 0.0365 | 0.0571 | 0.8429 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.5312 | 1.0 | 1.0 | 0.0 | 0.0194 | 0.8806 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.5432 | 1.0223 | 0.9778 | 0.0222 | 0.044 | 0.856 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9367 | 1.1054 | 0.8946 | 0.1054 | 0.9218 | 0.0509 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9027 | 1.0 | 1.0 | 0.0 | 0.8877 | 0.0173 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9319 | 1.0861 | 0.9139 | 0.0861 | 0.9172 | 0.046 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9262 | 1.1152 | 0.8848 | 0.1152 | 0.9012 | 0.0493 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.8893 | 1.0 | 1.0 | 0.0 | 0.8656 | 0.0344 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.9205 | 1.0944 | 0.9056 | 0.0944 | 0.8952 | 0.0438 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8922 | 1.1462 | 0.8538 | 0.1462 | 0.8296 | 0.0776 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.8448 | 1.0 | 1.0 | 0.0 | 0.7708 | 0.1292 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8851 | 1.1178 | 0.8822 | 0.1178 | 0.82 | 0.0831 |
| offensive | banglabert | full_ft | test | source_temp_scaled | aps | 0.9475 | 1.0901 | 0.9099 | 0.0901 | 0.9367 | 0.0586 |
| offensive | banglabert | full_ft | test | source_temp_scaled | lac | 0.9185 | 1.0 | 1.0 | 0.0 | 0.902 | 0.036 |
| offensive | banglabert | full_ft | test | source_temp_scaled | raps | 0.943 | 1.0723 | 0.9277 | 0.0723 | 0.9311 | 0.0555 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.9501 | 1.1071 | 0.8929 | 0.1071 | 0.9473 | 0.0529 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.9162 | 1.0 | 1.0 | 0.0 | 0.9068 | 0.0259 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.9468 | 1.0911 | 0.9089 | 0.0911 | 0.9432 | 0.0505 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8979 | 1.2024 | 0.7976 | 0.2024 | 0.8524 | 0.05 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8269 | 1.0 | 1.0 | 0.0 | 0.7717 | 0.1283 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8901 | 1.1709 | 0.8291 | 0.1709 | 0.8424 | 0.0576 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9485 | 1.1399 | 0.8601 | 0.1399 | 0.9331 | 0.0644 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.9007 | 1.0 | 1.0 | 0.0 | 0.8692 | 0.0337 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9431 | 1.1202 | 0.8798 | 0.1202 | 0.9275 | 0.0592 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | aps | 0.5767 | 1.1008 | 0.8992 | 0.1007 | 0.1136 | 0.7864 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | lac | 0.525 | 1.0 | 1.0 | 0.0 | 0.0052 | 0.8948 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | raps | 0.562 | 1.067 | 0.933 | 0.067 | 0.0827 | 0.8173 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9445 | 1.1368 | 0.8632 | 0.1368 | 0.9344 | 0.0538 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.9003 | 1.0 | 1.0 | 0.0 | 0.8903 | 0.0105 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9391 | 1.1148 | 0.8852 | 0.1148 | 0.9289 | 0.0484 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9319 | 1.1524 | 0.8476 | 0.1524 | 0.9073 | 0.0545 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8818 | 1.0 | 1.0 | 0.0 | 0.8499 | 0.0501 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9259 | 1.1286 | 0.8714 | 0.1286 | 0.9008 | 0.049 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9039 | 1.1883 | 0.8117 | 0.1883 | 0.8399 | 0.0664 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8398 | 1.0 | 1.0 | 0.0 | 0.7537 | 0.1463 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8945 | 1.1568 | 0.8432 | 0.1568 | 0.826 | 0.0743 |
| offensive | banglabert | lora | test | source_temp_scaled | aps | 0.9512 | 1.111 | 0.889 | 0.111 | 0.9451 | 0.0576 |
| offensive | banglabert | lora | test | source_temp_scaled | lac | 0.9163 | 1.0 | 1.0 | 0.0 | 0.907 | 0.0264 |
| offensive | banglabert | lora | test | source_temp_scaled | raps | 0.9472 | 1.0936 | 0.9064 | 0.0936 | 0.9408 | 0.0539 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | aps | 0.9296 | 1.0985 | 0.9015 | 0.0985 | 0.9249 | 0.0342 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | lac | 0.8953 | 1.0 | 1.0 | 0.0 | 0.8881 | 0.013 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | raps | 0.925 | 1.0799 | 0.9201 | 0.0799 | 0.9193 | 0.0306 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8646 | 1.1643 | 0.8357 | 0.1643 | 0.8335 | 0.0665 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | lac | 0.805 | 1.0 | 1.0 | 0.0 | 0.7733 | 0.1267 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8561 | 1.1352 | 0.8648 | 0.1352 | 0.8252 | 0.0748 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9274 | 1.1174 | 0.8826 | 0.1174 | 0.9179 | 0.037 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | lac | 0.886 | 1.0 | 1.0 | 0.0 | 0.8705 | 0.0295 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9213 | 1.0946 | 0.9054 | 0.0946 | 0.9103 | 0.0326 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.544 | 1.0547 | 0.9452 | 0.0547 | 0.0576 | 0.8424 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.5195 | 1.0 | 1.0 | 0.0 | 0.0173 | 0.8827 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.537 | 1.038 | 0.962 | 0.038 | 0.044 | 0.856 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9265 | 1.1117 | 0.8883 | 0.1117 | 0.9117 | 0.04 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.8888 | 1.0 | 1.0 | 0.0 | 0.8726 | 0.0274 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9214 | 1.0908 | 0.9092 | 0.0908 | 0.9063 | 0.0354 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9177 | 1.1363 | 0.8637 | 0.1363 | 0.9027 | 0.0315 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.8714 | 1.0 | 1.0 | 0.0 | 0.8514 | 0.0486 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.9115 | 1.1133 | 0.8867 | 0.1133 | 0.8943 | 0.0273 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8889 | 1.1783 | 0.8217 | 0.1783 | 0.8743 | 0.0264 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.8215 | 1.0 | 1.0 | 0.0 | 0.8008 | 0.0992 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8795 | 1.1473 | 0.8527 | 0.1473 | 0.8644 | 0.0356 |
| offensive | indicbert | full_ft | test | source_temp_scaled | aps | 0.9381 | 1.0916 | 0.9084 | 0.0916 | 0.9272 | 0.0481 |
| offensive | indicbert | full_ft | test | source_temp_scaled | lac | 0.9078 | 1.0 | 1.0 | 0.0 | 0.8954 | 0.0192 |
| offensive | indicbert | full_ft | test | source_temp_scaled | raps | 0.9345 | 1.076 | 0.924 | 0.076 | 0.9233 | 0.0448 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.9322 | 1.1571 | 0.8429 | 0.1571 | 0.9285 | 0.0361 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.8834 | 1.0012 | 0.9988 | 0.0012 | 0.8746 | 0.0254 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.9286 | 1.1371 | 0.8629 | 0.1371 | 0.9242 | 0.0332 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8843 | 1.2388 | 0.7612 | 0.2388 | 0.8629 | 0.0371 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8019 | 1.0016 | 0.9984 | 0.0016 | 0.7786 | 0.1214 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8762 | 1.2075 | 0.7925 | 0.2075 | 0.854 | 0.046 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9324 | 1.1762 | 0.8238 | 0.1762 | 0.9218 | 0.0433 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8737 | 1.0006 | 0.9994 | 0.0006 | 0.8506 | 0.0494 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9269 | 1.1505 | 0.8495 | 0.1505 | 0.9141 | 0.04 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | aps | 0.6266 | 1.2325 | 0.7675 | 0.2325 | 0.2467 | 0.6533 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | lac | 0.5203 | 1.0006 | 0.9994 | 0.0006 | 0.0576 | 0.8424 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | raps | 0.6062 | 1.1906 | 0.8094 | 0.1906 | 0.2081 | 0.6919 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9305 | 1.159 | 0.841 | 0.159 | 0.9185 | 0.0416 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8813 | 1.001 | 0.999 | 0.001 | 0.8653 | 0.0347 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9252 | 1.134 | 0.866 | 0.134 | 0.9134 | 0.0361 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9206 | 1.1952 | 0.8048 | 0.1952 | 0.9081 | 0.0322 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.86 | 1.0008 | 0.9992 | 0.0008 | 0.8436 | 0.0564 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9136 | 1.1651 | 0.8349 | 0.1651 | 0.9012 | 0.025 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.901 | 1.2652 | 0.7348 | 0.2652 | 0.8917 | 0.0133 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8153 | 1.0022 | 0.9978 | 0.0022 | 0.7971 | 0.1029 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8904 | 1.2291 | 0.7709 | 0.2291 | 0.8794 | 0.0206 |
| offensive | indicbert | lora | test | source_temp_scaled | aps | 0.9394 | 1.1265 | 0.8735 | 0.1265 | 0.9284 | 0.0495 |
| offensive | indicbert | lora | test | source_temp_scaled | lac | 0.9015 | 1.0005 | 0.9995 | 0.0005 | 0.8874 | 0.0156 |
| offensive | indicbert | lora | test | source_temp_scaled | raps | 0.9351 | 1.1068 | 0.8932 | 0.1068 | 0.9236 | 0.0457 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | aps | 0.9402 | 1.1077 | 0.8923 | 0.1077 | 0.931 | 0.0495 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | lac | 0.9091 | 1.0 | 1.0 | 0.0 | 0.8927 | 0.0259 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | raps | 0.9361 | 1.0911 | 0.9089 | 0.0911 | 0.9271 | 0.0453 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8982 | 1.1801 | 0.8199 | 0.1801 | 0.8744 | 0.0302 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | lac | 0.835 | 1.0 | 1.0 | 0.0 | 0.8095 | 0.0905 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8902 | 1.1512 | 0.8488 | 0.1512 | 0.8651 | 0.0359 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9386 | 1.1359 | 0.8641 | 0.1359 | 0.9219 | 0.0556 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8919 | 1.0 | 1.0 | 0.0 | 0.8584 | 0.0416 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9339 | 1.1162 | 0.8838 | 0.1162 | 0.9162 | 0.052 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | aps | 0.5718 | 1.094 | 0.906 | 0.094 | 0.1099 | 0.7901 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | lac | 0.528 | 1.0 | 1.0 | 0.0 | 0.033 | 0.867 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | raps | 0.5605 | 1.0703 | 0.9298 | 0.0702 | 0.088 | 0.812 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9429 | 1.1217 | 0.8783 | 0.1217 | 0.9343 | 0.0509 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9041 | 1.0 | 1.0 | 0.0 | 0.8959 | 0.0119 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9384 | 1.1014 | 0.8986 | 0.1014 | 0.9295 | 0.0466 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.9361 | 1.1402 | 0.8598 | 0.1402 | 0.9225 | 0.0487 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.891 | 1.0 | 1.0 | 0.0 | 0.8753 | 0.0247 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.9308 | 1.1178 | 0.8822 | 0.1178 | 0.916 | 0.0444 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.92 | 1.1876 | 0.8124 | 0.1876 | 0.8981 | 0.0403 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.8599 | 1.0 | 1.0 | 0.0 | 0.8335 | 0.0665 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.9128 | 1.1587 | 0.8413 | 0.1587 | 0.8903 | 0.0336 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | aps | 0.9485 | 1.1012 | 0.8988 | 0.1012 | 0.9446 | 0.0522 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | lac | 0.9149 | 1.0 | 1.0 | 0.0 | 0.9084 | 0.0217 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | raps | 0.9446 | 1.0843 | 0.9157 | 0.0843 | 0.94 | 0.0489 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.9404 | 1.1538 | 0.8462 | 0.1538 | 0.9324 | 0.0487 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.8929 | 1.0 | 1.0 | 0.0 | 0.8841 | 0.0159 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.9351 | 1.1334 | 0.8666 | 0.1334 | 0.9268 | 0.0437 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.9049 | 1.2531 | 0.7469 | 0.2531 | 0.8893 | 0.0201 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.818 | 1.0 | 1.0 | 0.0 | 0.7893 | 0.1107 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8961 | 1.2215 | 0.7785 | 0.2215 | 0.8792 | 0.0208 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9328 | 1.1862 | 0.8138 | 0.1862 | 0.9164 | 0.0495 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8782 | 1.0 | 1.0 | 0.0 | 0.8519 | 0.0481 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9263 | 1.1617 | 0.8383 | 0.1617 | 0.9082 | 0.0449 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | aps | 0.5798 | 1.1085 | 0.8915 | 0.1085 | 0.1262 | 0.7738 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | lac | 0.5247 | 1.0 | 1.0 | 0.0 | 0.0225 | 0.8775 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | raps | 0.5712 | 1.0865 | 0.9135 | 0.0865 | 0.1084 | 0.7916 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9434 | 1.1572 | 0.8428 | 0.1572 | 0.934 | 0.052 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8977 | 1.0 | 1.0 | 0.0 | 0.8835 | 0.0165 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9388 | 1.1343 | 0.8657 | 0.1343 | 0.9286 | 0.0481 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9371 | 1.1832 | 0.8168 | 0.1832 | 0.9263 | 0.0473 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8801 | 1.0 | 1.0 | 0.0 | 0.8618 | 0.0382 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9317 | 1.1581 | 0.8419 | 0.1581 | 0.9201 | 0.0425 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9259 | 1.2529 | 0.7471 | 0.2529 | 0.9121 | 0.0386 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8504 | 1.0 | 1.0 | 0.0 | 0.8239 | 0.0761 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.9185 | 1.2193 | 0.7807 | 0.2193 | 0.9031 | 0.0327 |
| offensive | xlmr_base | lora | test | source_temp_scaled | aps | 0.947 | 1.1296 | 0.8704 | 0.1296 | 0.9399 | 0.0536 |
| offensive | xlmr_base | lora | test | source_temp_scaled | lac | 0.9061 | 1.0 | 1.0 | 0.0 | 0.8965 | 0.0151 |
| offensive | xlmr_base | lora | test | source_temp_scaled | raps | 0.9425 | 1.1106 | 0.8894 | 0.1106 | 0.9356 | 0.049 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | aps | 0.9134 | 1.8408 | 0.2587 | 0.0995 | 0.8852 | 0.0475 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | lac | 0.9234 | 1.7851 | 0.399 | 0.1841 | 0.881 | 0.0506 |
| sentiment | banglabert | full_ft | shift_d1_test | source_temp_scaled | raps | 0.9055 | 1.8219 | 0.2428 | 0.0647 | 0.8707 | 0.0482 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.7749 | 1.9373 | 0.1756 | 0.1129 | 0.4248 | 0.4752 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.8552 | 2.0706 | 0.2142 | 0.2848 | 0.656 | 0.244 |
| sentiment | banglabert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.7604 | 1.9075 | 0.1634 | 0.0709 | 0.3809 | 0.5191 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.8909 | 1.7732 | 0.2965 | 0.0697 | 0.877 | 0.023 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9074 | 1.7268 | 0.4162 | 0.1429 | 0.8677 | 0.0424 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.89 | 1.7649 | 0.2825 | 0.0474 | 0.8771 | 0.0229 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.885 | 1.8216 | 0.2528 | 0.0744 | 0.8578 | 0.0422 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.9056 | 1.7963 | 0.3649 | 0.1612 | 0.8733 | 0.0377 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.88 | 1.8087 | 0.2388 | 0.0475 | 0.8506 | 0.0494 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8824 | 1.9122 | 0.1831 | 0.0953 | 0.8421 | 0.0657 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.9109 | 1.9393 | 0.2765 | 0.2157 | 0.872 | 0.0562 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8753 | 1.8909 | 0.1703 | 0.0612 | 0.8302 | 0.0718 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | aps | 0.899 | 1.7415 | 0.3243 | 0.0657 | 0.8683 | 0.0318 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | lac | 0.9085 | 1.6576 | 0.476 | 0.1337 | 0.8418 | 0.0582 |
| sentiment | banglabert | full_ft | test | source_temp_scaled | raps | 0.8971 | 1.7394 | 0.3057 | 0.0451 | 0.8732 | 0.0277 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | aps | 0.9134 | 1.8706 | 0.2905 | 0.1612 | 0.8863 | 0.0406 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | lac | 0.9164 | 1.7831 | 0.393 | 0.1761 | 0.8839 | 0.0424 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | raps | 0.9095 | 1.8507 | 0.2736 | 0.1244 | 0.8789 | 0.0444 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | aps | 0.8254 | 2.2206 | 0.0731 | 0.2938 | 0.5362 | 0.3659 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | lac | 0.8704 | 2.295 | 0.1259 | 0.4209 | 0.6645 | 0.2753 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | raps | 0.808 | 2.1716 | 0.0612 | 0.2328 | 0.4816 | 0.4184 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9024 | 1.8196 | 0.3029 | 0.1225 | 0.8853 | 0.0244 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.9076 | 1.7241 | 0.4254 | 0.1496 | 0.877 | 0.0341 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8994 | 1.8107 | 0.2859 | 0.0966 | 0.8794 | 0.0265 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9026 | 1.9032 | 0.2454 | 0.1487 | 0.868 | 0.0411 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.9062 | 1.8232 | 0.3582 | 0.1815 | 0.8772 | 0.0393 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8968 | 1.8887 | 0.2266 | 0.1153 | 0.8588 | 0.0466 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9035 | 2.0485 | 0.1529 | 0.2015 | 0.853 | 0.0688 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.9168 | 2.0031 | 0.2531 | 0.2562 | 0.8772 | 0.0454 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8968 | 2.0162 | 0.1407 | 0.1569 | 0.8409 | 0.0709 |
| sentiment | banglabert | lora | test | source_temp_scaled | aps | 0.9079 | 1.7719 | 0.3412 | 0.1131 | 0.8898 | 0.025 |
| sentiment | banglabert | lora | test | source_temp_scaled | lac | 0.9071 | 1.6681 | 0.4675 | 0.1356 | 0.8566 | 0.0442 |
| sentiment | banglabert | lora | test | source_temp_scaled | raps | 0.906 | 1.7635 | 0.3253 | 0.0888 | 0.892 | 0.0239 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | aps | 0.8637 | 2.0418 | 0.1393 | 0.1811 | 0.7963 | 0.1037 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | lac | 0.8706 | 1.9433 | 0.3562 | 0.2995 | 0.8065 | 0.0935 |
| sentiment | indicbert | full_ft | shift_d1_test | source_temp_scaled | raps | 0.8617 | 2.0557 | 0.0965 | 0.1522 | 0.7901 | 0.1099 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | aps | 0.7047 | 1.9323 | 0.2614 | 0.1938 | 0.386 | 0.514 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.6582 | 1.7435 | 0.5813 | 0.3249 | 0.3337 | 0.5663 |
| sentiment | indicbert | full_ft | shift_d2_test | source_temp_scaled | raps | 0.7129 | 1.9779 | 0.1928 | 0.1706 | 0.3836 | 0.5164 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.8887 | 1.9993 | 0.1718 | 0.171 | 0.868 | 0.032 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9066 | 1.911 | 0.3716 | 0.2826 | 0.8591 | 0.0438 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.8882 | 2.0187 | 0.1253 | 0.144 | 0.8628 | 0.0374 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.8816 | 2.0184 | 0.1585 | 0.1769 | 0.845 | 0.0551 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.8853 | 1.9054 | 0.3866 | 0.2921 | 0.8265 | 0.0735 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.8815 | 2.0396 | 0.1121 | 0.1516 | 0.8394 | 0.061 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8475 | 2.0279 | 0.1363 | 0.1643 | 0.7479 | 0.1521 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.8446 | 1.8668 | 0.4172 | 0.284 | 0.7674 | 0.1326 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8472 | 2.0441 | 0.0944 | 0.1385 | 0.7394 | 0.1606 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | aps | 0.8994 | 1.9915 | 0.175 | 0.1665 | 0.8781 | 0.0255 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | lac | 0.9193 | 1.9203 | 0.3535 | 0.2738 | 0.8874 | 0.0415 |
| sentiment | indicbert | full_ft | test | source_temp_scaled | raps | 0.9003 | 2.0129 | 0.131 | 0.144 | 0.8735 | 0.0378 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | aps | 0.8796 | 2.1761 | 0.1045 | 0.2806 | 0.8272 | 0.0809 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | lac | 0.8627 | 2.0826 | 0.202 | 0.2846 | 0.7919 | 0.1081 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | raps | 0.8756 | 2.1602 | 0.0806 | 0.2408 | 0.8099 | 0.0939 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | aps | 0.6366 | 1.7391 | 0.3983 | 0.1373 | 0.2342 | 0.6658 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | lac | 0.5823 | 1.5468 | 0.5789 | 0.1256 | 0.1698 | 0.7302 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | raps | 0.6383 | 1.7622 | 0.351 | 0.1132 | 0.2078 | 0.6922 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.896 | 2.1334 | 0.1359 | 0.2693 | 0.8695 | 0.0522 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8956 | 2.0409 | 0.2449 | 0.2857 | 0.8672 | 0.0402 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8926 | 2.1235 | 0.1132 | 0.2368 | 0.8602 | 0.0621 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.8826 | 2.1482 | 0.1187 | 0.2669 | 0.8234 | 0.0777 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8787 | 2.0469 | 0.2274 | 0.2743 | 0.8288 | 0.0712 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8781 | 2.1343 | 0.0965 | 0.2307 | 0.8099 | 0.0913 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8566 | 2.1846 | 0.0884 | 0.2729 | 0.7252 | 0.1748 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.85 | 2.0604 | 0.2024 | 0.2628 | 0.7319 | 0.1681 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8528 | 2.1678 | 0.0697 | 0.2375 | 0.7113 | 0.1887 |
| sentiment | indicbert | lora | test | source_temp_scaled | aps | 0.9022 | 2.111 | 0.1532 | 0.2643 | 0.8696 | 0.0571 |
| sentiment | indicbert | lora | test | source_temp_scaled | lac | 0.9024 | 2.0222 | 0.2607 | 0.2829 | 0.874 | 0.0342 |
| sentiment | indicbert | lora | test | source_temp_scaled | raps | 0.8991 | 2.1031 | 0.1328 | 0.2359 | 0.862 | 0.0643 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | aps | 0.8816 | 1.9522 | 0.2179 | 0.1701 | 0.8691 | 0.0309 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | lac | 0.8975 | 1.8796 | 0.3502 | 0.2299 | 0.8774 | 0.0271 |
| sentiment | xlmr_base | full_ft | shift_d1_test | source_temp_scaled | raps | 0.8736 | 1.9652 | 0.1741 | 0.1393 | 0.8554 | 0.0481 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | aps | 0.8162 | 1.9157 | 0.2776 | 0.1933 | 0.6564 | 0.2436 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | lac | 0.7806 | 1.7813 | 0.5296 | 0.3109 | 0.4278 | 0.4722 |
| sentiment | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | raps | 0.8244 | 1.9445 | 0.2219 | 0.1664 | 0.6682 | 0.2318 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | aps | 0.9046 | 1.9237 | 0.2207 | 0.1444 | 0.8859 | 0.0228 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | lac | 0.9156 | 1.8228 | 0.3787 | 0.2015 | 0.8859 | 0.0442 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | raps | 0.9063 | 1.9349 | 0.1862 | 0.121 | 0.8858 | 0.0273 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | aps | 0.8982 | 1.9468 | 0.2099 | 0.1566 | 0.891 | 0.0113 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | lac | 0.9063 | 1.8515 | 0.3744 | 0.2259 | 0.8766 | 0.0376 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | raps | 0.8972 | 1.9571 | 0.175 | 0.1321 | 0.8885 | 0.0176 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | aps | 0.8888 | 2.01 | 0.1662 | 0.1762 | 0.861 | 0.039 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | lac | 0.8943 | 1.9003 | 0.3535 | 0.2538 | 0.8546 | 0.0502 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | raps | 0.8862 | 2.0149 | 0.1326 | 0.1475 | 0.8543 | 0.0457 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | aps | 0.9094 | 1.8969 | 0.2394 | 0.1363 | 0.8828 | 0.0337 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | lac | 0.9162 | 1.8143 | 0.3774 | 0.1916 | 0.8874 | 0.0482 |
| sentiment | xlmr_base | full_ft | test | source_temp_scaled | raps | 0.909 | 1.9106 | 0.2056 | 0.1162 | 0.8769 | 0.0351 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | aps | 0.9055 | 2.0209 | 0.21 | 0.2308 | 0.8864 | 0.0246 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | lac | 0.8846 | 1.8896 | 0.3552 | 0.2448 | 0.8277 | 0.0748 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | raps | 0.8935 | 2.008 | 0.1831 | 0.191 | 0.8697 | 0.0314 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | aps | 0.8007 | 1.7575 | 0.4124 | 0.1699 | 0.5615 | 0.3385 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | lac | 0.7545 | 1.6522 | 0.603 | 0.2552 | 0.3626 | 0.5374 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | raps | 0.7998 | 1.752 | 0.3903 | 0.1423 | 0.5679 | 0.3321 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9137 | 1.9515 | 0.2365 | 0.1879 | 0.8915 | 0.0311 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.9137 | 1.8563 | 0.3721 | 0.2284 | 0.8743 | 0.0469 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9104 | 1.9401 | 0.215 | 0.1551 | 0.8928 | 0.0279 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9037 | 1.9841 | 0.2256 | 0.2097 | 0.8678 | 0.0386 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.9068 | 1.8819 | 0.3699 | 0.2518 | 0.8588 | 0.053 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8996 | 1.9662 | 0.2075 | 0.1737 | 0.8733 | 0.0314 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.896 | 2.0522 | 0.1812 | 0.2334 | 0.8436 | 0.0602 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8959 | 1.9141 | 0.3619 | 0.276 | 0.8298 | 0.0784 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8913 | 2.0259 | 0.1637 | 0.1896 | 0.8379 | 0.0621 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | aps | 0.9116 | 1.9196 | 0.2601 | 0.1797 | 0.8901 | 0.0351 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | lac | 0.9156 | 1.8456 | 0.3703 | 0.2159 | 0.8871 | 0.0418 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | raps | 0.9087 | 1.911 | 0.239 | 0.15 | 0.8878 | 0.0305 |

