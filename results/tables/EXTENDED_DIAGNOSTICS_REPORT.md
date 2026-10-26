# Extended Calibration, Conformal, and Representation Diagnostics

## 1. Correlation of Shift Statistics with Miscalibration

| task | model | ft_method | n_conditions | spearman_frag_vs_nll | p_value_frag_vs_nll_exact | pearson_frag_vs_nll | spearman_jaccard_vs_nll | spearman_frag_vs_classwise_ece | spearman_frag_vs_overconfidence | spearman_maha_pct_vs_nll |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | 8 | 0.9048 | 0.0046 | 0.7131 | -0.6429 | 0.9048 | 0.9048 | 0.9048 |
| offensive | banglabert | lora | 8 | 0.9048 | 0.0046 | 0.7332 | -0.6429 | 0.9524 | 0.9048 | 0.9524 |
| offensive | indicbert | full_ft | 8 | 0.9762 | 0.0004 | 0.8055 | -0.8095 | 0.9762 | 0.9762 | 0.9524 |
| offensive | indicbert | lora | 8 | 0.9762 | 0.0004 | 0.795 | -0.8095 | 0.9524 | 0.9762 | 0.881 |
| offensive | xlmr_base | full_ft | 8 | 0.9524 | 0.0011 | 0.6918 | -0.7381 | 0.9524 | 0.9762 | 0.8571 |
| offensive | xlmr_base | lora | 8 | 0.9762 | 0.0004 | 0.6809 | -0.8095 | 0.8571 | 0.9286 | -0.2143 |
| sentiment | banglabert | full_ft | 6 | 1.0 | 0.0028 | 0.8702 | -0.8286 | 0.9429 | 0.9429 | 1.0 |
| sentiment | banglabert | lora | 6 | 1.0 | 0.0028 | 0.8562 | -0.8286 | 0.7714 | 0.8857 | 1.0 |
| sentiment | indicbert | full_ft | 6 | 1.0 | 0.0028 | 0.8033 | -0.8286 | 0.7714 | 0.7714 | 0.8286 |
| sentiment | indicbert | lora | 6 | 0.9429 | 0.0167 | 0.8414 | -0.6571 | 0.0857 | 0.8286 | 0.6571 |
| sentiment | xlmr_base | full_ft | 6 | 0.8286 | 0.0583 | 0.8208 | -1.0 | 0.7714 | 0.8286 | 0.0286 |
| sentiment | xlmr_base | lora | 6 | 0.8286 | 0.0583 | 0.8199 | -1.0 | 0.6 | 0.7714 | -0.0857 |


## 2. Split-Conformal Cross-Check Against MAPIE

| run_id | task | model | ft_method | seed | ours_coverage | ours_set_size | ours_mapie_rule_coverage | ours_mapie_rule_set_size | mapie_version | mapie_coverage | mapie_set_size | status | abs_coverage_diff | abs_set_size_diff | abs_coverage_diff_same_rule | abs_set_size_diff_same_rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment_banglabert_full_ft_seed42 | sentiment | banglabert | full_ft | 42 | 0.9117647058823528 | 1.6573529411764707 | 0.9117647058823528 | 1.660294117647059 | 1.1.0 | 0.9117647058823528 | 1.660294117647059 | ok | 0.0 | 0.0029411764705882 | 0.0 | 0.0 |
| sentiment_banglabert_full_ft_seed123 | sentiment | banglabert | full_ft | 123 | 0.9051470588235294 | 1.6477941176470587 | 0.9088235294117648 | 1.6720588235294118 | 1.1.0 | 0.9088235294117648 | 1.6720588235294118 | ok | 0.0036764705882352 | 0.024264705882353 | 0.0 | 0.0 |
| sentiment_banglabert_full_ft_seed2026 | sentiment | banglabert | full_ft | 2026 | 0.9154411764705882 | 1.6955882352941176 | 0.9154411764705882 | 1.701470588235294 | 1.1.0 | 0.9154411764705882 | 1.701470588235294 | ok | 0.0 | 0.0058823529411764 | 0.0 | 0.0 |
| sentiment_banglabert_full_ft_seed7 | sentiment | banglabert | full_ft | 7 | 0.9198529411764704 | 1.7051470588235291 | 0.9198529411764704 | 1.7051470588235291 | 1.1.0 | 0.9198529411764704 | 1.7051470588235291 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_banglabert_full_ft_seed99 | sentiment | banglabert | full_ft | 99 | 0.9 | 1.598529411764706 | 0.9007352941176472 | 1.6095588235294118 | 1.1.0 | 0.9007352941176472 | 1.6095588235294118 | ok | 0.000735294117647 | 0.0110294117647058 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed42 | sentiment | banglabert | lora | 42 | 0.899264705882353 | 1.6345588235294115 | 0.9 | 1.6419117647058823 | 1.1.0 | 0.9 | 1.6419117647058823 | ok | 0.000735294117647 | 0.0073529411764705 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed123 | sentiment | banglabert | lora | 123 | 0.9125 | 1.6779411764705885 | 0.913235294117647 | 1.678676470588235 | 1.1.0 | 0.913235294117647 | 1.678676470588235 | ok | 0.000735294117647 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed2026 | sentiment | banglabert | lora | 2026 | 0.9088235294117648 | 1.675735294117647 | 0.9088235294117648 | 1.6764705882352942 | 1.1.0 | 0.9088235294117648 | 1.6764705882352942 | ok | 0.0 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed7 | sentiment | banglabert | lora | 7 | 0.8933823529411765 | 1.6191176470588236 | 0.8933823529411765 | 1.6191176470588236 | 1.1.0 | 0.8933823529411765 | 1.6191176470588236 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_banglabert_lora_seed99 | sentiment | banglabert | lora | 99 | 0.9007352941176472 | 1.6544117647058822 | 0.9036764705882352 | 1.661029411764706 | 1.1.0 | 0.9036764705882352 | 1.661029411764706 | ok | 0.0029411764705882 | 0.0066176470588237 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed42 | sentiment | indicbert | full_ft | 42 | 0.924264705882353 | 2.1051470588235293 | 0.924264705882353 | 2.1051470588235293 | 1.1.0 | 0.924264705882353 | 2.1051470588235293 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed123 | sentiment | indicbert | full_ft | 123 | 0.9080882352941176 | 2.065441176470588 | 0.9080882352941176 | 2.065441176470588 | 1.1.0 | 0.9080882352941176 | 2.065441176470588 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed2026 | sentiment | indicbert | full_ft | 2026 | 0.9088235294117648 | 2.115441176470588 | 0.9088235294117648 | 2.115441176470588 | 1.1.0 | 0.9088235294117648 | 2.115441176470588 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed7 | sentiment | indicbert | full_ft | 7 | 0.9227941176470588 | 2.1213235294117645 | 0.9227941176470588 | 2.1227941176470586 | 1.1.0 | 0.9227941176470588 | 2.1227941176470586 | ok | 0.0 | 0.0014705882352941 | 0.0 | 0.0 |
| sentiment_indicbert_full_ft_seed99 | sentiment | indicbert | full_ft | 99 | 0.9176470588235294 | 2.219117647058824 | 0.9176470588235294 | 2.223529411764706 | 1.1.0 | 0.9176470588235294 | 2.223529411764706 | ok | 0.0 | 0.0044117647058823 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed42 | sentiment | indicbert | lora | 42 | 0.8948529411764706 | 2.3713235294117645 | 0.8963235294117647 | 2.373529411764706 | 1.1.0 | 0.8963235294117647 | 2.373529411764706 | ok | 0.0014705882352941 | 0.0022058823529413 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed123 | sentiment | indicbert | lora | 123 | 0.9029411764705882 | 2.425735294117647 | 0.9029411764705882 | 2.425735294117647 | 1.1.0 | 0.9029411764705882 | 2.425735294117647 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed2026 | sentiment | indicbert | lora | 2026 | 0.9051470588235294 | 2.425735294117647 | 0.9058823529411764 | 2.4286764705882358 | 1.1.0 | 0.9058823529411764 | 2.4286764705882358 | ok | 0.000735294117647 | 0.0029411764705882 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed7 | sentiment | indicbert | lora | 7 | 0.9154411764705882 | 2.4808823529411765 | 0.9154411764705882 | 2.4808823529411765 | 1.1.0 | 0.9154411764705882 | 2.4808823529411765 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_indicbert_lora_seed99 | sentiment | indicbert | lora | 99 | 0.8963235294117647 | 2.360294117647059 | 0.8963235294117647 | 2.360294117647059 | 1.1.0 | 0.8963235294117647 | 2.360294117647059 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed42 | sentiment | xlmr_base | full_ft | 42 | 0.9154411764705882 | 1.786764705882353 | 0.9154411764705882 | 1.786764705882353 | 1.1.0 | 0.9154411764705882 | 1.786764705882353 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed123 | sentiment | xlmr_base | full_ft | 123 | 0.9213235294117648 | 1.7727941176470587 | 0.9220588235294118 | 1.775 | 1.1.0 | 0.9220588235294118 | 1.775 | ok | 0.0007352941176471 | 0.0022058823529411 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed2026 | sentiment | xlmr_base | full_ft | 2026 | 0.9154411764705882 | 1.7970588235294118 | 0.9154411764705882 | 1.7977941176470589 | 1.1.0 | 0.9154411764705882 | 1.7977941176470589 | ok | 0.0 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed7 | sentiment | xlmr_base | full_ft | 7 | 0.923529411764706 | 1.8338235294117649 | 0.924264705882353 | 1.8411764705882352 | 1.1.0 | 0.924264705882353 | 1.8411764705882352 | ok | 0.000735294117647 | 0.0073529411764705 | 0.0 | 0.0 |
| sentiment_xlmr_base_full_ft_seed99 | sentiment | xlmr_base | full_ft | 99 | 0.9117647058823528 | 1.7889705882352942 | 0.9125 | 1.7904411764705883 | 1.1.0 | 0.9125 | 1.7904411764705883 | ok | 0.000735294117647 | 0.0014705882352941 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed42 | sentiment | xlmr_base | lora | 42 | 0.9051470588235294 | 1.902941176470588 | 0.9051470588235294 | 1.9044117647058825 | 1.1.0 | 0.9051470588235294 | 1.9044117647058825 | ok | 0.0 | 0.0014705882352941 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed123 | sentiment | xlmr_base | lora | 123 | 0.9147058823529411 | 1.827941176470588 | 0.9161764705882351 | 1.8330882352941176 | 1.1.0 | 0.9161764705882351 | 1.8330882352941176 | ok | 0.0014705882352941 | 0.0051470588235293 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed2026 | sentiment | xlmr_base | lora | 2026 | 0.9198529411764704 | 1.886764705882353 | 0.9198529411764704 | 1.8875 | 1.1.0 | 0.9198529411764704 | 1.8875 | ok | 0.0 | 0.000735294117647 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed7 | sentiment | xlmr_base | lora | 7 | 0.8970588235294118 | 1.7713235294117646 | 0.8970588235294118 | 1.7727941176470587 | 1.1.0 | 0.8970588235294118 | 1.7727941176470587 | ok | 0.0 | 0.0014705882352941 | 0.0 | 0.0 |
| sentiment_xlmr_base_lora_seed99 | sentiment | xlmr_base | lora | 99 | 0.925 | 1.877941176470588 | 0.9257352941176472 | 1.8808823529411764 | 1.1.0 | 0.9257352941176472 | 1.8808823529411764 | ok | 0.000735294117647 | 0.0029411764705882 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed42 | offensive | banglabert | full_ft | 42 | 0.9184667513765352 | 1.0 | 0.9184667513765352 | 1.0 | 1.1.0 | 0.9184667513765352 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed123 | offensive | banglabert | full_ft | 123 | 0.9195256247352817 | 1.0 | 0.9195256247352817 | 1.0 | 1.1.0 | 0.9195256247352817 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed2026 | offensive | banglabert | full_ft | 2026 | 0.9129606099110548 | 1.0 | 0.9129606099110548 | 1.0 | 1.1.0 | 0.9129606099110548 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed7 | offensive | banglabert | full_ft | 7 | 0.9212198221092758 | 1.0 | 0.9212198221092758 | 1.0 | 1.1.0 | 0.9212198221092758 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_full_ft_seed99 | offensive | banglabert | full_ft | 99 | 0.9199491740787802 | 1.0 | 0.9199491740787802 | 1.0 | 1.1.0 | 0.9199491740787802 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed42 | offensive | banglabert | lora | 42 | 0.914866581956798 | 1.0 | 0.914866581956798 | 1.0 | 1.1.0 | 0.914866581956798 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed123 | offensive | banglabert | lora | 123 | 0.9161372299872936 | 1.0 | 0.9161372299872936 | 1.0 | 1.1.0 | 0.9161372299872936 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed2026 | offensive | banglabert | lora | 2026 | 0.9161372299872936 | 1.0 | 0.9161372299872936 | 1.0 | 1.1.0 | 0.9161372299872936 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed7 | offensive | banglabert | lora | 7 | 0.915713680643795 | 1.0 | 0.915713680643795 | 1.0 | 1.1.0 | 0.915713680643795 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_banglabert_lora_seed99 | offensive | banglabert | lora | 99 | 0.9171961033460398 | 1.0 | 0.9171961033460398 | 1.0 | 1.1.0 | 0.9171961033460398 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed42 | offensive | indicbert | full_ft | 42 | 0.9047013977128336 | 1.0406607369758576 | 0.9047013977128336 | 1.0406607369758576 | 1.1.0 | 0.9047013977128336 | 1.0406607369758576 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed123 | offensive | indicbert | full_ft | 123 | 0.8989834815756036 | 1.0252011859381618 | 0.8989834815756036 | 1.0252011859381618 | 1.1.0 | 0.8989834815756036 | 1.0252011859381618 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed2026 | offensive | indicbert | full_ft | 2026 | 0.9027954256670904 | 1.0230834392206691 | 0.9027954256670904 | 1.0230834392206691 | 1.1.0 | 0.9027954256670904 | 1.0230834392206691 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed7 | offensive | indicbert | full_ft | 7 | 0.905124947056332 | 1.0349428208386278 | 0.905124947056332 | 1.0349428208386278 | 1.1.0 | 0.905124947056332 | 1.0349428208386278 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_full_ft_seed99 | offensive | indicbert | full_ft | 99 | 0.9000423549343498 | 1.0294366793731469 | 0.9000423549343498 | 1.0294366793731469 | 1.1.0 | 0.9000423549343498 | 1.0294366793731469 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed42 | offensive | indicbert | lora | 42 | 0.8989834815756036 | 1.1543837357052096 | 0.8989834815756036 | 1.1543837357052096 | 1.1.0 | 0.8989834815756036 | 1.1543837357052096 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed123 | offensive | indicbert | lora | 123 | 0.903430749682338 | 1.2196103346039813 | 0.903430749682338 | 1.2196103346039813 | 1.1.0 | 0.903430749682338 | 1.2196103346039813 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed2026 | offensive | indicbert | lora | 2026 | 0.8966539601863617 | 1.177467174925879 | 0.8966539601863617 | 1.177467174925879 | 1.1.0 | 0.8966539601863617 | 1.177467174925879 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed7 | offensive | indicbert | lora | 7 | 0.9006776789495976 | 1.1802202456586193 | 0.9006776789495976 | 1.1802202456586193 | 1.1.0 | 0.9006776789495976 | 1.1802202456586193 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_indicbert_lora_seed99 | offensive | indicbert | lora | 99 | 0.8981363828886065 | 1.17429055484964 | 0.8981363828886065 | 1.17429055484964 | 1.1.0 | 0.8981363828886065 | 1.17429055484964 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed42 | offensive | xlmr_base | full_ft | 42 | 0.9142312579415502 | 1.0 | 0.9142312579415502 | 1.0 | 1.1.0 | 0.9142312579415502 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed123 | offensive | xlmr_base | full_ft | 123 | 0.9199491740787802 | 1.0 | 0.9199491740787802 | 1.0 | 1.1.0 | 0.9199491740787802 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed2026 | offensive | xlmr_base | full_ft | 2026 | 0.912537060567556 | 1.0 | 0.912537060567556 | 1.0 | 1.1.0 | 0.912537060567556 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed7 | offensive | xlmr_base | full_ft | 7 | 0.9167725540025412 | 1.0 | 0.9167725540025412 | 1.0 | 1.1.0 | 0.9167725540025412 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_full_ft_seed99 | offensive | xlmr_base | full_ft | 99 | 0.9112664125370604 | 1.0 | 0.9112664125370604 | 1.0 | 1.1.0 | 0.9112664125370604 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed42 | offensive | xlmr_base | lora | 42 | 0.9055484963998306 | 1.0 | 0.9055484963998306 | 1.0 | 1.1.0 | 0.9055484963998306 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed123 | offensive | xlmr_base | lora | 123 | 0.905124947056332 | 1.0 | 0.905124947056332 | 1.0 | 1.1.0 | 0.905124947056332 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed2026 | offensive | xlmr_base | lora | 2026 | 0.9063955950868277 | 1.0 | 0.9063955950868277 | 1.0 | 1.1.0 | 0.9063955950868277 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed7 | offensive | xlmr_base | lora | 7 | 0.903430749682338 | 1.0 | 0.903430749682338 | 1.0 | 1.1.0 | 0.903430749682338 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |
| offensive_xlmr_base_lora_seed99 | offensive | xlmr_base | lora | 99 | 0.9080897924608216 | 1.0 | 0.9080897924608216 | 1.0 | 1.1.0 | 0.9080897924608216 | 1.0 | ok | 0.0 | 0.0 | 0.0 | 0.0 |


## 3. Extended Calibration Metrics (Adaptive ECE, Classwise ECE, Over/Under-Confidence)

| task | model | ft_method | condition | calibration_regime | nll_mean | top_label_ece_15_mean | adaptive_ece_15_mean | classwise_ece_15_mean | overconfidence_ece_15_mean | underconfidence_ece_15_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.2333 | 0.021 | 0.0284 | 0.0302 | 0.0168 | 0.0042 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.2332 | 0.0239 | 0.0267 | 0.0323 | 0.0127 | 0.0113 |
| offensive | banglabert | full_ft | shift_d1_barishal_test | uncalibrated | 0.3138 | 0.0576 | 0.0575 | 0.0605 | 0.0572 | 0.0004 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.43 | 0.0672 | 0.068 | 0.0826 | 0.0662 | 0.001 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.4067 | 0.0336 | 0.0395 | 0.0645 | 0.0218 | 0.0119 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.6289 | 0.1228 | 0.1217 | 0.1276 | 0.1222 | 0.0006 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.2439 | 0.0297 | 0.0316 | 0.034 | 0.02 | 0.0097 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.2454 | 0.0303 | 0.0323 | 0.0376 | 0.0127 | 0.0176 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.3198 | 0.0607 | 0.0601 | 0.0628 | 0.0599 | 0.0009 |
| offensive | banglabert | full_ft | shift_d2_test | source_temp_scaled | 1.4578 | 0.4142 | 0.4125 | 0.4153 | 0.4131 | 0.001 |
| offensive | banglabert | full_ft | shift_d2_test | target_temp_scaled | 0.719 | 0.1106 | 0.1216 | 0.1119 | 0.1075 | 0.003 |
| offensive | banglabert | full_ft | shift_d2_test | uncalibrated | 2.5718 | 0.4558 | 0.4541 | 0.4565 | 0.455 | 0.0009 |
| offensive | banglabert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.2643 | 0.0259 | 0.0344 | 0.0381 | 0.0227 | 0.0032 |
| offensive | banglabert | full_ft | shift_d3_noise05 | uncalibrated | 0.3587 | 0.0667 | 0.0672 | 0.0692 | 0.0666 | 0.0001 |
| offensive | banglabert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.3059 | 0.0315 | 0.0371 | 0.0495 | 0.0304 | 0.0011 |
| offensive | banglabert | full_ft | shift_d3_noise10 | uncalibrated | 0.4255 | 0.0796 | 0.0795 | 0.0838 | 0.0794 | 0.0002 |
| offensive | banglabert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.4173 | 0.0653 | 0.0673 | 0.0877 | 0.0652 | 0.0001 |
| offensive | banglabert | full_ft | shift_d3_noise20 | uncalibrated | 0.6094 | 0.1211 | 0.1211 | 0.1269 | 0.121 | 0.0001 |
| offensive | banglabert | full_ft | test | source_temp_scaled | 0.2249 | 0.0225 | 0.0324 | 0.0278 | 0.0174 | 0.0051 |
| offensive | banglabert | full_ft | test | uncalibrated | 0.2996 | 0.0564 | 0.0566 | 0.0575 | 0.0561 | 0.0003 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | 0.2253 | 0.0155 | 0.0194 | 0.0219 | 0.0143 | 0.0012 |
| offensive | banglabert | lora | shift_d1_barishal_test | target_temp_scaled | 0.2248 | 0.0162 | 0.0174 | 0.0219 | 0.009 | 0.0072 |
| offensive | banglabert | lora | shift_d1_barishal_test | uncalibrated | 0.2789 | 0.0484 | 0.0481 | 0.0488 | 0.048 | 0.0004 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.4194 | 0.0525 | 0.0513 | 0.0595 | 0.0509 | 0.0016 |
| offensive | banglabert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.4034 | 0.0239 | 0.0292 | 0.0444 | 0.0142 | 0.0097 |
| offensive | banglabert | lora | shift_d1_chittagong_test | uncalibrated | 0.5391 | 0.1007 | 0.0988 | 0.1029 | 0.0997 | 0.001 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.2471 | 0.0153 | 0.0187 | 0.0305 | 0.011 | 0.0042 |
| offensive | banglabert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.2477 | 0.0158 | 0.0183 | 0.0337 | 0.0066 | 0.0092 |
| offensive | banglabert | lora | shift_d1_noakhali_test | uncalibrated | 0.2938 | 0.0504 | 0.0494 | 0.0568 | 0.0487 | 0.0017 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | 1.1387 | 0.337 | 0.3349 | 0.3375 | 0.3351 | 0.0019 |
| offensive | banglabert | lora | shift_d2_test | target_temp_scaled | 0.7017 | 0.0691 | 0.0864 | 0.0699 | 0.0657 | 0.0034 |
| offensive | banglabert | lora | shift_d2_test | uncalibrated | 1.7133 | 0.3994 | 0.3982 | 0.4003 | 0.3983 | 0.0011 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.2491 | 0.0136 | 0.0182 | 0.0185 | 0.0119 | 0.0018 |
| offensive | banglabert | lora | shift_d3_noise05 | uncalibrated | 0.2987 | 0.0513 | 0.0505 | 0.0518 | 0.0506 | 0.0006 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.2919 | 0.0175 | 0.0196 | 0.0251 | 0.0155 | 0.002 |
| offensive | banglabert | lora | shift_d3_noise10 | uncalibrated | 0.3564 | 0.0596 | 0.0582 | 0.0604 | 0.0587 | 0.0009 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.3815 | 0.0379 | 0.041 | 0.0537 | 0.0362 | 0.0018 |
| offensive | banglabert | lora | shift_d3_noise20 | uncalibrated | 0.4794 | 0.0879 | 0.0886 | 0.091 | 0.0874 | 0.0005 |
| offensive | banglabert | lora | test | source_temp_scaled | 0.2135 | 0.013 | 0.0182 | 0.0173 | 0.011 | 0.002 |
| offensive | banglabert | lora | test | uncalibrated | 0.2571 | 0.0441 | 0.0442 | 0.0449 | 0.0437 | 0.0004 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.3375 | 0.0342 | 0.0398 | 0.0429 | 0.0273 | 0.0069 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.337 | 0.0347 | 0.038 | 0.0443 | 0.0203 | 0.0144 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | uncalibrated | 0.424 | 0.0903 | 0.0883 | 0.0935 | 0.0892 | 0.0011 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.5112 | 0.0959 | 0.0997 | 0.0996 | 0.0947 | 0.0012 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.481 | 0.0492 | 0.0589 | 0.0548 | 0.0329 | 0.0163 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | uncalibrated | 0.711 | 0.1669 | 0.1665 | 0.1672 | 0.1667 | 0.0002 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.3519 | 0.0408 | 0.0475 | 0.046 | 0.0357 | 0.0051 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.3518 | 0.046 | 0.0468 | 0.0548 | 0.0236 | 0.0224 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | uncalibrated | 0.4467 | 0.0992 | 0.0995 | 0.1003 | 0.0992 | 0.0 |
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | 0.9413 | 0.2729 | 0.277 | 0.2757 | 0.2726 | 0.0003 |
| offensive | indicbert | full_ft | shift_d2_test | target_temp_scaled | 0.6943 | 0.0412 | 0.0695 | 0.0495 | 0.0403 | 0.0009 |
| offensive | indicbert | full_ft | shift_d2_test | uncalibrated | 1.3658 | 0.372 | 0.37 | 0.3729 | 0.3698 | 0.0022 |
| offensive | indicbert | full_ft | shift_d3_noise05 | source_temp_scaled | 0.3273 | 0.0374 | 0.0414 | 0.0379 | 0.0298 | 0.0075 |
| offensive | indicbert | full_ft | shift_d3_noise05 | uncalibrated | 0.4139 | 0.0889 | 0.0886 | 0.0901 | 0.0886 | 0.0002 |
| offensive | indicbert | full_ft | shift_d3_noise10 | source_temp_scaled | 0.3727 | 0.0438 | 0.0484 | 0.0474 | 0.0403 | 0.0035 |
| offensive | indicbert | full_ft | shift_d3_noise10 | uncalibrated | 0.4875 | 0.1065 | 0.1052 | 0.107 | 0.1058 | 0.0007 |
| offensive | indicbert | full_ft | shift_d3_noise20 | source_temp_scaled | 0.4468 | 0.0632 | 0.067 | 0.0666 | 0.0625 | 0.0007 |
| offensive | indicbert | full_ft | shift_d3_noise20 | uncalibrated | 0.6015 | 0.1339 | 0.1335 | 0.1343 | 0.1337 | 0.0002 |
| offensive | indicbert | full_ft | test | source_temp_scaled | 0.291 | 0.0314 | 0.0366 | 0.034 | 0.0207 | 0.0107 |
| offensive | indicbert | full_ft | test | uncalibrated | 0.3571 | 0.0747 | 0.0746 | 0.0753 | 0.0745 | 0.0002 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | 0.448 | 0.0296 | 0.0363 | 0.0365 | 0.0258 | 0.0038 |
| offensive | indicbert | lora | shift_d1_barishal_test | target_temp_scaled | 0.4467 | 0.0247 | 0.0322 | 0.0345 | 0.0158 | 0.0089 |
| offensive | indicbert | lora | shift_d1_barishal_test | uncalibrated | 0.4504 | 0.036 | 0.0421 | 0.0379 | 0.0343 | 0.0017 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | 0.6003 | 0.1045 | 0.1056 | 0.1049 | 0.104 | 0.0005 |
| offensive | indicbert | lora | shift_d1_chittagong_test | target_temp_scaled | 0.5618 | 0.0445 | 0.0567 | 0.0599 | 0.0381 | 0.0064 |
| offensive | indicbert | lora | shift_d1_chittagong_test | uncalibrated | 0.612 | 0.1155 | 0.1154 | 0.1156 | 0.115 | 0.0006 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | 0.4544 | 0.0305 | 0.0435 | 0.0367 | 0.0279 | 0.0026 |
| offensive | indicbert | lora | shift_d1_noakhali_test | target_temp_scaled | 0.4543 | 0.0322 | 0.0419 | 0.0395 | 0.0102 | 0.022 |
| offensive | indicbert | lora | shift_d1_noakhali_test | uncalibrated | 0.4568 | 0.0375 | 0.0484 | 0.0405 | 0.0368 | 0.0007 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | 0.8907 | 0.2337 | 0.2366 | 0.2384 | 0.2326 | 0.0011 |
| offensive | indicbert | lora | shift_d2_test | target_temp_scaled | 0.697 | 0.0399 | 0.0657 | 0.0292 | 0.0394 | 0.0005 |
| offensive | indicbert | lora | shift_d2_test | uncalibrated | 0.9134 | 0.2436 | 0.2462 | 0.2472 | 0.2426 | 0.001 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | 0.4319 | 0.0239 | 0.0331 | 0.0328 | 0.0217 | 0.0022 |
| offensive | indicbert | lora | shift_d3_noise05 | uncalibrated | 0.4343 | 0.0314 | 0.0398 | 0.0389 | 0.0308 | 0.0006 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | 0.4582 | 0.0308 | 0.038 | 0.0355 | 0.03 | 0.0008 |
| offensive | indicbert | lora | shift_d3_noise10 | uncalibrated | 0.4619 | 0.0415 | 0.0464 | 0.0432 | 0.0407 | 0.0008 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | 0.5082 | 0.0485 | 0.0532 | 0.05 | 0.0477 | 0.0008 |
| offensive | indicbert | lora | shift_d3_noise20 | uncalibrated | 0.5143 | 0.0591 | 0.0628 | 0.0603 | 0.0584 | 0.0007 |
| offensive | indicbert | lora | test | source_temp_scaled | 0.4048 | 0.0161 | 0.0253 | 0.0248 | 0.0111 | 0.005 |
| offensive | indicbert | lora | test | uncalibrated | 0.4059 | 0.0222 | 0.0311 | 0.03 | 0.0195 | 0.0027 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | source_temp_scaled | 0.2435 | 0.0177 | 0.0266 | 0.025 | 0.0158 | 0.0019 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | target_temp_scaled | 0.2439 | 0.021 | 0.0242 | 0.0301 | 0.01 | 0.0111 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | uncalibrated | 0.2984 | 0.0584 | 0.0578 | 0.0595 | 0.0579 | 0.0005 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | source_temp_scaled | 0.3981 | 0.0549 | 0.0574 | 0.0604 | 0.0548 | 0.0001 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | target_temp_scaled | 0.385 | 0.0264 | 0.031 | 0.0426 | 0.0151 | 0.0113 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | uncalibrated | 0.5154 | 0.1101 | 0.1096 | 0.1114 | 0.1098 | 0.0002 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | source_temp_scaled | 0.2686 | 0.0287 | 0.0332 | 0.0406 | 0.0235 | 0.0052 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | target_temp_scaled | 0.2709 | 0.0317 | 0.0336 | 0.0527 | 0.0113 | 0.0204 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | uncalibrated | 0.3282 | 0.0667 | 0.066 | 0.0695 | 0.0661 | 0.0005 |
| offensive | xlmr_base | full_ft | shift_d2_test | source_temp_scaled | 1.2312 | 0.3748 | 0.3708 | 0.3764 | 0.3724 | 0.0024 |
| offensive | xlmr_base | full_ft | shift_d2_test | target_temp_scaled | 0.7051 | 0.0821 | 0.0949 | 0.0811 | 0.0775 | 0.0046 |
| offensive | xlmr_base | full_ft | shift_d2_test | uncalibrated | 1.9049 | 0.4372 | 0.433 | 0.4391 | 0.4351 | 0.0021 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | source_temp_scaled | 0.2498 | 0.0195 | 0.0225 | 0.0239 | 0.015 | 0.0045 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | uncalibrated | 0.3011 | 0.0576 | 0.0566 | 0.0587 | 0.057 | 0.0006 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | source_temp_scaled | 0.2819 | 0.0213 | 0.0255 | 0.0275 | 0.0191 | 0.0022 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | uncalibrated | 0.3423 | 0.0671 | 0.0667 | 0.0676 | 0.0669 | 0.0002 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | source_temp_scaled | 0.3442 | 0.0264 | 0.0306 | 0.0335 | 0.0259 | 0.0005 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | uncalibrated | 0.4207 | 0.0828 | 0.0823 | 0.0833 | 0.0825 | 0.0002 |
| offensive | xlmr_base | full_ft | test | source_temp_scaled | 0.2253 | 0.0181 | 0.0237 | 0.0231 | 0.0144 | 0.0037 |
| offensive | xlmr_base | full_ft | test | uncalibrated | 0.2725 | 0.0532 | 0.0529 | 0.0542 | 0.0529 | 0.0003 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | 0.2642 | 0.0175 | 0.017 | 0.0308 | 0.0113 | 0.0062 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | target_temp_scaled | 0.2654 | 0.0182 | 0.0177 | 0.0353 | 0.0066 | 0.0116 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | uncalibrated | 0.2849 | 0.0449 | 0.0437 | 0.0488 | 0.0441 | 0.0008 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | 0.4001 | 0.0353 | 0.035 | 0.0413 | 0.0335 | 0.0018 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | target_temp_scaled | 0.3963 | 0.0206 | 0.0252 | 0.0331 | 0.0147 | 0.0058 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | uncalibrated | 0.444 | 0.0793 | 0.0779 | 0.0825 | 0.0786 | 0.0008 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | 0.2925 | 0.0188 | 0.0218 | 0.0524 | 0.0113 | 0.0075 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | target_temp_scaled | 0.2946 | 0.0202 | 0.0237 | 0.0591 | 0.005 | 0.0152 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | uncalibrated | 0.312 | 0.0467 | 0.046 | 0.0592 | 0.0458 | 0.0009 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | 1.2441 | 0.3717 | 0.3694 | 0.3725 | 0.3706 | 0.0011 |
| offensive | xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.6996 | 0.0622 | 0.0684 | 0.0618 | 0.0622 | 0.0 |
| offensive | xlmr_base | lora | shift_d2_test | uncalibrated | 1.6927 | 0.4203 | 0.4187 | 0.4213 | 0.4195 | 0.0008 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.2612 | 0.0129 | 0.0134 | 0.0186 | 0.0066 | 0.0063 |
| offensive | xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.2802 | 0.0403 | 0.0395 | 0.0435 | 0.0395 | 0.0008 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.2973 | 0.0097 | 0.0118 | 0.0192 | 0.006 | 0.0037 |
| offensive | xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.3207 | 0.0448 | 0.044 | 0.0493 | 0.0441 | 0.0007 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.3515 | 0.0095 | 0.0144 | 0.0267 | 0.0057 | 0.0038 |
| offensive | xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.3745 | 0.0501 | 0.0486 | 0.0535 | 0.0494 | 0.0008 |
| offensive | xlmr_base | lora | test | source_temp_scaled | 0.2393 | 0.0127 | 0.0151 | 0.0174 | 0.0091 | 0.0036 |
| offensive | xlmr_base | lora | test | uncalibrated | 0.2611 | 0.0418 | 0.0405 | 0.0437 | 0.041 | 0.0008 |
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
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | 0.7076 | 0.0746 | 0.107 | 0.0885 | 0.0334 | 0.0412 |
| sentiment | banglabert | lora | shift_d1_test | target_temp_scaled | 0.7083 | 0.0893 | 0.1085 | 0.0916 | 0.043 | 0.0463 |
| sentiment | banglabert | lora | shift_d1_test | uncalibrated | 0.7963 | 0.1293 | 0.1321 | 0.1134 | 0.1235 | 0.0058 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | 1.0932 | 0.1327 | 0.1299 | 0.1468 | 0.1285 | 0.0042 |
| sentiment | banglabert | lora | shift_d2_test | target_temp_scaled | 1.026 | 0.0914 | 0.102 | 0.1199 | 0.0591 | 0.0324 |
| sentiment | banglabert | lora | shift_d2_test | uncalibrated | 1.3878 | 0.2498 | 0.2488 | 0.2087 | 0.2493 | 0.0005 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | 0.6905 | 0.0298 | 0.0432 | 0.0435 | 0.0093 | 0.0205 |
| sentiment | banglabert | lora | shift_d3_noise05 | uncalibrated | 0.7886 | 0.1165 | 0.114 | 0.0849 | 0.1149 | 0.0015 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | 0.7482 | 0.0394 | 0.0461 | 0.0538 | 0.0204 | 0.019 |
| sentiment | banglabert | lora | shift_d3_noise10 | uncalibrated | 0.8572 | 0.1308 | 0.1295 | 0.0968 | 0.1302 | 0.0006 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | 0.8171 | 0.051 | 0.0545 | 0.0748 | 0.0246 | 0.0264 |
| sentiment | banglabert | lora | shift_d3_noise20 | uncalibrated | 0.9088 | 0.128 | 0.127 | 0.1062 | 0.1274 | 0.0006 |
| sentiment | banglabert | lora | test | source_temp_scaled | 0.6664 | 0.0412 | 0.0432 | 0.0443 | 0.0191 | 0.0221 |
| sentiment | banglabert | lora | test | uncalibrated | 0.7741 | 0.1222 | 0.1207 | 0.0884 | 0.1214 | 0.0007 |
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
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | 1.0288 | 0.0637 | 0.1083 | 0.0607 | 0.0261 | 0.0376 |
| sentiment | indicbert | lora | shift_d1_test | target_temp_scaled | 1.0299 | 0.0696 | 0.112 | 0.061 | 0.0302 | 0.0394 |
| sentiment | indicbert | lora | shift_d1_test | uncalibrated | 1.0305 | 0.0642 | 0.1097 | 0.0622 | 0.0296 | 0.0346 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | 1.1097 | 0.1027 | 0.1112 | 0.068 | 0.0955 | 0.0072 |
| sentiment | indicbert | lora | shift_d2_test | target_temp_scaled | 1.0698 | 0.0215 | 0.0474 | 0.0235 | 0.0099 | 0.0115 |
| sentiment | indicbert | lora | shift_d2_test | uncalibrated | 1.1174 | 0.1108 | 0.1178 | 0.0735 | 0.1035 | 0.0073 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | 1.0324 | 0.0372 | 0.0627 | 0.0341 | 0.0126 | 0.0246 |
| sentiment | indicbert | lora | shift_d3_noise05 | uncalibrated | 1.0327 | 0.0392 | 0.0634 | 0.0337 | 0.0164 | 0.0228 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | 1.0349 | 0.0367 | 0.0576 | 0.029 | 0.0144 | 0.0223 |
| sentiment | indicbert | lora | shift_d3_noise10 | uncalibrated | 1.0356 | 0.038 | 0.0586 | 0.0302 | 0.0179 | 0.0201 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | 1.0451 | 0.0393 | 0.0546 | 0.0322 | 0.0185 | 0.0208 |
| sentiment | indicbert | lora | shift_d3_noise20 | uncalibrated | 1.0467 | 0.0409 | 0.0594 | 0.0326 | 0.0222 | 0.0188 |
| sentiment | indicbert | lora | test | source_temp_scaled | 1.0287 | 0.0391 | 0.0618 | 0.0332 | 0.0095 | 0.0296 |
| sentiment | indicbert | lora | test | uncalibrated | 1.0287 | 0.0408 | 0.0609 | 0.0345 | 0.0131 | 0.0277 |
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
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | 0.8226 | 0.0781 | 0.0945 | 0.0838 | 0.0462 | 0.0319 |
| sentiment | xlmr_base | lora | shift_d1_test | target_temp_scaled | 0.8251 | 0.0806 | 0.1018 | 0.0843 | 0.054 | 0.0266 |
| sentiment | xlmr_base | lora | shift_d1_test | uncalibrated | 0.8459 | 0.0953 | 0.1189 | 0.095 | 0.082 | 0.0134 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | 1.0564 | 0.2099 | 0.2096 | 0.195 | 0.2049 | 0.0049 |
| sentiment | xlmr_base | lora | shift_d2_test | target_temp_scaled | 0.9391 | 0.0988 | 0.1093 | 0.1315 | 0.0624 | 0.0364 |
| sentiment | xlmr_base | lora | shift_d2_test | uncalibrated | 1.2067 | 0.2625 | 0.2592 | 0.2243 | 0.258 | 0.0045 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | 0.7485 | 0.0333 | 0.0399 | 0.0405 | 0.0088 | 0.0245 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | uncalibrated | 0.7589 | 0.054 | 0.0595 | 0.0535 | 0.0458 | 0.0082 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | 0.7738 | 0.035 | 0.0431 | 0.0411 | 0.0179 | 0.017 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | uncalibrated | 0.7866 | 0.0646 | 0.0686 | 0.0549 | 0.0598 | 0.0048 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | 0.8192 | 0.0362 | 0.0404 | 0.0503 | 0.0225 | 0.0137 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | uncalibrated | 0.835 | 0.0677 | 0.0697 | 0.064 | 0.0663 | 0.0014 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | 0.7399 | 0.0281 | 0.0382 | 0.0491 | 0.0093 | 0.0188 |
| sentiment | xlmr_base | lora | test | uncalibrated | 0.7508 | 0.0529 | 0.056 | 0.0563 | 0.0478 | 0.0051 |


## 4. Conformal Score Functions (LAC, APS, RAPS) and Class-Conditional Coverage

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
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.9499 | 1.1077 | 0.8923 | 0.1077 | 0.9467 | 0.0532 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.9136 | 1.0 | 1.0 | 0.0 | 0.9067 | 0.0206 |
| offensive | banglabert | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.9469 | 1.0932 | 0.9068 | 0.0932 | 0.9436 | 0.0502 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8993 | 1.2228 | 0.7772 | 0.2228 | 0.8606 | 0.0449 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8219 | 1.0 | 1.0 | 0.0 | 0.7768 | 0.1232 |
| offensive | banglabert | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8911 | 1.1939 | 0.8061 | 0.1939 | 0.8507 | 0.0493 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9482 | 1.1476 | 0.8524 | 0.1476 | 0.9344 | 0.0624 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.9013 | 1.0 | 1.0 | 0.0 | 0.87 | 0.0337 |
| offensive | banglabert | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.9427 | 1.126 | 0.874 | 0.126 | 0.9277 | 0.0581 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | aps | 0.6447 | 1.24 | 0.76 | 0.24 | 0.2602 | 0.6398 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | lac | 0.5222 | 1.0 | 1.0 | 0.0 | 0.0152 | 0.8848 |
| offensive | banglabert | lora | shift_d2_test | source_temp_scaled | raps | 0.6262 | 1.1995 | 0.8005 | 0.1995 | 0.222 | 0.678 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.945 | 1.1457 | 0.8543 | 0.1457 | 0.9356 | 0.0536 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8997 | 1.0 | 1.0 | 0.0 | 0.8903 | 0.0101 |
| offensive | banglabert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9399 | 1.1235 | 0.8765 | 0.1235 | 0.9302 | 0.0489 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9352 | 1.1661 | 0.8339 | 0.1661 | 0.9141 | 0.0545 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8847 | 1.0 | 1.0 | 0.0 | 0.8571 | 0.0429 |
| offensive | banglabert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9294 | 1.1421 | 0.8579 | 0.1421 | 0.9072 | 0.05 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9106 | 1.2175 | 0.7825 | 0.2175 | 0.8576 | 0.0597 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.839 | 1.0 | 1.0 | 0.0 | 0.758 | 0.142 |
| offensive | banglabert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.9019 | 1.1842 | 0.8158 | 0.1842 | 0.8444 | 0.0633 |
| offensive | banglabert | lora | test | source_temp_scaled | aps | 0.952 | 1.116 | 0.884 | 0.116 | 0.9467 | 0.0576 |
| offensive | banglabert | lora | test | source_temp_scaled | lac | 0.916 | 1.0 | 1.0 | 0.0 | 0.9056 | 0.0272 |
| offensive | banglabert | lora | test | source_temp_scaled | raps | 0.9479 | 1.0991 | 0.9009 | 0.0991 | 0.943 | 0.0532 |
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
| offensive | indicbert | full_ft | shift_d2_test | source_temp_scaled | lac | 0.5775 | 1.0988 | 0.9012 | 0.0987 | 0.2074 | 0.6926 |
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
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.8932 | 1.2883 | 0.7117 | 0.2883 | 0.8877 | 0.0144 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.8879 | 1.2196 | 0.7804 | 0.2196 | 0.8795 | 0.0205 |
| offensive | indicbert | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.8889 | 1.2672 | 0.7328 | 0.2672 | 0.883 | 0.017 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.8378 | 1.3029 | 0.6971 | 0.3029 | 0.7617 | 0.1383 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8157 | 1.2233 | 0.7767 | 0.2233 | 0.7225 | 0.1775 |
| offensive | indicbert | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8278 | 1.2788 | 0.7212 | 0.2788 | 0.7461 | 0.1539 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.8964 | 1.3033 | 0.6967 | 0.3033 | 0.8823 | 0.0255 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8855 | 1.2243 | 0.7757 | 0.2243 | 0.8646 | 0.0385 |
| offensive | indicbert | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.889 | 1.2766 | 0.7234 | 0.2766 | 0.8732 | 0.0281 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | aps | 0.7543 | 1.5165 | 0.4835 | 0.5165 | 0.6047 | 0.2953 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | lac | 0.7245 | 1.4658 | 0.5342 | 0.4658 | 0.5264 | 0.3736 |
| offensive | indicbert | lora | shift_d2_test | source_temp_scaled | raps | 0.7392 | 1.4888 | 0.5112 | 0.4888 | 0.5809 | 0.3191 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.8995 | 1.2733 | 0.7267 | 0.2733 | 0.8655 | 0.0374 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8888 | 1.1939 | 0.8061 | 0.1939 | 0.8475 | 0.0525 |
| offensive | indicbert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8937 | 1.2504 | 0.7496 | 0.2504 | 0.8582 | 0.0418 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.8903 | 1.2846 | 0.7154 | 0.2846 | 0.8544 | 0.0456 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8817 | 1.2065 | 0.7935 | 0.2065 | 0.8381 | 0.0619 |
| offensive | indicbert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8848 | 1.2613 | 0.7387 | 0.2613 | 0.8458 | 0.0542 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8754 | 1.3122 | 0.6878 | 0.3122 | 0.8393 | 0.0607 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8635 | 1.2318 | 0.7682 | 0.2318 | 0.8166 | 0.0834 |
| offensive | indicbert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8689 | 1.2879 | 0.7121 | 0.2879 | 0.8311 | 0.0689 |
| offensive | indicbert | lora | test | source_temp_scaled | aps | 0.9081 | 1.261 | 0.739 | 0.261 | 0.8765 | 0.0381 |
| offensive | indicbert | lora | test | source_temp_scaled | lac | 0.8996 | 1.1812 | 0.8188 | 0.1812 | 0.8617 | 0.0391 |
| offensive | indicbert | lora | test | source_temp_scaled | raps | 0.9034 | 1.2385 | 0.7615 | 0.2385 | 0.8695 | 0.0374 |
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
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | aps | 0.9393 | 1.1524 | 0.8476 | 0.1524 | 0.9269 | 0.052 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | lac | 0.8919 | 1.0 | 1.0 | 0.0 | 0.8745 | 0.0255 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | source_temp_scaled | raps | 0.9355 | 1.1336 | 0.8664 | 0.1336 | 0.9232 | 0.048 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | aps | 0.9057 | 1.255 | 0.745 | 0.255 | 0.8989 | 0.0123 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | lac | 0.8174 | 1.0 | 1.0 | 0.0 | 0.8069 | 0.0931 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | source_temp_scaled | raps | 0.8955 | 1.221 | 0.779 | 0.221 | 0.8882 | 0.0128 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | aps | 0.9345 | 1.1907 | 0.8093 | 0.1907 | 0.9146 | 0.0548 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | lac | 0.8757 | 1.0 | 1.0 | 0.0 | 0.8417 | 0.0583 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | source_temp_scaled | raps | 0.929 | 1.1665 | 0.8335 | 0.1665 | 0.9082 | 0.0503 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | aps | 0.5892 | 1.125 | 0.875 | 0.125 | 0.1482 | 0.7518 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | lac | 0.5232 | 1.0 | 1.0 | 0.0 | 0.0267 | 0.8733 |
| offensive | xlmr_base | lora | shift_d2_test | source_temp_scaled | raps | 0.5752 | 1.0947 | 0.9052 | 0.0947 | 0.1209 | 0.7791 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.943 | 1.1584 | 0.8416 | 0.1584 | 0.937 | 0.0486 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.8962 | 1.0 | 1.0 | 0.0 | 0.8876 | 0.0124 |
| offensive | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.938 | 1.1344 | 0.8656 | 0.1344 | 0.932 | 0.0436 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9365 | 1.1874 | 0.8126 | 0.1874 | 0.9302 | 0.0424 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8799 | 1.0 | 1.0 | 0.0 | 0.8705 | 0.0295 |
| offensive | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.9305 | 1.1606 | 0.8394 | 0.1606 | 0.9233 | 0.0372 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.9264 | 1.2558 | 0.7442 | 0.2558 | 0.9196 | 0.0326 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8491 | 1.0 | 1.0 | 0.0 | 0.8413 | 0.0587 |
| offensive | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.9186 | 1.2224 | 0.7776 | 0.2224 | 0.9121 | 0.0247 |
| offensive | xlmr_base | lora | test | source_temp_scaled | aps | 0.9463 | 1.1304 | 0.8696 | 0.1304 | 0.9411 | 0.0511 |
| offensive | xlmr_base | lora | test | source_temp_scaled | lac | 0.9057 | 1.0 | 1.0 | 0.0 | 0.8991 | 0.0122 |
| offensive | xlmr_base | lora | test | source_temp_scaled | raps | 0.9415 | 1.1108 | 0.8892 | 0.1108 | 0.9363 | 0.0465 |
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
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | aps | 0.9134 | 1.8428 | 0.3005 | 0.1433 | 0.8879 | 0.0372 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | lac | 0.9134 | 1.7692 | 0.4 | 0.1692 | 0.8736 | 0.0468 |
| sentiment | banglabert | lora | shift_d1_test | source_temp_scaled | raps | 0.9045 | 1.8229 | 0.2915 | 0.1144 | 0.8791 | 0.0334 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | aps | 0.7811 | 2.094 | 0.141 | 0.2351 | 0.5547 | 0.3453 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | lac | 0.8095 | 2.1219 | 0.2057 | 0.3276 | 0.6271 | 0.2729 |
| sentiment | banglabert | lora | shift_d2_test | source_temp_scaled | raps | 0.7644 | 2.0498 | 0.1323 | 0.1821 | 0.5179 | 0.3821 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.9012 | 1.8009 | 0.3076 | 0.1085 | 0.8852 | 0.0222 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.9046 | 1.7143 | 0.4197 | 0.134 | 0.877 | 0.036 |
| sentiment | banglabert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8987 | 1.786 | 0.2966 | 0.0826 | 0.8798 | 0.025 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.8863 | 1.8688 | 0.2538 | 0.1226 | 0.8515 | 0.0488 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8954 | 1.7972 | 0.3562 | 0.1534 | 0.862 | 0.0481 |
| sentiment | banglabert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8824 | 1.8476 | 0.2434 | 0.091 | 0.8417 | 0.0583 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8915 | 2.0013 | 0.1613 | 0.1626 | 0.8448 | 0.0628 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.9021 | 1.9541 | 0.2559 | 0.21 | 0.8568 | 0.0593 |
| sentiment | banglabert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8853 | 1.9691 | 0.1507 | 0.1199 | 0.8328 | 0.0722 |
| sentiment | banglabert | lora | test | source_temp_scaled | aps | 0.9062 | 1.7594 | 0.3457 | 0.1051 | 0.8806 | 0.0343 |
| sentiment | banglabert | lora | test | source_temp_scaled | lac | 0.9015 | 1.6491 | 0.474 | 0.1231 | 0.8569 | 0.0431 |
| sentiment | banglabert | lora | test | source_temp_scaled | raps | 0.9021 | 1.7479 | 0.3334 | 0.0813 | 0.8821 | 0.027 |
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
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | aps | 0.8935 | 2.4328 | 0.006 | 0.4388 | 0.5 | 0.4 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | lac | 0.8836 | 2.3264 | 0.1502 | 0.4766 | 0.581 | 0.319 |
| sentiment | indicbert | lora | shift_d1_test | source_temp_scaled | raps | 0.8935 | 2.4348 | 0.003 | 0.4378 | 0.5 | 0.4 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | aps | 0.847 | 2.3261 | 0.0045 | 0.3306 | 0.3465 | 0.5535 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | lac | 0.7266 | 1.8781 | 0.1886 | 0.0667 | 0.062 | 0.838 |
| sentiment | indicbert | lora | shift_d2_test | source_temp_scaled | raps | 0.8468 | 2.3256 | 0.0005 | 0.3261 | 0.3412 | 0.5588 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.8954 | 2.4907 | 0.005 | 0.4957 | 0.5723 | 0.3277 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.9006 | 2.4129 | 0.116 | 0.529 | 0.6535 | 0.2465 |
| sentiment | indicbert | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.8959 | 2.4915 | 0.001 | 0.4925 | 0.5711 | 0.3289 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.8931 | 2.4837 | 0.0071 | 0.4907 | 0.5606 | 0.3394 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.8938 | 2.3916 | 0.1179 | 0.5096 | 0.6351 | 0.2649 |
| sentiment | indicbert | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8922 | 2.4859 | 0.0021 | 0.4879 | 0.5557 | 0.3443 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8909 | 2.4819 | 0.0056 | 0.4875 | 0.552 | 0.348 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8912 | 2.3972 | 0.1135 | 0.5107 | 0.6258 | 0.2742 |
| sentiment | indicbert | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8907 | 2.4822 | 0.0015 | 0.4837 | 0.5483 | 0.3517 |
| sentiment | indicbert | lora | test | source_temp_scaled | aps | 0.8971 | 2.4903 | 0.0054 | 0.4957 | 0.5785 | 0.3215 |
| sentiment | indicbert | lora | test | source_temp_scaled | lac | 0.9028 | 2.411 | 0.1166 | 0.5276 | 0.6634 | 0.2366 |
| sentiment | indicbert | lora | test | source_temp_scaled | raps | 0.8972 | 2.4916 | 0.0015 | 0.4931 | 0.5766 | 0.3234 |
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
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | aps | 0.8836 | 1.995 | 0.2249 | 0.2199 | 0.87 | 0.03 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | lac | 0.8925 | 1.9035 | 0.3502 | 0.2537 | 0.8506 | 0.0638 |
| sentiment | xlmr_base | lora | shift_d1_test | source_temp_scaled | raps | 0.8766 | 1.9771 | 0.205 | 0.1821 | 0.8467 | 0.0533 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | aps | 0.7863 | 1.7878 | 0.3861 | 0.1739 | 0.5615 | 0.3385 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | lac | 0.741 | 1.6463 | 0.5935 | 0.2398 | 0.3476 | 0.5524 |
| sentiment | xlmr_base | lora | shift_d2_test | source_temp_scaled | raps | 0.7841 | 1.7848 | 0.358 | 0.1428 | 0.5754 | 0.3246 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | aps | 0.905 | 1.9621 | 0.2274 | 0.1894 | 0.8691 | 0.0352 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | lac | 0.911 | 1.8653 | 0.3626 | 0.2279 | 0.8686 | 0.0455 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | source_temp_scaled | raps | 0.9035 | 1.9497 | 0.2088 | 0.1585 | 0.8711 | 0.0336 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | aps | 0.9024 | 2.0021 | 0.2134 | 0.2154 | 0.8725 | 0.0289 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | lac | 0.9097 | 1.9082 | 0.3518 | 0.26 | 0.861 | 0.0476 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | source_temp_scaled | raps | 0.8982 | 1.9829 | 0.1938 | 0.1768 | 0.8724 | 0.029 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | aps | 0.8957 | 2.0721 | 0.1706 | 0.2426 | 0.8396 | 0.0604 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | lac | 0.8969 | 1.9356 | 0.351 | 0.2866 | 0.8446 | 0.0656 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | source_temp_scaled | raps | 0.8918 | 2.0484 | 0.1537 | 0.2021 | 0.8394 | 0.0606 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | aps | 0.9054 | 1.9356 | 0.2478 | 0.1834 | 0.8683 | 0.0417 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | lac | 0.9124 | 1.8541 | 0.3616 | 0.2157 | 0.8772 | 0.0492 |
| sentiment | xlmr_base | lora | test | source_temp_scaled | raps | 0.9038 | 1.9253 | 0.2275 | 0.1528 | 0.8684 | 0.0417 |


## 5. Mahalanobis Distance Percentiles Relative to the Calibration Split

| task | model | ft_method | condition | mean_maha_percentile_mean | median_maha_percentile_mean | frac_above_95th_ref_pct_mean | maha_error_gap_pct_mean | spearman_maha_nll_rho_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offensive | banglabert | full_ft | shift_d1_barishal_test | 53.0628 | 53.0733 | 0.0885 | 26.5027 | 0.3952 |
| offensive | banglabert | full_ft | shift_d1_chittagong_test | 59.402 | 62.6833 | 0.116 | 19.8369 | 0.531 |
| offensive | banglabert | full_ft | shift_d1_noakhali_test | 53.5302 | 54.841 | 0.0913 | 26.2452 | 0.495 |
| offensive | banglabert | full_ft | shift_d2_test | 86.5929 | 88.1369 | 0.3728 | -1.0028 | 0.019 |
| offensive | banglabert | full_ft | shift_d3_noise05 | 55.527 | 58.262 | 0.1014 | 25.7367 | 0.4248 |
| offensive | banglabert | full_ft | shift_d3_noise10 | 57.4177 | 60.0805 | 0.1011 | 22.2492 | 0.4629 |
| offensive | banglabert | full_ft | shift_d3_noise20 | 63.5659 | 67.5922 | 0.1184 | 18.1679 | 0.5246 |
| offensive | banglabert | full_ft | test | 54.9203 | 57.9716 | 0.1044 | 27.8574 | 0.3498 |
| offensive | banglabert | lora | shift_d1_barishal_test | 56.2967 | 58.4231 | 0.153 | 28.0912 | 0.4557 |
| offensive | banglabert | lora | shift_d1_chittagong_test | 64.2825 | 71.827 | 0.1579 | 20.8114 | 0.5627 |
| offensive | banglabert | lora | shift_d1_noakhali_test | 57.4483 | 61.8398 | 0.153 | 24.3876 | 0.5807 |
| offensive | banglabert | lora | shift_d2_test | 97.7625 | 99.3069 | 0.86 | -0.5624 | -0.0036 |
| offensive | banglabert | lora | shift_d3_noise05 | 59.742 | 64.8792 | 0.1616 | 26.1395 | 0.5499 |
| offensive | banglabert | lora | shift_d3_noise10 | 61.9846 | 67.8805 | 0.1589 | 23.6763 | 0.5929 |
| offensive | banglabert | lora | shift_d3_noise20 | 67.4529 | 74.7563 | 0.1642 | 19.4982 | 0.5926 |
| offensive | banglabert | lora | test | 56.7898 | 60.017 | 0.1567 | 28.2163 | 0.4438 |
| offensive | indicbert | full_ft | shift_d1_barishal_test | 54.6349 | 55.9093 | 0.0861 | 14.7481 | 0.2497 |
| offensive | indicbert | full_ft | shift_d1_chittagong_test | 55.4954 | 57.5074 | 0.0977 | 11.7601 | 0.2755 |
| offensive | indicbert | full_ft | shift_d1_noakhali_test | 55.2303 | 57.3591 | 0.0898 | 15.2869 | 0.261 |
| offensive | indicbert | full_ft | shift_d2_test | 73.3247 | 75.9517 | 0.3222 | 1.4524 | 0.079 |
| offensive | indicbert | full_ft | shift_d3_noise05 | 54.1146 | 55.5935 | 0.0891 | 14.9223 | 0.2174 |
| offensive | indicbert | full_ft | shift_d3_noise10 | 54.9629 | 57.0941 | 0.0964 | 13.0411 | 0.2366 |
| offensive | indicbert | full_ft | shift_d3_noise20 | 56.4225 | 58.8279 | 0.1069 | 12.4614 | 0.2833 |
| offensive | indicbert | full_ft | test | 53.7386 | 54.9958 | 0.0859 | 15.1905 | 0.1782 |
| offensive | indicbert | lora | shift_d1_barishal_test | 55.8224 | 59.3048 | 0.0817 | 5.8022 | 0.0452 |
| offensive | indicbert | lora | shift_d1_chittagong_test | 57.4181 | 60.7037 | 0.0826 | 3.061 | -0.0084 |
| offensive | indicbert | lora | shift_d1_noakhali_test | 57.005 | 60.9538 | 0.0928 | 5.3017 | 0.0371 |
| offensive | indicbert | lora | shift_d2_test | 95.6472 | 98.1009 | 0.785 | 0.7309 | 0.0349 |
| offensive | indicbert | lora | shift_d3_noise05 | 55.4606 | 58.1878 | 0.0916 | 7.3928 | 0.07 |
| offensive | indicbert | lora | shift_d3_noise10 | 55.5901 | 57.908 | 0.0865 | 7.0238 | 0.0817 |
| offensive | indicbert | lora | shift_d3_noise20 | 56.8032 | 59.3769 | 0.0868 | 5.8666 | 0.088 |
| offensive | indicbert | lora | test | 55.1669 | 57.8889 | 0.0925 | 7.6095 | 0.0592 |
| offensive | xlmr_base | full_ft | shift_d1_barishal_test | 50.7483 | 51.0089 | 0.0749 | 16.4465 | 0.2664 |
| offensive | xlmr_base | full_ft | shift_d1_chittagong_test | 54.5337 | 56.8673 | 0.0784 | 16.4793 | 0.3661 |
| offensive | xlmr_base | full_ft | shift_d1_noakhali_test | 50.9934 | 51.4116 | 0.0732 | 14.3081 | 0.3251 |
| offensive | xlmr_base | full_ft | shift_d2_test | 57.8241 | 56.1742 | 0.0357 | -5.8966 | -0.1445 |
| offensive | xlmr_base | full_ft | shift_d3_noise05 | 53.8048 | 54.9407 | 0.0888 | 17.6117 | 0.2998 |
| offensive | xlmr_base | full_ft | shift_d3_noise10 | 54.1176 | 55.7037 | 0.093 | 17.5791 | 0.335 |
| offensive | xlmr_base | full_ft | shift_d3_noise20 | 55.5096 | 57.6007 | 0.0908 | 16.1273 | 0.3748 |
| offensive | xlmr_base | full_ft | test | 53.4455 | 54.2433 | 0.092 | 17.8815 | 0.2493 |
| offensive | xlmr_base | lora | shift_d1_barishal_test | 49.4208 | 48.2026 | 0.078 | -6.3805 | -0.1295 |
| offensive | xlmr_base | lora | shift_d1_chittagong_test | 46.5808 | 44.3832 | 0.0709 | -3.8859 | -0.1731 |
| offensive | xlmr_base | lora | shift_d1_noakhali_test | 48.8016 | 47.872 | 0.0696 | -4.927 | -0.106 |
| offensive | xlmr_base | lora | shift_d2_test | 63.4173 | 66.2293 | 0.0722 | -3.2316 | -0.0894 |
| offensive | xlmr_base | lora | shift_d3_noise05 | 54.3307 | 55.195 | 0.1096 | -5.7762 | -0.1333 |
| offensive | xlmr_base | lora | shift_d3_noise10 | 52.5373 | 53.1984 | 0.0972 | -5.1187 | -0.1537 |
| offensive | xlmr_base | lora | shift_d3_noise20 | 49.296 | 48.7643 | 0.0742 | -3.9327 | -0.1894 |
| offensive | xlmr_base | lora | test | 56.5253 | 58.6668 | 0.1316 | -7.2036 | -0.1135 |
| sentiment | banglabert | full_ft | shift_d1_test | 78.0239 | 98.8668 | 0.594 | 18.6766 | 0.4521 |
| sentiment | banglabert | full_ft | shift_d2_test | 99.8977 | 100.0 | 0.994 | 0.1649 | 0.0897 |
| sentiment | banglabert | full_ft | shift_d3_noise05 | 77.8471 | 99.1759 | 0.5793 | 18.2814 | 0.4609 |
| sentiment | banglabert | full_ft | shift_d3_noise10 | 81.6228 | 99.5143 | 0.6309 | 14.2257 | 0.3837 |
| sentiment | banglabert | full_ft | shift_d3_noise20 | 88.2355 | 99.9558 | 0.7499 | 10.7087 | 0.3292 |
| sentiment | banglabert | full_ft | test | 76.2474 | 98.7344 | 0.5571 | 20.6389 | 0.4911 |
| sentiment | banglabert | lora | shift_d1_test | 85.4867 | 100.0 | 0.7204 | 16.5289 | 0.5394 |
| sentiment | banglabert | lora | shift_d2_test | 99.7398 | 100.0 | 0.9893 | 0.4236 | 0.1505 |
| sentiment | banglabert | lora | shift_d3_noise05 | 83.9272 | 100.0 | 0.6935 | 15.7226 | 0.5245 |
| sentiment | banglabert | lora | shift_d3_noise10 | 87.5609 | 100.0 | 0.7469 | 12.0286 | 0.455 |
| sentiment | banglabert | lora | shift_d3_noise20 | 92.2946 | 100.0 | 0.8357 | 8.2583 | 0.3957 |
| sentiment | banglabert | lora | test | 82.0117 | 100.0 | 0.6743 | 17.377 | 0.5415 |
| sentiment | indicbert | full_ft | shift_d1_test | 83.3483 | 93.3333 | 0.5831 | 7.944 | 0.3085 |
| sentiment | indicbert | full_ft | shift_d2_test | 99.7287 | 100.0 | 0.9851 | -0.1228 | 0.005 |
| sentiment | indicbert | full_ft | shift_d3_noise05 | 79.2967 | 92.8182 | 0.5241 | 8.0977 | 0.2439 |
| sentiment | indicbert | full_ft | shift_d3_noise10 | 79.7949 | 92.6637 | 0.5262 | 7.4973 | 0.2135 |
| sentiment | indicbert | full_ft | shift_d3_noise20 | 81.8911 | 94.1575 | 0.5482 | 5.9526 | 0.1882 |
| sentiment | indicbert | full_ft | test | 78.5861 | 92.0235 | 0.5128 | 8.4592 | 0.2443 |
| sentiment | indicbert | lora | shift_d1_test | 64.7454 | 70.1104 | 0.3701 | -3.7598 | -0.1931 |
| sentiment | indicbert | lora | shift_d2_test | 95.5034 | 95.688 | 0.805 | -0.4742 | -0.0264 |
| sentiment | indicbert | lora | shift_d3_noise05 | 58.4998 | 59.1244 | 0.2884 | -9.1824 | -0.2784 |
| sentiment | indicbert | lora | shift_d3_noise10 | 59.8661 | 61.7807 | 0.3081 | -8.7073 | -0.256 |
| sentiment | indicbert | lora | shift_d3_noise20 | 60.6286 | 63.5099 | 0.3131 | -7.4827 | -0.2336 |
| sentiment | indicbert | lora | test | 58.1655 | 58.624 | 0.2882 | -9.4603 | -0.2924 |
| sentiment | xlmr_base | full_ft | shift_d1_test | 69.0383 | 80.4415 | 0.3592 | 19.9871 | 0.4495 |
| sentiment | xlmr_base | full_ft | shift_d2_test | 76.0837 | 83.6939 | 0.3301 | 10.8813 | 0.2665 |
| sentiment | xlmr_base | full_ft | shift_d3_noise05 | 69.5749 | 82.8477 | 0.3647 | 21.3781 | 0.5146 |
| sentiment | xlmr_base | full_ft | shift_d3_noise10 | 67.8398 | 78.1383 | 0.3243 | 20.6396 | 0.4825 |
| sentiment | xlmr_base | full_ft | shift_d3_noise20 | 66.3269 | 74.312 | 0.2799 | 18.3732 | 0.4253 |
| sentiment | xlmr_base | full_ft | test | 72.0096 | 87.6233 | 0.4115 | 21.8196 | 0.5298 |
| sentiment | xlmr_base | lora | shift_d1_test | 65.1248 | 71.6998 | 0.2915 | -1.9819 | -0.0375 |
| sentiment | xlmr_base | lora | shift_d2_test | 79.605 | 90.6549 | 0.4493 | -7.3469 | -0.2445 |
| sentiment | xlmr_base | lora | shift_d3_noise05 | 71.6772 | 82.8477 | 0.375 | -1.919 | -0.0854 |
| sentiment | xlmr_base | lora | shift_d3_noise10 | 67.816 | 76.7918 | 0.3269 | -2.8346 | -0.1083 |
| sentiment | xlmr_base | lora | shift_d3_noise20 | 62.2619 | 66.1957 | 0.2607 | -3.4244 | -0.121 |
| sentiment | xlmr_base | lora | test | 75.8223 | 89.5953 | 0.4313 | -2.664 | -0.0735 |

