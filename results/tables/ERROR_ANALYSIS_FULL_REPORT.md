# Complete Error Analysis Report (Handbook Part K.1 – K.5)

## 1. Part K.2 — Human Gold-Standard Agreement & Adjudicated Breakdown

| task | n_total | human_kappa | exact_match_pct | adjudicated_category | count | percentage |
| --- | --- | --- | --- | --- | --- | --- |
| sentiment | 117 | 0.2829 | 42.74 | heavy code-mixing | 39 | 33.33 |
| sentiment | 117 | 0.2829 | 42.74 | other | 30 | 25.64 |
| sentiment | 117 | 0.2829 | 42.74 | negation | 17 | 14.53 |
| sentiment | 117 | 0.2829 | 42.74 | sarcasm | 14 | 11.97 |
| sentiment | 117 | 0.2829 | 42.74 | unclear true label | 10 | 8.55 |
| sentiment | 117 | 0.2829 | 42.74 | dialect-specific word | 7 | 5.98 |
| offensive | 234 | 0.8023 | 85.9 | dialect-specific word | 114 | 48.72 |
| offensive | 234 | 0.8023 | 85.9 | other | 49 | 20.94 |
| offensive | 234 | 0.8023 | 85.9 | heavy code-mixing | 41 | 17.52 |
| offensive | 234 | 0.8023 | 85.9 | negation | 20 | 8.55 |
| offensive | 234 | 0.8023 | 85.9 | unclear true label | 7 | 2.99 |
| offensive | 234 | 0.8023 | 85.9 | sarcasm | 3 | 1.28 |


## 2. Part K.3 — Blind Automated Categorizer Validation vs. Human Gold Standard

| task | category | support_gold | precision | recall | f1_score | overall_ai_kappa | overall_macro_f1 | clears_k3_validation_bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | dialect-specific word | 7 | 0.4286 | 0.4286 | 0.4286 | 0.6065 | 0.5512 | False |
| sentiment | heavy code-mixing | 39 | 0.9512 | 1.0 | 0.975 | 0.6065 | 0.5512 | False |
| sentiment | negation | 17 | 0.7895 | 0.8824 | 0.8333 | 0.6065 | 0.5512 | False |
| sentiment | other | 30 | 0.525 | 0.7 | 0.6 | 0.6065 | 0.5512 | False |
| sentiment | sarcasm | 14 | 0.6667 | 0.1429 | 0.2353 | 0.6065 | 0.5512 | False |
| sentiment | unclear true label | 10 | 0.2857 | 0.2 | 0.2353 | 0.6065 | 0.5512 | False |
| offensive | dialect-specific word | 114 | 0.95 | 1.0 | 0.9744 | 0.8909 | 0.6575 | False |
| offensive | heavy code-mixing | 41 | 0.9318 | 1.0 | 0.9647 | 0.8909 | 0.6575 | False |
| offensive | negation | 20 | 0.9444 | 0.85 | 0.8947 | 0.8909 | 0.6575 | False |
| offensive | other | 49 | 0.88 | 0.898 | 0.8889 | 0.8909 | 0.6575 | False |
| offensive | sarcasm | 3 | 0.0 | 0.0 | 0.0 | 0.8909 | 0.6575 | False |
| offensive | unclear true label | 7 | 0.5 | 0.1429 | 0.2222 | 0.8909 | 0.6575 | False |


## 3. Part K.4 — Pooled Unique Error Scaling Across All 5 Seeds

| task | category | total_5seed_sampled | pooled_unique_errors | category_count | percentage |
| --- | --- | --- | --- | --- | --- |
| sentiment | heavy code-mixing | 780 | 546 | 160 | 29.3 |
| sentiment | other | 780 | 546 | 154 | 28.21 |
| sentiment | negation | 780 | 546 | 102 | 18.68 |
| sentiment | dialect-specific word | 780 | 546 | 94 | 17.22 |
| sentiment | unclear true label | 780 | 546 | 25 | 4.58 |
| sentiment | sarcasm | 780 | 546 | 11 | 2.01 |
| offensive | dialect-specific word | 1170 | 907 | 452 | 49.83 |
| offensive | other | 1170 | 907 | 200 | 22.05 |
| offensive | heavy code-mixing | 1170 | 907 | 155 | 17.09 |
| offensive | negation | 1170 | 907 | 66 | 7.28 |
| offensive | unclear true label | 1170 | 907 | 24 | 2.65 |
| offensive | sarcasm | 1170 | 907 | 10 | 1.1 |


## 4. Part K.5 — Curated Illustrative Examples (Original Bangla + English Translation)

| task | category | error_id | condition | true_label | predicted_label | confidence | original_bangla_text | english_translation | linguistic_mechanism |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sentiment | heavy code-mixing | sentiment_shift_d2_test_123 | shift_d2_test (D2) | Positive | Negative | 0.4348 | I just want to complete this assignment যত দ্রুত সম্ভব but this harsh comments making me down कठोर आलोचना प्राप्त करना जीवन का एक हिस्सा है... নতুন project এর জন্য critical comment পেয়ে আমার আর কাজ করতে ইচ্ছা করতেছে না | I just want to complete this assignment as fast as possible, but these harsh comments are making me down. Receiving harsh criticism is part of life... After getting critical comments on the new project, I no longer feel like working. | Tri-script switching (Bangla + Latin + Devanagari) and phonetic transliteration fragment subwords (2.16 subwords/word), inverting polarity. |
| sentiment | heavy code-mixing | sentiment_shift_d2_test_389 | shift_d2_test (D2) | Positive | Neutral | 0.469 | মেরি প্রমোশন উনলোগণে বিনা কোই ওয়াজঃ কি মেরি জুনিয়র কো দে দিয়া... Time to bounce back and keep striving for greatness... hotash kintu porajito noi. ami safoller jonno chesta chaliye jabo | They gave my promotion to my junior without any reason... Time to bounce back and keep striving for greatness... Disappointed but not defeated; I will keep striving for success. | Hindi-in-Bengali script ('মেরি প্রমোশন উনলোগণে') followed by Romanized Bangla ('hotash kintu porajito noi') obscures the resilient positive conclusion. |
| sentiment | sarcasm | sentiment_D0_source_1116 | D0_source | Neutral | Negative | 0.5237 | সব ঠিক আছে বাট গেস্টিক তারাতারি হবে আরকি | Everything is fine, except you will get gastric ulcers pretty fast, that's all. | Ironic praise ('সব ঠিক আছে' / everything is fine) paired with a sarcastic health jab ('গেস্টিক তারাতারি হবে') confuses literal polarity cues. |
| sentiment | sarcasm | sentiment_D0_source_212 | D0_source | Neutral | Negative | 0.4574 | দাদা আমাদের পুরান ঢাকায় আইসো খাবার কি বুঝবা | Brother, come visit our Old Dhaka—then you will understand what real food actually is! | Colloquial rhetorical challenge ('খাবার কি বুঝবা') is misread by the classifier as hostile/negative rather than an enthusiastic invitation. |
| sentiment | negation | sentiment_D0_source_845 | D0_source | Negative | Neutral | 0.4714 | সেই ক্ষেত্রে , ব্যাংক স্টেট্মেন্ট না দিয়ে , ব্যাংক থেকে পাসপোর্টে ডলার এন্ডোর্স করাটাই বেটার | In that case, instead of giving a bank statement, endorsing dollars in the passport from the bank is better. | Subordinate participial negation ('না দিয়ে' / instead of giving) modifies only the first clause, shifting predicted class toward Neutral. |
| sentiment | negation | sentiment_D0_source_1213 | D0_source | Positive | Negative | 0.4491 | আপনি আমেরিকা থেকে আসার পথে বলছিলেন কাতারের অনেক ভিডিও আছে যেগুলো দেখাবেন । তারপর থাইল্যান্ড ভিডিও আসার কথা । কিন্তু কিছুই তো পাচ্ছিনা | On your way back from America you said you had many Qatar videos to show, and then Thailand videos. But I am not finding anything yet! | Terminal negated verb ('পাচ্ছিনা' / not getting) overrides the enthusiastic fan anticipation in the preceding sentences. |
| sentiment | dialect-specific word | sentiment_shift_d3_noise10_916 | shift_d3_noise10 (D3) | Negative | Positive | 0.4727 | বাংলাদেশের হইলে টানন হেঁচড়া কডতেে হইতো ভাই হাহাহা | If this were in Bangladesh, one would have had to drag and pull them around, brother, hahaha! | Colloquial verb inflection ('হইলে', 'হইতো') combined with character noise ('কডতেে') and 'হাহাহা' causes a false Positive prediction. |
| sentiment | unclear true label | sentiment_D0_source_206 | D0_source | Neutral | Positive | 0.4882 | এইটাকা উদ্যোক্তাদের মধ্যেই ভাগাভাগি হয়েছে । আরেকটু খতিয়ে দেখলে সব কিছু ফাঁস হবে । একটা দিন মজুরের টাকা এই ভাবে যারা মেরে খাই তাদের বিচার করা উচিৎ | This money was divided among the organizers. Looking closely will expose everything. Those who embezzle a day-laborer's money like this should be punished. | Ground-truth label in SentNoB is 'Neutral', even though the text expresses strong moral indignation against embezzlement. |
| offensive | dialect-specific word | offensive_shift_d1_chittagong_test_1284 | shift_d1_chittagong_test (D1) | Offensive | Non-Offensive | 0.8081 | আলার ফুয়া অক্কলরে ফিডা দরহার ।আলার ক্ষেত বিচারক হত্তুন। | These wretched brats need a thrashing! Where did these incompetent judges come from? | Chittagonian lexical items ('আলার ফুয়া', 'অক্কলরে', 'ফিডা দরহার', 'হত্তুন') are unseen in Standard Bangla, causing a high-confidence (0.81) false negative. |
| offensive | other | offensive_shift_d3_noise10_3236 | shift_d3_noise10 (D3) | Offensive | Non-Offensive | 0.7897 | আমি এই নাস্্তিকের বাশিচাই | I demand the hanging [corrupted to 'বাশিচাই'] of this atheist. | 10% synthetic keyboard noise corrupts 'ফাঁসি চাই' (demand hanging) into 'বাশিচাই', masking the violent threat. |

