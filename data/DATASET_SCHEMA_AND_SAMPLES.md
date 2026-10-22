# Dataset Schema, Split Counts, and Illustrative Samples

This document provides the exact schema, row counts, label encodings, and representative samples for all datasets in `data/processed/` so reviewers and automated tools can inspect the data structure without loading multi-megabyte raw CSV files.

## 1. Split Overview & Exact Row Counts

| task | file_name | n_rows | columns |
| --- | --- | --- | --- |
| sentiment | data/processed/sentiment/cal.csv | 1359 | text, labels, text_clean |
| sentiment | data/processed/sentiment/d1_dialectal.csv | 401 | source_d0_index, source_d0_text, translated_text, text_clean, labels |
| sentiment | data/processed/sentiment/d1_dialectal_cal.csv | 200 | source_d0_index, source_d0_text, translated_text, text_clean, labels |
| sentiment | data/processed/sentiment/d1_dialectal_test.csv | 201 | source_d0_index, source_d0_text, translated_text, text_clean, labels |
| sentiment | data/processed/sentiment/d2_codemixed_cal.csv | 201 | text_clean, labels |
| sentiment | data/processed/sentiment/d2_codemixed_test.csv | 804 | text_clean, labels |
| sentiment | data/processed/sentiment/synthetic_noise_05pct.csv | 1360 | text, labels, text_clean |
| sentiment | data/processed/sentiment/synthetic_noise_10pct.csv | 1360 | text, labels, text_clean |
| sentiment | data/processed/sentiment/synthetic_noise_20pct.csv | 1360 | text, labels, text_clean |
| sentiment | data/processed/sentiment/test.csv | 1360 | text, labels, text_clean |
| sentiment | data/processed/sentiment/train.csv | 9520 | text, labels, text_clean |
| sentiment | data/processed/sentiment/val.csv | 1361 | text, labels, text_clean |
| offensive | data/processed/offensive/cal.csv | 4718 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/d1_barishal_cal.csv | 611 | text_clean, labels |
| offensive | data/processed/offensive/d1_barishal_test.csv | 2445 | text_clean, labels |
| offensive | data/processed/offensive/d1_chittagong_cal.csv | 610 | text_clean, labels |
| offensive | data/processed/offensive/d1_chittagong_test.csv | 2443 | text_clean, labels |
| offensive | data/processed/offensive/d1_noakhali_cal.csv | 611 | text_clean, labels |
| offensive | data/processed/offensive/d1_noakhali_test.csv | 2445 | text_clean, labels |
| offensive | data/processed/offensive/d2_codemixed_cal.csv | 200 | text_clean, labels |
| offensive | data/processed/offensive/d2_codemixed_test.csv | 800 | text_clean, labels |
| offensive | data/processed/offensive/synthetic_noise_05pct.csv | 4722 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/synthetic_noise_10pct.csv | 4722 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/synthetic_noise_20pct.csv | 4722 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/test.csv | 4722 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/train.csv | 33057 | text, target, type, labels, text_clean |
| offensive | data/processed/offensive/val.csv | 4723 | text, target, type, labels, text_clean |

## 2. Representative 3-Row Samples per Condition

### `sentiment/test.csv`

| text | labels | text_clean |
| --- | --- | --- |
| আল্লাহ্ তুমি আমাদের বাংলাদেশে সঠিক ইসলামীক সাশন কায়েম করার তৌফিক দাও আমিন | 1 | আল্লাহ্ তুমি আমাদের বাংলাদেশে সঠিক ইসলামীক সাশন কায়েম করার তৌফিক দাও আমিন |
| দয়া করে এই অভিযান বন্দ করবেন না | 0 | দয়া করে এই অভিযান বন্দ করবেন না |
| লজ্জা না থাকলে যা হয় তাদের চাকরি থেকে বহিস্কার করা হওক | 2 | লজ্জা না থাকলে যা হয় তাদের চাকরি থেকে বহিস্কার করা হওক |

### `sentiment/d1_dialectal_test.csv`

| source_d0_index | source_d0_text | translated_text | text_clean | labels |
| --- | --- | --- | --- | --- |
| 298 | হত্যা , ঘুষ , সুদ , ডাকাতি , ভরে গেছে | হত্যা , ঘুষ , সুদ , ডাকাতি , ভরি গেছে | হত্যা , ঘুষ , সুদ , ডাকাতি , ভরি গেছে | 2 |
| 70 | বেচরামের দেউরিতে আপনাদের সবাইকে দেখে খুব ভাল লাগল । সবার ভিডিও আমার খুব প্রিয় সাথে ঐ নান্না বিরিয়ানিও | বেচরামর দেউরিত আপনারার সবরে দেখিয়া বউত ভালা লাগল । সবর ভিডিও আমার বউত প্রিয় লগে অউ নান্না বিরিয়ানিও | বেচরামর দেউরিত আপনারার সবরে দেখিয়া বউত ভালা লাগল । সবর ভিডিও আমার বউত প্রিয় লগে অউ নান্না বিরিয়ানিও | 1 |
| 115 | নীলা মার্কেট ঢাকার কোথায় | নীলা মার্কেট ঢাকার কুনানো | নীলা মার্কেট ঢাকার কুনানো | 1 |

### `sentiment/d2_codemixed_test.csv`

| text_clean | labels |
| --- | --- |
| একটি ভয় বা ব্যক্তিগত chellenge অতিক্রম করার জন্য অভিনন্দন! খুদকি বিকাস অর জিবান মে সাফালতা পানেকে লিয়ে খুদ কো আগে বারহানা চাহিয়ে अपने डर का सामना करने और खुद को अपने कम्फर्ट जोन से बाहर निकालने के लिए आपको खुद पर गर्व महसूस करना चाहिए। The feeling of accomplishment that comes with overcoming a challenge can be incredibly fulfilling and আপনার আত্মবিশ্বাসকে বাড়িয়ে তুলতে পারে।. nijer subidhar jayga theke ber hoye notun kichu korar jonno onek sahos ar proyojon hoy | 1 |
| আজ, ম্যায়নে আপনে নাই ডিভাইস কে লিয়ে উপযোগকর্তা পুস্তক পড়নে মে কিছু সময় বিতায়া অর মুঝে বাস্তব মে ইয়া কাফি জানকারীপূর্ন অর মাদাদাগার লাগা। যদিও এটি সবচেয়ে exciting task নাও হতে পারে, আমাদের যন্ত্রপাতি এবং devices কীভাবে কাজ করে তা বোঝার জন্য সময় নেওয়া দীর্ঘমেয়াদে আমাদের অনেক হতাশা এবং সময় বাঁচাতে পারে। I am grateful for the resources available to us, such as user ম্যানুয়াল, that can help us maximize the potential of our ডিভাইস and ensure they are functioning properly. यहाँ उन उपकरणों को सीखने और समझने के लिए समय निकालना है जिनका हम दैनिक आधार पर उपयोग करते हैं, और हमारे लिए उपलब्ध तकनीक का अधिकतम लाभ उठाते हैं। Asuna sikhate ebam br̥d'dhi karara icchara sathe natuna cyalenjagulira sathe yogayoga caliye ya'i ebam pratiti kajake byaktigata bikasa ebam atma-unnatira suyoga hisabe dekhi. | 0 |
| প্রধান কি পার বিশ্বাস না কর সক্ত! সবী পরিকল্পনা অউর প্রতিয়াশাওঁ ক্যা বাদ, কর্মক্রম কো শেষ সময় ম্যায় রাদ্দ কার দিয় জাত হ্যায়! সময় অর অরজ কাইয়াই বারবাদাইয়াই কি হ্যায়। আমি সবাইকে দেখে খুব excited ছিলাম এবং now সব কিছুর জন্য Why do these জিনিস সবসময় happen to me? मैं अभी बहुत निराश और निराश हूं। Hatasara dirgha talikara madhye eti arekati patana." | 2 |

### `sentiment/synthetic_noise_10pct.csv`

| text | labels | text_clean |
| --- | --- | --- |
| আল্লাহ্ তুমি আমাদের বাংলাদেশে সঠিক ইসলামীক সাশন কায়েম করার তৌফিক দাও আমিন | 1 | আল্লাহ্ তুমি আমাদের বাংলাাধেশে সঠিক ইসসলামীক সাশব কায়েম করার তৌফিক দাও আমিন |
| দয়া করে এই অভিযান বন্দ করবেন না | 0 | দয়া করে এই ভিযান বন্দ করবেন না |
| লজ্জা না থাকলে যা হয় তাদের চাকরি থেকে বহিস্কার করা হওক | 2 | লজ্জা নাথাকলে যা হয় তাের চাকরি থেকে বহিসকার করা হওক |

### `offensive/test.csv`

| text | target | type | labels | text_clean |
| --- | --- | --- | --- | --- |
| ঔ খানকির পোলারা কই গান কয় | group | slander | 1 | ঔ খানকির পোলারা কই গান কয় |
| পাপনের দুই গালে জুতা মরো তালে তালে। | male | callToViolence | 1 | পাপনের দুই গালে জুতা মরো তালে তালে। |
| আল্লাহ উপর ভরসা কর | nan | nan | 0 | আল্লাহ উপর ভরসা কর |

### `offensive/d1_chittagong_test.csv`

| text_clean | labels |
| --- | --- |
| কমেন্ট পরিত আসসি।এককা সাইড দন | 0 |
| বস, চালায় জন। | 0 |
| বয়ন বয়ন, বয় জন গয় ইন ,হনডিক্কা জিকির আবার? | 0 |

### `offensive/d2_codemixed_test.csv`

| text_clean | labels |
| --- | --- |
| রাজনীতি কি খুব কি সোয়ার্থ কে লিয়ে নির্বাচন কি ছবি ইচাও কো জ্যাদা মান দিয়া যত হে সরকারি কর্মকর্তারা রাজনৈতিক লাভের চেয়ে people over political gain দেবেন এই কামনা করছি। Feeling powerless in the face of a government shutdown or budget impasse and দৈনন্দিন জীবনে এর প্রভাব आश्चर्य है कि क्या राजनीतिक व्यवस्था टूट गई है और महत्वपूर्ण सुधार की आवश्यकता है। asa kora jay je rajnitibid ra tader motamot ke ak jaygay ak sathe rekhe kaj korbe | 1 |
| Excited to স্বেচ্ছাসেবক at the community service project আগামীকাল! समुदाय को वापस देना और जरूरतमंद लोगों की मदद करना बहुत अच्छा लगता है। টি ওয়ার্ল্ড নীডস মোর পিপল ওহ অরে উইলিং তো লেন্ড এ হেল্পিং হ্যান্ড. অন্যদের help করার জন্য একত্রিত হওয়া একটি community এর অংশ হতে পেরে গর্বিত। hamaare pados par sakaaraatmak prabhaav daalane ke lie tatpar hain. | 0 |
| জাস্ট অ্যাটেনডেড এন ইনক্রেডিবল লোকাল স্টোরিটেলিং ইভেন্ট এন্ড ফিলিং ইন্সপিরেড বিটি টি অফ পার্সোনাল স্টোরিজ! আমাদের নিজস্ব ব্যক্তিগত গল্প share করা অন্যদের সাথে সংযোগ স্থাপন, build empathy, and foster understanding বৃদ্ধির একটি শক্তিশালী উপায় হতে পারে। Whether it's through কথ্য শব্দ, লিখিত স্মৃতিচারণ বা চাক্ষুষ গল্প there are countless ways to share our ব্যক্তিগত অভিজ্ঞতা and make our voices heard. यदि आप अपने आप को अभिव्यक्त करने और दूसरों के साथ जुड़ने के तरीके की तलाश कर रहे हैं, तो मैं अत्यधिक सलाह देता हूं कि स्थानीय कहानी कार्यक्रम में भाग लें या अपनी निजी कहानियों को साझा करें। aaie ham sabhee apane anoothe drshtikon ko saajha karana jaaree rakhen aur maanav anubhav kee vividhata ka jashn manaen! #kahaanee #vyaktigat kahaaniyaan #aatmabhivyakti #sahaanubhooti #vividhata | 0 |

### `offensive/synthetic_noise_10pct.csv`

| text | target | type | labels | text_clean |
| --- | --- | --- | --- | --- |
| ঔ খানকির পোলারা কই গান কয় | group | slander | 1 | ঔ খানকিড ফোলারা কই গান কয় |
| পাপনের দুই গালে জুতা মরো তালে তালে। | male | callToViolence | 1 | পাফননের দুই গালে জুতা মরো রালে তালে। |
| আল্লাহ উপর ভরসা কর | nan | nan | 0 | আল্লাহ উফর ভরসা কর |
