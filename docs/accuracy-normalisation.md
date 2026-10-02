# Accuracy scoring normalisation

Raw WER and CER are always reported alongside normalised scores. Egyptian Arabic spelling variation and Arabic/English code-switching make raw WER overly pessimistic: the same spoken name or word may be written with different alef forms, optional diacritics, or English spelling. Normalisation reduces these predictable orthographic differences while retaining the raw score for auditability.

The published normalisation rules are applied in this order:

1. Strip optional line-leading `[mm:ss]` or `[hh:mm:ss]` timestamps.
2. Remove Arabic tashkeel U+064B–U+0652, superscript alef U+0670, and tatweel U+0640.
3. Map أ إ آ ٱ to ا; ى to ي; ة to ه; ؤ to و; ئ to ي.
4. Map Arabic-Indic U+0660–U+0669 and Eastern Arabic-Indic U+06F0–U+06F9 digits to ASCII digits.
5. Lowercase Latin text, remove punctuation (Arabic and Latin punctuation), and collapse whitespace.
6. If Vocabulary is provided, map every alias to its canonical spelling in both the reference and hypothesis before scoring. Vocabulary scoring separately counts either canonical or alias as recognised and exact canonical spelling as canonical.

Numbers are compared as multisets. The hand-written number-word coverage includes common Egyptian/MSA forms from zero through twenty (including تلات/تلاتة, تمانية, عشرة/عشره), tens, MSA hundred forms, Egyptian 100–900 forms including مية/ميه and ميتين through تسعمية, and thousand forms including ألفين/الفين = 2,000. It does not parse arbitrary compound number phrases (such as “twenty-five” spoken as words), gender/declension variants not listed in the implementation, fractions, or spoken decimals; digit strings support integers, decimals, and comma-separated thousands.
