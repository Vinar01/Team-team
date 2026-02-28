
spacing, filler words, punctuations, number to word, contractions, 
word + character error rate



//khushi
whisper
    remove spacing, remove punctuation, change num to wrod, change contraction.

//vises
grammar
   remove filler

weights (linguistic / acoustic / length(pujit))
    tukka (mostly)
    nn (unlikely)

//lenght consistency

//assumption : all transcripts are of the same language, anf the language is given, and the audio languae is the same as the transcript language

![alt text](image.png)

1. Overall Normalization Pipeline (Order Matters!)

The correct order should be:

1. Normalize encoding (UTF-8)
2. Lowercase (if case-insensitive task)
3. Expand contractions
4. Convert numbers to words
5. Remove punctuation
6. Normalize whitespace

Why this order?

Contractions contain punctuation (don't, I'm)

Numbers may contain commas/periods (1,000, 3.14)

Removing punctuation too early breaks logic
