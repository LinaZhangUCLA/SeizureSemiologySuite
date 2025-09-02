#!/usr/bin/env python
# coding: utf-8

# In[3]:


# pip install wordcloud matplotlib
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def tokenize(text):
    """
    Tokenize and clean a sentence:
    - lowercase
    - keep alphabetic tokens
    - drop very short tokens (len <= 1)
    """
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    return [t for t in tokens if len(t) > 1]

def generate_wordcloud(text, width=1000, height=500, output_path=None):
    """
    Build and display a word cloud from text.
    If output_path is provided, also save the image.
    """
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        stopwords=STOPWORDS
    ).generate(text)

    plt.figure(figsize=(width/100, height/100))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if output_path:
        wc.to_file(output_path)  # e.g., "wordcloud.png"
    plt.show()

if __name__ == "__main__":
    sentence = "Tokenize this sentence and generate a nice word cloud showing the most frequent words in this example sentence."
    # 1) Tokenize
    tokens = tokenize(sentence)
    print("Tokens:", tokens)

    # 2) Word cloud (uses the original sentence; you could also use ' ' .join(tokens))
    generate_wordcloud(sentence, output_path=None)  # set to "wordcloud.png" to save


# In[9]:


# pip install pandas wordcloud matplotlib
import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# 1. Load CSV
df = pd.read_csv("Behavioral description - Sheet1.csv")

# 2. Correct column name (with space)
text_column = "Behavioral description"

all_text = " ".join(df[text_column].astype(str))

# 3. Tokenize & clean
def tokenize(text):
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    return [t for t in tokens if len(t) > 1]

cleaned_text = " ".join(tokenize(all_text))

# 4. Generate word cloud
wc = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    stopwords=STOPWORDS
).generate(cleaned_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()





# In[17]:


# pip install pandas wordcloud matplotlib openpyxl
import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

# 1. Load Excel file
df = pd.read_excel("Seizure_Data_Scrutinized.xlsx", engine="openpyxl")

# 2. Check column names
print("Columns in file:", df.columns)

# 3. Pick the column with text (adjust if needed)
# Example: if the column is "Description" or "Notes"
text_column = "Behavioral description"  # <-- change this if different in new file

all_text = " ".join(df[text_column].astype(str))

# 4. Tokenize & clean
def tokenize(text):
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    return [t for t in tokens if len(t) > 1]

tokens = tokenize(all_text)
cleaned_text = " ".join(tokens)

# 5. Generate word cloud
wc = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    stopwords=STOPWORDS
).generate(cleaned_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

# 6. Get exact word counts
word_counts = Counter(tokens)
freq_df = pd.DataFrame(word_counts.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)

# Show top 20
print(freq_df.head(20))

# Save to Excel/CSV
freq_df.to_csv("word_frequencies.csv", index=False)



# In[21]:


# pip install pandas wordcloud matplotlib openpyxl
import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

# 1. Load Excel file
df = pd.read_excel("Seizure_Data_Scrutinized.xlsx", engine="openpyxl")

# 2. Check column names
print("Columns in file:", df.columns)

# 3. Pick the column with text (adjust if needed)
text_column = "Behavioral description"  # <-- change if different in your file
all_text = " ".join(df[text_column].astype(str))

# 4. Tokenize & clean
def tokenize(text):
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    return [t for t in tokens if len(t) > 1]

tokens = tokenize(all_text)

# 5. Define custom words to remove
custom_remove = {"movement","patient", "minute",
"behavioral description",
"description provided",
"noted",
"phone",
"nurse",
"exhibit",
"later",
"around",
"room",
"chair",
"asked",
"blanket",
"state",
"pillow",
"toward",
"button",
"initially",
"tested",
"call button",
"develop",
"mom",
"subsequent",
"following offset",
"registered nurse",
"often",
"word",
"call",
"hand will",
"nursing staff",
"eeg",
"greater",
"slightly"}

# Merge with default stopwords
stopwords = STOPWORDS.union(custom_remove)

# Filter tokens
tokens = [t for t in tokens if t not in stopwords]
cleaned_text = " ".join(tokens)

# 6. Generate word cloud
wc = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    stopwords=stopwords
).generate(cleaned_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

# 7. Get exact word counts (after filtering)
word_counts = Counter(tokens)
freq_df = pd.DataFrame(word_counts.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)

# Show top 20
print(freq_df.head(20))

# Save to Excel/CSV
freq_df.to_csv("word_frequencies.csv", index=False)



# In[1]:


# pip install pandas wordcloud matplotlib openpyxl
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

# ============= 1) Load & choose column ============
df = pd.read_excel("Seizure_Data_Scrutinized.xlsx", engine="openpyxl")
text_col = "Behavioral description"   # change if needed
raw = " ".join(df[text_col].astype(str))

# ============= 2) Normalize text ==================
text = raw.lower()

# unify hyphen/underscore handling (so "left-arm" -> "left arm")
text = re.sub(r"[\-_/]", " ", text)

# collapse weird whitespace
text = re.sub(r"\s+", " ", text).strip()

# ============= 3) Remove multi-word stop-phrases BEFORE tokenizing =========
# Put any phrases you want gone here. (All should be lowercase.)
stop_phrases = {
    "behavioral description", "description provided", "call button",
    "registered nurse", "nursing staff", "hand will", "following offset",
    "at this time", "able to", "states that", "mom reports"
}
# remove them with word-boundary safe regex
for p in sorted(stop_phrases, key=len, reverse=True):
    text = re.sub(rf"\b{re.escape(p)}\b", " ", text)

# ============= 4) Tokenize (letters only), keep words >= 2 chars ===========
tokens = re.findall(r"[a-z]+", text)
tokens = [t for t in tokens if len(t) >= 2]

# ============= 5) Build n-grams (bigrams + trigrams) =======================
def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

bigrams  = ngrams(tokens, 2)
trigrams = ngrams(tokens, 3)

# simple frequency thresholds to keep meaningful phrases
min_bigram  = 8   # tune these based on your corpus size
min_trigram = 5

bigram_counts  = Counter([bg for bg in bigrams  if bg.split()[0] != bg.split()[1]])
trigram_counts = Counter([tg for tg in trigrams if len(set(tg.split())) > 1])

bigram_kept  = {k:v for k,v in bigram_counts.items()  if v >= min_bigram}
trigram_kept = {k:v for k,v in trigram_counts.items() if v >= min_trigram}

# ============= 6) Build the display text for WordCloud =====================
# We replace spaces with underscores so WordCloud treats phrases as single tokens.
def underscore_join(d):
    return {k.replace(" ", "_"): v for k, v in d.items()}

phrase_freqs = {}
phrase_freqs.update(underscore_join(bigram_kept))
phrase_freqs.update(underscore_join(trigram_kept))

# Also include unigrams if you want a mixed cloud (comment out to show phrases-only)
# Remove boring unigrams with STOPWORDS + your custom words
custom_unigram_stops = {
    "patient","movement","minute","nurse","asked","initially","later","around","room",
    "chair","button","state","tested","develop","mom","often","word","call","eeg",
    "slightly","greater","provided","description","behavioral","event","episode","appears"
}
unigram_stops = STOPWORDS.union(custom_unigram_stops)

unigram_counts = Counter([t for t in tokens if t not in unigram_stops])
# keep top N unigrams so phrases dominate visually
top_unigrams = dict(unigram_counts.most_common(60))

# merge (phrases get higher weight so they pop)
combined_freqs = {}
combined_freqs.update({k: v*3 for k,v in phrase_freqs.items()})  # boost phrases
combined_freqs.update(top_unigrams)

# ============= 7) Draw the cloud ==========================================
wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    collocations=False,   # important: we *control* phrases ourselves
)
wc.generate_from_frequencies(combined_freqs)

plt.figure(figsize=(14,7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.show()

# ============= 8) Export frequencies ======================================
freq_df = (
    pd.DataFrame(
        [(k.replace("_"," "), v, "phrase" if "_" in k else "unigram")
         for k,v in combined_freqs.items()],
        columns=["token","count","type"]
    )
    .sort_values(["type","count"], ascending=[True, False])
)
freq_df.to_csv("word_frequencies_phrases.csv", index=False)
print(freq_df.head(20))


# In[4]:


# pip install pandas wordcloud matplotlib openpyxl
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

# ============= 1) Load Excel file ============
df = pd.read_excel("Seizure_Data_Scrutinized.xlsx", engine="openpyxl")
text_col = "Behavioral description"   # adjust if needed
raw = " ".join(df[text_col].astype(str))

# ============= 2) Normalize text ============
text = raw.lower()
text = re.sub(r"[\-_/]", " ", text)           # replace hyphens/underscores with space
text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace

# ============= 3) Tokenize words ============
tokens = re.findall(r"[a-z]+", text)          # keep only letters
tokens = [t for t in tokens if len(t) >= 2]   # drop single letters

# ============= 4) Remove stopwords ==========
custom_remove = {
    "movement","patient","minute","nurse","asked","initially","later","around","room",
    "chair","button","state","tested","develop","mom","often","word","call","eeg",
    "slightly","greater","provided","description","behavioral","event","episode","appears",
    "offset","time","report","staff","phone","seen","test","exam", "able", "non", "unable", "seconds", 
    "sometimes", "bed"   
}
stopwords = STOPWORDS.union(custom_remove)

tokens = [t for t in tokens if t not in stopwords]

# ============= 5) Count word frequencies =====
word_counts = Counter(tokens)

# ============= 6) Generate word cloud ========
wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=stopwords,
    collocations=False   # ensures no multi-word combos appear
).generate_from_frequencies(word_counts)

plt.figure(figsize=(14,7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.show()

# ============= 7) Save frequencies ===========
freq_df = pd.DataFrame(word_counts.items(), columns=["Word","Count"]).sort_values(by="Count", ascending=False)
freq_df.to_csv("word_frequencies_unigrams.csv", index=False)
print(freq_df.head(20))


# In[ ]:




