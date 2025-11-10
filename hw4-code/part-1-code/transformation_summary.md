# Text Transformation Methods for Adversarial Testing

## 1. Synonym Replacement Transformation

### Implementation Details:

**Parameters:**
- Replacement probability: 50% (prob=0.5)
- Target: All non-stopword tokens in the sentence

**Algorithm:**
1. **Tokenization**: Split the input sentence into individual words using NLTK's word tokenizer
2. **Word Selection**: For each word in the sentence:
   - Skip if the word is a stopword (common words like "the", "is", "and", etc.)
   - Skip if the word is not alphabetic
   - With 50% probability, select the word for replacement
3. **Synonym Finding**: For selected words:
   - Query WordNet for up to 3 synsets (synonym sets) for the word
   - Collect all lemmas (word forms) from these synsets
   - Filter out the original word itself
   - Replace underscores with spaces in multi-word synonyms
4. **Replacement**: If valid synonyms exist, randomly choose one synonym to replace the original word
5. **Reconstruction**: Join the modified word list back into a sentence with spaces

**Example:**
- Original: "This movie was really great and entertaining"
- Transformed: "This film was genuinely great and entertaining" (movie→film, really→genuinely)

### Why This Is Reasonable:

**Semantic Preservation**: Synonyms maintain the core meaning of the sentence. WordNet provides linguistically validated synonym relationships, ensuring that replacements are semantically appropriate in most contexts.

**Real-world Validity**: Different people naturally use different words to express the same sentiment. For example, "good/great/excellent" or "bad/terrible/awful" are commonly interchanged in reviews.

**Tests Robustness**: A robust sentiment classifier should recognize that "This film is excellent" and "This movie is great" convey the same positive sentiment. If synonym replacement significantly degrades performance, it suggests the model may be overfitting to specific vocabulary rather than understanding semantic meaning.

**Human Comprehensibility**: The transformed text remains natural and readable, similar to paraphrasing that humans might produce.

---

## 2. Typo Introduction Transformation

### Implementation Details:

**Parameters:**
- Word selection probability: 25% (prob=0.25)
- Character replacement rate per selected word: 40% (typo_per_word=0.4)
- Minimum word length: 3 characters

**Algorithm:**
1. **Tokenization**: Split the input sentence into words by whitespace
2. **Word Selection**: For each word:
   - With 25% probability, select the word for typo introduction
   - Skip words with 2 or fewer characters
3. **Typo Introduction**: For each selected word:
   - Calculate number of typos: max(1, int(word_length × 0.4))
   - Randomly sample character positions to modify
   - For each selected position:
     - Convert character to lowercase
     - If character exists in keyboard neighbor map, replace with random adjacent key
     - Preserve case from original if uppercase
4. **Reconstruction**: Join all words back into a sentence

**Keyboard Neighbor Map**: Based on physical QWERTY keyboard layout:
```
'a' → ['s','q','w','z']
'b' → ['v','g','h','n']
'c' → ['x','d','f','v']
'd' → ['s','e','r','f','c','x']
'e' → ['w','s','d','r']
'f' → ['d','r','t','g','v','c']
'g' → ['f','t','y','h','b','v']
'h' → ['g','y','u','j','n','b']
'i' → ['u','j','k','o']
'j' → ['h','u','i','k','n','m']
'k' → ['j','i','o','l','m']
'l' → ['k','o','p']
'm' → ['n','j','k']
'n' → ['b','h','j','m']
'o' → ['i','k','l','p']
'p' → ['o','l']
'q' → ['w','a']
'r' → ['e','d','f','t']
's' → ['a','w','e','d','x','z']
't' → ['r','f','g','y']
'u' → ['y','h','j','i']
'v' → ['c','f','g','b']
'w' → ['q','a','s','e']
'x' → ['z','s','d','c']
'y' → ['t','g','h','u']
'z' → ['a','s','x']
```

**Example:**
- Original: "This movie was really great"
- Transformed: "Thos movid was reakly grfat" (i→o, e→d, l→k, e→f)

### Why This Is Reasonable:

**Real-world Occurrence**: Typos are extremely common in user-generated content, especially in informal contexts like social media reviews, mobile typing, or fast typing. Studies show that 10-20% of words in informal text contain typos.

**Physical Plausibility**: Using keyboard neighbors simulates genuine typing errors where users accidentally hit adjacent keys. This is more realistic than random character substitution.

**Tests Generalization**: A good sentiment classifier should be robust to minor spelling variations. Humans can easily understand "grest" as "great" or "mivie" as "movie" - the model should demonstrate similar robustness.

**Character-level Understanding**: This tests whether the model relies too heavily on exact character sequences versus learning more robust word representations. Modern transformers with subword tokenization (like BERT) should theoretically handle this better.

**Controlled Degradation**: The parameters (25% word selection, 40% character replacement per word) create enough noise to challenge the model while keeping most text recognizable. This results in approximately 10% of all characters being modified.

---

## Experimental Results

Both transformations were tested on BERT-based sentiment classification on the IMDB dataset:

| Transformation | Accuracy | Drop from Baseline | Effective? |
|----------------|----------|-------------------|------------|
| Baseline (No transformation) | 92.252% | - | - |
| Synonym Replacement | 86.576% | 5.676% | ✅ Yes |
| Typo Introduction | 86.932% | 5.320% | ✅ Yes |

Both transformations achieved **>4 percentage point accuracy drops**, meeting the effectiveness threshold.

---

## Comparison and Analysis

**Synonym Replacement (86.576%)**:
- Tests semantic robustness and vocabulary invariance
- More effective at degrading performance
- Maintains grammaticality and readability
- Challenges the model's understanding of word meanings

**Typo Introduction (86.932%)**:
- Tests character-level robustness and spelling variation handling
- Slightly less effective
- Creates more human-like errors
- Challenges the model's subword tokenization and character pattern recognition

The typo transformation was slightly less effective, which makes sense because:
1. Modern transformers use subword tokenization (WordPiece/BPE) that can partially handle character-level variations
2. BERT's character-level patterns in subwords may still capture partial word identity
3. Synonym replacement fundamentally changes the tokens the model sees, while typos may still map to similar subword representations

Both transformations are "reasonable" because they simulate realistic variations in human-generated text while maintaining the underlying sentiment, making them valuable for testing model robustness.
