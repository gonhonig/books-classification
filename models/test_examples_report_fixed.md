
## Accuracy Summary (Using Actual Test Data)

### Single-Label Sentences
- **Correct Predictions**: 45/60
- **Accuracy**: 0.750 (75.0%)

### Multi-Label Sentences
- **Correct Predictions**: 48/60
- **Accuracy**: 0.800 (80.0%)

### Overall Performance
- **Total Examples**: 120
- **Total Correct**: 93
- **Overall Accuracy**: 0.775

# Test Examples Analysis (Fixed - Using Actual Test Data)

## Important Note
This analysis uses ONLY sentences that were held out as test data during training.
These are the actual test sentences that the models have never seen before.

## Single-Label Sentence Examples

These sentences belong to only one book:

### Example 1

**Sentence**: Love for my life urged a compliance; I stepped over the threshold to wait till the others should enter.

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.015) ✅
- Wuthering Heights: 0 (prob: 0.180) ❌
- Frankenstein: 1 (prob: 0.953) ❌
- The Adventures of Alice in Wonderland: 0 (prob: 0.187) ✅

---

### Example 2

**Sentence**: Well, but you don t need it, I should fancy.

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.686) ✅
- Wuthering Heights: 1 (prob: 0.643) ❌
- Frankenstein: 0 (prob: 0.094) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 0.734) ❌

---

### Example 3

**Sentence**: No, but for all sorts of nervous invalids.

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.234) ❌
- Wuthering Heights: 0 (prob: 0.117) ✅
- Frankenstein: 0 (prob: 0.264) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.164) ✅

---

### Example 4

**Sentence**: Why, this spring Natalia Golitzina died from having an ignorant doctor.

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.697) ✅
- Wuthering Heights: 0 (prob: 0.022) ✅
- Frankenstein: 0 (prob: 0.024) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.009) ✅

---

### Example 5

**Sentence**: I getten summut else to do, he answered, and continued his work; moving his lantern jaws meanwhile, and surveying my dress and countenance (the former a great deal too fine, but the latter, I m sure, ...

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.008) ✅
- Wuthering Heights: 1 (prob: 0.705) ✅
- Frankenstein: 1 (prob: 0.898) ❌
- The Adventures of Alice in Wonderland: 0 (prob: 0.117) ✅

---

### Example 6

**Sentence**: Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next.

**Original Book**: The Adventures of Alice in Wonderland
**Original Label**: 0

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.154) ✅
- Wuthering Heights: 1 (prob: 0.576) ❌
- Frankenstein: 0 (prob: 0.112) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 0.990) ✅

---

### Example 7

**Sentence**: Altogether, I fancy that in the people s ideas there are very clear and definite notions of certain, as they call it, gentlemanly lines of action.

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.074) ❌
- Wuthering Heights: 0 (prob: 0.040) ✅
- Frankenstein: 0 (prob: 0.427) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.262) ✅

---

### Example 8

**Sentence**: And they don t sanction the gentry s moving outside bounds clearly laid down in their ideas.

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.028) ❌
- Wuthering Heights: 0 (prob: 0.095) ✅
- Frankenstein: 0 (prob: 0.004) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.059) ✅

---

### Example 9

**Sentence**: The princess began talking to him, but he did not hear her.

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.960) ✅
- Wuthering Heights: 0 (prob: 0.163) ✅
- Frankenstein: 0 (prob: 0.054) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.331) ✅

---

### Example 10

**Sentence**: Why, I wouldn t say anything about it, even if I fell off the top of the house!

**Original Book**: The Adventures of Alice in Wonderland
**Original Label**: 0

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.035) ✅
- Wuthering Heights: 1 (prob: 0.779) ❌
- Frankenstein: 0 (prob: 0.045) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 0.762) ✅

---

### Example 11

**Sentence**: (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.)

**Original Book**: The Adventures of Alice in Wonderland
**Original Label**: 0

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.012) ✅
- Wuthering Heights: 0 (prob: 0.062) ✅
- Frankenstein: 1 (prob: 0.938) ❌
- The Adventures of Alice in Wonderland: 1 (prob: 0.996) ✅

---

### Example 12

**Sentence**: The Antipathies, I think (she was rather glad there _was_ no one listening, this time, as it didn t sound at all the right word) but I shall have to ask them what the name of the country is, you know.

**Original Book**: The Adventures of Alice in Wonderland
**Original Label**: 0

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.615) ❌
- Wuthering Heights: 0 (prob: 0.214) ✅
- Frankenstein: 0 (prob: 0.117) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 0.942) ✅

---

### Example 13

**Sentence**: I questioned with myself where must I turn for comfort?

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.114) ✅
- Wuthering Heights: 1 (prob: 0.531) ✅
- Frankenstein: 1 (prob: 0.909) ❌
- The Adventures of Alice in Wonderland: 0 (prob: 0.120) ✅

---

### Example 14

**Sentence**: I had sought shelter at Wuthering Heights, almost gladly, because I was secured by that arrangement from living alone with him; but he knew the people we were coming amongst, and he did not fear their...

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.011) ✅
- Wuthering Heights: 1 (prob: 0.901) ✅
- Frankenstein: 0 (prob: 0.213) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.003) ✅

---

### Example 15

**Sentence**: I listened to detect a woman s voice in the house, and filled the interim with wild regrets and dismal anticipations, which, at last, spoke audibly in irrepressible sighing and weeping.

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.502) ❌
- Wuthering Heights: 1 (prob: 0.915) ✅
- Frankenstein: 1 (prob: 0.989) ❌
- The Adventures of Alice in Wonderland: 0 (prob: 0.423) ✅

---

## Multi-Label Sentence Examples

These sentences belong to multiple books:

### Example 1

**Sentence**: We can t talk to Kitty about it!

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.867) ✅
- Wuthering Heights: 0 (prob: 0.011) ✅
- Frankenstein: 0 (prob: 0.000) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 0.994) ✅

---

### Example 2

**Sentence**: mim!

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 1
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.962) ✅
- Wuthering Heights: 1 (prob: 0.995) ✅
- Frankenstein: 1 (prob: 0.911) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 0.998) ✅

---

### Example 3

**Sentence**: Oh dear!

**Original Book**: The Adventures of Alice in Wonderland
**Original Label**: 0

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 1
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.999) ✅
- Wuthering Heights: 1 (prob: 1.000) ✅
- Frankenstein: 1 (prob: 0.998) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 1.000) ✅

---

### Example 4

**Sentence**: Mincing un munching!

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 1
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.092) ❌
- Wuthering Heights: 1 (prob: 0.813) ✅
- Frankenstein: 1 (prob: 0.545) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 0.983) ✅

---

### Example 5

**Sentence**: he demanded, grimly.

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.960) ✅
- Wuthering Heights: 1 (prob: 0.984) ✅
- Frankenstein: 1 (prob: 0.958) ❌
- The Adventures of Alice in Wonderland: 1 (prob: 0.996) ✅

---

### Example 6

**Sentence**: (Which was very likely true.)

**Original Book**: The Adventures of Alice in Wonderland
**Original Label**: 0

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.716) ✅
- Wuthering Heights: 1 (prob: 0.687) ✅
- Frankenstein: 0 (prob: 0.432) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 0.900) ✅

---

### Example 7

**Sentence**: Down, down, down.

**Original Book**: The Adventures of Alice in Wonderland
**Original Label**: 0

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.937) ✅
- Wuthering Heights: 0 (prob: 0.411) ❌
- Frankenstein: 1 (prob: 0.721) ❌
- The Adventures of Alice in Wonderland: 1 (prob: 0.999) ✅

---

### Example 8

**Sentence**: Is he come back, then?

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.915) ✅
- Wuthering Heights: 1 (prob: 0.954) ✅
- Frankenstein: 1 (prob: 0.721) ❌
- The Adventures of Alice in Wonderland: 1 (prob: 0.843) ❌

---

### Example 9

**Sentence**: It s well the hellish villain has kept his word!

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.079) ❌
- Wuthering Heights: 1 (prob: 0.832) ✅
- Frankenstein: 1 (prob: 0.774) ❌
- The Adventures of Alice in Wonderland: 1 (prob: 0.856) ❌

---

### Example 10

**Sentence**: He walked up and down, with his hands in his pockets, apparently quite forgetting my presence; and his abstraction was evidently so deep, and his whole aspect so misanthropical, that I shrank from dis...

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 0
- Wuthering Heights: 1
- Frankenstein: 1
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 0 (prob: 0.014) ✅
- Wuthering Heights: 1 (prob: 0.906) ✅
- Frankenstein: 1 (prob: 0.980) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.048) ✅

---

### Example 11

**Sentence**: By all means, please, and I shall come too, said Kitty, and she blushed.

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.926) ✅
- Wuthering Heights: 0 (prob: 0.053) ✅
- Frankenstein: 0 (prob: 0.044) ✅
- The Adventures of Alice in Wonderland: 1 (prob: 0.999) ✅

---

### Example 12

**Sentence**: Do you think you could manage it?)

**Original Book**: The Adventures of Alice in Wonderland
**Original Label**: 0

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 0
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.556) ✅
- Wuthering Heights: 0 (prob: 0.354) ✅
- Frankenstein: 0 (prob: 0.085) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.307) ❌

---

### Example 13

**Sentence**: He was only afraid his brother might ask him some question which would make it evident he had not heard.

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.536) ✅
- Wuthering Heights: 0 (prob: 0.294) ❌
- Frankenstein: 0 (prob: 0.231) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.100) ✅

---

### Example 14

**Sentence**: Yes, of course.

**Original Book**: Anna Karenina
**Original Label**: 1

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 0
- The Adventures of Alice in Wonderland: 1

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.997) ✅
- Wuthering Heights: 1 (prob: 0.974) ✅
- Frankenstein: 1 (prob: 0.557) ❌
- The Adventures of Alice in Wonderland: 1 (prob: 0.998) ✅

---

### Example 15

**Sentence**: I sobbed; I was beyond regarding self-respect, weighed down by fatigue and wretchedness.

**Original Book**: Wuthering Heights
**Original Label**: 2

**True Labels**:
- Anna Karenina: 1
- Wuthering Heights: 1
- Frankenstein: 1
- The Adventures of Alice in Wonderland: 0

**Model Predictions**:
- Anna Karenina: 1 (prob: 0.537) ✅
- Wuthering Heights: 1 (prob: 0.990) ✅
- Frankenstein: 1 (prob: 0.997) ✅
- The Adventures of Alice in Wonderland: 0 (prob: 0.263) ✅

---

