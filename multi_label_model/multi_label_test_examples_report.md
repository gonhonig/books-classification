# Multi-Label Neural Network - Test Examples Report

**Single-label Accuracy**: 0.800
**Multi-label Accuracy**: 0.333

## Single-Label Examples

### Example 1
**Sentence**: I know, continued the unhappy victim, how heavily and fatally this one circumstance weighs against me, but I have no power of explaining it; and when I have expressed my utter ignorance, I am only left to conjecture concerning the probabilities by which it might have been placed in my pocket.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.027 (0) | ✓ |
| Frankenstein | 1 | 0.992 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.006 (0) | ✓ |
| Wuthering Heights | 0 | 0.032 (0) | ✓ |

### Example 2
**Sentence**: No, I didn t, said Alice: I don t think it s at all a pity.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.103 (0) | ✓ |
| Frankenstein | 0 | 0.009 (0) | ✓ |
| The Adventures of Alice in Wonderland | 1 | 0.977 (1) | ✓ |
| Wuthering Heights | 0 | 0.111 (0) | ✓ |

### Example 3
**Sentence**: These wonderful narrations inspired me with strange feelings.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.075 (0) | ✓ |
| Frankenstein | 1 | 0.997 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.010 (0) | ✓ |
| Wuthering Heights | 0 | 0.284 (0) | ✓ |

### Example 4
**Sentence**: You are a dog in the manger, Cathy, and desire no one to be loved but yourself!

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.276 (0) | ✓ |
| Frankenstein | 0 | 0.620 (1) | ✗ |
| The Adventures of Alice in Wonderland | 0 | 0.067 (0) | ✓ |
| Wuthering Heights | 1 | 0.690 (1) | ✓ |

### Example 5
**Sentence**: So Bill s got to come down the chimney, has he?

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.355 (0) | ✓ |
| Frankenstein | 0 | 0.008 (0) | ✓ |
| The Adventures of Alice in Wonderland | 1 | 0.225 (0) | ✗ |
| Wuthering Heights | 0 | 0.879 (1) | ✗ |

### Example 6
**Sentence**: I would have made a pilgrimage to the highest peak of the Andes, could I, when there, have precipitated him to their base.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.008 (0) | ✓ |
| Frankenstein | 1 | 0.994 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.004 (0) | ✓ |
| Wuthering Heights | 0 | 0.024 (0) | ✓ |

### Example 7
**Sentence**: I had sufficient leisure for these and many other reflections during my journey to Ingolstadt, which was long and fatiguing.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.004 (0) | ✓ |
| Frankenstein | 1 | 0.997 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.003 (0) | ✓ |
| Wuthering Heights | 0 | 0.008 (0) | ✓ |

### Example 8
**Sentence**: But he has already recovered his spirits, and is reported to be on the point of marrying a lively pretty Frenchwoman, Madame Tavernier.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.118 (0) | ✓ |
| Frankenstein | 1 | 0.828 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.019 (0) | ✓ |
| Wuthering Heights | 0 | 0.085 (0) | ✓ |

### Example 9
**Sentence**: how sincerely you did love me, and endeavour to elevate my mind until it was on a level with your own.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.243 (0) | ✓ |
| Frankenstein | 1 | 0.919 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.028 (0) | ✓ |
| Wuthering Heights | 0 | 0.238 (0) | ✓ |

### Example 10
**Sentence**: Elizabeth read my anguish in my countenance, and kindly taking my hand, said, My dearest friend, you must calm yourself.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.006 (0) | ✓ |
| Frankenstein | 1 | 0.985 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.006 (0) | ✓ |
| Wuthering Heights | 0 | 0.028 (0) | ✓ |

### Example 11
**Sentence**: Sometimes they were the expressive eyes of Henry, languishing in death, the dark orbs nearly covered by the lids and the long black lashes that fringed them; sometimes it was the watery, clouded eyes of the monster, as I first saw them in my chamber at Ingolstadt.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.012 (0) | ✓ |
| Frankenstein | 1 | 0.987 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.008 (0) | ✓ |
| Wuthering Heights | 0 | 0.020 (0) | ✓ |

### Example 12
**Sentence**: I remained for several years their only child.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.005 (0) | ✓ |
| Frankenstein | 1 | 0.992 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.004 (0) | ✓ |
| Wuthering Heights | 0 | 0.011 (0) | ✓ |

### Example 13
**Sentence**: When my hunger was appeased, I directed my steps towards the well-known path that conducted to the cottage.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.007 (0) | ✓ |
| Frankenstein | 1 | 0.976 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.009 (0) | ✓ |
| Wuthering Heights | 0 | 0.035 (0) | ✓ |

### Example 14
**Sentence**: If you like Isabella, you shall marry her.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.737 (1) | ✗ |
| Frankenstein | 0 | 0.061 (0) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.125 (0) | ✓ |
| Wuthering Heights | 1 | 0.678 (1) | ✓ |

### Example 15
**Sentence**: Well, perhaps you haven t found it so yet, said Alice; but when you have to turn into a chrysalis you will some day, you know and then after that into a butterfly, I should think you ll feel it a little queer, won t you?

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.035 (0) | ✓ |
| Frankenstein | 0 | 0.025 (0) | ✓ |
| The Adventures of Alice in Wonderland | 1 | 0.980 (1) | ✓ |
| Wuthering Heights | 0 | 0.041 (0) | ✓ |

## Multi-Label Examples

### Example 1
**Sentence**: Poor papa!

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.976 (1) | ✓ |
| Frankenstein | 1 | 0.406 (0) | ✗ |
| The Adventures of Alice in Wonderland | 0 | 0.292 (0) | ✓ |
| Wuthering Heights | 1 | 0.964 (1) | ✓ |

### Example 2
**Sentence**: I said, passionately.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.650 (1) | ✗ |
| Frankenstein | 1 | 0.836 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.118 (0) | ✓ |
| Wuthering Heights | 1 | 0.916 (1) | ✓ |

### Example 3
**Sentence**: Go away!

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.989 (1) | ✓ |
| Frankenstein | 1 | 0.867 (1) | ✓ |
| The Adventures of Alice in Wonderland | 1 | 0.447 (0) | ✗ |
| Wuthering Heights | 1 | 0.986 (1) | ✓ |

### Example 4
**Sentence**: The infant had been placed with these good people to nurse: they were better off then.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.067 (0) | ✗ |
| Frankenstein | 1 | 0.806 (1) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.028 (0) | ✓ |
| Wuthering Heights | 0 | 0.158 (0) | ✓ |

### Example 5
**Sentence**: But, away with you!

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.977 (1) | ✓ |
| Frankenstein | 0 | 0.508 (1) | ✗ |
| The Adventures of Alice in Wonderland | 0 | 0.464 (0) | ✓ |
| Wuthering Heights | 1 | 0.969 (1) | ✓ |

### Example 6
**Sentence**: she continued.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.974 (1) | ✓ |
| Frankenstein | 0 | 0.135 (0) | ✓ |
| The Adventures of Alice in Wonderland | 1 | 0.422 (0) | ✗ |
| Wuthering Heights | 1 | 0.952 (1) | ✓ |

### Example 7
**Sentence**: She looked nice.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.972 (1) | ✓ |
| Frankenstein | 0 | 0.131 (0) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.064 (0) | ✓ |
| Wuthering Heights | 1 | 0.915 (1) | ✓ |

### Example 8
**Sentence**: Then she turned up.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.989 (1) | ✓ |
| Frankenstein | 0 | 0.178 (0) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.329 (0) | ✓ |
| Wuthering Heights | 1 | 0.938 (1) | ✓ |

### Example 9
**Sentence**: said Alice, in a great hurry to change the subject of conversation.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.119 (0) | ✗ |
| Frankenstein | 0 | 0.001 (0) | ✓ |
| The Adventures of Alice in Wonderland | 1 | 1.000 (1) | ✓ |
| Wuthering Heights | 0 | 0.027 (0) | ✓ |

### Example 10
**Sentence**: he asked.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 1.000 (1) | ✓ |
| Frankenstein | 0 | 0.026 (0) | ✓ |
| The Adventures of Alice in Wonderland | 1 | 0.777 (1) | ✓ |
| Wuthering Heights | 1 | 0.994 (1) | ✓ |

### Example 11
**Sentence**: Ah, there he is!

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.979 (1) | ✓ |
| Frankenstein | 0 | 0.631 (1) | ✗ |
| The Adventures of Alice in Wonderland | 0 | 0.391 (0) | ✓ |
| Wuthering Heights | 1 | 0.961 (1) | ✓ |

### Example 12
**Sentence**: Away after them!

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.729 (1) | ✓ |
| Frankenstein | 0 | 0.441 (0) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.439 (0) | ✓ |
| Wuthering Heights | 1 | 0.721 (1) | ✓ |

### Example 13
**Sentence**: It is over!

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.944 (1) | ✓ |
| Frankenstein | 0 | 0.360 (0) | ✓ |
| The Adventures of Alice in Wonderland | 0 | 0.192 (0) | ✓ |
| Wuthering Heights | 1 | 0.875 (1) | ✓ |

### Example 14
**Sentence**: She d soon fetch it back!

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 1 | 0.358 (0) | ✗ |
| Frankenstein | 0 | 0.087 (0) | ✓ |
| The Adventures of Alice in Wonderland | 1 | 0.339 (0) | ✗ |
| Wuthering Heights | 1 | 0.818 (1) | ✓ |

### Example 15
**Sentence**: I followed.

| Book | True Label | Prediction | Correct |
|------|------------|------------|--------|
| Anna Karenina | 0 | 0.219 (0) | ✓ |
| Frankenstein | 0 | 0.689 (1) | ✗ |
| The Adventures of Alice in Wonderland | 1 | 0.101 (0) | ✗ |
| Wuthering Heights | 1 | 0.546 (1) | ✓ |

