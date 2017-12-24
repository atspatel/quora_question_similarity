# quora_question_similarity
Programming assignment For NanoNets

This is the simple Nueral Network code to find binary similarity between 2 questions from quora.

FLow of Networks follows as:

input: List of pair of questions.
output : List of 1 and 0s

Network:

Break sentence (question) in words and lemmatize each words.
created a padded sentence matrix using word embedding

A lstm Layer has been applied on sentence matrix and taking output of last word as sentence LSTM output, which will be act as input to 2 layer dense network.
Output of dense network is now the final vector represent of question.

Now, cosine similarity has been calculated between pair of sentences. Cosine score threshold will be treated as 1.

Loss for network:
if input_label is 1:
  loss = (1 - cosine_score^2)/4

if input_label is 0:
  loss = (cosine_score^2) if cosine_score > threshold else 0
