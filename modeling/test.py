import torch
from modeling.modeling_encoder import TextEncoder

encoder = TextEncoder('bert', model_name='bert-base-uncased', output_token_states=False)

sentence = "This is a sample sentence to encode."

input_ids = torch.tensor(encoder.tokenizer.encode(sentence)).unsqueeze(0)

output = encoder(input_ids)

sentence_embedding = output[0]

print(sentence_embedding)