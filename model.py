import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_p=0.1):
        super(DecoderRNN, self).__init__()
        
        # Model attributes
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
    
        # Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        # LSTM
        # batch_first=True:
        # in: (batch_size, caption_length, in_features/embedding_dim)
        # out: (batch_size, caption_length, out/hidden)
        self.lstm = nn.LSTM(self.embed_size,
                            self.hidden_size,
                            self.num_layers, 
                            dropout=drop_p,
                            batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(drop_p)
        
        # Fully-connected output layer
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Softmax
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features, captions):
        
        # We take the first n-1 tokens
        # The last token <end> is removed
        # That is how it is depicted in the notebook image
        # and also in the reference paper
        captions = captions[:, :-1]
        
        # Embedding
        # After applying the embedding
        # we have a sequence of embedded vectors
        embed = self.embedding(captions)
        
        # Resize image features: features.size(0) == batch_size
        features = features.view(features.size(0), 1, -1)
        
        # Concatenate features and embedded caption tokens
        # The image features have the embedding size
        # and they should be passed to the LSTM first
        # so they come to the front
        inputs = torch.cat((features, embed), dim=1) 
        # inputs size:
        # (batch_size, sequence_length, embedding_size)
        
        # LSTM: Pass the batch of sequences: [image, token_1, token_2, ..., token_(n-1)]
        # If we don't pass any hidden state, it defaults to 0
        # but a hidden state is always returned!
        lstm_out, hidden = self.lstm(inputs)
        # lstm_out size: (batch_size, sequence_length, hidden_size)
        # hidden -> (h, c)
        # h size: (1, batch_size, hidden_size)
        # c size: (1, batch_size, hidden_size)

        # Size of lstm_out:
        # (batch_size, sequence_length, hidden_dimension)
        # To pass to the linear layer we need to have (-1, hidden_dimension),1
        # i.e., (batch_size*sequence_length, hidden_dimension)
        out = lstm_out.reshape(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))
        
        # Pass through a dropout layer
        out = self.dropout(out)
        
        # Put out through the fully-connected layer
        # Map from the hidden dimension space to the vocabulary dimension space
        out = self.fc(out)
        
        # Size of out: (batch_size*sequence_length, vocabulary_size)
        # We need to resize it
        out = out.view(lstm_out.size(0), lstm_out.size(1), -1)
        
        # Log-Softmax / SoftMax: dim=2
        #out = self.softmax(out) # a probability for each token in the vocabulary
        
        # Return the final output
        # We don't return the hidden state because we won't re-use it!
        return out
    
    def sample(self, inputs, states=None, max_len=20):
        """Accepts pre-processed image tensor (inputs)
        and returns predicted sentence,
        i.e., list of tensor ids of length max_len)
        
        Arguments:
            inputs (tensor): pre-processed image tensor
            states (tensor): hidden state to initialize
            max_len (int): maximum length of the returned caption integers
            
        Returns:
            outputs (list): list of caption integers; length: max_len
        """
        
        # Initialize output as a list of <end> tokens 
        outputs = [1]*max_len
        
        # Initial hidden state is 0
        hidden = None
        
        # Pass the image and obtain the token sequence
        # This is similar to the forward pass
        with torch.no_grad():
            for i in range(max_len):
                # lstm_out size: (batch_size=1, sequence_length=1, hidden_size)
                lstm_out, hidden = self.lstm(inputs, hidden)
                # out size: (1, hidden_size) 
                out = lstm_out.reshape(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))
                # fc_out size (1, vocabulary_size)
                fc_out = self.fc(out)
                
                # Get probabilities
                p = F.softmax(fc_out, dim=1).data

                # Move to cpu
                p = p.cpu() 
                # Use top_k sampling to get the index of the next word
                top_k = 5
                p, top_indices = p.topk(top_k)
                #print(top_indices)
                top_indices = top_indices.numpy().squeeze()
                # Select the likely next word index with some element of randomness
                p = p.numpy().squeeze()
                token_id = int(np.random.choice(top_indices, p=p/p.sum()))
                
                # Save to output sequence
                outputs[i] = token_id
                
                # Create next input from output token
                # inputs size: (1, 1, embedding_size=512)
                input_token = torch.Tensor([[token_id]]).long()
                inputs = self.embedding(input_token)

                if token_id == 1:
                    # <end>
                    break
        
        return outputs