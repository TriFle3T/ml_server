import torch
import numpy as np
import pandas as pd
from utils.dataset import BERTDataset

class BERTpredict():
    def __init__(self, model, tok, vocab, max_len, batch_size, device):
        self.model = model
        self.tok = tok
        self.vocab = vocab
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        
                 
    def predict(self, predict_sentence):

        emotion = {0:'행복',1:'분노',2:'혐오',3:'두려움',4:'중립',5:'슬픔',6:'놀람'}
        data = [predict_sentence, '0']
        dataset_another = [data]

        another_test = BERTDataset(dataset_another, 0, 1, self.tok, self.vocab, self.max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=self.batch_size, num_workers=5)
        
        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)

            valid_length= valid_length
            label = label.long().to(self.device)

            out = self.model(token_ids, valid_length, segment_ids)

            test_per=[]
            for i in out:
                logits = i.detach().cpu().numpy()  
                for logit in logits:
                  test_per.append(round((1/(1+np.exp((-1)*logit)))*100))
            
            em = pd.Series(test_per,index = [0,1,2,3,4,5,6])
            res = em.values.tolist()
            res.append(int(em.sort_values().index[6]))
            idx = 5 if res[7] == 6 else 6
            res.append(int(em.sort_values().index[idx]))
            return res
