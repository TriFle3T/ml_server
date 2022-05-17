import torch
import numpy as np

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

        emotion = {0:'두려움',1:'놀람',2:'분노',3:'슬픔',4:'중립',5:'행복',6:'혐오'}
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

            test_eval=[]
            for i in out:
                logits = i.detach().cpu().numpy()  
                test_eval.append(emotion[np.argmax(logits)])
            
            test_per=[]
            for i,logit in enumerate(logits):
                test_per.append(round((1/(1+np.exp((-1)*logit)))*100))
                print("{}:{}%".format(emotion[i],test_per[i]))
            print(">> 입력하신 내용에서 " + test_eval[0] + "가(이) 느껴집니다.")
            return test_per[5],test_per[2],test_per[6],test_per[0],test_per[3],test_eval[0]