from transformers import (
    BertModel,
    EsmModel,
    T5EncoderModel
)
import torch
import torch.nn as nn

class PLMEncoder(nn.Module):

    def __init__(self, args):

        super().__init__()

        if args.plm_type == 'bert':
            self.plm = BertModel.from_pretrained(args.plm_dir)
        elif args.plm_type in ['esm', 'saprot']:
            self.plm = EsmModel.from_pretrained(args.plm_dir)
        elif args.plm_type in ['ankh', 't5']:
            self.plm = T5EncoderModel.from_pretrained(args.plm_dir)
        else:
            raise ValueError("Invalid PLM type.")

        if args.plm_freeze:
            for param in self.plm.parameters():
                param.requires_grad = False

        self.output_dim = self.plm.config.hidden_size

    def forward(self, batch):
        
        return self.get_embedding(batch[0], batch[1])
    
    @torch.no_grad()
    def get_embedding(self, input_ids, attention_mask):
        
        plm_output = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        
        return plm_output.last_hidden_state