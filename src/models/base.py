class VenusModelOutput:
    '''
    BaseModelOutput contains the output of a BaseModel.
    '''
    
    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)

        self.hidden_states = kwargs.get('hidden_states', None)

        self.logits = kwargs.get('logits', None)

        self.loss = kwargs.get('loss', None)