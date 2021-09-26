from .MEDFE import MEDFE
from .RefineNet import RefineNet

def create_model_I(opt):
    model = MEDFE(opt)
    print("model [%s] was created" % (model.name()))
    return model

def create_model_R(opt):
    model = RefineNet(opt)
    print("model [%s] was created" % (model.name()))
    return model

