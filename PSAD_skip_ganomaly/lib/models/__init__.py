##
import importlib
from .. import plus_variable
##
def load_model(opt, dataloader):
    """ Load model based on the model name.

    Arguments:
        opt {[argparse.Namespace]} -- options
        dataloader {[dict]} -- dataloader class

    Returns:
        [model] -- Returned model
    """
    model_name = opt.model
    model_path = f"lib.models.{model_name}"
    model_lib  = importlib.import_module(model_path)
    
    if 'late' in plus_variable.version:
        model = getattr(model_lib, 'PSAD_latefusion_Skipganomaly')
    else:
        model = getattr(model_lib, 'PSAD_others_Skipganomaly')
    return model(opt, dataloader)