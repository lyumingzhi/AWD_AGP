MODEL_ORDER = [
    "RFRnet",
    "GMCNNnet",
    "Crfillnet",
    "EdgeConnet",
    "Gennet",
    "FcFnet",
    "Matnet",
    "WDModel",
    "DBWEModel",
    "SLBRModel",
]


def _build_model(name, opt):
    if name == "RFRnet":
        from inpainting.AWD_AGP.source_models.RFR import RFRNetModelAPI
        return RFRNetModelAPI(opt.dataset, opt)
    if name == "GMCNNnet":
        from inpainting.AWD_AGP.source_models.GMCNN import GMCNNAPI
        return GMCNNAPI(opt.dataset, opt)
    if name == "Crfillnet":
        from inpainting.AWD_AGP.source_models.Crfill import CrfillAPI
        return CrfillAPI(opt.dataset, opt)
    if name == "EdgeConnet":
        from inpainting.AWD_AGP.source_models.EdgeCon import ConnectEdgeAPI
        return ConnectEdgeAPI(opt)
    if name == "Gennet":
        from inpainting.AWD_AGP.source_models.Generative import Generative
        return Generative(dataset=opt.dataset, opt=opt)
    if name == "FcFnet":
        from inpainting.AWD_AGP.source_models.FcF import FcFnetAPI
        return FcFnetAPI(dataset=opt.dataset, device="cuda", opt=opt)
    if name == "Matnet":
        from inpainting.AWD_AGP.source_models.Mat import MatAPI
        return MatAPI(opt)
    if name == "WDModel":
        from inpainting.AWD_AGP.source_models.WDnet import WDnet
        return WDnet(opt)
    if name == "DBWEModel":
        from inpainting.AWD_AGP.source_models.DBWEModel import DBWRnet
        return DBWRnet(opt)
    if name == "SLBRModel":
        from inpainting.AWD_AGP.source_models.SLBRnet import SLBRnet
        return SLBRnet(opt)
    raise ValueError(f"Unknown model name {name!r}. Choose from {MODEL_ORDER}")


def load_models(opt, model_names=None, as_dict=False):
    selected = MODEL_ORDER if model_names is None else list(dict.fromkeys(model_names))
    models = {}
    for name in selected:
        model = _build_model(name, opt)
        if hasattr(model, "eval"):
            model.eval()
        models[name] = model
    if as_dict:
        return models
    return tuple(models.get(name) for name in MODEL_ORDER)
