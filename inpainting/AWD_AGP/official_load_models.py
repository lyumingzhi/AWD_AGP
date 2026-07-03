def load_models(opt):
    from inpainting.AWD_AGP.source_models.Crfill import CrfillAPI
    from inpainting.AWD_AGP.source_models.FcF import FcFnetAPI
    from inpainting.AWD_AGP.source_models.RFR import RFRNetModelAPI
    from inpainting.AWD_AGP.source_models.GMCNN import GMCNNAPI
    from inpainting.AWD_AGP.source_models.EdgeCon import ConnectEdgeAPI
    from inpainting.AWD_AGP.source_models.Generative import Generative
    from inpainting.AWD_AGP.source_models.Mat import MatAPI
    from inpainting.AWD_AGP.source_models.WDnet import WDnet
    from inpainting.AWD_AGP.source_models.DBWEModel import DBWRnet
    from inpainting.AWD_AGP.source_models.SLBRnet import SLBRnet

    RFRnet = RFRNetModelAPI(opt.dataset, opt)
    GMCNNnet = GMCNNAPI(opt.dataset, opt)
    Crfillnet = CrfillAPI(opt.dataset, opt)
    EdgeConnet = ConnectEdgeAPI(opt)
    Gennet = Generative(dataset=opt.dataset, opt=opt)
    FcFnet = FcFnetAPI(dataset=opt.dataset, device='cuda', opt=opt)
    Matnet = MatAPI(opt)

    WDModel = WDnet(opt)
    DBWEModel = DBWRnet(opt)
    SLBRModel = SLBRnet(opt)

    RFRnet.eval()
    GMCNNnet.eval()
    Crfillnet.eval()
    EdgeConnet.eval()
    Gennet.eval()
    FcFnet.eval()
    Matnet.eval()

    return RFRnet, GMCNNnet, Crfillnet, EdgeConnet, Gennet, FcFnet, Matnet, WDModel, DBWEModel, SLBRModel
