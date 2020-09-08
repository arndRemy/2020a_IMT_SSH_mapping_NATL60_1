from DINAE import *

def save_Models(dict_global_Params,genFilename,NBProjCurrent,encoder,decoder,iter,*args):   

    if len(args)>0:
        gradModel = args[0] 
        gradMaskModel = args[1] 
        NBGradCurrent = args[2]

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    alpha=[1.,0.,0.]
    genSuffixModel = '_Alpha%03d'%(100*alpha[0]+10*alpha[1]+alpha[2])

    if flagTrOuputWOMissingData == 1:
        genSuffixModel = genSuffixModel+'_AETRwoMissingData'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))
    else:
        genSuffixModel = genSuffixModel+'_AE'+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))+'_Nproj'+str('%02d'%(NBProjCurrent))

    fileMod = dirSAVE+genFilename+genSuffixModel+'_Encoder_iter%03d'%(iter)+'.mod'
    print('.................. Encoder '+fileMod)
    encoder.save(fileMod)
    fileMod = dirSAVE+genFilename+genSuffixModel+'_Decoder_iter%03d'%(iter)+'.mod'
    print('.................. Decoder '+fileMod)
    decoder.save(fileMod)

    return genSuffixModel
