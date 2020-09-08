from DINAE import *

def FP_solver(dict_global_Params,genFilename,x_train,x_train_missing,mask_train,gt_train,\
                        meanTr,stdTr,x_test,x_test_missing,mask_test,gt_test,lday_test,x_train_OI,x_test_OI,encoder,decoder,model_AE,DimCAE):

    # import Global Parameters
    for key,val in dict_global_Params.items():
        exec("globals()['"+key+"']=val")

    
    # ***************** #
    # model compilation #
    # ***************** #

    # model fit
    NbProjection   = [0,2,5,7]
    if flagTrOuputWOMissingData==0:
        lrUpdate   = [1e-4,1e-5,1e-6,1e-7]
    else:
        lrUpdate   = [1e-3,1e-4,1e-5,1e-6]
    IterUpdate     = [0,3,10,15,20,25,30,35,40]
    val_split      = 0.1
    
    ## initialization
    x_train_init = np.copy(x_train_missing)
    x_test_init  = np.copy(x_test_missing)

    comptUpdate = 0

    # ******************** #
    # Start Learning model #
    # ******************** #
        
    print("..... Start learning AE model %d FP/Grad %s"%(flagAEType,flagOptimMethod))
    for iter in range(0,Niter):
        if iter == IterUpdate[comptUpdate]:
            # update DINConvAE model
            NBProjCurrent = NbProjection[comptUpdate]
            print("..... Update/initialize number of projections in DINCOnvAE model # %d"%(NbProjection[comptUpdate]))
            global_model_FP,global_model_FP_Masked = define_DINConvAE(NbProjection[comptUpdate],model_AE,x_train.shape,\
                                                                          flag_MultiScaleAEModel,flagUseMaskinEncoder,\
                                                                          size_tw,include_covariates,N_cov)
            if flagTrOuputWOMissingData == 1:
                global_model_FP.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
            else:
                global_model_FP_Masked.compile(loss='mean_squared_error',optimizer=keras.optimizers.Adam(lr=lrUpdate[comptUpdate]))
            if comptUpdate < len(NbProjection)-1:
                comptUpdate += 1
        
        # gradient descent iteration            
        if flagTrOuputWOMissingData == 1:
            history = global_model_FP.fit([x_train_init,mask_train],gt_train,
                  batch_size=batch_size,
                  epochs = NbEpoc,
                  verbose = 1, 
                  validation_split=val_split)
        else:
            history = global_model_FP_Masked.fit([x_train_init,mask_train],[np.zeros((x_train_init.shape[0],1)),gt_train],
                  batch_size=batch_size,
                  epochs = NbEpoc,
                  verbose = 1, 
                  validation_split=val_split)

        # *********************** #
        # Prediction on test data #
        # *********************** #

        # trained full-model
        x_train_pred    = global_model_FP.predict([x_train_init,mask_train])
        x_test_pred     = global_model_FP.predict([x_test_init,mask_test])

        # trained AE applied to gap-free data
        rec_AE_Tr       = model_AE.predict([x_train,np.ones((mask_train.shape))])
        rec_AE_Tt       = model_AE.predict([x_test,np.ones((mask_test.shape))])

        # remove additional covariates from variables
        if include_covariates == True:
            mask_train_wc, x_train_wc, x_train_init_wc, x_train_missing_wc,\
            mask_test_wc, x_test_wc, x_test_init_wc, x_test_missing_wc,\
            meanTr_wc, stdTr_wc=\
            mask_train, x_train, x_train_init, x_train_missing,\
            mask_test, x_test, x_test_init, x_test_missing,\
            meanTr, stdTr
            index = np.arange(0,(N_cov+1)*size_tw,(N_cov+1))
            mask_train      = mask_train[:,:,:,index]
            x_train         = x_train[:,:,:,index]
            x_train_init    = x_train_init[:,:,:,index]
            x_train_missing = x_train_missing[:,:,:,index]
            mask_test      = mask_test[:,:,:,index]
            x_test         = x_test[:,:,:,index]
            x_test_init    = x_test_init[:,:,:,index]
            x_test_missing = x_test_missing[:,:,:,index]
            meanTr = meanTr[0]
            stdTr  = stdTr[0]

        # save models
        genSuffixModel=save_Models(dict_global_Params,genFilename,NBProjCurrent,encoder,decoder,iter)
 
        idT = int(np.floor(x_test.shape[3]/2))
        saved_path = dirSAVE+'/saved_path_%03d'%(iter)+'_FP_'+suf1+'_'+suf2+'.pickle'
        saved_path_Tr = dirSAVE+'/saved_path_%03d'%(iter)+'_FP_'+suf1+'_'+suf2+'_train.pickle'
        if flagloadOIData == 1:
            # Save DINAE result         
            with open(saved_path, 'wb') as handle:
                pickle.dump([((gt_test*stdTr)+meanTr+x_test_OI)[:,:,:,idT],((x_test_missing*stdTr)+meanTr+x_test_OI)[:,:,:,idT],\
                         ((x_test_pred*stdTr)+meanTr+x_test_OI)[:,:,:,idT],((rec_AE_Tt*stdTr)+meanTr+x_test_OI)[:,:,:,idT],x_test_OI[:,:,:,idT]], handle)
            with open(saved_path_Tr, 'wb') as handle:
                pickle.dump([((gt_train*stdTr)+meanTr+x_train_OI)[:,:,:,idT],((x_train_missing*stdTr)+meanTr+x_train_OI)[:,:,:,idT],\
                         ((x_train_pred*stdTr)+meanTr+x_train_OI)[:,:,:,idT],((rec_AE_Tr*stdTr)+meanTr+x_train_OI)[:,:,:,idT],x_train_OI[:,:,:,idT]], handle)
        else:
            # Save DINAE result         
            with open(saved_path, 'wb') as handle:
                pickle.dump([((gt_test*stdTr)+meanTr)[:,:,:,idT],((x_test_missing*stdTr)+meanTr)[:,:,:,idT],\
                         ((x_test_pred*stdTr)+meanTr)[:,:,:,idT],((rec_AE_Tt*stdTr)+meanTr)[:,:,:,idT], x_test_OI[:,:,:,idT]], handle)
            with open(saved_path_Tr, 'wb') as handle:
                pickle.dump([((gt_train*stdTr)+meanTr)[:,:,:,idT],((x_train_missing*stdTr)+meanTr)[:,:,:,idT],\
                         ((x_train_pred*stdTr)+meanTr)[:,:,:,idT],((rec_AE_Tr*stdTr)+meanTr)[:,:,:,idT], x_train_OI[:,:,:,idT]], handle)

        # reset variables with additional covariates
        if include_covariates == True:
            mask_train, x_train, x_train_init, x_train_missing,\
            mask_test, x_test, x_test_init, x_test_missing,\
            meanTr, stdTr=\
            mask_train_wc, x_train_wc, x_train_init_wc, x_train_missing_wc,\
            mask_test_wc, x_test_wc, x_test_init_wc, x_test_missing_wc,\
            meanTr_wc, stdTr_wc

