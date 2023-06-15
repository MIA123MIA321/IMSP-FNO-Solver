from Fourier import *

        
def train(x_train,y_train,x_test,y_test,device,model_pth='',name='',
          threshold=0.01,early_stop=False,start_lr = 2e-3,end_lr = 1e-5,
          batch_size = 16,epochs=200,loss_level=1,FNO_ARG = (12,32,4),
          NET_DIR = '/data/liuziyang/Programs/pde_solver/Network/'):
    y_data_len = len(y_train)
    if y_data_len == 1:
        train_data = Data.TensorDataset(x_train[0],x_train[1], y_train[0])
        test_data = Data.TensorDataset(x_test[0],x_test[1], y_test[0])
    elif y_data_len == 2:
        train_data = Data.TensorDataset(x_train[0],x_train[1], y_train[0], y_train[1])
        test_data = Data.TensorDataset(x_test[0],x_test[1], y_test[0], y_test[1])
    train_loader = Data.DataLoader(dataset = train_data, batch_size = batch_size)
    test_loader = Data.DataLoader(dataset = test_data, batch_size = batch_size)
    SEED_SET(1234)
    if model_pth=='':
        model = FNO2d_modified(FNO_ARG[0],FNO_ARG[1],FNO_ARG[2]).to(device)
    else:
        model = torch.load(NET_DIR + model_pth + '.pth')
    myloss = LpLoss()
    optimizer = optim.Adam(model.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=end_lr)
    
    def ERROR(model,data,loss_level):
        q = data[0][:,0]
        f = data[1]
        u0 = data[2]
        loss_rel, loss_rel1 = 0., 0.
        if loss_level >=1:
            for i in range(f.shape[1]):
                out = model(torch.cat([q,f[:,i]],1))
                loss_rel += myloss.rel(out,u0[:,i])
            loss_rel /= f.shape[1]
        if loss_level >=2:
            u1 = data[3]
            for i in range(f.shape[1]):
                out1 = model(torch.cat([q,u0[:,i]],1))
                loss_rel1 += myloss.rel(out1,u1[:,i])
            loss_rel1 /= f.shape[1]
        if loss_level == 1:
            return loss_rel
        elif loss_level==2:
            return loss_rel,loss_rel1
        
    for ep in range(epochs):
        t1 = default_timer()
        train_l2 = np.zeros(loss_level)
        for data in train_loader:
            loss = 0.
            optimizer.zero_grad()
            LOSS = ERROR(model,data,loss_level)
            if loss_level == 1:
                loss += LOSS
            elif loss_level ==2:
                loss += LOSS[0] + 0.1*(ep/epochs)**2*LOSS[1]
            loss.backward()
            optimizer.step()
            for i in range(loss_level):
                try:
                    train_l2[i] += LOSS[i].item()
                except:
                    train_l2[i] += LOSS.item()
        scheduler.step()

        model.eval()
        test_l2 = np.zeros(loss_level)
        with torch.no_grad():
            for dataaa in test_loader:
                LOSSS = ERROR(model,dataaa,loss_level)
                for i in range(loss_level):
                    try:
                        test_l2[i] += LOSSS[i].item()
                    except:
                        test_l2[i] += LOSSS.item()
        
        
        train_l2 = tuple(np.around(train_l2*100/ntrain, decimals=2))
        test_l2 = tuple(np.around(test_l2*100/ntest, decimals=2))
        
        t2 = default_timer()
        if early_stop:
            if train_l2[0]/100<threshold and test_l2[0]/100<threshold:
                print ('Epoch [{}/{}], train: {:.2f}%, test: {:.2f}%, Time per epoch: {:.4f}' 
                        .format(ep+1, epochs, train_l2_1,train_l2_2,test_l2_1,test_l2_2, t2 - t1))
                break
        if ep==0 or (ep + 1) % 10 == 0:
            print ('Epoch [{}/{}], train: {}%, test: {}%, Time per epoch: {:.4f}' 
                    .format(ep+1, epochs, train_l2,test_l2, t2 - t1))
    if name != '':
        torch.save(model,NET_DIR + name + '.pth')
        
if __name__ == '__main__':
    k = 20
    qmethod = 'G'
    R = 200
    label1 = 'R200'
    label2 = 'R200'
    scheme = 'PML'
    angle_total = 64
    angle_TYPE = 'P'
    angle_for_test = 4
    outputsize = 64
    angle_mode = 'uniform'
    NS_return = 'T'
    ntrain = 1024
    ntest = 32
    maxq = 0.1
    FNO_ARG = (12,32,4)
    train_filename = 'k{}_{}_{}_{}_{},{},{}_{}_{}_NS{}_{}'.format(k, scheme, ntrain, angle_TYPE, angle_for_test,angle_total,
                                                 angle_mode, qmethod, maxq, NS_return, label1)
    test_filename = 'k{}_{}_{}_{}_{},{},{}_{}_{}_NS{}_{}'.format(k, scheme, ntest, angle_TYPE, angle_for_test,angle_total,
                                             angle_mode, qmethod, maxq, NS_return, label2)
    NET_name = 'k{}_{}_{}_{},{},{}_{}_{}_NS{}_{}_{},{},{}'.format(k, scheme, angle_TYPE, angle_for_test,angle_total,
                                             angle_mode, qmethod, maxq, NS_return, label2,
                                             FNO_ARG[0],FNO_ARG[1],FNO_ARG[2])
    q_train, wave_train, u_train0, u_train1 = load_data(train_filename, device = device,
                    NS_return = NS_return, Usage = 'Train',output_size = outputsize)
    q_test, wave_test, u_test0, u_test1 = load_data(test_filename, device = device,
                        NS_return = NS_return, Usage = 'Train',output_size = outputsize)
    x_train,x_test = (q_train,wave_train),(q_test,wave_test)
    y_train,y_test = (u_train0,u_train1),(u_test0,u_test1)

    
    train(x_train,y_train,x_test,y_test,device,
      '',NET_name+'_0',FNO_ARG=FNO_ARG,
      start_lr=2e-3,end_lr=1e-5,epochs=500,early_stop=False,loss_level=1)
    
    train(x_train,y_train,x_test,y_test,device,
      NET_name+'_0', NET_name+'_1',
      start_lr=1e-4,end_lr=1e-5,epochs=200,early_stop=False,loss_level=2)
    
    
    # nohup python -u src/train.py >> .train.log 2>&1 &
    
    
    
    
    
    
    # train(x_train,y_train,x_test,y_test,device,
    #   NET_name+'_1', NET_name+'_2',
    #   start_lr=1e-4,end_lr=1e-5,epochs=200,early_stop=False,loss_level=1)