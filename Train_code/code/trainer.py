import os
import torch
from decimal import Decimal

import utility
from model.IKSConv import IKSConv
from tqdm import tqdm


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

        # store psnr and sparsity and lastDecayEpoch
        self.avg_psnr=[]
        self.sparsity=[]
        self.lastDecayEpoch=1
        self.myBestPSE=[-1,-1,-1]


    def train(self):
        # self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        # todo user warning
        lr = self.scheduler.get_last_lr()[0]

        # test pre_trained model #################################################################
        if epoch==0:
            name=''
            if self.args.sparseMode not in ['lw','fw','kw','pw']:
                name+='Pre '
            else:
                name+='Pre+ IKS({}:{})'.format(self.args.sparseMode,self.args.t0)
                if self.args.targetIKSSparsity == 0:
                    name+='+ noAD'
                else:
                    name+='+ AD(D{}) + k{}'.format(self.args.decayDistance,self.args.k)

            self.ckp.write_log('modules: {}'.format(name))

            # calc params
            TOT_param=sum(p.numel() for p in self.model.parameters())
            IKS_thre_param=0
            for n, m in self.model.named_modules():
                if isinstance(m, IKSConv):
                    IKS_thre_param+=m.threshold.numel()
                    # todo, init decay for finetune
                    if self.args.init_decay != 1:
                        m.decayThreshold(self.args.init_decay)

            TOT_param-=IKS_thre_param
            print('Origin_model parameters:\t{:.2f}M\nIKS_threshold parameters:\t{:.2f}M\nIKS_rate:\t\t\t{:.4f}'.
                  format(TOT_param/1e6,IKS_thre_param/1e6,IKS_thre_param/TOT_param))

            # test PSNR on test set
            self.model.eval()
            with torch.no_grad():
                avg_acc=0
                for idx_data, d in enumerate(self.loader_test):
                    for idx_scale, scale in enumerate(self.scale):
                        d.dataset.set_scale(idx_scale)
                        eval_acc = 0
                        for lr0, hr, filename in tqdm(d, ncols=80):
                            lr0, hr = self.prepare([lr0, hr])
                            sr = self.model(lr0, idx_scale)
                            sr = utility.quantize(sr, self.args.rgb_range)

                            eval_acc += utility.calc_psnr(
                                sr, hr, scale, self.args.rgb_range
                            )

                        self.ckp.write_log(
                            'Pretrained Model: [{} x{}]\tPSNR: {:.6f}'.format(
                                d.dataset.name,
                                scale,
                                eval_acc / len(d)))
                        avg_acc+=eval_acc / len(d)
                avg_acc/=len(self.loader_test)
                self.ckp.write_log('Pretrained Model Avg PSNR: {:.6f}'.format(avg_acc))
        ############################################################################################################

        self.ckp.write_log(
            '\n[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
            # todo : test the feasibility of modified code
            # break

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        self.scheduler.step()


    def test(self, ssim=False):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('Evaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test),len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            avg_acc=0
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare([lr, hr])
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )

                        if self.args.save_results:
                            self.ckp.save_results(d.dataset.name, filename[0], save_list, scale)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    avg_acc+=self.ckp.log[-1, idx_data, idx_scale]
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale]+1
                        )
                    )

                    # add PSNR to tensorboard result
                    self.ckp.writer.add_scalar("EPOCH_PSNR_x{}_{}".format(scale, d.dataset.name), self.ckp.log[-1, idx_scale], epoch)
                    # psnr in self.ckp.log[epoch, idx_data, idx_scale]

            avg_acc /= len(self.loader_test)
            self.ckp.write_log('Average PSNR: {:.6f}'.format(avg_acc))
        self.ckp.write_log(
            'Total time: {:.2f}s'.format(timer_test.toc()), refresh=True
        )

        # Storing sparsity and threshold statistics for IKSConv models
        count = 0
        remain = 0.0
        for n, m in self.model.named_modules():
            if isinstance(m, IKSConv):
                sparsity, total_params, thresh = m.getSparsity()
                remain += int(((100 - sparsity) / 100) * total_params)
                count += total_params
        if count!=0:
            # IKS_rate: RCANG10R20x4-0.9717 RCANG5R10x4-0.9211 EDSRx4-0.8897, now IKSSparsity for all
            IKS_sparsity = (100 - (100 * remain / count))
            self.ckp.writer.add_scalar("EPOCH_sparsity", IKS_sparsity, epoch)
            self.ckp.write_log("IKS_sparsity:{:.8f}".format(IKS_sparsity))
            self.sparsity.append(IKS_sparsity)
        # ***

        # save my best model
        if not self.args.test_only:
            psnr_tmp=0
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    psnr_tmp+=self.ckp.log[-1, idx_data, idx_scale]
            self.avg_psnr.append(psnr_tmp/len(self.loader_test))

            if count==0:
                best = self.ckp.log.max(0)
                self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
            else:
                my_best=(self.sparsity[-1]>self.args.targetIKSSparsity-5 and self.avg_psnr[-1]>self.myBestPSE[0])
                if my_best:
                    self.myBestPSE=[self.avg_psnr[-1],self.sparsity[-1],epoch]
                self.ckp.save(self, epoch, is_best=my_best,lde=-1)
                self.ckp.write_log(
                    '(MyBest: {:.3f} and {:.2f}% @epoch {})'.format(
                        self.myBestPSE[0],
                        self.myBestPSE[1],
                        self.myBestPSE[2]
                    )
                )
        # ***

        # # save multi-stage model
        # if not self.args.test_only:
        #     if epoch==self.lastDecayEpoch+1:
        #         self.currentPsnrBase=0
        #     if self.psnr[-1]>self.currentPsnrBase :
        #         self.currentPsnrBase=self.psnr[-1]
        #         self.ckp.save(self, epoch, is_best=True, lde=self.lastDecayEpoch)
        #         self.ckp.write_log(
        #             '(Multi-stage model- lde{} saved: {:.3f} and {:.2f}% @epoch {})'.format(
        #                 self.lastDecayEpoch,self.psnr[-1],self.sparsity[-1],epoch
        #             )
        #         )
        # # ***

        # # threshold decay ****************************************************
        D = self.args.decayDistance
        K = self.args.k
        if self.args.targetIKSSparsity != 0 and self.args.sparseMode in ['lw', 'fw', 'kw', 'pw']:
            if epoch - self.lastDecayEpoch > D and epoch < self.args.epochs - D:
                # Calculate S_cur: already in self.sparsity[-1]
                if abs(self.args.targetIKSSparsity + 0.5 - self.sparsity[-1]) > 0.5: # tar~tar+1
                    self.lastDecayEpoch = epoch
                    decayRatio = 1 + K * (self.sparsity[-1] - self.args.targetIKSSparsity)  # linear
                    for name, param in self.model.named_modules():
                        if isinstance(param, IKSConv):
                            param.decayThreshold(decayRatio)
        # *******************************************************************************

        # ###todo SA ****************************************************
        # D = self.args.decayDistance
        # K = self.args.k
        # if self.args.targetIKSSparsity != 0 and self.args.sparseMode in ['lw', 'fw', 'kw', 'pw']:
        #     if epoch - self.lastDecayEpoch > D and epoch < self.args.epochs - D:
        #         # Calculate S_cur: already in self.sparsity[-1]
        #         if self.sparsity[-1] < self.args.targetIKSSparsity: # tar~tar+1
        #             self.lastDecayEpoch = epoch
        #             decayRatio = 0.95
        #             for name, param in self.model.named_modules():
        #                 if isinstance(param, IKSConv):
        #                     param.decayThreshold(decayRatio)
        # # *******************************************************************************

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test(ssim=True)
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs

