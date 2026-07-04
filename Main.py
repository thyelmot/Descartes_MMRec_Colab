import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, GCNLayer, SpAdjDropEdge
from DataHandler import DataHandler
from VelocityModel import VelocityModel
from FlowMatching import GraphFlowMatching
from Diffusion import GaussianDiffusion, FlowMatching_Original, Denoise
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
from scipy.sparse import coo_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')

        recallMax = 0
        ndcgMax = 0
        precisionMax = 0
        bestEpoch = 0
        start_epoch = 0

        # ── Auto-resume từ checkpoint ──────────────────────────────────────
        checkpoint_dir = getattr(args, 'checkpoint_dir', '.')
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{args.model_type}_{args.data}.pth")
        if os.path.exists(checkpoint_path):
            log(f'[Resume] Tìm thấy checkpoint: {checkpoint_path}')
            start_epoch = self.load_checkpoint(checkpoint_path)
            log(f'[Resume] Tiếp tục từ epoch {start_epoch}')
        else:
            log('Model Initialized (no checkpoint found, training from scratch)')
        # ──────────────────────────────────────────────────────────────────

        try:
            for ep in range(start_epoch, args.epoch):
                # --- Descartes-MMRec: Giai đoạn 1 - Lọc Hoài Nghi (Core Graph) ---
                log(f'Epoch {ep}: Cắt tỉa đồ thị (Doubt Pruning) tạo Core Graph...')
                iEmbeds = self.model.getItemEmbeds().detach()
                uEmbeds = self.model.getUserEmbeds().detach()
                image_feats = self.model.getImageFeats().detach()
                text_feats = self.model.getTextFeats().detach()
                
                # Gọi Doubt Evaluator để lấy Doubt Score (Không làm thưa đồ thị)
                self.S_doubt = self.model.doubt_evaluator(
                    self.handler.trnMat, 
                    uEmbeds, image_feats, text_feats
                )
                # -----------------------------------------------------------------
                
                tstFlag = (ep % args.tstEpoch == 0)
                reses = self.trainEpoch(ep)
                log(self.makePrint('Train', ep, reses, tstFlag))
                if tstFlag:
                    reses = self.testEpoch()
                    if (reses['Recall'] > recallMax):
                        recallMax = reses['Recall']
                        ndcgMax = reses['NDCG']
                        precisionMax = reses['Precision']
                        bestEpoch = ep
                        self.save_checkpoint(ep)
                    log(self.makePrint('Test', ep, reses, tstFlag))
                print()
        except KeyboardInterrupt:
            print("\n[CẢNH BÁO] Training bị ngắt bởi người dùng. Sẽ tiến hành lưu lại kết quả tốt nhất hiện tại...")

        print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)
        return {
            'best_epoch': bestEpoch,
            'recall': recallMax,
            'ndcg': ndcgMax,
            'precision': precisionMax
        }

    def save_checkpoint(self, epoch):
        checkpoint = {
            'model': self.model.state_dict(),
            'epoch': epoch + 1,  # lưu epoch tiếp theo để resume đúng
            'recall_max': 0,  # placeholder
        }
        if args.model_type == 'flowmatching_optimized':
            checkpoint['velocity_image'] = self.velocity_model_image.state_dict()
            checkpoint['velocity_text'] = self.velocity_model_text.state_dict()
            if args.data == 'tiktok':
                checkpoint['velocity_audio'] = self.velocity_model_audio.state_dict()
        else:
            checkpoint['denoise_image'] = self.denoise_model_image.state_dict()
            checkpoint['denoise_text'] = self.denoise_model_text.state_dict()
            if args.data == 'tiktok':
                checkpoint['denoise_audio'] = self.denoise_model_audio.state_dict()

        checkpoint_dir = getattr(args, 'checkpoint_dir', '.')
        os.makedirs(checkpoint_dir, exist_ok=True)
        out_file = os.path.join(checkpoint_dir, f"checkpoint_{args.model_type}_{args.data}.pth")
        torch.save(checkpoint, out_file)
        log(f'Checkpoint saved to {out_file} at epoch {epoch}')

    def load_checkpoint(self, path):
        """Load checkpoint và trả về epoch bắt đầu."""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model'])
        if args.model_type == 'flowmatching_optimized':
            if 'velocity_image' in checkpoint:
                self.velocity_model_image.load_state_dict(checkpoint['velocity_image'])
            if 'velocity_text' in checkpoint:
                self.velocity_model_text.load_state_dict(checkpoint['velocity_text'])
            if args.data == 'tiktok' and 'velocity_audio' in checkpoint:
                self.velocity_model_audio.load_state_dict(checkpoint['velocity_audio'])
        else:
            if 'denoise_image' in checkpoint:
                self.denoise_model_image.load_state_dict(checkpoint['denoise_image'])
            if 'denoise_text' in checkpoint:
                self.denoise_model_text.load_state_dict(checkpoint['denoise_text'])
            if args.data == 'tiktok' and 'denoise_audio' in checkpoint:
                self.denoise_model_audio.load_state_dict(checkpoint['denoise_audio'])
        return checkpoint.get('epoch', 0)

    def prepareModel(self):
        if args.data == 'tiktok':
            self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).to(device)
        else:
            self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        if args.model_type == 'flowmatching_optimized':
            self.flow_matching = GraphFlowMatching(sigma_min=1e-4).to(device)
            out_dims = eval(args.dims) + [args.item]
            in_dims = out_dims[::-1]
            self.velocity_model_image = VelocityModel(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)
            self.velocity_opt_image = torch.optim.Adam(self.velocity_model_image.parameters(), lr=args.lr, weight_decay=0)
            self.velocity_model_text = VelocityModel(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)
            self.velocity_opt_text = torch.optim.Adam(self.velocity_model_text.parameters(), lr=args.lr, weight_decay=0)
            if args.data == 'tiktok':
                self.velocity_model_audio = VelocityModel(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)
                self.velocity_opt_audio = torch.optim.Adam(self.velocity_model_audio.parameters(), lr=args.lr, weight_decay=0)
        else:
            if args.model_type == 'flowmatching_original':
                self.diffusion_model = FlowMatching_Original(args.steps).to(device)
            else:
                self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).to(device)

            out_dims = eval(args.dims) + [args.item]
            in_dims = out_dims[::-1]
            self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)
            self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)
            self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)
            self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)
            if args.data == 'tiktok':
                self.denoise_model_audio = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).to(device)
                self.denoise_opt_audio = torch.optim.Adam(self.denoise_model_audio.parameters(), lr=args.lr, weight_decay=0)

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def buildUIMatrix(self, u_list, i_list, edge_list):
        mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return torch.sparse.FloatTensor(idxs, vals, shape).to(device)

    def trainEpoch(self, current_epoch):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epRecLoss, epClLoss, epAdvLoss = 0, 0, 0, 0
        epDiLoss = 0
        epDiLoss_image, epDiLoss_text = 0, 0
        if args.data == 'tiktok':
            epDiLoss_audio = 0
        steps = trnLoader.dataset.__len__() // args.batch

        import time
        # --- Fast DataLoader replacement ---
        dataset_tensor = self.handler.diffusionData.data
        dataset_size = dataset_tensor.shape[0]
        perm_indices = torch.randperm(dataset_size)
        num_batches = (dataset_size + args.batch - 1) // args.batch

        log(f'Batch size: {args.batch}, Num batches: {num_batches}')
        
        for i in range(num_batches):
            t0 = time.time()
            batch_index = perm_indices[i*args.batch : (i+1)*args.batch]
            batch_item = dataset_tensor[batch_index]
            batch_item, batch_index = batch_item.to(device), batch_index.to(device)

            iEmbeds = self.model.getItemEmbeds().detach()
            uEmbeds = self.model.getUserEmbeds().detach()

            image_feats = self.model.getImageFeats().detach()
            text_feats = self.model.getTextFeats().detach()
            if args.data == 'tiktok':
                audio_feats = self.model.getAudioFeats().detach()

            if args.model_type == 'flowmatching_optimized':
                self.velocity_opt_image.zero_grad()
                self.velocity_opt_text.zero_grad()
                if args.data == 'tiktok':
                    self.velocity_opt_audio.zero_grad()

                cfm_loss_image, msi_loss_image = self.flow_matching.training_losses(
                    self.velocity_model_image, batch_item, iEmbeds, batch_index, image_feats, S_doubt=getattr(self, 'S_doubt', None), omega=args.omega)
                cfm_loss_text, msi_loss_text = self.flow_matching.training_losses(
                    self.velocity_model_text, batch_item, iEmbeds, batch_index, text_feats, S_doubt=getattr(self, 'S_doubt', None), omega=args.omega)
                if args.data == 'tiktok':
                    cfm_loss_audio, msi_loss_audio = self.flow_matching.training_losses(
                        self.velocity_model_audio, batch_item, iEmbeds, batch_index, audio_feats, S_doubt=getattr(self, 'S_doubt', None), omega=args.omega)

                loss_image = cfm_loss_image.mean() + msi_loss_image.mean() * args.e_loss
                loss_text = cfm_loss_text.mean() + msi_loss_text.mean() * args.e_loss
                if args.data == 'tiktok':
                    loss_audio = cfm_loss_audio.mean() + msi_loss_audio.mean() * args.e_loss
            else:
                self.denoise_opt_image.zero_grad()
                self.denoise_opt_text.zero_grad()
                if args.data == 'tiktok':
                    self.denoise_opt_audio.zero_grad()

                diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(
                    self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, S_doubt=getattr(self, 'S_doubt', None), omega=args.omega)
                diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(
                    self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats, S_doubt=getattr(self, 'S_doubt', None), omega=args.omega)
                if args.data == 'tiktok':
                    diff_loss_audio, gc_loss_audio = self.diffusion_model.training_losses(
                        self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats, S_doubt=getattr(self, 'S_doubt', None), omega=args.omega)

                loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
                loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
                if args.data == 'tiktok':
                    loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * args.e_loss

            epDiLoss_image += loss_image.item()
            epDiLoss_text += loss_text.item()
            if args.data == 'tiktok':
                epDiLoss_audio += loss_audio.item()

            if args.data == 'tiktok':
                loss = loss_image + loss_text + loss_audio
            else:
                loss = loss_image + loss_text

            loss.backward()

            if args.model_type == 'flowmatching_optimized':
                self.velocity_opt_image.step()
                self.velocity_opt_text.step()
                if args.data == 'tiktok':
                    self.velocity_opt_audio.step()
                log('FlowMatching Step %d/%d' % (i, num_batches), save=False, oneline=True)
            else:
                self.denoise_opt_image.step()
                self.denoise_opt_text.step()
                if args.data == 'tiktok':
                    self.denoise_opt_audio.step()
                t1 = time.time()
                log('Diffusion Step %d/%d (%.2fs)' % (i, num_batches, t1 - t0), save=False, oneline=False)

        log('')
        log('Start to re-build UI matrix using Euler Solver')

        with torch.no_grad():

            u_list_image = []
            i_list_image = []
            edge_list_image = []

            u_list_text = []
            i_list_text = []
            edge_list_text = []

            if args.data == 'tiktok':
                u_list_audio = []
                i_list_audio = []
                edge_list_audio = []

        # --- Fast DataLoader replacement ---
            for i_batch in range(num_batches):
                t0 = time.time()
                batch_index = perm_indices[i_batch*args.batch : (i_batch+1)*args.batch]
                batch_item = dataset_tensor[batch_index]
                batch_item, batch_index = batch_item.to(device), batch_index.to(device)

                if args.model_type == 'flowmatching_optimized':
                    x_start = torch.randn_like(batch_item)
                    denoised_batch_image = self.flow_matching.euler_solve(self.velocity_model_image, x_start, steps=args.steps)
                    denoised_batch_text = self.flow_matching.euler_solve(self.velocity_model_text, x_start, steps=args.steps)
                    if args.data == 'tiktok':
                        denoised_batch_audio = self.flow_matching.euler_solve(self.velocity_model_audio, x_start, steps=args.steps)
                else:
                    denoised_batch_image = self.diffusion_model.p_sample(self.denoise_model_image, batch_item, args.sampling_steps, args.sampling_noise)
                    denoised_batch_text = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, args.sampling_steps, args.sampling_noise)
                    if args.data == 'tiktok':
                        denoised_batch_audio = self.diffusion_model.p_sample(self.denoise_model_audio, batch_item, args.sampling_steps, args.sampling_noise)

                top_item, indices_ = torch.topk(denoised_batch_image, k=args.rebuild_k)
                batch_u = batch_index.repeat_interleave(args.rebuild_k).cpu().tolist()
                
                u_list_image.extend(batch_u)
                i_list_image.extend(indices_.flatten().cpu().tolist())
                edge_list_image.extend([1.0] * len(batch_u))

                top_item, indices_ = torch.topk(denoised_batch_text, k=args.rebuild_k)
                u_list_text.extend(batch_u)
                i_list_text.extend(indices_.flatten().cpu().tolist())
                edge_list_text.extend([1.0] * len(batch_u))

                if args.data == 'tiktok':
                    top_item, indices_ = torch.topk(denoised_batch_audio, k=args.rebuild_k)
                    u_list_audio.extend(batch_u)
                    i_list_audio.extend(indices_.flatten().cpu().tolist())
                    edge_list_audio.extend([1.0] * len(batch_u))
                t1 = time.time()
                log('Euler Solver Step %d/%d (%.2fs)' % (i_batch, num_batches, t1 - t0), save=False, oneline=False)

            # image
            u_list_image = np.array(u_list_image)
            i_list_image = np.array(i_list_image)
            edge_list_image = np.array(edge_list_image)
            self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
            self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

            # text
            u_list_text = np.array(u_list_text)
            i_list_text = np.array(i_list_text)
            edge_list_text = np.array(edge_list_text)
            self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
            self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

            if args.data == 'tiktok':
                # audio
                u_list_audio = np.array(u_list_audio)
                i_list_audio = np.array(i_list_audio)
                edge_list_audio = np.array(edge_list_audio)
                self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
                self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

        log('UI matrix built!')

        for i, tem in enumerate(trnLoader):
            ancs, poss, negs = tem
            ancs = ancs.long().to(device)
            poss = poss.long().to(device)
            negs = negs.long().to(device)

            self.opt.zero_grad()

            if args.data == 'tiktok':
                usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
            else:
                usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]
            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
            regLoss = self.model.reg_loss() * args.reg
            loss = bprLoss + regLoss

            epRecLoss += bprLoss.item()
            epLoss += loss.item()

            if args.data == 'tiktok':
                usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
            else:
                usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
            if args.data == 'tiktok':
                clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg
                clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, args.temp)) * args.ssl_reg
                clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, args.temp)) * args.ssl_reg
            else:
                clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

            clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
            clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
            if args.data == 'tiktok':
                clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, args.temp)) * args.ssl_reg
                clLoss_ = clLoss1 + clLoss2 + clLoss3
            else:
                clLoss_ = clLoss1 + clLoss2

            if args.cl_method == 1:
                clLoss = clLoss_

            # --- Descartes-MMRec: Giai đoạn 2 - Phản Biện (Adversarial Training) ---
            adv_loss = torch.tensor(0.0).to(device)
            if self.model.adversarial_trainer.is_active(current_epoch):
                # Bật requires_grad cho đặc trưng để attacker tính gradient
                v_feat_adv = self.handler.image_feats.clone().detach().requires_grad_(True)
                t_feat_adv = self.handler.text_feats.clone().detach().requires_grad_(True)
                
                # Forward pass phụ để tính loss cho attacker
                if args.trans == 0 or args.trans == 2:
                    v_e = self.model.leakyrelu(torch.mm(v_feat_adv, self.model.image_trans))
                else:
                    v_e = self.model.image_trans(v_feat_adv)
                    
                if args.trans == 0:
                    t_e = self.model.leakyrelu(torch.mm(t_feat_adv, self.model.text_trans))
                else:
                    t_e = self.model.text_trans(t_feat_adv)
                
                # Tính BPR Loss giả lập trên các đặc trưng này
                adv_pos_embeds = (v_e + t_e)[poss]
                adv_neg_embeds = (v_e + t_e)[negs]
                
                adv_scoreDiff = pairPredict(usrEmbeds[ancs], adv_pos_embeds, adv_neg_embeds)
                adv_bprLoss = - (adv_scoreDiff).sigmoid().log().sum() / args.batch
                
                # Tạo nhiễu
                perturb_v = self.model.adversarial_trainer.generate_perturbation(adv_bprLoss, v_feat_adv)
                perturb_t = self.model.adversarial_trainer.generate_perturbation(adv_bprLoss, t_feat_adv)
                
                # Cộng nhiễu vào features ban đầu để ép mô hình phòng thủ
                v_feat_rob = self.handler.image_feats + perturb_v
                t_feat_rob = self.handler.text_feats + perturb_t
                
                if args.trans == 0 or args.trans == 2:
                    v_e_rob = self.model.leakyrelu(torch.mm(v_feat_rob, self.model.image_trans))
                else:
                    v_e_rob = self.model.image_trans(v_feat_rob)
                    
                if args.trans == 0:
                    t_e_rob = self.model.leakyrelu(torch.mm(t_feat_rob, self.model.text_trans))
                else:
                    t_e_rob = self.model.text_trans(t_feat_rob)
                    
                # Defender Loss
                rob_pos_embeds = (v_e_rob + t_e_rob)[poss]
                rob_neg_embeds = (v_e_rob + t_e_rob)[negs]
                rob_scoreDiff = pairPredict(usrEmbeds[ancs], rob_pos_embeds, rob_neg_embeds)
                
                adv_loss = - (rob_scoreDiff).sigmoid().log().sum() / args.batch
                epAdvLoss += adv_loss.item()
            # ------------------------------------------------------------------------

            loss += clLoss + args.lambda_adv * adv_loss

            epClLoss += clLoss.item()

            loss.backward()
            self.opt.step()

            log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ; adv : %.3f' % (
                i,
                steps,
                bprLoss.item(),
                regLoss.item(),
                clLoss.item(),
                adv_loss.item()
                ), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['BPR Loss'] = epRecLoss / steps
        ret['CL loss'] = epClLoss / steps
        ret['Adv loss'] = epAdvLoss / steps
        ret['CFM image loss'] = epDiLoss_image / num_batches
        ret['CFM text loss'] = epDiLoss_text / num_batches
        if args.data == 'tiktok':
            ret['CFM audio loss'] = epDiLoss_audio / num_batches
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg, epPrecision = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat

        if args.data == 'tiktok':
            usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
        else:
            usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().to(device)
            trnMask = trnMask.to(device)
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = torch.topk(allPreds, args.topk)
            recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            epPrecision += precision
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        ret['Precision'] = epPrecision / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = allPrecision = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = precision = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
                    precision += 1
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            precision = precision / args.topk
            allRecall += recall
            allNdcg += ndcg
            allPrecision += precision
        return allRecall, allNdcg, allPrecision

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def train_colab(config_dict, dataset_base_path=None):
    """
    Entry point để gọi từ Colab notebook thay cho CLI.

    Tham số:
        config_dict (dict): Các hyperparameter, ví dụ:
            {
                'data': 'baby',
                'model_type': 'flowmatching_optimized',
                'epoch': 50,
                'lr': 1e-3,
                'checkpoint_dir': '/content/drive/MyDrive/DiffMM_Checkpoints',
                'log_file': '/content/drive/MyDrive/DiffMM_Checkpoints/train.log',
                ...
            }
        dataset_base_path (str): Đường dẫn tới thư mục gốc chứa Datasets/
            Ví dụ: '/content/drive/MyDrive/DiffMM_Data'

    Trả về:
        dict: {'best_epoch', 'recall', 'ndcg', 'precision'}
    """
    from Params import get_config_from_dict
    import Params

    # Cập nhật args toàn cục
    new_args = get_config_from_dict(config_dict)
    Params.args = new_args

    # Cập nhật args trong tất cả module đã import
    import DataHandler as DataHandler_module
    import Model as Model_module
    import Diffusion as Diffusion_module

    DataHandler_module.args = new_args
    Model_module.args = new_args
    Diffusion_module.args = new_args

    # Cập nhật global args cho module hiện tại
    global args
    args = new_args

    # Setup log file nếu có
    log_file = getattr(args, 'log_file', None)
    if log_file:
        logger.set_log_file(log_file)

    seed_it(args.seed)

    gpu_id = getattr(args, 'gpu', '0')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    logger.saveDefault = True

    log(f'=== Start Training: {args.model_type} on {args.data} ===')

    handler = DataHandler(base_path=dataset_base_path)
    handler.LoadData()
    log('Load Data done')

    coach = Coach(handler)
    results = coach.run()

    import json
    checkpoint_dir = getattr(args, 'checkpoint_dir', '.')
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_file = os.path.join(checkpoint_dir, f"results_{args.model_type}_{args.data}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

    return results


if __name__ == '__main__':
    # ── Chế độ CLI (chạy từ terminal) ─────────────────────────────────────
    seed_it(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True

    log('Start')
    handler = DataHandler(base_path=getattr(args, 'dataset_path', './Datasets'))
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    results = coach.run()

    import json
    checkpoint_dir = getattr(args, 'checkpoint_dir', '.')
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_file = os.path.join(checkpoint_dir, f"results_{args.model_type}_{args.data}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")
