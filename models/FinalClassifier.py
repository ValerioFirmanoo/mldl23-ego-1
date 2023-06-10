from torch import nn
import torch
from math import ceil
from models import I3D
from torch.autograd import Function
from torch.distributions import Categorical
from scipy.stats import entropy
import numpy as np


class Classifier(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.avg_modality = model_args.avg_modality
        self.num_classes = model_args.num_classes
        self.num_clips = model_args.num_clips
        self.baseline_type = model_args.baseline_type
        self.beta = model_args.beta
        self.domain_adapt_strategy=model_args.domain_adapt_strategy

        self.AvgPool = nn.AdaptiveAvgPool2d((1,512))

        self.TRN = RelationModuleMultiScale(512, 512, self.num_clips)

        #self.temporal_aggregation = temporal_aggregation

        self.relation_domain_classifier_all = nn.ModuleList()
        for i in range(self.num_clips-1):
            relation_domain_classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
                nn.Softmax()
            )
            self.relation_domain_classifier_all += [relation_domain_classifier]

        self.fc0 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
            nn.ReLU())

        self.GSD = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.ReLU(),
            nn.Softmax()
            )

        self.GVD = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.ReLU(),
            nn.Softmax()
        )

        self.fc_classifier_frame = nn.Sequential(
            nn.Linear(512, self.num_classes),
            nn.ReLU(),
            #nn.Softmax()
        )

        self.fc_classifier_video = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
            nn.ReLU(),
            #nn.Softmax()
        )

    def domain_classifier_frame(self, feat, beta):
        feat_fc_domain_frame = GradReverse.apply(feat, beta)
        pred_fc_domain_frame = self.GSD(feat_fc_domain_frame)

        return pred_fc_domain_frame

    def domain_classifier_relation(self, feat_relation, beta):
        # 32x4x1024 --> (32x4)x2
        attention=[]
        pred_fc_domain_relation_video = None
        for i in range(len(self.relation_domain_classifier_all)):
            feat_relation_single = feat_relation[:,i,:].squeeze(1) # 32x1x1024 -> 32x1024
            feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single, beta) # the same beta for all relations (for now)

            pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)
            #print('pred_fc_domain_relation_single: ',pred_fc_domain_relation_single)
            entropies = Categorical(probs=pred_fc_domain_relation_single).entropy()
            #print('entropies: ', entropies)
            weight_factor = 0.5
            attention.append((1-entropies) * weight_factor)

            if pred_fc_domain_relation_video is None:
                pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1,1,2)
            else:
                pred_fc_domain_relation_video = torch.cat((pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1,1,2)), 1)

        pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1,2)

        return attention, pred_fc_domain_relation_video

    def domain_classifier_video(self, feat, beta):
        feat_fc_domain_video = GradReverse.apply(feat, beta)
        pred_fc_domain_video = self.GVD(feat_fc_domain_video)

        return pred_fc_domain_video

    def temporal_aggregation(self, x):

        x=x.view(-1, self.num_clips, 512) #restore the original shape of the tensor

        if self.avg_modality == 'Pooling':
            x = self.AvgPool(x)
            #print('before sum',x.shape)
            #x = x.view(-1, 1024)
        elif self.avg_modality == 'TRN':
            x = self.TRN(x)
        return x

    def final_output(self, pred, pred_video, num_segments):
        if self.baseline_type == 'video':
            base_out = pred_video
        else:
            base_out = pred

        output = base_out
        return output

    def get_attn_feat_relation(self,feat_fc_video_relation, att_source):
        #print(feat_fc_video_relation.shape)
        multiplier = 0.1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feat_fc_video_relation_att=torch.rand(size=feat_fc_video_relation.shape).to(device)
        #print(feat_fc_video_relation_att.shape)
        #print(att_source[0].reshape(-1,1).shape)
        #print((att_source[0].reshape(-1, 1)* feat_fc_video_relation[:,0,:]).shape)
        #print(feat_fc_video_relation_att[:,0,:].shape)
        #print(len(att_source))
        for i in range(0, len(att_source)):
            # print('att_source[i]: ',att_source[i].shape)
            # feat_fc_video_relation_source_att[:,i,:]=torch.matmul(att_source[i].reshape(1,-1),feat_fc_video_relation_source[:,i,:])
            # PENSO CHE L'ERRORE SIA QUI
            # penso che il calcolo qui sopra dovrebbe essere:
            att_source[i] = att_source[i] * multiplier
            feat_fc_video_relation_att[:,i,:] = (1+att_source[i].reshape(-1,1)) * feat_fc_video_relation[:,i,:]
            # feat_fc_video_relation_target_att[:,i,:]=torch.matmul(att_target[i].reshape(1,-1),feat_fc_video_relation_target[:,i,:])
        #print('feat_fc_video_relation_source_att[:,i,:]: ',feat_fc_video_relation_att[:,2,:])
        #print('feat_fc_video_relation_source_att: ',feat_fc_video_relation_att.shape)
        #import pdb; pdb.set_trace()
        feat_fc_video_relation_att = feat_fc_video_relation_att.view(-1, 1, 4, 512)
        feat_fc_video_relation_att = nn.AvgPool2d([4,1])(feat_fc_video_relation_att)# 32 x 4 x 1024 --> 32 x 1024
        feat_fc_video_relation_att = feat_fc_video_relation_att.squeeze(1).squeeze(1)
        return feat_fc_video_relation_att

    def forward(self, input_source, input_target):
        beta = self.beta
        batch_source = input_source.size()[0]
        batch_target = input_target.size()[0]
        num_segments = self.num_clips
        num_class = self.num_classes

        feat_all_source = []
        feat_all_target = []
        pred_domain_all_source = []
        pred_domain_all_target = []
        pred_fc_video_source_att=[]
        pred_fc_video_target_att=[]

        # Here we reshape dimensions as batch x num_clip x feat_dim --> (batch * num_clip) x feat_dim because we
        # want to process every clip independently not as part of a batch of clips that refers to the same video because
        # we are at a frame level#
        feat_base_source = input_source.view(-1, input_source.size()[-1]) # e.g. 32 x 5 x 1024 --> 160 x 1024
        feat_base_target = input_target.view(-1, input_target.size()[-1]) # e.g. 32 x 5 x 1024 --> 160 x 1024

        #Adaptive Batch Normalization and shared layers to ask, at the moment we put:
        # Qua questione shared layer non si capisce
        feat_fc_source = self.fc0(feat_base_source) # 160 x 1024 --> 160 x 1024
        feat_fc_target = self.fc0(feat_base_target) # 160 x 1024 --> 160 x 1024
        #feat_fc_source = feat_base_source
        #feat_fc_target = feat_base_target

        # === adversarial branch (frame-level), in our case clip-level ===#
        pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta) # 160 x 1024 --> 160 x 2
        #print('pred_fc_domain_frame_source: ',pred_fc_domain_frame_source)
        pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta) # 160 x 1024 --> 160 x 2
        #print('pred_fc_domain_frame_target: ',pred_fc_domain_frame_source)

    # Da capire un attimo le dimensioni di questo append !!!
        pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
        pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

    #=== prediction (frame-level) ===#
        pred_fc_source = self.fc_classifier_frame(feat_fc_source) # 160 x 1024 --> 160 x num_classes
        pred_fc_target = self.fc_classifier_frame(feat_fc_target) # 160 x 1024 --> 160 x num_classes

        ### aggregate the frame-based features to relation-based features ###
        feat_fc_video_relation_source = self.temporal_aggregation(feat_fc_source)
        feat_fc_video_relation_target = self.temporal_aggregation(feat_fc_target)

        if self.avg_modality == 'TRN': #we have 4 frames relations
            [att_source, pred_fc_domain_video_relation_source] = self.domain_classifier_relation(feat_fc_video_relation_source, beta) # 32 x 4 x 1024 --> 32 x 2
            [att_target, pred_fc_domain_video_relation_target] = self.domain_classifier_relation(feat_fc_video_relation_target, beta) # 32 x 4 x 1024 --> 32 x 2
        #print('att_source: ',att_source[2])

        # === prediction (video-level) ===#
        #aggregate the frame-based features to video-based features, we can use sum() even in AVGPOOL case because we have
        # alredy only 1 "clip" dimension (batch x feat_dim)

        #QUA INSERIRE MOLTIPLIC PER ATTENTION
        #=== attention module ===#
        if 'ATT' in self.domain_adapt_strategy:
            #feat_fc_video_relation_source_att = feat_fc_video_relation_source # 32 x 4 x 512
            #feat_fc_video_relation_target_att = feat_fc_video_relation_target
            #print('feat_fc_video_relation_source: ',feat_fc_video_relation_source[:,2,:])
            feat_fc_video_source_att = self.get_attn_feat_relation(feat_fc_video_relation_source, att_source)
            feat_fc_video_target_att = self.get_attn_feat_relation(feat_fc_video_relation_target, att_target)
            #for i in range(0,len(att_source)):
                #print('att_source[i]: ',att_source[i].shape)
                #feat_fc_video_relation_source_att[:,i,:]=torch.matmul(att_source[i].reshape(1,-1),feat_fc_video_relation_source[:,i,:])
                #PENSO CHE L'ERRORE SIA QUI
                #penso che il calcolo qui sopra dovrebbe essere:
                #feat_fc_video_relation_source_att[:,i,:]=(1+att_source[i].reshape(-1,1)) * feat_fc_video_relation_source[:,i,:]
                #print('feat_fc_video_relation_source_att[:,i,:]: ',feat_fc_video_relation_source_att[:,i,:])
                #feat_fc_video_relation_target_att[:,i,:]=torch.matmul(att_target[i].reshape(1,-1),feat_fc_video_relation_target[:,i,:])
                #feat_fc_video_relation_target_att[:,i,:]=(1+att_source[i].reshape(-1,1)) * feat_fc_video_relation_target[:,i,:]
            #feat_fc_video_source_att = feat_fc_video_relation_source_att.sum(1)  # 32 x 4 x 1024 --> 32 x 1024
            #feat_fc_video_target_att = feat_fc_video_relation_target_att.sum(1)  # 32 x 4 x 1024 --> 32 x 1024
            pred_fc_video_source = self.fc_classifier_video(feat_fc_video_source_att)
            #print('pred_fc_video_source_att: ',pred_fc_video_source_att)
            pred_fc_video_target = self.fc_classifier_video(feat_fc_video_target_att)
            pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source_att, beta)
            pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target_att, beta)


        elif 'ATT' not in self.domain_adapt_strategy:
            feat_fc_video_source = feat_fc_video_relation_source.sum(1) # 32 x 4 x 1024 --> 32 x 1024
            feat_fc_video_target = feat_fc_video_relation_target.sum(1) # 32 x 4 x 1024 --> 32 x 1024
            #print('after sum',feat_fc_video_source.shape)

            pred_fc_video_source = self.fc_classifier_video(feat_fc_video_source)
            pred_fc_video_target = self.fc_classifier_video(feat_fc_video_target)

            pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)
            pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)
            #print('pred_fc_domain_video_source: ',pred_fc_domain_video_source)
        #what does he do in the code: he append the DOMAIN predictions of the frame-level and video-level indipendentemente
        #from the aggregation method, then appends domain_relation_predictions only if we have used TRN as aggregation method
        # or another time the same domain_video_predictions if we have used AVGPOOL as aggregation method

        if self.avg_modality == 'TRN': #append domain_relation_predictions

            num_relation = feat_fc_video_relation_source.size()[1]
            pred_domain_all_source.append(pred_fc_domain_video_relation_source.view((batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]))
            pred_domain_all_target.append(pred_fc_domain_video_relation_target.view((batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]))

        elif self.avg_modality == 'Pooling': #append domain_video_predictions again
            pred_domain_all_source.append(pred_fc_domain_video_source) # if not trn-m, add dummy tensors for relation features
            pred_domain_all_target.append(pred_fc_domain_video_target) # if not trn-m, add dummy tensors for relation features

        pred_domain_all_source.append(pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
        pred_domain_all_target.append(pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))

        #=== final output ===#
        #output_source = self.final_output(pred_fc_source, pred_fc_video_source, num_segments)
        #output_target = self.final_output(pred_fc_target, pred_fc_video_target, num_segments)

        results = {
            'domain_source':pred_domain_all_source ,
            'domain_target': pred_domain_all_target,
            'pred_frame_source': pred_fc_source,
            'pred_video_source': pred_fc_video_source,
            'pred_frame_target': pred_fc_target,
            'pred_video_target': pred_fc_video_target,
            #'pred_video_source_att':pred_fc_video_source_att,
            #'pred_video_target_att': pred_fc_video_target_att
        }

        return results, {}
        #return  pred_fc_source, pred_fc_video_source, pred_domain_all_source, pred_fc_target, pred_fc_video_target , pred_domain_all_target


class RelationModuleMultiScale(nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        # self.num_class = num_class
        self.num_frames = num_frames
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_scale_1 = input[:, self.relations_scales[0][0] , :]
        act_scale_1 = act_scale_1.view(act_scale_1.size(0), self.scales[0] * self.img_feature_dim)
        act_scale_1 = self.fc_fusion_scales[0](act_scale_1)
        act_scale_1 = act_scale_1.unsqueeze(1) # add one dimension for the later concatenation
        act_all = act_scale_1.clone()

        for scaleID in range(1, len(self.scales)):
            act_relation_all = torch.zeros_like(act_scale_1)
            # iterate over the scales
            num_total_relations = len(self.relations_scales[scaleID])
            num_select_relations = self.subsample_scales[scaleID]
            idx_relations_evensample = [int(ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]

            #for idx in idx_relations_randomsample:
            for idx in idx_relations_evensample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = act_relation.unsqueeze(1)  # add one dimension for the later concatenation
                act_relation_all += act_relation

            act_all = torch.cat((act_all, act_relation_all), 1)
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None

