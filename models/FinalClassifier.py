from torch import nn
import torch
from math import ceil
from models import I3D


class Classifier(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.avg_modality = model_args.avg_modality
        self.num_classes = model_args.num_classes
        self.num_clips = model_args.num_clips

        self.AvgPool = nn.AdaptiveAvgPool2d((1,1024))

        self.TRN=RelationModuleMultiScale(1024, 1024, self.num_clips)

        def temporal_aggregation(self, x):

            x=x.view(-1, self.num_clips, 1024) #restore the original shape of the tensor

            if self.avg_modality == 'Pooling':
                x = self.AvgPool(x)
                x = x.view(-1, 1024)
            elif self.avg_modality == 'TRN':
                x = self.TRN(x)
                x = x.sum(1)
            return x

        self.temporal_aggregation = temporal_aggregation

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes),
            nn.ReLU())

        #self.softmax = nn.Softmax(dim=1)

    def forward(self, input_source, input_target, beta, is_train):
        batch_source = input_source.size()[0]
		batch_target = input_target.size()[0]
        num_segments = self.num_clips
        num_class = self.num_classes

        feat_all_source = []
		feat_all_target = []
		pred_domain_all_source = []
		pred_domain_all_target = []

        # Here we reshape dimensions as batch x num_clip x feat_dim --> (batch * num_clip) x feat_dim because we
        # want to process every clip independently not as part of a batch of clips that refers to the same video because
        # we are at a frame level#
        feat_base_source = input_source.view(-1, input_source.size()[-1]) # e.g. 32 x 5 x 1024 --> 160 x 1024
		feat_base_target = input_target.view(-1, input_target.size()[-1]) # e.g. 32 x 5 x 1024 --> 160 x 1024

        #Adaptive Batch Normalization and shared layers to ask, at the moment we put:
        feat_fc_source = feat_base_source
        feat_fc_target = feat_base_target

        # === adversarial branch (frame-level), in our case clip-level ===#
		pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta) # 160 x 1024 --> 2
		pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta) # 160 x 1024 --> 2

        # Da capire un attimo le dimensioni di questo append !!!
        pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
		pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

        #=== prediction (frame-level) ===#
		pred_fc_source = self.fc_classifier_source(feat_fc_source) # 160 x 1024 --> num_classes
		pred_fc_target = self.fc_classifier_target(feat_fc_target) # 160 x 1024 --> num_classes

        ### aggregate the frame-based features to relation-based features ###
		feat_fc_relation_source = self.temporal_aggregation(feat_fc_source)
		feat_fc_relation_target = self.temporal_aggregation(feat_fc_target)

        # Here if we have used AvgPool we obtain a tensor of size batch x feat_dim, otherwise if we have used TRN we obtain
        # a tensor of size batch x num_relations x feat_dim and we have to implement a domain classifier for the TRN case
        # so i think we need to do something as:
        # if self.avg_modality == 'TRN':
        #  pred_fc_domain_relation_source = self.domain_classifier_relation(feat_fc_relation_source, beta)
        #  pred_fc_domain_relation_target = self.domain_classifier_relation(feat_fc_relation_target, beta)
        # where domain_classifier_relation have to be implemented as fc where num_relation x feat_dim -> 2
        # pay attention that aggregated data with AvgPool are yet at video level (1x1024) so doing a relation domain classifier
        # on them is useful because we will do it later so we have to put
        # elif self.avg_modality == 'Pooling':
        # feat_fc_video_source = feat_fc_relation_source
        # feat_fc_video_target = feat_fc_relation_target




    def forward(self, x):

        x = self.temporal_aggregation(self,x)

        x = self.fc1(x)
        x= self.fc2(x)

        return x, {}

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
