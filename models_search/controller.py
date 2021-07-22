# -*- coding: utf-8 -*-
# @Date    : 2019-09-29
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_search.building_blocks_search import CONV_TYPE, NORM_TYPE, UP_TYPE, SHORT_CUT_TYPE, SKIP_TYPE


class Controller(nn.Module):
    def __init__(self, args):
        """
        init
        :param args:
        :param cur_stage: varies from 0 to ...
        """
        super(Controller, self).__init__()
        self.hid_size = args.latent_dim
        self.total_cells = args.total_cells
        self.hx = []
        self.cx = []

        self.tokens0 = [len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE), len(SHORT_CUT_TYPE)]
        self.tokens1 = [len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE), len(SHORT_CUT_TYPE), len(SKIP_TYPE) ** 1]
        self.tokens2 = [len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE), len(SHORT_CUT_TYPE), len(SKIP_TYPE) ** 2]

        self.tokens = [self.tokens0, self.tokens1, self.tokens2]
        # self.encoders = nn.ModuleList([nn.Embedding(sum(self.tokens), self.hid_size) for i in range(self.cur_stage)])
        # self.decoders = nn.ModuleList([nn.Linear(self.hid_size, token) for token in self.tokens])
        self.lstm = nn.LSTMCell(self.hid_size, self.hid_size)
        self.cell0_encoder = nn.Embedding(sum(self.tokens0), self.hid_size)
        #self.cell0_lstm = nn.LSTMCell(self.hid_size, self.hid_size)
        self.cell0_decoder = nn.ModuleList([nn.Linear(self.hid_size, token) for token in self.tokens0])

        self.cell1_encoder = nn.Embedding(sum(self.tokens1), self.hid_size)
        #self.cell1_lstm = nn.LSTMCell(self.hid_size, self.hid_size)
        self.cell1_decoder = nn.ModuleList([nn.Linear(self.hid_size, token) for token in self.tokens1])

        self.cell2_encoder = nn.Embedding(sum(self.tokens2), self.hid_size)
        #self.cell2_lstm = nn.LSTMCell(self.hid_size, self.hid_size)
        self.cell2_decoder = nn.ModuleList([nn.Linear(self.hid_size, token) for token in self.tokens2])

        self.cell_encoders_list = [self.cell0_encoder, self.cell1_encoder, self.cell2_encoder]
        self.cell_decoders_list = [self.cell0_decoder, self.cell1_decoder, self.cell2_decoder]

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hid_size, requires_grad=False).cuda()

    def forward(self, x, hidden, cur_cell, index, entire=False):  # index represents the output_type of CONV_TYPE/NORM_TYPE...
        #         print(cur_cell,index)
        #         if index == 0:
        #             embed = x
        #         else:
        #             embed = self.encoder(x)
        #         hx, cx = self.lstm(embed, hidden)

        #         # decode
        #         logit = self.decoders[index](hx)

        #         return logit, (hx, cx)
        if cur_cell == 0:
            if index == 0:
                embed = x
            else:
                embed = self.cell0_encoder(x)
            hx, cx = self.lstm(embed, hidden)

            # decode
            logit = self.cell0_decoder[index](hx)
        elif cur_cell == 1:
            if index == 0:
                embed = x
            else:
                embed = self.cell1_encoder(x)
            hx, cx = self.lstm(embed, hidden)

            # decode
            logit = self.cell1_decoder[index](hx)
        else:
            if index == 0:
                embed = x
            else:
                embed = self.cell2_encoder(x)
            hx, cx = self.lstm(embed, hidden)

            # decode
            logit = self.cell2_decoder[index](hx)

        return logit, (hx, cx)

    def sample(self, batch_size, with_hidden=False):
        # x = self.initHidden(batch_size)
        hidden = (self.initHidden(batch_size), self.initHidden(batch_size))
        archs = []
        entropies = []
        selected_log_probs = []
        hiddens = []
        for cur_cell in range(self.total_cells):
            x = self.initHidden(batch_size)
            if archs:
                prev_archs = torch.cat(archs, -1)
                prev_hxs, prev_cxs = hidden
                selected_idx = np.random.choice(len(prev_archs), batch_size)  # TODO: replace=False
                selected_idx = [int(x) for x in selected_idx]

                #selected_archs = []
                selected_hxs = []
                selected_cxs = []

                for s_idx in selected_idx:
                    #selected_archs.append(prev_archs[s_idx].unsqueeze(0))
                    selected_hxs.append(prev_hxs[s_idx].unsqueeze(0))
                    selected_cxs.append(prev_cxs[s_idx].unsqueeze(0))
                #selected_archs = torch.cat(selected_archs, 0)
                hidden = (torch.cat(selected_hxs, 0), torch.cat(selected_cxs, 0))
            entropy = []
            actions = []
            selected_log_prob = []
            for decode_idx in range(len(self.cell_decoders_list[cur_cell])):
                # print(cur_cell,decode_idx)
                logit, hidden = self.forward(x, hidden, cur_cell, decode_idx)
                prob = F.softmax(logit, dim=-1)  # bs * logit_dim
                log_prob = F.log_softmax(logit, dim=-1)
                # print("log_prob:{}".format(log_prob))
                entropy.append(-(log_prob * prob).sum(1, keepdim=True))  # list[array(bs * 1)]
                action = prob.multinomial(1)  # list[bs * 1]
                actions.append(action)
                # print("action_data:{}".format(action.data))
                op_log_prob = log_prob.gather(1, action.data)  # list[bs * 1]
                #                Example:
                #               >>> t = torch.Tensor([[1,2],[3,4]])
                #               >>> torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))
                #                   1  1
                #                   4  3
                #                   [torch.FloatTensor of size 2x2]
                # print("op_log_prob:{}".format(op_log_prob))
                selected_log_prob.append(op_log_prob)
                tokens = self.tokens[cur_cell]
                x = action.view(batch_size) + sum(tokens[:decode_idx])
                x = x.requires_grad_(False)

            hiddens.append(hidden[1])
            archs.append(torch.cat(actions, -1))  # batch_size * len(self.decoders)
            selected_log_probs.append(torch.cat(selected_log_prob, -1))  # list(batch_size * len(self.decoders))
            entropies.append(torch.cat(entropy, -1))  # list(bs * 1)

        #hiddens = torch.cat(hiddens, -1)
        archs = torch.cat(archs, -1)
        selected_log_probs = torch.cat(selected_log_probs, -1)
        entropies = torch.cat(entropies, -1)

        if with_hidden:
            return archs, selected_log_probs, entropies, hiddens

        return archs, selected_log_probs, entropies

