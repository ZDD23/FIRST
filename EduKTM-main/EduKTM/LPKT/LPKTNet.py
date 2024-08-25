# coding: utf-8
# 2021/8/17 @ sone

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LPKTNet(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, dropout=0.2):
        super(LPKTNet, self).__init__()

        # batch_size = 32
        # n_at = 1326
        # n_it = 2839
        # n_question = 102
        # n_exercise = 3162
        # seqlen = 500
        # d_k = 128
        # d_a = 50
        # d_e = 128
        # q_gamma = 0.03
        # dropout = 0.2

        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.q_matrix = q_matrix
        self.n_question = n_question

        self.at_embed = nn.Embedding(n_at + 10, d_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(n_it + 10, d_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(n_exercise + 10, d_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)

        self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, e_data, at_data, a_data, it_data):
        batch_size, seq_len = e_data.size(0), e_data.size(1) # 32 , 500
        e_embed_data = self.e_embed(e_data)   # e_emb_data\at...\it..  (32, 500, 128)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        # (32, 500, 50)[batch_size, sequence_length, self.d_a]

        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question + 1, self.d_k)).repeat(batch_size, 1, 1).to(device)
        # h_pre (32, 103, 128)  [batch_size, self.n_question + 1, self.d_k]

        h_tilde_pre = None

        # 基本学习单元，拼接练习、做题时间、作答情况
        all_learning = self.linear_1(torch.cat((e_embed_data, at_embed_data, a_data), 2))
        # (32, 500, 128+128+50 ——> 128)
        learning_pre = torch.zeros(batch_size, self.d_k).to(device)
        # (32, 128)
        pred = torch.zeros(batch_size, seq_len).to(device)
        # (32, 500)

        for t in range(0, seq_len - 1):
            e = e_data[:, t]
            # e (32,)
            # q_e: (bs, 1, n_skill) self.q_matrix[e]: (32, num_skill)
            # view 函数用于对张量的形状进行调整。batch_size 表示批量大小，1 用于在第二个维度插入一个额外的维度，
            # 而 -1 表示 PyTorch 会根据剩余的维度自动计算这个维度的大小。
            q_e = self.q_matrix[e].view(batch_size, 1, -1)
            it = it_embed_data[:, t] # (32,128)
            # Learning Module
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)
            # bmm 是 PyTorch 中的批量矩阵乘法操作（batch matrix multiplication）。
            # 它用于对多个矩阵进行批量计算，类似于矩阵乘法（dot product），但可以同时处理多个矩阵。
            # q_e.bmm(h_pre): q_e (32,1,128) * h_pre (32, 103, 128) --->  (32, 1, 103)

            learning = all_learning[:, t]  # t=0:(32,128)
            learning_gain = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            # (32, 128 + 128 + 128 +128 ---> 128 )
            learning_gain = self.tanh(learning_gain)
            gamma_l = self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            # (32, 128 + 128 + 128 +128 ---> 128 )
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))
            # q_e.transpose (32,1,103)---> (32,103,1)
            # bmm(LG.view(batch_size, 1, -1) (32, 1, 128)   LG_tilde(32,103,128)





            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)
            n_skill = LG_tilde.size(1)
            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                # (batch_size, d_k) ---> (batch_size, d_k * n_skill) ----> (batch_size, n_skill, d_k)
                it.repeat(1, n_skill).view(batch_size, -1, self.d_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre

            # Predicting Module
            h_tilde = self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k)
            y = self.sig(self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y

            # prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        return pred
