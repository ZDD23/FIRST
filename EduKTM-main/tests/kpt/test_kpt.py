# coding: utf-8
# 2021/4/5 @ liujiayu
from EduKTM import KPT


def test_train(data, tmp_path):
    train_set, q_m, stu_num, prob_num, know_num, time_window_num = data

    cdm = KPT(q_m, stu_num, prob_num, know_num, time_window_num=time_window_num)

    cdm.train(train_set, epoch=30, lr=0.001, lr_b=0.0001, epsilon=1e-1, init_method='mean')
    rmse, mae = cdm.eval([{'user_id': 0, 'item_id': 0, 'score': 1.0}])
    filepath = tmp_path / "kpt.params"
    cdm.save(filepath)
    cdm.load(filepath)
