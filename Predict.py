"""
使用 best_model.pt 进行推荐预测
三个功能：
1. 给指定用户推荐 Top-N 首歌
2. 预测用户对某首歌的喜好分数
3. 批量评估测试集（验证模型没有损坏）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score


# ================================================================
# 第一步：把模型结构和数据重新准备好
# （和 model.py 完全一样，必须一致才能正确加载权重）
# ================================================================

class KGNNLayer(nn.Module):
    def __init__(self, n_entities, n_relations, embed_dim, neighbor_size):
        super().__init__()
        self.embed_dim     = embed_dim
        self.neighbor_size = neighbor_size
        self.entity_emb    = nn.Embedding(n_entities, embed_dim)
        self.relation_emb  = nn.Embedding(n_relations, embed_dim)
        self.W             = nn.Linear(embed_dim, embed_dim, bias=False)

    def sample_neighbors(self, entity_ids, kg_dict):
        batch_ents, batch_rels = [], []
        for eid in entity_ids:
            neighbors = kg_dict.get(eid, [])
            if len(neighbors) == 0:
                ents = [0] * self.neighbor_size
                rels = [0] * self.neighbor_size
            elif len(neighbors) >= self.neighbor_size:
                idx  = np.random.choice(
                    len(neighbors), self.neighbor_size, replace=False)
                ents = [neighbors[i][0] for i in idx]
                rels = [neighbors[i][1] for i in idx]
            else:
                idx  = np.random.choice(
                    len(neighbors), self.neighbor_size, replace=True)
                ents = [neighbors[i][0] for i in idx]
                rels = [neighbors[i][1] for i in idx]
            batch_ents.append(ents)
            batch_rels.append(rels)
        return (torch.LongTensor(batch_ents),
                torch.LongTensor(batch_rels))

    def forward(self, user_emb, entity_ids, kg_dict):
        neigh_ent_ids, neigh_rel_ids = self.sample_neighbors(
            entity_ids, kg_dict)
        neigh_emb  = self.entity_emb(neigh_ent_ids)
        rel_emb    = self.relation_emb(neigh_rel_ids)
        user_proj  = self.W(user_emb).unsqueeze(1)
        scores     = (user_proj * rel_emb).sum(dim=-1)
        weights    = F.softmax(scores, dim=-1).unsqueeze(-1)
        aggregated = (weights * neigh_emb).sum(dim=1)
        self_emb   = self.entity_emb(torch.LongTensor(entity_ids))
        return F.relu(self_emb + aggregated)


class KGNN(nn.Module):
    def __init__(self, n_users, n_entities, n_relations,
                 embed_dim=64, n_layers=2, neighbor_size=8):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.layers   = nn.ModuleList([
            KGNNLayer(n_entities, n_relations, embed_dim, neighbor_size)
            for _ in range(n_layers)
        ])

    def forward(self, user_ids, entity_ids, kg_dict):
        u = self.user_emb(torch.LongTensor(user_ids))
        for layer in self.layers:
            v_emb = layer(u, entity_ids, kg_dict)
        return torch.sigmoid((u * v_emb).sum(dim=-1))


# ================================================================
# 第二步：加载数据
# ================================================================

def load_data():
    item2entity = {}
    entity2item = {}
    with open('item_index2entity_id.txt', 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            item2entity[int(p[0])] = int(p[1])
            entity2item[int(p[1])] = int(p[0])

    relation2id  = {}
    relation_cnt = 0
    kg_triples   = []
    with open('kg.txt', 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            head, rel_name, tail = int(p[0]), p[1], int(p[2])
            if rel_name not in relation2id:
                relation2id[rel_name] = relation_cnt
                relation_cnt += 1
            kg_triples.append((head, relation2id[rel_name], tail))

    kg_dict = defaultdict(list)
    for head, rid, tail in kg_triples:
        kg_dict[head].append((tail, rid))
        kg_dict[tail].append((head, rid))

    user_pos_items = defaultdict(list)
    all_items      = set()
    with open('user_artists.dat', 'r') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            uid, aid = int(p[0]), int(p[1])
            if aid in item2entity:
                user_pos_items[uid].append(aid)
                all_items.add(aid)

    all_entities = set()
    for h, _, t in kg_triples:
        all_entities.add(h)
        all_entities.add(t)

    return {
        'kg_dict':        kg_dict,
        'item2entity':    item2entity,
        'entity2item':    entity2item,
        'user_pos_items': user_pos_items,
        'all_items':      sorted(all_items),
        'n_users':        max(user_pos_items.keys()) + 1,
        'n_entities':     max(all_entities) + 1,
        'n_relations':    len(relation2id),
    }


# ================================================================
# 第三步：加载模型权重
# 关键：模型结构必须和训练时完全一致
# ================================================================

def load_model(data, model_path='best_model.pt'):
    model = KGNN(
        n_users      = data['n_users'],
        n_entities   = data['n_entities'],
        n_relations  = data['n_relations'],
        embed_dim    = 64,
        n_layers     = 2,
        neighbor_size= 8,
    )

    # 把 .pt 文件里存的数字填回模型
    # map_location='cpu' 表示不管训练时用没用GPU，都加载到CPU
    model.load_state_dict(
        torch.load(model_path, map_location='cpu'))

    # 切换到推理模式
    # 推理模式和训练模式的区别：
    # 训练模式：会计算梯度，更新参数（慢）
    # 推理模式：只做前向计算，不更新参数（快）
    model.eval()

    print(f"模型加载成功：{model_path}")
    return model


# ================================================================
# 功能1：给指定用户推荐 Top-N 首歌
# ================================================================

def recommend_topn(model, user_id, data, top_n=10):
    """
    给 user_id 这个用户，从所有他没听过的歌里，推荐分数最高的 top_n 首

    流程：
    1. 找出用户听过的歌（排除掉，不重复推荐）
    2. 对剩余所有歌打分
    3. 按分数排序，取前 top_n 首
    """
    kg_dict     = data['kg_dict']
    item2entity = data['item2entity']
    all_items   = data['all_items']

    # 用户已经听过的歌，推荐时要排除
    listened = set(data['user_pos_items'].get(user_id, []))

    # 候选歌曲 = 所有歌 - 听过的歌
    candidates = [item for item in all_items if item not in listened]

    if len(candidates) == 0:
        print(f"用户 {user_id} 已经听过所有歌曲！")
        return []

    # 批量打分（分批处理，防止一次性太大撑爆内存）
    batch_size  = 512
    all_scores  = []

    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch_items  = candidates[start: start + batch_size]
            batch_size_  = len(batch_items)

            # 用户 id 重复 batch_size 次（每首歌都要和同一个用户配对）
            user_ids    = [user_id] * batch_size_
            entity_ids  = [item2entity.get(aid, 0)
                           for aid in batch_items]

            scores = model(user_ids, entity_ids, kg_dict)
            all_scores.extend(scores.cpu().numpy().tolist())

    # 把歌曲和分数配对，按分数从高到低排序
    scored_items = sorted(
        zip(candidates, all_scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_items = scored_items[:top_n]

    print(f"\n给用户 {user_id} 的 Top-{top_n} 推荐：")
    print(f"（该用户已听过 {len(listened)} 首，候选池 {len(candidates)} 首）")
    print(f"{'排名':<6} {'artistID':<12} {'推荐分数':<10}")
    print('-' * 30)
    for rank, (artist_id, score) in enumerate(top_items, 1):
        print(f"  {rank:<4} {artist_id:<12} {score:.4f}")

    return top_items


# ================================================================
# 功能2：预测用户对某首歌的喜好分数
# ================================================================

def predict_score(model, user_id, artist_id, data):
    """
    预测 user_id 对 artist_id 的喜好分数（0~1之间）
    分数越高，代表越可能喜欢
    """
    item2entity = data['item2entity']
    kg_dict     = data['kg_dict']

    # 检查这首歌是否在知识图谱里
    if artist_id not in item2entity:
        print(f"artistID={artist_id} 不在知识图谱里，无法预测")
        return None

    entity_id = item2entity[artist_id]

    with torch.no_grad():
        score = model([user_id], [entity_id], kg_dict)

    score_val = score.item()

    # 判断用户有没有听过这首歌
    listened     = data['user_pos_items'].get(user_id, [])
    already_seen = artist_id in listened

    print(f"\n预测结果：")
    print(f"  用户 {user_id} 对 歌手 {artist_id} 的喜好分数：{score_val:.4f}")
    print(f"  用户是否已听过：{'是' if already_seen else '否'}")

    if score_val >= 0.7:
        print(f"  → 强烈推荐（分数很高）")
    elif score_val >= 0.5:
        print(f"  → 推荐（分数较高）")
    else:
        print(f"  → 不推荐（分数较低）")

    return score_val


# ================================================================
# 功能3：批量评估（验证加载的模型和训练时结果一致）
# ================================================================

def evaluate(model, data, n_samples=2000):
    """
    从 user_pos_items 里随机抽样构造测试集，计算 AUC
    用来验证：加载的模型权重是否正确
    """
    kg_dict     = data['kg_dict']
    item2entity = data['item2entity']
    all_items   = data['all_items']

    # 随机构造正负样本
    np.random.seed(0)
    samples = []
    users   = list(data['user_pos_items'].keys())

    while len(samples) < n_samples:
        uid      = np.random.choice(users)
        pos_list = data['user_pos_items'][uid]
        if len(pos_list) == 0:
            continue

        # 正样本
        pos_item = np.random.choice(pos_list)
        samples.append((uid, pos_item, 1))

        # 负样本
        pos_set = set(pos_list)
        neg_item = np.random.choice(all_items)
        while neg_item in pos_set:
            neg_item = np.random.choice(all_items)
        samples.append((uid, neg_item, 0))

    # 批量打分
    all_labels, all_scores = [], []
    batch_size = 512

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch      = samples[start: start + batch_size]
            user_ids   = [s[0] for s in batch]
            artist_ids = [s[1] for s in batch]
            labels     = [s[2] for s in batch]
            entity_ids = [item2entity.get(aid, 0) for aid in artist_ids]

            scores = model(user_ids, entity_ids, kg_dict)
            all_labels.extend(labels)
            all_scores.extend(scores.cpu().numpy().tolist())

    auc = roc_auc_score(all_labels, all_scores)
    print(f"\n验证结果（{n_samples*2} 条样本）：AUC = {auc:.4f}")
    print("（应该和训练时最终 AUC 接近，说明模型加载正确）")
    return auc


# ================================================================
# 主程序：演示三个功能
# ================================================================

if __name__ == '__main__':

    print("加载数据...")
    data = load_data()

    print("加载模型...")
    model = load_model(data, model_path='best_model.pt')

    # ── 功能3：先验证模型加载正确 ──
    print("\n【验证模型】")
    evaluate(model, data, n_samples=2000)

    # ── 功能1：给用户2推荐10首歌 ──
    print("\n【Top-N 推荐】")
    recommend_topn(model, user_id=2, data=data, top_n=10)

    # ── 功能2：预测用户2对几首具体歌手的喜好 ──
    print("\n【单个预测】")

    # 取用户2听过的一首歌和没听过的一首歌，对比分数
    listened_by_2 = data['user_pos_items'][2]
    not_listened  = [item for item in data['all_items']
                     if item not in set(listened_by_2)]

    if listened_by_2:
        print("--- 用户2听过的歌（预期分数较高）---")
        predict_score(model, user_id=2,
                      artist_id=listened_by_2[0], data=data)

    if not_listened:
        print("--- 用户2没听过的歌（预期分数较低）---")
        predict_score(model, user_id=2,
                      artist_id=not_listened[0], data=data)