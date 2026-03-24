"""
KGNN-LS 模型代码
把 preprocess.py 里的 data 字典传进来即可使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


# ================================================================
# 核心模块：单层 KGNN 聚合
# 对应流程中的步骤3、4、5
# ================================================================

class KGNNLayer(nn.Module):

    def __init__(self, n_entities, n_relations, embed_dim, neighbor_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.neighbor_size = neighbor_size

        # 实体 embedding 表：9366 × 64
        self.entity_emb = nn.Embedding(n_entities, embed_dim)
        # 关系 embedding 表：60 × 64
        self.relation_emb = nn.Embedding(n_relations, embed_dim)
        # 用于计算注意力分数的线性变换
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)

        # 初始化：用均匀分布，范围 [-0.1, 0.1]
        nn.init.uniform_(self.entity_emb.weight,   -0.1, 0.1)
        nn.init.uniform_(self.relation_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.W.weight,            -0.1, 0.1)

    def sample_neighbors(self, entity_ids, kg_dict):
        """
        给一批实体，每个采样 K 个邻居
        entity_ids: list，长度 = batch_size
        返回:
          neighbor_entities  [B, K]  邻居的 entity_id
          neighbor_relations [B, K]  对应的关系编号
        """
        batch_ents, batch_rels = [], []

        for eid in entity_ids:
            neighbors = kg_dict.get(eid, [])

            if len(neighbors) == 0:
                # 没有邻居：用0填充（entity=0，relation=0）
                ents = [0] * self.neighbor_size
                rels = [0] * self.neighbor_size
            elif len(neighbors) >= self.neighbor_size:
                # 邻居够多：随机不重复采样 K 个
                idx = np.random.choice(len(neighbors),
                                       self.neighbor_size, replace=False)
                ents = [neighbors[i][0] for i in idx]
                rels = [neighbors[i][1] for i in idx]
            else:
                # 邻居不够：有放回地补满 K 个
                idx = np.random.choice(len(neighbors),
                                       self.neighbor_size, replace=True)
                ents = [neighbors[i][0] for i in idx]
                rels = [neighbors[i][1] for i in idx]

            batch_ents.append(ents)
            batch_rels.append(rels)

        return (torch.LongTensor(batch_ents),      # [B, K]
                torch.LongTensor(batch_rels))       # [B, K]

    def forward(self, user_emb, entity_ids, kg_dict):
        """
        步骤3：查邻居
        步骤4：算注意力权重（Softmax）
        步骤5：加权聚合

        user_emb:   [B, D]  用户 embedding
        entity_ids: list，长度 B，每个是 entity_id 整数
        返回:       [B, D]  聚合更新后的实体表示
        """
        # --- 步骤3：采样邻居 ---
        neigh_ent_ids, neigh_rel_ids = self.sample_neighbors(entity_ids, kg_dict)
        # neigh_ent_ids: [B, K]
        # neigh_rel_ids: [B, K]

        # 查 embedding 表
        neigh_emb = self.entity_emb(neigh_ent_ids)    # [B, K, D]
        rel_emb   = self.relation_emb(neigh_rel_ids)  # [B, K, D]

        # --- 步骤4：计算注意力权重 ---
        # 用户 embedding 经过线性变换后扩展维度：[B, D] → [B, 1, D]
        user_proj = self.W(user_emb).unsqueeze(1)     # [B, 1, D]

        # 用户和每条关系做点积：[B, 1, D] × [B, K, D] → [B, K]
        # 这就是"用户2对每种关系打分"
        scores  = (user_proj * rel_emb).sum(dim=-1)   # [B, K]
        weights = F.softmax(scores, dim=-1)            # [B, K]，加起来=1

        # --- 步骤5：加权聚合邻居 ---
        weights  = weights.unsqueeze(-1)               # [B, K, 1]
        aggregated = (weights * neigh_emb).sum(dim=1)  # [B, D]

        # 加上实体自身的 embedding
        self_emb = self.entity_emb(
            torch.LongTensor(entity_ids))              # [B, D]
        output = F.relu(self_emb + aggregated)         # [B, D]

        return output


# ================================================================
# 完整模型：堆叠多层 KGNNLayer
# ================================================================

class KGNN(nn.Module):

    def __init__(self, n_users, n_entities, n_relations,
                 embed_dim=64, n_layers=2, neighbor_size=8):
        super().__init__()

        self.n_layers = n_layers

        # 用户 embedding 表：2101 × 64
        self.user_emb = nn.Embedding(n_users, embed_dim)
        nn.init.uniform_(self.user_emb.weight, -0.1, 0.1)

        # 堆叠 n_layers 层聚合（默认2层：1跳邻居 + 2跳邻居）
        self.layers = nn.ModuleList([
            KGNNLayer(n_entities, n_relations, embed_dim, neighbor_size)
            for _ in range(n_layers)
        ])

    def forward(self, user_ids, item_entity_ids, kg_dict):
        """
        步骤1~6 的完整前向传播

        user_ids:         [B]  用户编号列表
        item_entity_ids:  list，长度 B，每个是歌曲对应的 entity_id
        返回:             [B]  推荐分数（0~1之间）
        """
        # 查用户 embedding
        u = self.user_emb(torch.LongTensor(user_ids))  # [B, D]

        # 逐层聚合：每一层都用用户 embedding 计算注意力
        # 第1层：聚合直接邻居（1跳）
        # 第2层：在第1层结果基础上再聚合（相当于看到2跳之外）
        v = item_entity_ids
        for layer in self.layers:
            v_emb = layer(u, v, kg_dict)  # [B, D]

        # 步骤6：用户 embedding 和 更新后实体 embedding 做点积
        # 再经过 Sigmoid 压到 0~1
        score = torch.sigmoid((u * v_emb).sum(dim=-1))  # [B]
        return score


# ================================================================
# 训练一个 epoch
# ================================================================

def train_one_epoch(model, train_data, item2entity, kg_dict,
                    optimizer, batch_size=1024):
    model.train()
    np.random.shuffle(train_data)
    total_loss = 0
    n_batches = 0

    for start in range(0, len(train_data), batch_size):
        batch = train_data[start: start + batch_size]

        # 拆开三列：用户、歌曲、标签
        user_ids = [s[0] for s in batch]
        artist_ids = [s[1] for s in batch]
        labels = torch.FloatTensor([s[2] for s in batch])

        # artistID → entity_id（用翻译本转换）
        entity_ids = [item2entity.get(aid, 0) for aid in artist_ids]

        # 前向传播，得到预测分数
        scores = model(user_ids, entity_ids, kg_dict)

        # 计算 BCE Loss（Binary Cross Entropy）
        # 就是之前讲的 -[y·log(ŷ) + (1-y)·log(1-ŷ)]
        loss = F.binary_cross_entropy(scores, labels)

        # 反向传播：自动计算所有参数的梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches  # 返回平均 loss


# ================================================================
# 评估：计算 AUC 和 Accuracy
# ================================================================

def evaluate(model, test_data, item2entity, kg_dict, batch_size=1024):
    model.eval()
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for start in range(0, len(test_data), batch_size):
            batch = test_data[start: start + batch_size]

            user_ids = [s[0] for s in batch]
            artist_ids = [s[1] for s in batch]
            labels = [s[2] for s in batch]
            entity_ids = [item2entity.get(aid, 0) for aid in artist_ids]

            scores = model(user_ids, entity_ids, kg_dict)

            all_labels.extend(labels)
            all_scores.extend(scores.cpu().numpy())

    auc = roc_auc_score(all_labels, all_scores)
    preds  = (np.array(all_scores) >= 0.5).astype(int)
    acc = (np.array(preds) == np.array(all_labels)).mean()
    return auc, acc


# ================================================================
# 主训练循环
# ================================================================

def train(data, embed_dim=64, n_layers=2, neighbor_size=8,
          lr=1e-3, n_epochs=30, batch_size=1024):

    # 初始化模型
    model = KGNN(
        n_users      = data['n_users'],
        n_entities   = data['n_entities'],
        n_relations  = data['n_relations'],
        embed_dim    = embed_dim,
        n_layers     = n_layers,
        neighbor_size= neighbor_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=1e-5)

    print(f"模型参数总量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"开始训练，共 {n_epochs} 轮...\n")

    best_auc = 0

    for epoch in range(1, n_epochs + 1):

        # 训练一轮
        avg_loss = train_one_epoch(
            model, data['train_data'], data['item2entity'],
            data['kg_dict'], optimizer, batch_size
        )

        # 每5轮评估一次
        if epoch % 5 == 0:
            auc, acc = evaluate(
                model, data['test_data'], data['item2entity'],
                data['kg_dict'], batch_size
            )
            print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | "
                  f"AUC {auc:.4f} | Acc {acc:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"           ↑ 新的最佳 AUC，模型已保存")
        else:
            print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f}")

    print(f"\n训练完成！最佳 AUC = {best_auc:.4f}")
    return model


# ================================================================
# 把 preprocess.py 的结果传进来，直接运行
# ================================================================

if __name__ == '__main__':

    # 先运行 preprocess.py 得到 data 字典，再在这里导入
    # 这里直接把两个文件合并运行，复制 preprocess.py 的内容
    import numpy as np
    from collections import defaultdict

    # ---- 复制 preprocess.py 的处理逻辑 ----
    item2entity = {}
    entity2item = {}
    with open('item_index2entity_id.txt', 'r') as f:
        for line in f:
            p = line.strip().split('\t')
            item2entity[int(p[0])] = int(p[1])
            entity2item[int(p[1])] = int(p[0])

    relation2id = {}
    relation_cnt = 0
    kg_triples = []
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
    all_items = set()
    with open('user_artists.dat', 'r') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            uid, aid = int(p[0]), int(p[1])
            if aid in item2entity:
                user_pos_items[uid].append(aid)
                all_items.add(aid)
    all_items = list(all_items)

    def gen_samples(upi, items, neg=1):
        samples = []
        for uid, pos_list in upi.items():
            pos_set = set(pos_list)
            for item in pos_list:
                samples.append((uid, item, 1))
            nc = 0
            while nc < len(pos_list) * neg:
                c = np.random.choice(items)
                if c not in pos_set:
                    samples.append((uid, c, 0))
                    nc += 1
        return samples

    np.random.seed(42)
    all_samples = gen_samples(user_pos_items, all_items)
    np.random.shuffle(all_samples)
    split = int(len(all_samples) * 0.8)

    all_entities = set()
    for h, _, t in kg_triples:
        all_entities.add(h); all_entities.add(t)

    data = {
        'train_data' : all_samples[:split],
        'test_data'  : all_samples[split:],
        'kg_dict'    : kg_dict,
        'item2entity': item2entity,
        'n_users'    : max(user_pos_items.keys()) + 1,
        'n_entities' : max(all_entities) + 1,
        'n_relations': len(relation2id),
    }
    # ---- 预处理结束 ----

    # 开始训练
    model = train(data)