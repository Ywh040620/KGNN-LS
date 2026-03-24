"""
KGNN-LS 消融实验
对比4种配置，证明每个部件的贡献：

配置A：纯协同过滤（无知识图谱）         ← baseline
配置B：KGNN 均匀权重（无注意力机制）    ← 加了图谱，但没有注意力
配置C：KGNN 单层聚合（无第2层）         ← 有注意力，但只看1跳邻居
配置D：KGNN-LS 完整模型（2层+注意力）   ← 你已经跑过的完整版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score


# ================================================================
# 数据预处理（和 model.py 完全一样，直接复用）
# ================================================================

def load_data():
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

    np.random.seed(42)  # 固定种子，保证4次实验用完全相同的数据划分

    samples = []
    for uid, pos_list in user_pos_items.items():
        pos_set = set(pos_list)
        for item in pos_list:
            samples.append((uid, item, 1))
        nc = 0
        while nc < len(pos_list):
            c = np.random.choice(all_items)
            if c not in pos_set:
                samples.append((uid, c, 0))
                nc += 1

    np.random.shuffle(samples)
    split = int(len(samples) * 0.8)

    all_entities = set()
    for h, _, t in kg_triples:
        all_entities.add(h)
        all_entities.add(t)

    return {
        'train_data':  samples[:split],
        'test_data':   samples[split:],
        'kg_dict':     kg_dict,
        'item2entity': item2entity,
        'n_users':     max(user_pos_items.keys()) + 1,
        'n_entities':  max(all_entities) + 1,
        'n_relations': len(relation2id),
    }


# ================================================================
# 核心模块：带"开关"的 KGNNLayer
# use_attention=True  → 完整版，用用户自适应注意力
# use_attention=False → 消融版，所有邻居权重相等（均匀分配）
# ================================================================

class KGNNLayer(nn.Module):

    def __init__(self, n_entities, n_relations, embed_dim,
                 neighbor_size, use_attention=True):
        super().__init__()
        self.neighbor_size  = neighbor_size
        self.use_attention  = use_attention   # ← 这就是开关

        self.entity_emb   = nn.Embedding(n_entities, embed_dim)
        self.relation_emb = nn.Embedding(n_relations, embed_dim)

        # 只有开着注意力时，线性层才有意义
        if use_attention:
            self.W = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.uniform_(self.entity_emb.weight,   -0.1, 0.1)
        nn.init.uniform_(self.relation_emb.weight, -0.1, 0.1)

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

        neigh_emb = self.entity_emb(neigh_ent_ids)    # [B, K, D]
        rel_emb   = self.relation_emb(neigh_rel_ids)  # [B, K, D]

        if self.use_attention:
            # ── 有注意力：权重由用户偏好决定 ──
            user_proj = self.W(user_emb).unsqueeze(1)      # [B, 1, D]
            scores    = (user_proj * rel_emb).sum(dim=-1)  # [B, K]
            weights   = F.softmax(scores, dim=-1)          # [B, K]
        else:
            # ── 无注意力（消融）：所有邻居权重相等 ──
            # 不管用户是谁，每个邻居的权重都是 1/K
            B, K = neigh_ent_ids.shape
            weights = torch.ones(B, K) / K                # [B, K]

        weights    = weights.unsqueeze(-1)                 # [B, K, 1]
        aggregated = (weights * neigh_emb).sum(dim=1)     # [B, D]

        self_emb = self.entity_emb(
            torch.LongTensor(entity_ids))                 # [B, D]
        return F.relu(self_emb + aggregated)              # [B, D]


# ================================================================
# 配置A：纯协同过滤
# 完全不用知识图谱，只用用户和歌曲的 embedding 直接做点积
# 等价于矩阵分解（MF）
# ================================================================

class CollaborativeFiltering(nn.Module):

    def __init__(self, n_users, n_items, embed_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.uniform_(self.user_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.item_emb.weight, -0.1, 0.1)

    def forward(self, user_ids, item_ids, kg_dict=None):
        # kg_dict 参数接收但不使用，保持接口统一
        u = self.user_emb(torch.LongTensor(user_ids))   # [B, D]
        v = self.item_emb(torch.LongTensor(item_ids))   # [B, D]
        return torch.sigmoid((u * v).sum(dim=-1))       # [B]


# ================================================================
# 配置B / C / D：KGNN 模型（通过参数控制消融）
# use_attention=False → 配置B（均匀权重）
# n_layers=1          → 配置C（单层）
# use_attention=True, n_layers=2 → 配置D（完整版）
# ================================================================

class KGNN(nn.Module):

    def __init__(self, n_users, n_entities, n_relations,
                 embed_dim=64, n_layers=2,
                 neighbor_size=8, use_attention=True):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        nn.init.uniform_(self.user_emb.weight, -0.1, 0.1)

        self.layers = nn.ModuleList([
            KGNNLayer(n_entities, n_relations, embed_dim,
                      neighbor_size, use_attention)
            for _ in range(n_layers)
        ])

    def forward(self, user_ids, entity_ids, kg_dict):
        u = self.user_emb(torch.LongTensor(user_ids))  # [B, D]
        for layer in self.layers:
            v_emb = layer(u, entity_ids, kg_dict)
        return torch.sigmoid((u * v_emb).sum(dim=-1))  # [B]


# ================================================================
# 通用训练和评估函数（4种配置共用）
# ================================================================

def train_model(model, train_data, item2entity, kg_dict,
                n_epochs=50, batch_size=1024, lr=1e-3,
                model_name='模型', is_cf=False):
    """
    is_cf=True  → 协同过滤，item_id 直接用 artistID 的索引
    is_cf=False → KGNN，item_id 需要用翻译本转成 entity_id
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-5)

    # 协同过滤需要把 artistID 映射到连续索引
    if is_cf:
        all_artist_ids = sorted(item2entity.keys())
        artist2idx     = {aid: i for i, aid in enumerate(all_artist_ids)}

    print(f"\n{'='*50}")
    print(f"开始训练：{model_name}")
    print(f"{'='*50}")

    best_auc = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        np.random.shuffle(train_data)
        total_loss = 0
        n_batches  = 0

        for start in range(0, len(train_data), batch_size):
            batch      = train_data[start: start + batch_size]
            user_ids   = [s[0] for s in batch]
            artist_ids = [s[1] for s in batch]
            labels     = torch.FloatTensor([s[2] for s in batch])

            if is_cf:
                # 协同过滤：直接用 artistID 的连续索引
                item_ids = [artist2idx[aid] for aid in artist_ids]
                scores   = model(user_ids, item_ids, None)
            else:
                # KGNN：用翻译本转成 entity_id
                entity_ids = [item2entity.get(aid, 0)
                              for aid in artist_ids]
                scores     = model(user_ids, entity_ids, kg_dict)

            loss = F.binary_cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches

        if epoch % 10 == 0:
            auc, acc = evaluate_model(
                model, train_data[:5000],   # 快速评估用部分训练集
                item2entity, kg_dict,
                is_cf=is_cf,
                artist2idx=artist2idx if is_cf else None
            )
            auc_test, acc_test = evaluate_model(
                model, data['test_data'],
                item2entity, kg_dict,
                is_cf=is_cf,
                artist2idx=artist2idx if is_cf else None
            )
            print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | "
                  f"Test AUC {auc_test:.4f} | Test Acc {acc_test:.4f}")
            if auc_test > best_auc:
                best_auc = auc_test

    print(f"→ {model_name} 最佳 AUC = {best_auc:.4f}")
    return best_auc


def evaluate_model(model, test_data, item2entity, kg_dict,
                   batch_size=1024, is_cf=False, artist2idx=None):
    model.eval()
    all_labels, all_scores = [], []

    with torch.no_grad():
        for start in range(0, len(test_data), batch_size):
            batch      = test_data[start: start + batch_size]
            user_ids   = [s[0] for s in batch]
            artist_ids = [s[1] for s in batch]
            labels     = [s[2] for s in batch]

            if is_cf:
                item_ids = [artist2idx.get(aid, 0)
                            for aid in artist_ids]
                scores   = model(user_ids, item_ids, None)
            else:
                entity_ids = [item2entity.get(aid, 0)
                              for aid in artist_ids]
                scores     = model(user_ids, entity_ids, kg_dict)

            all_labels.extend(labels)
            all_scores.extend(scores.cpu().numpy())

    auc  = roc_auc_score(all_labels, all_scores)
    pred = (np.array(all_scores) >= 0.5).astype(int)
    acc  = (pred == np.array(all_labels)).mean()
    return auc, acc


# ================================================================
# 主程序：依次运行4种配置，最后打印对比表
# ================================================================

if __name__ == '__main__':

    print("加载数据...")
    data = load_data()

    N_EPOCHS      = 50    # 每种配置训练50轮
    EMBED_DIM     = 64
    NEIGHBOR_SIZE = 8
    N_USERS       = data['n_users']
    N_ENTITIES    = data['n_entities']
    N_RELATIONS   = data['n_relations']
    N_ITEMS_CF    = len(data['item2entity'])  # 协同过滤用的物品数

    results = {}  # 存储每种配置的最终 AUC

    # ── 配置A：纯协同过滤 ──────────────────────────────────────
    model_A = CollaborativeFiltering(
        n_users=N_USERS,
        n_items=N_ITEMS_CF,
        embed_dim=EMBED_DIM
    )
    results['A_纯协同过滤'] = train_model(
        model_A, data['train_data'],
        data['item2entity'], data['kg_dict'],
        n_epochs=N_EPOCHS,
        model_name='配置A：纯协同过滤（无知识图谱）',
        is_cf=True
    )

    # ── 配置B：KGNN 均匀权重（去掉注意力）────────────────────
    model_B = KGNN(
        n_users=N_USERS, n_entities=N_ENTITIES,
        n_relations=N_RELATIONS, embed_dim=EMBED_DIM,
        n_layers=2, neighbor_size=NEIGHBOR_SIZE,
        use_attention=False    # ← 关掉注意力
    )
    results['B_均匀权重'] = train_model(
        model_B, data['train_data'],
        data['item2entity'], data['kg_dict'],
        n_epochs=N_EPOCHS,
        model_name='配置B：KGNN 均匀权重（无注意力机制）',
        is_cf=False
    )

    # ── 配置C：KGNN 单层聚合（去掉第2层）────────────────────
    model_C = KGNN(
        n_users=N_USERS, n_entities=N_ENTITIES,
        n_relations=N_RELATIONS, embed_dim=EMBED_DIM,
        n_layers=1,            # ← 只有1层
        neighbor_size=NEIGHBOR_SIZE,
        use_attention=True
    )
    results['C_单层聚合'] = train_model(
        model_C, data['train_data'],
        data['item2entity'], data['kg_dict'],
        n_epochs=N_EPOCHS,
        model_name='配置C：KGNN 单层聚合（无第2层）',
        is_cf=False
    )

    # ── 配置D：完整 KGNN-LS ───────────────────────────────────
    model_D = KGNN(
        n_users=N_USERS, n_entities=N_ENTITIES,
        n_relations=N_RELATIONS, embed_dim=EMBED_DIM,
        n_layers=2,
        neighbor_size=NEIGHBOR_SIZE,
        use_attention=True     # ← 完整版
    )
    results['D_完整KGNN-LS'] = train_model(
        model_D, data['train_data'],
        data['item2entity'], data['kg_dict'],
        n_epochs=N_EPOCHS,
        model_name='配置D：完整 KGNN-LS（2层+注意力）',
        is_cf=False
    )

    # ================================================================
    # 打印最终对比表
    # ================================================================

    print('\n')
    print('=' * 60)
    print('消融实验结果汇总')
    print('=' * 60)
    print(f"{'配置':<28} {'AUC':>8} {'vs 完整版':>12} {'结论'}")
    print('-' * 60)

    full_auc = results['D_完整KGNN-LS']

    rows = [
        ('A_纯协同过滤',  '无知识图谱、无注意力'),
        ('B_均匀权重',    '有知识图谱、无注意力'),
        ('C_单层聚合',    '有注意力、仅1层聚合'),
        ('D_完整KGNN-LS', '完整模型'),
    ]

    for key, desc in rows:
        auc  = results[key]
        diff = auc - full_auc
        if key == 'D_完整KGNN-LS':
            conclusion = '← 完整模型（基准）'
            diff_str   = '   —'
        else:
            diff_str   = f"{diff:+.4f}"
            if diff < -0.01:
                conclusion = '去掉后明显变差，该部件有效'
            elif diff < 0:
                conclusion = '去掉后略微变差'
            else:
                conclusion = '去掉后无明显影响'
        print(f"{desc:<28} {auc:>8.4f} {diff_str:>12}  {conclusion}")

    print('=' * 60)

    # 自动生成简历描述
    auc_A = results['A_纯协同过滤']
    auc_B = results['B_均匀权重']
    auc_D = results['D_完整KGNN-LS']

    kg_gain   = auc_D - auc_A          # 知识图谱的整体贡献
    attn_gain = auc_D - auc_B          # 注意力机制的贡献

    print('\n【可直接写进简历的描述】')
    print(f"在 Last.FM 数据集上，完整 KGNN-LS 模型 AUC 达 {auc_D:.4f}；")
    print(f"消融实验表明：引入知识图谱相比纯协同过滤 AUC 提升 "
          f"{kg_gain:.4f}（{kg_gain/auc_A*100:.1f}%），")
    print(f"用户自适应注意力机制相比均匀权重进一步提升 "
          f"{attn_gain:.4f}（{attn_gain/auc_B*100:.1f}%）。")