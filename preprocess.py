"""
KGNN 数据预处理
"""

import numpy as np
from collections import defaultdict

# 第一步：读取"翻译本" item_index2entity_id.txt
# 作用：把 user_artists.dat 里的 artistID 翻译成 kg.txt 里的实体编号
item2entity = {}   # artistID -> entity_id
entity2item = {}   # entity_id -> artistID

with open('item_index2entity_id.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        artist_id = int(parts[0])    # user_artists.dat 里的 artistID
        entity_id = int(parts[1])    # kg.txt 里的实体编号
        item2entity[artist_id] = entity_id
        entity2item[entity_id] = artist_id

print("=== 翻译本 ===")
print(f"共有 {len(item2entity)} 首歌有对应的知识图谱实体")
print("前5条：", list(item2entity.items())[:5])


# 第二步：读取知识图谱 kg.txt
# 作用：记录每个实体的邻居是谁、通过什么关系连接

# kg.txt 的关系是文字，需要先把文字转成数字编号
relation2id = {}   # "music.artist.origin" -> 0
relation_cnt = 0

kg_triples = []    # 存所有三元组 (头实体, 关系编号, 尾实体)

with open('kg.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        head = int(parts[0])      # 头实体编号
        relation = parts[1]           # 关系名称（文字）
        tail = int(parts[2])      # 尾实体编号

        # 如果这个关系名还没见过，给它分配一个新编号
        if relation not in relation2id:
            relation2id[relation] = relation_cnt
            relation_cnt += 1

        relation_id = relation2id[relation]
        kg_triples.append((head, relation_id, tail))

print("\n=== 知识图谱 ===")
print(f"共有 {len(kg_triples)} 条关系三元组")
print(f"共有 {len(relation2id)} 种关系类型")
print("关系类型举例：")
for name, idx in list(relation2id.items())[:5]:
    print(f"  {idx}: {name}")
print("前3条三元组（头实体, 关系编号, 尾实体）：", kg_triples[:3])


# ================================================================
# 第三步：把知识图谱整理成"邻居字典"
# 作用：给定一个实体，能快速查到它的所有邻居
# ================================================================

# kg_dict[实体] = [(邻居实体, 关系编号), (邻居实体, 关系编号), ...]
kg_dict = defaultdict(list)

for head, relation_id, tail in kg_triples:
    kg_dict[head].append((tail, relation_id))
    kg_dict[tail].append((head, relation_id))   # 双向，两边都能查

print("\n=== 邻居字典 ===")
# 查一下 entity=0（对应 artistID=2）的邻居
example_entity = 0
neighbors = kg_dict[example_entity]
print(f"entity={example_entity} 有 {len(neighbors)} 个邻居")
print(f"前3个邻居（邻居实体, 关系编号）：{neighbors[:3]}")


# ================================================================
# 第四步：读取用户行为数据 user_artists.dat
# 作用：知道每个用户听过哪些歌（正样本）
# ================================================================

# user_artists.dat 格式：userID  artistID  weight
# weight 是听歌次数，我们只用它来判断"听过"还是"没听过"
# 只保留在翻译本里出现过的歌（有知识图谱信息的歌）

user_pos_items = defaultdict(list)   # user -> [听过的 artistID 列表]
all_items = set()                    # 所有出现过的 artistID

skipped = 0
with open('user_artists.dat', 'r') as f:
    next(f)   # 跳过第一行表头 "userID artistID weight"
    for line in f:
        parts = line.strip().split('\t')
        user_id = int(parts[0])
        artist_id = int(parts[1])
        # weight  = int(parts[2])   # 暂时不用

        # 只保留有知识图谱信息的歌
        if artist_id in item2entity:
            user_pos_items[user_id].append(artist_id)
            all_items.add(artist_id)
        else:
            skipped += 1

all_items = list(all_items)

print("\n=== 用户行为 ===")
print(f"共有 {len(user_pos_items)} 个用户")
print(f"共有 {len(all_items)} 首有图谱信息的歌")
print(f"跳过了 {skipped} 条（歌曲没有知识图谱信息）")
print(f"用户2 听过的歌（前5首）：{user_pos_items[2][:5]}")


# ================================================================
# 第五步：生成训练样本（正样本 + 负样本）
# 正样本：用户听过的歌，标签=1
# 负样本：用户没听过的歌，随机抽，标签=0
# ================================================================

def generate_samples(user_pos_items, all_items, neg_ratio=1):
    """
    user_pos_items: {user_id: [听过的歌列表]}
    all_items:      所有歌的列表
    neg_ratio:      每个正样本对应几个负样本（默认1:1）
    返回：[(user_id, artist_id, label), ...]
    """
    samples = []
    all_items_set_map = {}   # 缓存每个用户的正样本集合，加速负采样

    for user_id, pos_list in user_pos_items.items():
        pos_set = set(pos_list)
        all_items_set_map[user_id] = pos_set

        # 加入正样本
        for item in pos_list:
            samples.append((user_id, item, 1))

        # 随机采等量负样本
        n_neg = len(pos_list) * neg_ratio
        neg_count = 0
        while neg_count < n_neg:
            candidate = np.random.choice(all_items)
            if candidate not in pos_set:
                samples.append((user_id, candidate, 0))
                neg_count += 1

    return samples

np.random.seed(42)   # 固定随机种子，保证结果可复现
all_samples = generate_samples(user_pos_items, all_items)

print("\n=== 训练样本 ===")
print(f"总样本数：{len(all_samples)}")
pos_count = sum(1 for _, _, label in all_samples if label == 1)
neg_count = sum(1 for _, _, label in all_samples if label == 0)
print(f"正样本（听过）：{pos_count}")
print(f"负样本（没听过）：{neg_count}")
print("前5条样本（用户, 歌曲, 标签）：")
for s in all_samples[:5]:
    print(f"  用户{s[0]} - 歌曲{s[1]} - {'听过✓' if s[2]==1 else '没听✗'}")


# ================================================================
# 第六步：划分训练集和测试集（80% 训练，20% 测试）
# ================================================================

np.random.shuffle(all_samples)
split = int(len(all_samples) * 0.8)
train_data = all_samples[:split]
test_data  = all_samples[split:]

print("\n=== 数据集划分 ===")
print(f"训练集：{len(train_data)} 条")
print(f"测试集：{len(test_data)} 条")


# ================================================================
# 第七步：统计各类数量，供模型初始化用
# ================================================================

# 找出所有实体编号的最大值
all_entities = set()
for head, _, tail in kg_triples:
    all_entities.add(head)
    all_entities.add(tail)

n_users = max(user_pos_items.keys()) + 1
n_items = len(all_items)
n_entities = max(all_entities) + 1
n_relations = len(relation2id)

print("\n=== 模型需要的基本参数 ===")
print(f"n_users    = {n_users}    （用户数量）")
print(f"n_items    = {n_items}    （歌曲数量）")
print(f"n_entities = {n_entities}  （知识图谱实体总数）")
print(f"n_relations = {n_relations}  （关系类型数量）")
print("\n预处理完成！下一步可以开始写模型代码了。")


# ================================================================
# 把处理好的数据打包，方便后续模块使用
# ================================================================

data = {
    'train_data'    : train_data,
    'test_data'     : test_data,
    'kg_dict'       : kg_dict,
    'item2entity'   : item2entity,
    'relation2id'   : relation2id,
    'all_items'     : all_items,
    'n_users'       : n_users,
    'n_entities'    : n_entities,
    'n_relations'   : n_relations,
}