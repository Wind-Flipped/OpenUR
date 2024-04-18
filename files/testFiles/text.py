import numpy as np
import re
from collections import defaultdict
from itertools import chain

# load text.npy
traj_sentences = np.load("str.npy")
print(traj_sentences.shape)
print(traj_sentences[56902:])
# 正则表达式来提取居住地和工作地
living_place_pattern = r"live in ((?:\w+[\s\w]+))"
work_place_pattern = r"go to ((?:\w+[\s\w]+(?:, )?)+)"

# 提取信息
extracted_places = []
for sentence in traj_sentences:
    living_place_match = re.search(living_place_pattern, sentence)
    work_place_match = re.search(work_place_pattern, sentence)

    if living_place_match and work_place_match:
        living_place = living_place_match.group(1)
        work_places = work_place_match.group(1).split(", ")
        extracted_places.append((living_place, work_places))
print("len of extracted_places")
print(len(extracted_places))
# 创建一个字典来存储居住地和工作地组合以及对应的用户列表
user_combinations = defaultdict(list)

# 遍历用户轨迹，将用户添加到对应的组合中
for user, (living_place, work_places) in enumerate(extracted_places):
    for work_place in work_places:
        user_combinations[(living_place, work_place)].append(user)

# 筛选出居住地和工作地都相同的用户
same_living_work_users = {k: v for k, v in user_combinations.items() if len(v) > 1}

# print(same_living_work_users)
print("----------------")
print("----------------")
print("----------------")
# 将所有用户序号添加到一个集合中去重
all_users = set(chain.from_iterable(same_living_work_users.values()))

# 将去重后的用户序号转换为列表并排序
sorted_users = sorted(all_users)

# print(sorted_users)
print("--------------")
print(len(sorted_users))

# qianmen_temple_users = []
#
# for user, (living_place, work_places) in enumerate(extracted_places):
#     if "Heping Street" in living_place and "Ditan park" in work_places:
#         qianmen_temple_users.append(user)
#
# print(qianmen_temple_users)