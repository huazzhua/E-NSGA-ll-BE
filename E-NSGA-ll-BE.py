import copy
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from crossover import *
from env_select import env_select
from evolved_mutate import *
from guide_select import *
from initPops import *
from nonDominationSort import *


# =========================
# 随机种子
# =========================
GLOBAL_RANDOM_SEED = 1
SPLIT_RANDOM_STATE = 608

np.random.seed(GLOBAL_RANDOM_SEED)
random.seed(GLOBAL_RANDOM_SEED)


# =========================
# 实验参数
# =========================
TEST_SIZE = 0.2
INNER_CV_SPLITS = 5
TOP_PERCENTAGE = 0.05

# 如果你的CSV已经是“最小-最大归一化”后的数据，这里建议保持False。
# 如果你的CSV是原始未归一化数据，可以改为True；
# 代码会只在训练集上fit scaler，再transform测试集，避免数据泄露。
USE_MINMAX_SCALE = False

KNN_NEIGHBORS = 3
KNN_ALGORITHM = "kd_tree"


# =========================
# 稀疏化参数
# =========================
# 允许最终准确率相对进化结束时最高准确率下降的绝对百分点。
# 例如0.04表示最多下降4个百分点；0.0001表示最多下降0.01个百分点。
MAX_ACCURACY_DROP = 0.0001

# 最多从多少个候选解开始做后向消除。
N_PRUNE_STARTS = 8

# 每轮优先评估ReliefF分数最低的多少个已选特征。
# 数值越大，删减更充分，但运行更慢。
PRUNE_CANDIDATE_LIMIT = 40

# 连续多少轮没有成功删除特征后停止。
MAX_NO_IMPROVE_ROUNDS = 2

FLOAT_TOL = 1e-12


def make_clf():
    """统一创建分类器，避免复用同一个clf对象带来的状态干扰。"""
    return KNeighborsClassifier(
        n_neighbors=KNN_NEIGHBORS,
        algorithm=KNN_ALGORITHM,
        n_jobs=-1
    )


def top_select(X, score, percentage=0.05):
    """
    根据特征的重要性评分，选择前percentage比例的特征。

    注意：
    这个函数应该只在训练集上调用。
    测试集只能使用训练集选出的selected_index做同样的列选择，不能重新计算score。

    :param X: ndarray, shape = (n_samples, n_features)
    :param score: ndarray/list, shape = (n_features,)
    :param percentage: float
    :return:
        X_selected: 训练集筛选后的特征矩阵
        score_selected: 对应被选中特征的ReliefF分数
        selected_index: 被选中特征在原始特征空间中的0-based索引
    """
    score = np.asarray(score).reshape(-1)

    if not 0 < percentage <= 1:
        raise ValueError("percentage必须在(0, 1]范围内")

    # 至少保留1个特征，避免小维度数据集出现0个特征。
    length = max(1, int(np.ceil(percentage * len(score))))

    sorted_index = np.argsort(-score)
    selected_index = sorted_index[:length]

    X_selected = X[:, selected_index]
    score_selected = score[selected_index]

    return X_selected, score_selected, selected_index


def reliefFScore(X, y, k=5):
    """
    计算每个特征的重要性，基于最近邻距离来评估特征。

    注意：
    这个函数只能在训练集上计算。
    如果在完整数据集上计算，再做训练/测试划分，就会产生数据泄露。
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples, n_features = X.shape

    distance = pairwise_distances(X, metric="manhattan")
    score = np.zeros(n_features)

    for idx in range(n_samples):
        near_hit = []
        near_miss = dict()

        self_fea = X[idx, :]
        labels = np.unique(y).tolist()

        stop_dict = dict()
        for label in labels:
            stop_dict[label] = 0

        # 删除当前样本所属类别，剩下的是miss类别。
        del labels[labels.index(y[idx])]

        p_dict = dict()
        p_label_idx = float(len(y[y == y[idx]])) / float(n_samples)

        for label in labels:
            p_label_c = float(len(y[y == label])) / float(n_samples)
            p_dict[label] = p_label_c / (1 - p_label_idx)
            near_miss[label] = []

        distance_sort = []
        distance[idx, idx] = np.max(distance[idx, :])

        for i in range(n_samples):
            distance_sort.append([distance[idx, i], int(i), y[i]])
        distance_sort.sort(key=lambda item: item[0])

        for i in range(n_samples):
            # find k nearest hit points
            if distance_sort[i][2] == y[idx]:
                if len(near_hit) < k:
                    near_hit.append(distance_sort[i][1])
                elif len(near_hit) == k:
                    stop_dict[y[idx]] = 1
            else:
                # find k nearest miss points for each label
                if len(near_miss[distance_sort[i][2]]) < k:
                    near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                else:
                    if len(near_miss[distance_sort[i][2]]) == k:
                        stop_dict[distance_sort[i][2]] = 1

            stop = True
            for _, value in stop_dict.items():
                if value != 1:
                    stop = False

            if stop:
                break

        # update ReliefF score
        near_hit_term = np.zeros(n_features)
        for ele in near_hit:
            near_hit_term = np.array(abs(self_fea - X[ele, :])) + np.array(near_hit_term)

        near_miss_term = dict()
        for label, miss_list in near_miss.items():
            near_miss_term[label] = np.zeros(n_features)
            for ele in miss_list:
                near_miss_term[label] = (
                    np.array(abs(self_fea - X[ele, :])) + np.array(near_miss_term[label])
                )
            score += near_miss_term[label] / (k * p_dict[label])

        score -= near_hit_term / k

    return score


def load_csv(filename):
    """
    读取CSV数据集，并分解为特征和类标签。
    默认最后一列为标签，其余列为特征。
    """
    data = pd.read_csv(filename)

    X = data.iloc[:, 0:-1].to_numpy(dtype=float)
    Y = data.iloc[:, -1].to_numpy()

    print("nSamples", X.shape[0])
    print("nFeatures", X.shape[1])

    _, Y = np.unique(Y, return_inverse=True)
    return X, Y


def safe_train_test_split(X, Y, test_size=0.2, random_state=666):
    """
    先划分训练集/测试集。

    尽量使用stratify=Y保持类别比例；
    如果数据集类别样本太少导致分层失败，则退回普通随机划分。
    """
    try:
        return train_test_split(
            X,
            Y,
            test_size=test_size,
            random_state=random_state,
            stratify=Y
        )
    except ValueError as exc:
        print("警告：分层训练/测试划分失败，将改用普通随机划分。")
        print("原因：", exc)
        return train_test_split(
            X,
            Y,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )


def build_inner_cv(Y_train, max_splits=5, random_state=666):
    """
    构造训练集内部交叉验证。

    这个CV只在训练集内部使用，用于E_NSGA2搜索和后向消除。
    独立测试集不参与这里的任何过程。
    """
    _, counts = np.unique(Y_train, return_counts=True)
    min_class_count = int(np.min(counts))

    n_splits = min(max_splits, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "训练集中某些类别样本数少于2，无法进行内部交叉验证。"
            "请增大数据集、调整test_size，或改用普通训练/验证划分。"
        )

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )


def function(pops, current_feature_list, X_search, Y_search, cv, initial_dim):
    """
    计算每个个体，也就是每个特征子集的适应度。

    重要修改：
    1. 不再在这里重新划分train_test_split；
    2. 不再使用完整数据集X, Y；
    3. 只使用训练集X_search, Y_search上的内部CV评价候选特征子集；
    4. 测试集完全不参与E_NSGA2搜索。

    current_feature_list:
        当前染色体各列对应于top_select后训练集X_train_top中的真实列位置。
    initial_dim:
        top_select之后的初始特征维度，用作稀疏性目标的分母。
    """
    if pops.ndim == 1:
        pops = pops.reshape(1, -1)

    if pops.shape[1] != len(current_feature_list):
        raise ValueError(
            f"染色体长度{pops.shape[1]}与feature_list长度"
            f"{len(current_feature_list)}不一致"
        )

    fits = np.zeros((len(pops), 2))
    feature_map = np.asarray(current_feature_list, dtype=int)

    for j in range(pops.shape[0]):
        local_index = np.flatnonzero(pops[j] != 0)

        if local_index.size == 0:
            # 没有选中任何特征，视为最差。
            fits[j][0] = 1
            fits[j][1] = 1
        else:
            # 当前压缩染色体位置 -> top_select后X_train_top中的真实列位置
            actual_index = feature_map[local_index]
            X_subset = X_search[:, actual_index]

            acc = cross_val_score(
                make_clf(),
                X_subset,
                Y_search,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1
            )

            # 第一个目标：特征比例，越小越好。
            fits[j][0] = local_index.size / initial_dim

            # 第二个目标：错误率，越小越好。
            fits[j][1] = 1 - acc.mean()

    return fits


def E_NSGA2(
        nIter,
        nChr,
        nPop,
        pc,
        pm,
        score,
        feature_list,
        X_search,
        Y_search,
        cv,
        initial_dim):
    """
    E-NSGA-II特征选择。

    重要修改：
    该函数只在训练集上搜索特征子集。
    测试集不能传入，也不能在这里使用。
    """
    Iter = 1

    score = np.asarray(score).reshape(-1)
    feature_list = list(feature_list)

    pops = initPops(nPop, nChr, score)
    fits = function(pops, feature_list, X_search, Y_search, cv, initial_dim)

    feature_counts = None

    while Iter < nIter:
        Iter += 1

        # 保留原7_29动态变异率
        min_pm = 0.05
        current_pm = pm * (1 - Iter / nIter) + min_pm * (Iter / nIter)

        ranks = nonDominationSort(pops, fits)
        distances = crowdingDistanceSort(pops, fits, ranks)
        pops, fits = g_select(nPop, pops, fits, ranks, distances)

        chr_pops = crossover(pops, pc)
        chr_pops = e_mutate(
            chr_pops,
            current_pm,
            scores=score,
            fits=fits
        )

        chr_fits = function(chr_pops, feature_list, X_search, Y_search, cv, initial_dim)
        pops, fits = env_select(pops, fits, chr_pops, chr_fits)

        # 保留原7_29：连续5代统计后，每15代删除始终未被使用的列
        if pops.shape[1] > 100:
            if (Iter + 4) % 15 == 0:
                feature_counts = np.sum(pops, axis=0)

            if (Iter + 3) % 15 == 0 and feature_counts is not None:
                feature_counts += np.sum(pops, axis=0)

            if (Iter + 2) % 15 == 0 and feature_counts is not None:
                feature_counts += np.sum(pops, axis=0)

            if (Iter + 1) % 15 == 0 and feature_counts is not None:
                feature_counts += np.sum(pops, axis=0)

            if Iter % 15 == 0 and feature_counts is not None:
                feature_counts += np.sum(pops, axis=0)
                keep_mask = feature_counts != 0

                # 防止极端情况下全部特征被删除
                if not np.any(keep_mask):
                    keep_mask[np.argmax(score)] = True

                pops = pops[:, keep_mask]

                # 同步当前位置到top_select后X_train_top的映射
                feature_list = np.asarray(
                    feature_list, dtype=int
                )[keep_mask].tolist()

                # 引导变异使用的ReliefF分数同步删列
                score = score[keep_mask]

                fits = function(pops, feature_list, X_search, Y_search, cv, initial_dim)
                feature_counts = None

                print(f"第{Iter}代动态删列后当前维度: {pops.shape[1]}")

    # 最终稀疏化统一交给后面的“准确率预算后向消除”完成。
    return pops, fits, feature_list, score


def evaluate_one(solution, feature_list, X_search, Y_search, cv, initial_dim):
    """评估单个二进制解并返回一维适应度。"""
    return function(
        np.asarray(solution, dtype=int),
        feature_list,
        X_search,
        Y_search,
        cv,
        initial_dim
    )[0]


def best_first_backward_elimination(
        solution,
        feature_list,
        current_score,
        reference_error,
        max_accuracy_drop,
        X_search,
        Y_search,
        cv,
        initial_dim,
        candidate_limit=40,
        max_no_improve_rounds=2):
    """
    在准确率预算内执行best-first后向消除。

    注意：
    后向消除仍然只使用训练集内部CV，不使用测试集。
    """
    candidate = np.asarray(solution, dtype=int).copy()
    current_fit = evaluate_one(candidate, feature_list, X_search, Y_search, cv, initial_dim)

    allowed_error = min(1.0, reference_error + max_accuracy_drop)
    score_array = np.asarray(current_score).reshape(-1)

    no_improve_rounds = 0

    while np.count_nonzero(candidate) > 1:
        selected_positions = np.flatnonzero(candidate == 1)

        # 优先尝试删除ReliefF分数最低的特征。
        selected_scores = score_array[selected_positions]
        order = np.argsort(selected_scores)
        ordered_positions = selected_positions[order]

        if candidate_limit is not None and candidate_limit > 0:
            ordered_positions = ordered_positions[:candidate_limit]

        feasible_moves = []

        for position in ordered_positions:
            trial = candidate.copy()
            trial[position] = 0

            if np.count_nonzero(trial) == 0:
                continue

            trial_fit = evaluate_one(trial, feature_list, X_search, Y_search, cv, initial_dim)

            if trial_fit[1] <= allowed_error + FLOAT_TOL:
                # 排序优先级：
                # 1. 删除后错误率更低；
                # 2. 被删特征ReliefF分数更低。
                feasible_moves.append(
                    (trial_fit[1], score_array[position], position, trial_fit)
                )

        if not feasible_moves:
            no_improve_rounds += 1

            # 候选池过小时，再完整扫描一次全部已选特征，
            # 避免某些高ReliefF但冗余的特征被漏掉。
            if (
                no_improve_rounds <= max_no_improve_rounds
                and len(ordered_positions) < len(selected_positions)
            ):
                candidate_limit = len(selected_positions)
                continue

            break

        feasible_moves.sort(key=lambda item: (item[0], item[1]))
        _, _, best_position, best_trial_fit = feasible_moves[0]

        candidate[best_position] = 0
        current_fit = best_trial_fit
        no_improve_rounds = 0

    current_fit = evaluate_one(candidate, feature_list, X_search, Y_search, cv, initial_dim)
    return candidate, current_fit


def select_and_prune_sparse_solution(
        final_pop,
        final_fits,
        final_feature_list,
        current_score,
        X_search,
        Y_search,
        cv,
        initial_dim,
        max_accuracy_drop=0.04,
        n_starts=8,
        candidate_limit=40):
    """
    从多个不同起点进行后向消除，最后选择满足准确率预算的最稀疏解。

    起点包括：
    - 训练集内部CV准确率最高的若干个体；
    - 特征较少且已位于准确率预算内的若干个体。

    注意：
    这里仍然不能使用测试集。
    """
    errors = final_fits[:, 1]
    feature_counts = np.sum(final_pop != 0, axis=1)

    reference_error = float(np.min(errors))
    reference_accuracy = 1.0 - reference_error
    allowed_error = min(1.0, reference_error + max_accuracy_drop)
    accuracy_floor = 1.0 - allowed_error

    eligible = np.flatnonzero(errors <= allowed_error + FLOAT_TOL)
    if eligible.size == 0:
        eligible = np.array([int(np.argmin(errors))])

    # 一部分从高准确率端取，一部分从低特征数端取。
    accuracy_order = eligible[np.argsort(errors[eligible])]
    sparse_order = eligible[
        np.lexsort((errors[eligible], feature_counts[eligible]))
    ]

    half = max(1, n_starts // 2)
    start_indices = list(accuracy_order[:half]) + list(sparse_order[:n_starts])

    # 去重
    unique_start_indices = []
    seen = set()

    for idx in start_indices:
        key = tuple(np.asarray(final_pop[idx], dtype=int).tolist())
        if key not in seen:
            seen.add(key)
            unique_start_indices.append(int(idx))

        if len(unique_start_indices) >= n_starts:
            break

    print("进化结束时训练集内部CV最高准确率:", reference_accuracy)
    print(
        f"允许下降 {max_accuracy_drop * 100:.4f} 个百分点，"
        f"训练集内部CV最低允许准确率: {accuracy_floor}"
    )
    print("后向消除起点数量:", len(unique_start_indices))

    pruned_solutions = []
    pruned_fits = []

    for number, idx in enumerate(unique_start_indices, start=1):
        start_count = int(np.count_nonzero(final_pop[idx]))

        pruned_solution, pruned_fit = best_first_backward_elimination(
            final_pop[idx],
            final_feature_list,
            current_score=current_score,
            reference_error=reference_error,
            max_accuracy_drop=max_accuracy_drop,
            X_search=X_search,
            Y_search=Y_search,
            cv=cv,
            initial_dim=initial_dim,
            candidate_limit=candidate_limit,
            max_no_improve_rounds=MAX_NO_IMPROVE_ROUNDS
        )

        end_count = int(np.count_nonzero(pruned_solution))
        end_accuracy = 1.0 - pruned_fit[1]

        print(
            f"起点{number}: {start_count} -> {end_count} 个特征, "
            f"训练集内部CV准确率 {end_accuracy}"
        )

        if pruned_fit[1] <= allowed_error + FLOAT_TOL:
            pruned_solutions.append(pruned_solution)
            pruned_fits.append(pruned_fit)

    if not pruned_solutions:
        best_index = int(np.argmin(errors))
        return (
            final_pop[best_index].copy(),
            final_fits[best_index].copy(),
            reference_accuracy,
            accuracy_floor
        )

    pruned_solutions = np.asarray(pruned_solutions, dtype=int)
    pruned_fits = np.asarray(pruned_fits, dtype=float)

    counts = np.sum(pruned_solutions != 0, axis=1)
    min_count = np.min(counts)
    sparsest = np.flatnonzero(counts == min_count)

    # 特征数相同则选择训练集内部CV准确率更高的。
    chosen = sparsest[np.argmin(pruned_fits[sparsest, 1])]

    return (
        pruned_solutions[chosen],
        pruned_fits[chosen],
        reference_accuracy,
        accuracy_floor
    )


def evaluate_final_solution_on_independent_test(
        solution,
        final_feature_list,
        X_train_top,
        Y_train,
        X_test_top,
        Y_test):
    """
    用最终选出的特征子集训练模型，并在独立测试集上评估。

    这是整个流程中唯一使用测试集的地方。
    """
    solution = np.asarray(solution, dtype=int)
    selected_pos = np.flatnonzero(solution == 1)

    if selected_pos.size == 0:
        raise ValueError("最终解没有选择任何特征，无法进行测试集评估。")

    feature_map = np.asarray(final_feature_list, dtype=int)

    if solution.shape[0] != feature_map.shape[0]:
        raise ValueError(
            f"最终解长度{solution.shape[0]}与final_feature_list长度"
            f"{feature_map.shape[0]}不一致"
        )

    # 当前染色体位置 -> top_select后X_train_top/X_test_top中的列位置
    selected_top_positions = feature_map[selected_pos]

    model = make_clf()
    model.fit(X_train_top[:, selected_top_positions], Y_train)

    train_accuracy = model.score(X_train_top[:, selected_top_positions], Y_train)
    test_accuracy = model.score(X_test_top[:, selected_top_positions], Y_test)

    return train_accuracy, test_accuracy, selected_top_positions


if __name__ == "__main__":
    # 开始计时
    start_time = time.perf_counter()

    nIter = 150
    nPop = 100
    pc = 0.7
    pm = 0.15

    # 修改为你的数据集路径
    file_path = r"E:\数据集\最小‐最大归一化方法数据集\ALLAML.csv"
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    X_all, Y_all = load_csv(file_path)

    # =====================================================
    # 1. 先划分训练集和独立测试集
    #    从这一刻开始，测试集不能参与任何特征评分、特征筛选和算法搜索。
    # =====================================================
    X_train_raw, X_test_raw, Y_train, Y_test = safe_train_test_split(
        X_all,
        Y_all,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE
    )

    print("训练集样本数:", X_train_raw.shape[0])
    print("测试集样本数:", X_test_raw.shape[0])

    # =====================================================
    # 2. 可选归一化：只能在训练集fit，再transform测试集
    # =====================================================
    if USE_MINMAX_SCALE:
        minmax = preprocessing.MinMaxScaler()
        X_train = minmax.fit_transform(X_train_raw)
        X_test = minmax.transform(X_test_raw)
    else:
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()

    # =====================================================
    # 3. 只在训练集上计算ReliefF和top_select
    # =====================================================
    f_score = reliefFScore(X_train, Y_train)

    X_train_top, score, selected_original_indices = top_select(
        X_train,
        f_score,
        percentage=TOP_PERCENTAGE
    )

    # 测试集只能使用训练集选出来的列，不能重新计算ReliefF或top_select。
    X_test_top = X_test[:, selected_original_indices]

    print("top_select后训练集特征数:", X_train_top.shape[1])

    # =====================================================
    # 4. 构造训练集内部CV
    #    用于E_NSGA2搜索和后向消除。
    # =====================================================
    inner_cv = build_inner_cv(
        Y_train,
        max_splits=INNER_CV_SPLITS,
        random_state=SPLIT_RANDOM_STATE
    )

    dim = X_train_top.shape[1]
    feature_list = np.arange(dim).tolist()

    # =====================================================
    # 5. 只在训练集上运行E_NSGA2
    # =====================================================
    final_pop, final_fits, final_feature_list, final_score = E_NSGA2(
        nIter=nIter,
        nChr=dim,
        nPop=nPop,
        pc=pc,
        pm=pm,
        score=score,
        feature_list=feature_list,
        X_search=X_train_top,
        Y_search=Y_train,
        cv=inner_cv,
        initial_dim=dim
    )

    # 结束计时：只统计搜索阶段和选择阶段之前的主要运行时间
    end_time = time.perf_counter()
    running_time = end_time - start_time

    print(f"算法运行时间: {running_time:.4f} 秒")

    # =====================================================
    # 6. 后向消除仍然只在训练集内部CV上完成
    # =====================================================
    best_solution, best_fitness, reference_accuracy, accuracy_floor = (
        select_and_prune_sparse_solution(
            final_pop=final_pop,
            final_fits=final_fits,
            final_feature_list=final_feature_list,
            current_score=final_score,
            X_search=X_train_top,
            Y_search=Y_train,
            cv=inner_cv,
            initial_dim=dim,
            max_accuracy_drop=MAX_ACCURACY_DROP,
            n_starts=N_PRUNE_STARTS,
            candidate_limit=PRUNE_CANDIDATE_LIMIT
        )
    )

    # =====================================================
    # 7. 最后一步，才使用独立测试集评估一次
    # =====================================================
    train_accuracy, test_accuracy, selected_top_positions = (
        evaluate_final_solution_on_independent_test(
            solution=best_solution,
            final_feature_list=final_feature_list,
            X_train_top=X_train_top,
            Y_train=Y_train,
            X_test_top=X_test_top,
            Y_test=Y_test
        )
    )

    selected_original = selected_original_indices[selected_top_positions]

    print("数据集名称:", dataset_name)
    print("最优特征子集:", np.array2string(best_solution, threshold=np.inf, max_line_width=np.inf))

    print("选中特征数量:", len(selected_original))
    print("选中特征原始索引(0-based):", selected_original.tolist())
    print("选中特征原始索引(1-based):", (selected_original + 1).tolist())

    # 计算稀疏性
    sparsity = np.sum(best_solution) / dim
    print(f"特征选择的稀疏性: {sparsity} (减少了 {100 * (1 - sparsity)}% 的特征)")

    # 输出训练集内部CV结果和独立测试集结果
    final_inner_cv_accuracy = 1 - best_fitness[1]

    print("进化结束时训练集内部CV最高准确率:", reference_accuracy)
    print("训练集内部CV最低允许准确率:", accuracy_floor)
    print("最终稀疏特征子集的训练集内部CV准确率:", test_accuracy)
    print("训练集内部CV实际下降百分点:", (reference_accuracy - final_inner_cv_accuracy) * 100)

    print("最终模型训练集准确率:", train_accuracy)
    print("最终模型独立测试集准确率:", final_inner_cv_accuracy)

    print("说明: 独立测试集没有参与ReliefF、top_select、E_NSGA2和后向消除，只在最后评估一次。")

    # 提取最终非支配目标集
    ranks = nonDominationSort(final_pop, final_fits)
    pareto_fits = final_fits[ranks == 0]

    # HV需要的参数
    hv_ref_point = np.array([1.1, 1.1])

    print("训练集内部CV Pareto目标值：")
    print(pareto_fits)

    print("HV参考点：")
    print(hv_ref_point)
