import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import itertools
import time
import decimal

def KNN_learn(T, cv):
    """
    Returns the K with the minimal prediction error (int)

    Arguments:
    T - training set (NumPy array)
    cv - number of folds for K-fold cross validation (int)
    """
    X_train = T[:, :-1]
    y_train = T[:, -1]
    k_nums, results = [], []
    for K in range(3, (int(len(X_train)/10))):
        mod = KNeighborsClassifier(n_neighbors=K)
        cv_results = cross_val_score(mod, X_train, y_train, cv=cv)
        avg = cv_results.mean()
        k_nums.append(K)
        results.append(avg)
    knum = results.index(max(results))
    the_k = k_nums[knum]
    return the_k

def KNN_predict(X_train, y_train, K, X_test):
    """
    Returns the list of predicted y values for each x test case

    Arguments:
    X_train, y_train - split training set T. Used in this format to speed up the code execution (NumPy arrays)
    K - how many neighbors to consider for the prediction (int)
    X_test - test set without the labels (NumPy array)
    """
    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def Falsify_Baseline(T, n, X_test, cv):
    """
    Returns a dictionary with keys 'outcome', 'violations'. 'violations' is a
    dictionary with keys 'index' and 'v_set' (violation set)

    Arguments:
    T - training data (numpy array)
    n - data poisoning limit (int)
    X_test - test data (numpy array)
    cv - amount of groups to consider for cross validation (int)
    """
    X_train = T[:, :-1]
    y_train = T[:, -1].astype(int)
    K = KNN_learn(T, cv)
    y_pred = KNN_predict(X_train, y_train, K, X_test)
    subsets = [np.array(list(combination))  for combination in itertools.combinations(T, len(X_train) - n)]
    outcome = np.full(len(X_test), "Unknown", dtype=object)
    result = {'outcome': outcome, 'violations': {'index': [], 'v_set': []}}
    start_time = time.time()
    for subset in subsets:
        X = subset[:, :-1]
        y = subset[:, -1].astype(int)
        # KNN_learn + predict
        Tk = np.column_stack((X, y))
        K_sub = KNN_learn(Tk, cv)
        y_pred_sub = KNN_predict(X, y, K_sub, X_test)
        # if condition
        if not np.array_equal(y_pred_sub, y_pred):
            diff_indices = np.where(y_pred_sub != y_pred)[0].astype(int)
            result['outcome'][diff_indices.astype(int)] = "Falsified"
            result["violations"]["index"].extend(diff_indices.tolist())
            result["violations"]["v_set"].extend(subset.tolist())
        end_time = time.time()
        total_time = end_time - start_time
        if total_time > 7200:
            break
    if int(total_time) < 7200:
        for i in range(len(result['outcome'])):
            if result['outcome'][i] == "Unknown":
                result['outcome'][i] = "Certified"
    return result

def solver(ind, T, n=0):
    """
    Returns a list with the most frequent labels for every x test case.

    Arguments:
    ind - indices of the nearest neighbors for every x. It is a result of .kneighbors for KNN model after fit.
    T - training set (NumPy array). Required to convert indices of nearest neighbors to labels.
    n - amount of how much to remove from the label counter (int). Default is 0.
    """
    labcount = []
    labset = []
    labels = T[:, -1].astype(int)
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    for lst in ind:
        counts = np.bincount(labels[lst], minlength=num_labels)
        counts[np.argmax(counts)] -= n
        labcount.append(counts.tolist())
        labset.append(np.argmax(counts))
    return labset

def QuickCertify(T, n, X_test):
    """
    It is the implementation of Algorithm 3
    Returns a list of Boolean value, 1 value per 1 test case

    Arguments:
    T - training set (NumPy array)
    n - poisoning threshold (int)
    X_test - test data (NumPy array)
    """
    X_train = T[:, :-1]
    y_train = T[:, -1]
    result = np.ones(len(X_test), dtype=bool)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    LabelSet = []
    for K in range(3, int(len(X_train)/10)):
        indices_0 = model.kneighbors(X_test, n_neighbors=K, return_distance=False)
        y_pred_0 = solver(indices_0, T, n=0)
        indices_1 = model.kneighbors(X_test, n_neighbors=(K+n), return_distance=False)
        y_pred_1 = solver(indices_1, T, n=n)
        diff_indices = np.flatnonzero(np.array(y_pred_0) != np.array(y_pred_1))
        result[diff_indices] = False
        if y_pred_0 not in LabelSet:
            LabelSet.append(y_pred_0)
    if len(LabelSet) > 1:
        LabelSet = np.array(LabelSet)
        non_matching = np.where(np.ptp(LabelSet, axis=0) != 0)[0]
        result[non_matching] = False
    return result

def partition(T, cv, random_state = None):
    """
    Returns a list of lists with indices. This list is used later to partition T into groups

    Arguments:
    T - training set (NumPy array)
    cv - amount of groups to consider for cross validation (int)
    random_state - a number which is used to fix the random processes (int)
    """
    np.random.seed(random_state)
    lst = [a for a in range(len(T))]
    np.random.shuffle(lst)
    partition_size = len(lst) // cv
    remaining_elements = len(lst) % cv
    partitions = [lst[b * partition_size:(b + 1) * partition_size] for b in range(cv)]
    for c in range(remaining_elements):
        partitions[c].append(lst[cv * partition_size + c])
    return partitions

def KNN_learn_init(T, cv, random_state = None):
    """
    It is the implementation of Algorithm 5
    Returns the dictionary with the keys 'K' - K value with minimal error, 
    'K_list' - the list of all K values checked,
    'Errors_of_groups' - error of each group in each K,
    'K_error' - the error calculated for each K,
    'ErrSet' - the set of training instances when the prediction was incorrect,
    'Groups' - groups used for cross validation

    Arguments:
    T - training set (NumPy array)
    cv - amount of groups to consider for cross validation (int)
    random_state - a number which is used to fix the random processes (int)
    """
    np.random.seed(random_state)
    result = {'K': None, 'K_list': [], 'Errors_of_groups': [], 'K_error': [], 'ErrSet': []}
    groups = partition(T, cv, random_state)
    result["Groups"] = groups
    for K in range(3, (len(T) // 10)):
        result["K_list"].append(K)
        K_err = []
        errset_k = []
        for partition in groups:
            partition = np.array(partition)
            errset = np.array([], dtype=int)
            # X and Y for predictions
            T_group = T[partition, :]
            X_group = T_group[:, :-1]
            y_group = T_group[:, -1].astype(int)
            # Making a training set T without the partitioned group
            T_rest = np.delete(T, partition, axis = 0)
            X_rest = T_rest[:, :-1]
            y_rest = T_rest[:, -1].astype(int)
            preds = KNN_predict(X_rest, y_rest, K, X_group)
            errset = np.where(preds != y_group)[0].tolist()
            errorG = len(errset) / len(partition)
            K_err.append(errorG)
            error_ind = partition[errset]
            errset_k.append(error_ind)
        result['Errors_of_groups'].append(K_err)
        error = sum(K_err) / cv
        result['K_error'].append(error)
        result['ErrSet'].append(errset_k)
    result['K'] = result["K_list"][result['K_error'].index(min(result['K_error']))]
    return result

def GenPromisingSubsets(T, n, x, y, max_iterations=15):
    """
    It is the implementation of Algorithm 4
    Returns 2 lists: one containing a list of removal sets R and another containing a list of promising subsets
    list of R is needed for Algorithm 6 and list of promising subsets is needed for Algorithm 2.

    Arguments:
    T - training set (NumPy array)
    n - poisoning threshold (int)
    x - test values (Numpy array)
    y - test labels (NumPy array)
    max_iterations - a limit of iterations to avoid (almost) infinite loops in case the result does not converge
    """
    promising_subsets, r_sub = [], []
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(T[:, :-1], T[:, -1])
    for K in range(3, int(len(T) / 10)):
        start, end, iterations = 0, n + 1, 0
        while start < end and iterations < max_iterations:
            mid = int((start + end) / 2)
            neigh = model.kneighbors(x, n_neighbors=(K + mid), return_distance=False)
            y_pred = solver(neigh, T, mid)
            if list(y) != y_pred:
                end = mid
            else:
                start = mid + 1
            iterations += 1 
        min_rmv = start
        if min_rmv <= n:
            ka = model.kneighbors(x, n_neighbors=(K + n + 1), return_distance=False)
            if min_rmv > 0:       
                for every_x in ka:
                    R1_list = list(itertools.combinations(every_x, min_rmv)) 
                    for R1 in R1_list:
                        remaining_size = n - len(R1)               
                        if remaining_size >= 1:
                            ids_col = np.arange(0, T.shape[0])
                            ids_col = ids_col.reshape(-1, 1)
                            T1 = np.hstack((ids_col, T))
                            T2 = np.delete(T1, every_x, axis=0)
                            T3 = T2[:, 0].astype(int).tolist()
                            R2_list = list(itertools.combinations(T3, remaining_size))           
                            for R2 in R2_list:
                                R = set(R1).union(set(R2))                         
                                if R not in r_sub:
                                    T4 = np.delete(T, list(R), axis=0)
                                    r_sub.append(R)
                                    promising_subsets.append(T4)
    return promising_subsets, r_sub

def KNN_learn_update(T, R, Error, n):
    """
    It is the implementation of Algorithm 6
    Returns the updated K value (int) according to the minimal error within K values

    Arguments:
    T - training set (NumPy array)
    R - removal sets (list of sets)
    Error - dictionary generated from the Algorithm 5
    n - poisoning threshold (int). Required to generate influSet (Algorithm 3 is used, it uses n)
    """
    # algo 6 line 1
    groups = Error["Groups"]
    errset = Error["ErrSet"]
    all_r_sets, influSet = [], []
    # algo 6 line 3: calculating T_prime (T')
    T_prime = np.delete(T, list(R), axis = 0)
    # algo 6 lines 2 and 4: calculating groups without R and influSet
    for g in range(len(groups)):
        # 1st influSet condition
        group = np.array(groups[g])
        mask = ~np.isin(np.arange(len(group)), list(R))
        result_group = group[mask]
        all_r_sets.append(result_group)
        X_test = T[result_group, :-1]
        y_test = T[result_group, -1]
        # 2nd influSet condition
        train_g = np.delete(T, result_group, axis = 0)
        X_train_g = train_g[:, :-1]
        y_train_g = train_g[:, -1]
        K_max = int(len(T) // 10)
        group_mod = KNeighborsClassifier(n_neighbors=K_max)
        group_mod.fit(X_train_g, y_train_g)
        ind = group_mod.kneighbors(X_test, n_neighbors=K_max, return_distance= False)
        mask = np.isin(ind, R)
        mask_any = np.any(mask, axis=1)
        indices = np.where(mask_any)[0]
        # 3rd influSet condition
        sett = []
        X_three = X_test[indices]
        y_three = y_test[indices]
        if len(X_three) == 0:
            influSet.append([])
            continue
        else:
            res = QuickCertify(X_train_g, y_train_g, n, X_three)
            indi = np.where(~np.array(res))[0]
            sett = list(X_three[indi])
        infx = X_three[sett]
        infy = y_three[sett]
        inf = np.column_stack((infx, infy))
        influSet.append(inf)
    # algo 6 lines 5-19: for cycle
    ks = []
    kerr = []
    for K in range(3, len(T)//10):
        gerr = []
        errset_K = errset[(K-3)]
        for i in range(len(all_r_sets)):
            # all_r_sets is just indices
            infl_group = influSet[i]
            group_r = T[all_r_sets[i], :]
            if len(infl_group) > 0:
                # T: x: tr_x and y: tr_y
                tr = np.delete(T, groups[i], axis = 0)
                tr_x = tr[:, :-1]
                tr_y = tr[:, -1]
                # T' x: t_g_x and y: t_g_y
                t_g = np.delete(T_prime, all_r_sets[i], axis=0)
                t_g_x = t_g[:, :-1]
                t_g_y = t_g[:, -1]
                need_x = infl_group[:, :-1]
                need_y = infl_group[:, -1]
                yo = KNN_predict(tr_x, tr_y, K, need_x)
                yn = KNN_predict(t_g_x, t_g_y, K, need_x)
                newSet_plus = list(infl_group[(yo == need_y) & (yn != need_y)])
                newSet_minus = list(infl_group[(yo != need_y) & (yn == need_y)])
                # errset works like this: first index for K, second index for group
                if len(errset_K) > 0:
                    errset_new = errset_K[i]
                    errset_new = np.delete(errset_new, list(R))
                    if len(newSet_minus) > 0:
                        errset_new = np.delete(errset_new, newSet_minus)
                    errset_neww = np.concatenate((errset_new, newSet_plus))
                else:
                    errset_neww = newSet_plus
                group_err_new = len(errset_neww) / len(group_r)
                gerr.append(group_err_new)
            else:
                group_err = 0
                gerr.append(group_err)
        ks.append(K)
        k_error = sum(gerr) / len(groups)
        kerr.append(k_error)
    errmin = min(kerr)
    hhh = kerr.index(errmin)
    best_K = ks[hhh]
    return best_K

def Falsify_New(T, n, X_test, cv, random_state = None):
    """
    It is the implementation of Algorithm 2
    Returns the result for each test case (list) and bad subsets (dictionary) with the x test array indice as key

    Arguments:
    T - test set (NumPy array)
    n - poisoning threshold (int)
    X_test - array of test cases (NumPy array)
    cv - amount of groups to consider for cross validation (int)
    random_state - a number which is used to fix the random processes (int)
    """
    result = ["Unknown" for _ in range(len(X_test))]
    X_train = T[:, :-1]
    y_train = T[:, -1] 
    quick_result = QuickCertify(T, n, X_test)
    for i, q in enumerate(quick_result):
        if q == True:
            result[i] = "Certified" 
    # reducing X_test space
    the_time, bad_subsets = 0, []
    ind = [result.index(a) for a in result if a == "Unknown"]
    X_test_new = X_test[ind] 
    if len(X_test_new) > 0:
        in_res = KNN_learn_init(T, cv, random_state)
        K = in_res['K']
        y_pred = KNN_predict(X_train, y_train, K, X_test_new)
        promising_subsets, r_sub = GenPromisingSubsets(T, n, X_test_new, y_pred)
        u = 0
        bad_subsets = dict()
        st_time = time.time()
        at = len(promising_subsets)
        for i in range(at):
            K_hat = KNN_learn_update(T, r_sub[i], in_res, n)
            prosub_df = promising_subsets[i]
            pro_x = prosub_df[:, :-1]
            pro_y = prosub_df[:, -1]
            y_hat = KNN_predict(pro_x, pro_y, K_hat, X_test_new)
            mismatches = y_hat != y_pred
            if np.any(mismatches):
                d = [ind[x] for x in range(len(X_test_new)) if mismatches[x]]
                result = np.array(result) 
                result[d] = "Falsified"
                result = result.tolist()
                d = []
                for x in range(len(X_test_new)):
                    if y_hat[x] != y_pred[x]:
                        x_ind = ind[x]
                        d.append(x_ind)
                        key = str(x_ind) + "_" +str(u)
                        bad_subsets[key] = promising_subsets[i]
                        u += 1
                        result[ind[x]] = "Falsified"
            nd_time = time.time()
            the_time = nd_time - st_time
            if int(the_time) > 1800:
                break
    if the_time < 1800:
        for v in range(len(result)):
            if result[v] == "Unknown":
                result[v] = "Certified"
    return result, bad_subsets

def add_up_to_n(original_data, n, random_state = None):
    """
    Data poisoning technique. The values and labels are generated within the range of
    original data values, rounding the values to the same amount of decimals as the original data
    Returns n rows of randomly generated data

    Arguments:
    original_data - clean data to poison (NumPy array)
    n - poisoning threshold (int)
    random_state - a number which is used to fix the random processes (int)
    """
    np.random.seed(random_state)
    num_rows, num_columns = original_data.shape
    random_data = np.empty((n, num_columns))
    for col_ind in range(num_columns - 1):
        first_number = original_data[0, col_ind]
        d = decimal.Decimal(str(first_number))
        decimal_places = d.as_tuple().exponent
        decimal_places = max(0, -decimal_places)
        if np.issubdtype(original_data[:, col_ind].dtype, np.integer):
            random_data[:, col_ind] = np.round(
                np.random.randint(
                    np.min(original_data[:, col_ind]),
                    np.max(original_data[:, col_ind]) + 1,
                    size=n
                    ), decimals = 0
            )
        else:
            random_data[:, col_ind] = np.round(
                np.random.uniform(
                    np.min(original_data[:, col_ind]),
                    np.max(original_data[:, col_ind]),
                    size=n
                    ), decimals=decimal_places
            )
    random_data[:, -1] = np.round(
                np.random.randint(
                    np.min(original_data[:, -1]),
                    np.max(original_data[:, -1]) + 1,
                    size=n
                    ), decimals = 0
            )
    return random_data

def change_n_labels(original_data, n, random_state = None):
    """
    Data poisoning technique. The labels are flipped within the range of
    original labels, making sure the new label for an observation is a different one
    Returns poisoned data with n labels flipped

    Arguments:
    original_data - clean data to poison (NumPy array)
    n - poisoning threshold (int)
    random_state - a number which is used to fix the random processes (int)
    """
    np.random.seed(random_state)
    changed_arr = original_data.copy()
    last_column = changed_arr[:, -1]
    for _ in range(n):
        index_to_change = np.random.randint(0, len(last_column))
        current_value = last_column[index_to_change]
        new_value = current_value
        while new_value == current_value:
            new_value = float(np.random.randint(np.min(last_column), np.max(last_column) + 1))
        last_column[index_to_change] = new_value
    changed_arr[:, -1] = last_column
    return changed_arr

def change_values(arr, n, random_state = None):
    """
    Data poisoning technique. All explanatory values of the n observations are changed, except the label
    The random values are within range of original data and rounded to resemble the original data
    Returns poisoned data values of n observations changed

    Arguments:
    original_data - clean data to poison (NumPy array)
    n - poisoning threshold (int)
    random_state - a number which is used to fix the random processes (int)
    """
    np.random.seed(random_state)
    changed_arr = arr.copy()
    num_rows, num_columns = changed_arr.shape
    indices_to_change = np.random.choice(num_rows, size=n, replace=False)
    for col_ind in range(num_columns - 1):
        first_number = changed_arr[0, col_ind]
        d = decimal.Decimal(str(first_number))
        decimal_places = d.as_tuple().exponent
        decimal_places = max(0, -decimal_places)
        new_values = np.round(
            np.random.uniform(
                np.min(changed_arr[:, col_ind]),
                np.max(changed_arr[:, col_ind]),
                size=n
            ),
            decimals=decimal_places
        )
        changed_arr[indices_to_change, col_ind] = new_values
    return changed_arr