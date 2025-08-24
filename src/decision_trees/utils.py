def gini(labels):
    impurity = 1
    for cls in set(labels):
        prob = labels.count(cls) / len(labels)
        impurity -= prob ** 2
    return impurity


def split_data(feature, labels, threshold):
    left_labels, right_labels = [], []
    for feat_val, lbl in zip(feature, labels):
        if feat_val <= threshold:
            left_labels.append(lbl)
        else:
            right_labels.append(lbl)
    return left_labels, right_labels


def split_gain(feature, labels, threshold):
    parent_imp = gini(labels)
    left_labels, right_labels = split_data(feature, labels, threshold)
    n_total = len(labels)

    if len(left_labels) == 0 or len(right_labels) == 0:
        return 0

    weighted_imp = (len(left_labels) / n_total) * gini(left_labels) + \
                   (len(right_labels) / n_total) * gini(right_labels)
    return parent_imp - weighted_imp


def split_dataset(features, labels, feature_name, threshold):
    left_features = {f: [] for f in features}
    right_features = {f: [] for f in features}
    left_labels, right_labels = [], []

    for i in range(len(labels)):
        value = features[feature_name][i]
        if value <= threshold:
            left_labels.append(labels[i])
            for f in features:
                left_features[f].append(features[f][i])
        else:
            right_labels.append(labels[i])
            for f in features:
                right_features[f].append(features[f][i])

    return (left_features, left_labels), (right_features, right_labels)


def best_split(feature, labels):
    best_gain, best_thre = -1, None
    for thre in set(feature):
        gain = split_gain(feature, labels, thre)
        if gain > best_gain:
            best_gain, best_thre = gain, thre
    return best_gain, best_thre


def best_feature_split(features, labels):
    best_gain, best_feature, best_thresh = -1, None, None
    for feat_name, feat_values in features.items():
        gain, thresh = best_split(feat_values, labels)
        if gain > best_gain:
            best_gain, best_feature, best_thresh = gain, feat_name, thresh
    return best_feature, best_thresh, best_gain
