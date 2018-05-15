import numpy as np


# calculate the f-score and others
class Acc_score(object):
    def __init__(self, query_label_names):
        self.hists = []
        self.query_label_names = query_label_names

    def reset(self):
        self.hists = []

    def collect(self, gt_labels, output_labels):
        # compute hist
        assert gt_labels.shape == output_labels.shape
        hist = self._fast_hist(gt_labels, output_labels,
                               len(self.query_label_names) + 1)
        self.hists.append(hist)

    def get_f1_results(self, eval_label_names=None):
        if eval_label_names is None:
            eval_label_names = self.query_label_names
        hists = np.stack(self.hists, axis=0)
        hist_sum = np.sum(hists, axis=0)
        # print(hist_sum)
        f1s = self._collect_f1s(hist_sum, self.query_label_names,
                                eval_label_names)
        return f1s

    @staticmethod
    def _fast_hist(a, b, n):
        '''
        fast histogram calculation
        ---
        * a, b: label ids, a.shape == b.shape
        * n: number of classes to measure
        '''
        k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
        return np.bincount(
            n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

    @staticmethod
    def _collect_f1s(hist_vals, label_names, eval_label_names):
        f1s = dict()
        for eval_names in eval_label_names:
            if isinstance(eval_names, str):
                eval_names = [eval_names]
            name = '+'.join(eval_names)
            label_ids = [label_names.index(n) + 1 for n in eval_names]
            # print(label_ids)
            intersected = 0
            for label_id1 in label_ids:
                for label_id2 in label_ids:
                    intersected += hist_vals[label_id1, label_id2]
            A = hist_vals[label_ids, :].sum()
            B = hist_vals[:, label_ids].sum()
            f1 = 2 * intersected / (A + B)
            f1s[name] = f1

        intersected = hist_vals[0, 0]
        A = hist_vals[0, :].sum()
        B = hist_vals[:, 0].sum()
        f1 = 2 * intersected / (A + B)
        f1s['other'] = f1
        return f1s