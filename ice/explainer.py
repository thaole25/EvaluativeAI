import matplotlib.pyplot as plt
import numpy as np
import pydotplus
import torch

from collections import defaultdict
import time
import pickle
import os
import shutil

from ice import channel_reducer
from preprocessing import params, initdata
import classifiers


class Explainer:
    def __init__(
        self,
        args=None,
        title="",
        layer_name="",
        class_names=None,
        utils=None,
        keep_feature_images=True,
        reducer_type="NMF",
        n_components=10,
        featuretopk=80,
        featureimgtopk=5,
        epsilon=1e-4,
    ):
        self.args = args
        self.title = title
        self.layer_name = layer_name
        self.class_names = class_names
        self.class_nos = len(class_names) if class_names is not None else 0
        self.keep_feature_images = keep_feature_images
        self.reducer_type = reducer_type
        self.featuretopk = featuretopk
        self.featureimgtopk = featureimgtopk  # number of images for a feature
        self.n_components = n_components
        self.epsilon = epsilon
        self.utils = utils
        self.reducer = None
        self.feature_distribution = None
        self.features = {}
        self.clf_y_preds = None

        self.exp_location = params.EXP_PATH
        self.font = params.FONT_SIZE

    def load(self):
        title = self.title
        with open(self.exp_location / title / (title + ".pickle"), "rb") as f:
            tdict = pickle.load(f)
            self.__dict__.update(tdict)

    def save(self):
        if not os.path.exists(self.exp_location):
            os.mkdir(self.exp_location)
        title = self.title
        if not os.path.exists(self.exp_location / title):
            os.mkdir(self.exp_location / title)
        with open(self.exp_location / title / (title + ".pickle"), "wb") as f:
            pickle.dump(self.__dict__, f)

    def train_model(self, model, processed_data):
        if self.args.reducer == "NA":
            self._train_concepts_on_classifier(model, processed_data)
        else:
            self._train_reducer(model, processed_data.balanced_per_class)
            if self.args.train_clf:
                self._train_concepts_on_classifier(model, processed_data)
            else:
                self._estimate_weight(model, processed_data.balanced_per_class)

    def _train_reducer(self, model, loaders):

        print("Training reducer:")

        if self.reducer is None:
            if not self.reducer_type in channel_reducer.ALGORITHM_NAMES:
                print("reducer not exist")
                return

            if channel_reducer.ALGORITHM_NAMES[self.reducer_type] == "decomposition":
                if self.reducer_type == "NMF":
                    self.reducer = channel_reducer.ChannelDecompositionReducer(
                        n_components=self.n_components,
                        reduction_alg=self.reducer_type,
                        max_iter=2000,
                    )
                else:
                    self.reducer = channel_reducer.ChannelDecompositionReducer(
                        n_components=self.n_components,
                        reduction_alg=self.reducer_type,
                    )
            else:
                self.reducer = channel_reducer.ChannelClusterReducer(
                    n_components=self.n_components, reduction_alg=self.reducer_type
                )

        X_features = []
        for loader in loaders:
            X_features.append(model.get_feature(loader, self.layer_name))
        print("1/5 Feature maps gathered.")

        if not self.reducer._is_fit:
            nX_feature = np.concatenate(X_features)
            total = np.product(nX_feature.shape)
            l = nX_feature.shape[0]
            if total > params.CALC_LIMIT:
                p = params.CALC_LIMIT / total
                print("dataset too big, train with {:.2f} instances".format(p))
                idx = np.random.choice(l, int(l * p), replace=False)
                nX_feature = nX_feature[idx]

            print("loading complete, with size of {}".format(nX_feature.shape))
            start_time = time.time()
            nX = self.reducer.fit_transform(nX_feature)

            print("2/5 Reducer trained, spent {} s.".format(time.time() - start_time))

        self.cavs = self.reducer._reducer.components_
        nX = nX.mean(axis=(1, 2))
        self.feature_distribution = {
            "overall": [
                (nX[:, i].mean(), nX[:, i].std(), nX[:, i].min(), nX[:, i].max())
                for i in range(self.n_components)
            ]
        }

        reX = []
        self.feature_distribution["classes"] = []
        for X_feature in X_features:
            t_feature = self.reducer.transform(X_feature)
            pred_feature = self._feature_filter(t_feature)
            self.feature_distribution["classes"].append(
                [
                    pred_feature.mean(axis=0),
                    pred_feature.std(axis=0),
                    pred_feature.min(axis=0),
                    pred_feature.max(axis=0),
                ]
            )
            reX.append(self.reducer.inverse_transform(t_feature))

        err = []
        for i in range(len(self.class_names)):
            res_true = model.feature_predict(X_features[i], layer_name=self.layer_name)[
                :, i
            ]
            res_recon = model.feature_predict(reX[i], layer_name=self.layer_name)[:, i]
            err.append(
                abs(res_true - res_recon).mean(axis=0)
                / (abs(res_true.mean(axis=0)) + self.epsilon)
            )

        self.reducer_err = np.array(err)
        if type(self.reducer_err) is not np.ndarray:
            self.reducer_err = np.array([self.reducer_err])

        print("3/5 Error estimated, fidelity: {}.".format(self.reducer_err))

        return self.reducer_err

    def _estimate_weight(self, model, loaders):
        if self.reducer is None:
            return

        X_features = []

        for loader in loaders:
            X_features.append(
                model.get_feature(loader, self.layer_name)  # [: params.ESTIMATE_NUM]
            )
        X_feature = np.concatenate(X_features)

        print("4/5 Weight estimator initialized.")

        self.test_weight = []
        for i in range(self.n_components):
            cav = self.cavs[i, :]

            res1 = model.feature_predict(
                X_feature - self.epsilon * cav, layer_name=self.layer_name
            )
            res2 = model.feature_predict(
                X_feature + self.epsilon * cav, layer_name=self.layer_name
            )

            res_dif = res2 - res1
            dif = res_dif.mean(axis=0) / (2 * self.epsilon)
            if type(dif) is not np.ndarray:
                dif = np.array([dif])
            self.test_weight.append(dif)

        print("5/5 Weight estimated.")

        self.test_weight = np.array(self.test_weight)

    def _train_concepts_on_classifier(self, model, processed_data):
        if self.reducer is None:
            X_features = model.get_feature(
                processed_data.balanced_X, layer_name=self.layer_name
            )
            X_test_features = model.get_feature(
                processed_data.X_test, layer_name=self.layer_name
            )
        else:
            X_features = self.reducer.transform(
                model.get_feature(processed_data.balanced_X, layer_name=self.layer_name)
            )
            X_test_features = self.reducer.transform(
                model.get_feature(processed_data.X_test, layer_name=self.layer_name)
            )
        if self.args.feature_type == "mean":
            X_features = X_features.mean(axis=(1, 2))
            X_test_features = X_test_features.mean(axis=(1, 2))
        else:
            X_features = X_features.max(axis=(1, 2))
            X_test_features = X_test_features.max(axis=(1, 2))
        remove_concepts = self.args.remove_concepts
        remove_concepts = [int(feat) for feat in remove_concepts]
        if remove_concepts:
            X_features = np.delete(X_features, remove_concepts, axis=1)
            X_test_features = np.delete(X_test_features, remove_concepts, axis=1)
        classifier = classifiers.factory(model_type=self.args.ice_clf)
        classifier.fit(X_features, processed_data.balanced_y)
        if self.args.ice_clf in ["lda", "logistic"]:
            self.test_weight = np.transpose(classifier.coef_)
        elif self.args.ice_clf == "gnb":
            self.test_weight = np.transpose(classifier.theta_)
        elif self.args.ice_clf == "mlp":
            self.test_weight = classifier.coefs_[0]
        y_test_pred = classifier.predict(X_test_features)
        y_preds = y_test_pred.tolist()
        self.clf_y_preds = y_preds
        return self.clf_y_preds

    def generate_features(self, model, loaders, threshold):
        self._visualise_features(model, loaders)
        self._save_features(threshold=threshold)
        if self.keep_feature_images == False:
            self.features = {}
        return

    def _feature_filter(self, featureMaps, threshold=None):
        # filter feature map to feature value with threshold for target value
        if self.args.feature_type == "mean":
            res = featureMaps.mean(axis=(1, 2))
        else:
            res = featureMaps.max(axis=(1, 2))
        if threshold is not None:
            res = -abs(res - threshold)
        return res

    def _update_feature_dict(self, x, h, nx, nh, threshold=None):

        if type(x) == type(None):
            return nx, nh
        else:
            x = np.concatenate([x, nx])
            h = np.concatenate([h, nh])

            nidx = self._feature_filter(h, threshold=threshold).argsort()[
                -self.featureimgtopk :
            ]
            x = x[nidx, ...]
            h = h[nidx, ...]
            return x, h

    def _visualise_features(self, model, loaders, featureIdx=None, inter_dict=None):
        path_lesion_dict = initdata.get_lesion_by_path()
        featuretopk = min(self.featuretopk, self.n_components)
        imgTopk = self.featureimgtopk
        if featureIdx is None:
            featureIdx = []
            tidx = []
            w = self.test_weight
            for i, _ in enumerate(self.class_names):
                tw = w[:, i]
                tidx += tw.argsort()[::-1][:featuretopk].tolist()
            featureIdx += list(set(tidx))

        nowIdx = set(self.features.keys())
        featureIdx = list(set(featureIdx) - nowIdx)
        featureIdx.sort()

        if len(featureIdx) == 0:
            print("All feature gathered")
            return

        print("visualising features:")
        print(featureIdx)

        features = {}
        for No in featureIdx:
            features[No] = [None, None]

        if inter_dict is not None:
            for k in inter_dict.keys():
                inter_dict[k] = [[None, None] for No in featureIdx]

        print("loading training data")
        for i, loader in enumerate(loaders):

            for X in loader:
                X_data, X_paths = X[0], X[2]

                X_unique = []  # select images not from the same lesion
                lesion_used = set()
                for ix, x in enumerate(X_data):
                    base_name = os.path.basename(X_paths[ix])
                    image_id = os.path.splitext(base_name)[0]
                    lesion = path_lesion_dict[image_id]
                    if lesion not in lesion_used:
                        X_unique.append(x)
                        lesion_used.add(lesion)

                X_unique = torch.stack(X_unique)
                featureMaps = self.reducer.transform(
                    model.get_feature(X_unique, self.layer_name)
                )

                X_feature = self._feature_filter(featureMaps)

                for No in featureIdx:
                    samples, heatmap = features[No]
                    idx = X_feature[:, No].argsort()[-imgTopk:]

                    nheatmap = featureMaps[idx, :, :, No]
                    nsamples = X_unique[idx, ...]

                    samples, heatmap = self._update_feature_dict(
                        samples, heatmap, nsamples, nheatmap
                    )

                    features[No] = [samples, heatmap]

                    if inter_dict is not None:
                        for k in inter_dict.keys():
                            vmin = self.feature_distribution["overall"][No][2]
                            vmax = self.feature_distribution["overall"][No][3]
                            temp_v = (vmax - vmin) * k + vmin
                            inter_dict[k][No] = self._update_feature_dict(
                                inter_dict[k][No][0],
                                inter_dict[k][No][1],
                                X_unique,
                                featureMaps[:, :, :, No],
                                threshold=temp_v,
                            )

            print(
                "Done with class: {}, {}/{}".format(
                    self.class_names[i], i + 1, len(loaders)
                )
            )
        # create repeat prototypes in case lack of samples
        for no, (x, h) in features.items():
            idx = h.mean(axis=(1, 2)).argmax()
            for i in range(h.shape[0]):
                if h[i].max() == 0:
                    x[i] = x[idx]
                    h[i] = h[idx]

        self.features.update(features)
        self.save()
        return inter_dict

    def _save_features(self, threshold=0.7, background=0.5, smooth=True):
        if os.path.exists(self.exp_location / self.title):
            shutil.rmtree(self.exp_location / self.title)
        os.mkdir(self.exp_location / self.title)
        feature_path = self.exp_location / self.title / "feature_imgs"

        if not os.path.exists(feature_path):
            os.mkdir(feature_path)

        for idx in self.features.keys():

            x, h = self.features[idx]
            # x = self.gen_masked_imgs(x,h,threshold=threshold,background = background,smooth = smooth)
            minmax = False
            if self.reducer_type == "PCA":
                minmax = True
            x, h = self.utils.img_filter(
                x,
                h,
                threshold=threshold,
                background=background,
                smooth=smooth,
                minmax=minmax,
            )
            img_width, img_height = self.utils.img_width, self.utils.img_height
            combined_img_height = img_height * self.featureimgtopk
            img_channels = self.utils.nchannels
            # the processing in ImageUtils handles channel last
            img_size_processing = [img_width, combined_img_height, img_channels]
            nimg = np.zeros(img_size_processing)
            nh = np.zeros([img_width, combined_img_height])
            num_examples = x.shape[0]  # should be equal to featureimgtopk
            for i in range(num_examples):
                timg = self.utils.deprocessing(x[i])
                if timg.max() > 1:
                    timg = timg / 255.0
                    timg = abs(timg)
                timg = np.clip(timg, 0, 1)
                max_h = self.utils.find_max_area_contour(h[i])
                nimg[:, i * img_width : (i + 1) * img_height, :] = timg
                nh[:, i * img_width : (i + 1) * img_height] = max_h
            fig = self.utils.contour_img(nimg, nh)
            fig.savefig(
                feature_path / (str(idx) + ".jpg"),
                bbox_inches="tight",
                pad_inches=0,
                format="jpg",
                dpi=params.DPI,
            )
            plt.close(fig)

    def global_explanations(self, concept_names=None, feature_only=True):
        title = self.title
        fpath = (self.exp_location / self.title / "feature_imgs").absolute()
        feature_topk = min(self.featuretopk, self.n_components)
        feature_weight = self.test_weight
        class_names = self.class_names
        Nos = range(self.class_nos)

        font = self.font

        def LR_graph(wlist, No, feature_only):
            def node_string(count, fidx, w, feature_only):
                if concept_names:
                    concept_name = concept_names[fidx]
                else:
                    concept_name = fidx
                nodestr = ""
                nodestr += '{} [label=< <table border="0">'.format(count)
                nodestr += "<tr>"
                nodestr += '<td><img src= "{}" /></td>'.format(
                    str(fpath / ("{}.jpg".format(fidx)))
                )
                nodestr += "</tr>"
                if not feature_only:
                    nodestr += '<tr><td><FONT POINT-SIZE="{}"> FeatureRank: {} </FONT></td></tr>'.format(
                        font, count
                    )
                    nodestr += '<tr><td><FONT POINT-SIZE="{}"> Concept: {}, Weight: {:.3f} </FONT></td></tr>'.format(
                        font, concept_name, w
                    )
                else:
                    if concept_names:
                        nodestr += '<tr><td><FONT POINT-SIZE="{}"> Concept {}: {}</FONT></td></tr>'.format(
                            font, fidx + 1, concept_name
                        )
                    else:
                        nodestr += '<tr><td><FONT POINT-SIZE="{}"> Concept {}</FONT></td></tr>'.format(
                            font, fidx + 1
                        )
                nodestr += "</table>  >];\n"
                return nodestr

            resstr = "digraph Tree {node [shape=box] ;rankdir = LR;\n"

            count = len(wlist)
            for k, v in wlist:
                resstr += node_string(count, k, v, feature_only)
                count -= 1

            if not feature_only:
                resstr += '0 [label=< <table border="0">'
                resstr += '<tr><td><FONT POINT-SIZE="{}"> ClassName: {} </FONT></td></tr>'.format(
                    font, class_names[No]
                )
                resstr += '<tr><td><FONT POINT-SIZE="{}"> Fidelity error: {:.3f} % </FONT></td></tr>'.format(
                    font, self.reducer_err[No] * 100
                )
                resstr += '<tr><td><FONT POINT-SIZE="{}"> First {} features out of {} </FONT></td></tr>'.format(
                    font, feature_topk, self.n_components
                )
                resstr += "</table>  >];\n"
            resstr += "}"

            return resstr

        if not os.path.exists(self.exp_location / title / "GE"):
            os.mkdir(self.exp_location / title / "GE")

        if not os.path.exists(params.EXAMPLE_PATH / "global"):
            os.mkdir(params.EXAMPLE_PATH / "global")

        print("Generate explanations with fullset condition")
        if feature_only:
            wlist = [
                (k, v) for k, v in enumerate(feature_weight[:, 0])
            ]  # Fixed wlist - quick fix
            graph = pydotplus.graph_from_dot_data(
                LR_graph(wlist, No=None, feature_only=feature_only)
            )
            if self.args.example:
                graph.write_jpg(
                    str(params.EXAMPLE_PATH / "global" / ("{}.jpg".format(title)))
                )
        else:
            for i in Nos:
                # wlist = [(j, feature_weight[j][i]) for j in feature_weight[:, i].argsort()[-feature_topk:]]
                wlist = [(k, v) for k, v in enumerate(feature_weight[:, i])]
                graph = pydotplus.graph_from_dot_data(LR_graph(wlist, i))
                graph.write_jpg(
                    str(
                        self.exp_location
                        / title
                        / "GE"
                        / ("{}.jpg".format(class_names[i]))
                    )
                )

    def segment_concept_image(self, x, h, feature_id, img_path, background=0.7):
        minmax = False
        if self.reducer_type == "PCA":
            minmax = True
        x1, h1 = self.utils.img_filter(
            x, np.array([h[:, :, feature_id]]), background=background, minmax=minmax
        )
        x1 = self.utils.deprocessing(x1)
        x1 = x1 / x1.max()
        x1 = abs(x1)
        max_h = self.utils.find_max_area_contour(h1[0])
        fig = self.utils.contour_img(x1[0], max_h)
        fig.savefig(
            img_path,
            format="jpg",
            dpi=params.DPI,
        )
        plt.close(fig)
        return img_path

    def local_explanations(
        self,
        x,
        model,
        background=0.7,
        instance_name=None,
        with_total=True,
        display_value=True,
        plot_img=True,
        concept_names=None,
    ):
        font = self.font
        featuretopk = min(self.featuretopk, self.n_components)
        target_classes = list(range(self.class_nos))
        w = self.test_weight
        pred = model.predict(x)[0][target_classes]

        if self.reducer is not None:
            h = self.reducer.transform(model.get_feature(x, self.layer_name))[0]
        else:
            h = model.get_feature(x, self.layer_name)[0]
        s = h.mean(axis=(0, 1))
        concept_contributions = defaultdict(
            dict
        )  # {class_id: {feature_id: contribution_score}}
        remove_features = [int(feat) for feat in self.args.remove_concepts]
        for cidx in target_classes:
            tw = w[:, cidx]
            tw_idx = tw.argsort()[::-1][:featuretopk]
            for fidx in tw_idx:
                if fidx not in remove_features:
                    concept_contributions[cidx][fidx] = s[fidx] * tw[fidx]

        if not plot_img:
            return concept_contributions, s

        fpath = self.exp_location / self.title / "explanations"
        if not os.path.exists(fpath):
            os.mkdir(fpath)

        afpath = fpath / "all"
        if not os.path.exists(afpath):
            os.mkdir(afpath)

        if instance_name is not None:
            fpath = fpath / instance_name
            if not os.path.exists(fpath):
                os.mkdir(fpath)
        else:
            count = 0
            while os.path.exists(fpath / str(count)):
                count += 1
            fpath = fpath / str(count)
            os.mkdir(fpath)
            instance_name = str(count)

        feature_idx = []
        for cidx in target_classes:
            tw = w[:, cidx]
            tw_idx = tw.argsort()[::-1][:featuretopk]
            feature_idx.append(tw_idx)
        feature_idx = list(set(np.concatenate(feature_idx).tolist()))
        new_feature_idx = [fid for fid in feature_idx if fid not in remove_features]
        feature_idx = new_feature_idx
        for k in feature_idx:
            self.segment_concept_image(
                x=x,
                h=h,
                feature_id=k,
                img_path=fpath / ("feature_{}.jpg".format(k)),
                background=background,
            )

        fpath = fpath.absolute()

        gpath = self.exp_location.absolute() / self.title / "feature_imgs"

        def node_string(fidx, score, weight):
            nodestr = ""
            nodestr += '<table border="0">\n'
            nodestr += "<tr>"
            nodestr += '<td><img src= "{}" /></td>'.format(
                str(fpath / ("feature_{}.jpg".format(fidx)))
            )
            nodestr += '<td><img src= "{}" /></td>'.format(
                str(gpath / ("{}.jpg".format(fidx)))
            )
            nodestr += "</tr>\n"
            if display_value:
                if concept_names:
                    concept_name = concept_names[fidx]
                else:
                    concept_name = fidx
                nodestr += '<tr><td colspan="2"><FONT POINT-SIZE="{}"> ClassName: {}, Concept: {}</FONT></td></tr>\n'.format(
                    font, self.class_names[cidx], concept_name
                )
                nodestr += '<tr><td colspan="2"><FONT POINT-SIZE="{}"> Similarity: {:.3f}, Weight: {:.3f}, Contribution: {:.3f}</FONT></td></tr> \n'.format(
                    font, score, weight, score * weight
                )
            nodestr += "</table>  \n"
            return nodestr

        for cidx in target_classes:
            tw = w[:, cidx]
            # tw_idx = tw.argsort()[::-1][:featuretopk]
            tw_idx = list(range(len(tw)))
            total = 0
            resstr = "digraph Tree {node [shape=plaintext] ;\n"
            resstr += '1 [label=< \n<table border="0"> \n'
            for fidx in tw_idx:
                resstr += "<tr><td>\n"
                resstr += node_string(fidx, s[fidx], tw[fidx])
                total += s[fidx] * tw[fidx]
                resstr += "</td></tr>\n"

            if with_total:
                resstr += '<tr><td><FONT POINT-SIZE="{}"> Total Contribution: {:.3f}, Prediction: {:.3f}</FONT></td></tr> \n'.format(
                    font, total, pred[cidx]
                )
            resstr += "</table> \n >];\n"
            resstr += "}"

            graph = pydotplus.graph_from_dot_data(resstr)
            graph.write_jpg(str(fpath / ("explanation_{}.jpg".format(cidx))))
            graph.write_jpg(str(afpath / ("{}_{}.jpg".format(instance_name, cidx))))

        if self.args.example:
            final_fig = plt.figure(
                layout="constrained", figsize=(16, len(feature_idx) * 2)
            )
            subfigs = final_fig.subfigures(1, 1, wspace=0.01)  # width_ratios=[1, 2]
            # axsLeft = subfigs[0].subplots(1, 1)
            # axsLeft.barh(np.arange(len(feature_idx)), feature_idx, align='center', color='coral')
            # axsLeft.set_yticks(np.arange(len(feature_idx)), labels=['Feature '+str(f) for f in  feature_idx])
            rows = len(feature_idx)
            cols = 2
            axsRight = subfigs.subplots(rows, cols, sharey=True, width_ratios=[1, 5])
            # final_fig.subplots_adjust(wspace=0.02)
            for i, feat in enumerate(feature_idx):
                img = plt.imread(fpath / ("feature_{}.jpg".format(feat)))
                axsRight[i][0].imshow(img)
                axsRight[i][0].get_xaxis().set_ticks([])
                axsRight[i][0].get_yaxis().set_ticks([])
                if concept_names:
                    axsRight[i][0].set_ylabel(
                        "Concept {}\n {}".format(feat + 1, concept_names[feat]),
                        rotation=0,
                        labelpad=75,
                    )
                else:
                    axsRight[i][0].set_ylabel(
                        "Concept " + str(feat + 1), rotation=0, labelpad=35
                    )

                img = plt.imread(gpath / ("{}.jpg".format(feat)))
                axsRight[i][1].imshow(img)
                axsRight[i][1].get_xaxis().set_ticks([])
                axsRight[i][1].get_yaxis().set_ticks([])

            axsRight[rows - 1][0].set_xlabel("Test image")
            axsRight[rows - 1][1].set_xlabel("Train images")
            example_img_name = self.title + "_" + instance_name + ".jpg"
            final_fig.savefig(
                params.EXAMPLE_PATH / (example_img_name), format="jpg", dpi=params.DPI
            )
            plt.close()

        return concept_contributions, s
