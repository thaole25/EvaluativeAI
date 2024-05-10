import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np

from tqdm import tqdm
from collections import defaultdict, OrderedDict

from woe import WoEImageGaussian, WoEExplainer
from ice import PytorchModelWrapper, Explainer, ImageUtils
from preprocessing import params
import classifiers
from pcbm.models import PosthocLinearCBM

# SELECTED_PRINT_TEST = (
#     [0, 2] + [21, 23] + [41, 43] + [61, 63] + [81, 83] + [101, 103] + [121, 123]
# )  # selected test instance to plot feature images

SELECTED_PRINT_TEST = []


def get_metrics(y_trues, y_preds):
    acc = accuracy_score(y_trues, y_preds)
    binary_ty = [params.IDX_TO_BINARY[i] for i in y_trues]
    binary_y_preds = [params.IDX_TO_BINARY[i] for i in y_preds]
    binary_acc = accuracy_score(binary_ty, binary_y_preds)
    tn, fp, fn, tp = confusion_matrix(binary_ty, binary_y_preds).ravel()
    sensitivity = tp / (tp + fn)  # recall
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    return acc, binary_acc, sensitivity, specificity, precision, f1_score


def print_results(method_name, y_true, y_pred, model_row, args):
    acc, binary_acc, sensitivity, specificity, precision, f1_score = get_metrics(
        y_true, y_pred
    )
    print("Accuracy on test {}: {:4.2f}%".format(method_name, acc * 100))
    model_row.append(acc * 100)
    model_row.append(binary_acc * 100)
    model_row.append(sensitivity * 100)
    model_row.append(specificity * 100)
    model_row.append(precision * 100)
    model_row.append(f1_score * 100)
    # cm = confusion_matrix(y_true, y_pred, normalize="true")
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # img_name = params.RESULT_PATH / "{}_matrix_{}_ncomp{}_seed{}.png".format(
    #     method_name, args.model, args.no_concepts, args.seed
    # )
    # disp.plot().figure_.savefig(img_name)


class ChannelPCBMReducer(object):
    def __init__(
        self, concept_bank, n_components=8, channel_last=False
    ):  # input channel is at last?
        if not isinstance(n_components, int):
            raise ValueError("n_components must be an int, not '%s'." % n_components)
        self.channel_last = channel_last
        self.cavs = concept_bank.vectors.cpu().detach().numpy()
        self.intercepts = concept_bank.intercepts.cpu().detach().numpy()
        self.norms = concept_bank.norms.cpu().detach().numpy()
        self.names = concept_bank.concept_names.copy()
        self.n_components = self.cavs.shape[0]
        self._is_fit = True

    def fit(self, x, y, verbose=False):
        return

    def transform(self, x):
        orig_shape = x.shape
        x = x.reshape([-1, x.shape[-1]])
        margins = ((np.matmul(self.cavs, x.T) + self.intercepts) / (self.norms)).T
        nshape = list(orig_shape[:-1]) + [-1]
        x = margins.reshape(nshape)
        return x

    def inverse_transform(self, x):
        return None


class ConceptUtils:
    def __init__(self, args, concept_names=None):
        if args.algo != "ice" and args.algo != "pcbm":
            ValueError("Method name is either ice or pcbm")
        self.method_name = args.algo
        self.args = args
        if args.algo == "ice":
            self.title = self.method_name.upper() + "_HAM_{}_ncomp{}_seed{}_{}".format(
                args.model, args.no_concepts, args.seed, args.reducer
            )
        elif args.algo == "pcbm":
            self.title = (
                self.method_name.upper()
                + "_HAM_{}_ncomp{}_seed{}_{}_{}".format(
                    args.model, args.no_concepts, args.seed, args.lr, args.n_samples
                )
            )
        self.concept_names = concept_names
        if args.example:
            self.title = (
                "A_"
                + self.method_name.upper()
                + "_HAM_{}_ncomp{}_seed{}".format(
                    args.model, args.no_concepts, args.seed
                )
            )

        self.EXP_SAVE = "{}_Exp_{}_ncomp{}_seed{}_{}_{}.sav".format(
            self.args.algo.upper(),
            self.args.model,
            self.args.no_concepts,
            self.args.seed,
            self.args.feature_type,
            self.args.clf,
        )

        self.CONCEPT_SAVE = "{}_concept_{}_ncomp{}_seed{}_{}_{}.sav".format(
            self.args.algo.upper(),
            self.args.model,
            self.args.no_concepts,
            self.args.seed,
            self.args.feature_type,
            self.args.clf,
        )

    def get_ice_model(
        self, backbone_model, balanced_train_dl_per_class, original_train_dl_per_class
    ):
        print("Running the concept model...")
        LAYER_NAME = params.ICE_CONCEPT_LAYER[self.args.model]

        ice_model = PytorchModelWrapper(
            backbone_model,
            batch_size=params.BATCH_SIZE,
            predict_target=params.TARGET_CLASSES,
            input_channel_first=True,
            model_channel_first=True,
            input_size=[params.INPUT_CHANNEL, params.INPUT_RESIZE, params.INPUT_RESIZE],
        )

        Exp = Explainer(
            args=self.args,
            title=self.title,
            layer_name=LAYER_NAME,
            class_names=params.CLASSES_NAMES,
            utils=ImageUtils(
                mode="ham10000",
                img_format="channels_first",
                img_size=(params.INPUT_RESIZE, params.INPUT_RESIZE),
                nchannels=params.INPUT_CHANNEL,
                mean=params.INPUT_MEAN,
                std=params.INPUT_STD,
            ),
            n_components=self.args.no_concepts,
            reducer_type=self.args.reducer,
        )

        if self.args.retrain_concept:
            print("Retrain the concept model...")
            # train reducer based on target classes
            Exp.train_model(ice_model, balanced_train_dl_per_class)
            if self.args.example or self.args.save_model:
                # generate features
                Exp.generate_features(
                    ice_model, original_train_dl_per_class, self.args.threshold
                )
                # generate global explanations
                # Exp.global_explanations()
                # save the explainer, use load to load it with the same title
                Exp.save()
        else:
            Exp.load()

        if self.args.save_model:
            print("Saving concept model...")
            torch.save(Exp, params.MODEL_PATH / self.EXP_SAVE)
            torch.save(ice_model, params.MODEL_PATH / self.CONCEPT_SAVE)

        return Exp, ice_model

    def get_pcbm_model(
        self,
        concept_bank,
        backbone_model,
        balanced_train_dl_per_class,
        original_train_dl_per_class,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        def linear_weight(
            X_train, y_train, X_test, y_test, model, posthoc_layer, layer_name
        ):
            """
            The output of embeddings_train is [9735, 7, 7, 2048]. This func is for the part "Projection onto the Concept Subspace"
            """
            embeddings_train = (
                torch.tensor(model.get_feature(X_train, layer_name=layer_name))
                .to(device=params.DEVICE)
                .detach()
            )
            embeddings_test = (
                torch.tensor(model.get_feature(X_test, layer_name=layer_name))
                .to(device=params.DEVICE)
                .detach()
            )
            print("linear layer: ", embeddings_train.shape)
            embeddings_train = embeddings_train.mean(axis=(1, 2))
            embeddings_test = embeddings_test.mean(axis=(1, 2))
            print(embeddings_train.shape)
            # embeddings_train = torch.squeeze(embeddings_train)
            # embeddings_test = torch.squeeze(embeddings_test)
            projs_train = (
                posthoc_layer.compute_dist(embeddings_train).detach().cpu().numpy()
            )
            projs_test = (
                posthoc_layer.compute_dist(embeddings_test).detach().cpu().numpy()
            )
            classifier = classifiers.factory(
                model_type=self.args.pcbm_classifier, seed=self.args.seed
            )
            classifier.fit(projs_train, y_train)
            print("linear weights, ", projs_train.shape, y_train.shape)
            print(
                "Linear layer - train accuracy:",
                classifier.score(projs_train, y_train) * 100,
            )
            print(
                "Linear layer - test accuracy:",
                classifier.score(projs_test, y_test) * 100,
            )
            return classifier.coef_, classifier.intercept_

        LAYER_NAME = params.PCBM_CONCEPT_LAYER[self.args.model]
        posthoc_layer = PosthocLinearCBM(
            concept_bank,
            backbone_name=f"{self.args.dataset}_{self.args.model}",
            idx_to_class=None,
            n_classes=len(params.TARGET_CLASSES),
        )
        posthoc_layer = posthoc_layer.to(params.DEVICE)
        m = torch.nn.Sequential(
            OrderedDict([("backbone", backbone_model), ("classifier", posthoc_layer)])
        )
        pcbm_model = PytorchModelWrapper(
            m,
            batch_size=params.BATCH_SIZE,
            predict_target=params.TARGET_CLASSES,
            input_channel_first=True,
            model_channel_first=True,
            input_size=[params.INPUT_CHANNEL, params.INPUT_RESIZE, params.INPUT_RESIZE],
        )
        pcbm_model.layer_dict = dict(pcbm_model.model.named_modules())
        weight, bias = linear_weight(
            X_train, y_train, X_test, y_test, pcbm_model, posthoc_layer, LAYER_NAME
        )
        pcbm_model.model.classifier.set_weights(weight, bias)
        Exp = Explainer(
            args=self.args,
            title=self.title,
            layer_name=LAYER_NAME,
            class_names=params.CLASSES_NAMES,
            utils=ImageUtils(
                mode="ham10000",
                img_format="channels_first",
                img_size=(params.INPUT_RESIZE, params.INPUT_RESIZE),
                nchannels=params.INPUT_CHANNEL,
                mean=params.INPUT_MEAN,
                std=params.INPUT_STD,
            ),
            n_components=concept_bank.vectors.shape[0],
            reducer_type=self.args.reducer,
        )
        if self.args.retrain_concept:
            print("Retrain the concept model...")
            r = ChannelPCBMReducer(concept_bank)
            Exp.reducer = r
            Exp.reducer_err = [0] * Exp.n_components
            Exp.cavs = r.cavs
            Exp._estimate_weight(pcbm_model, balanced_train_dl_per_class)
            if self.args.example or self.args.save_model:
                # generate features
                Exp.generate_features(
                    pcbm_model,
                    original_train_dl_per_class,
                    threshold=self.args.threshold,
                )
                # generate global explanations
                # Exp.global_explanations(concept_names=self.concept_names)
                Exp.save()
        else:
            Exp.load()

        if self.args.save_model:
            print("Saving concept model...")
            torch.save(Exp, params.MODEL_PATH / self.EXP_SAVE)
            torch.save(pcbm_model, params.MODEL_PATH / self.CONCEPT_SAVE)

        return Exp, pcbm_model

    def evaluate_concept_model(self, test_concepts, ty, model_row):
        ty = ty.tolist()
        all_concepts = test_concepts
        all_concepts["sum"] = all_concepts[
            [str(c) for c in range(self.args.no_concepts)]
        ].sum(axis=1)
        max_concept_contribution_df = all_concepts.groupby(["id"]).agg(
            max_sum=("sum", "max")
        )
        max_concept_contribution_df["y"] = all_concepts.merge(
            max_concept_contribution_df, left_on=["sum"], right_on=["max_sum"]
        )["y"]
        y_preds = max_concept_contribution_df["y"].to_list()
        print_results(
            method_name=self.method_name,
            y_true=ty,
            y_pred=y_preds,
            model_row=model_row,
            args=self.args,
        )

    def format_concept_data(self, X, y, X_paths, Exp, concept_model):
        """
        Get the concept contributions of the training set
        """
        target_classes = params.TARGET_CLASSES
        num_instanes = len(X)
        data_concepts = defaultdict(list)
        data_similarities = defaultdict(list)
        for i in tqdm(range(num_instanes)):
            instance_name = str(i)
            y_i = y[i]
            instance_x = X[i]
            if self.args.example and i in SELECTED_PRINT_TEST:
                print("Test ID: ", i, " ground truth: ", y_i, " path: ", X_paths[i])
                concept_contributions, similarities = Exp.local_explanations(
                    instance_x,
                    concept_model,
                    instance_name=instance_name,
                    plot_img=True,
                    display_value=False,
                    with_total=False,
                    concept_names=self.concept_names,
                    remove_features=self.args.remove_concepts,
                )
            else:
                concept_contributions, similarities = Exp.local_explanations(
                    instance_x,
                    concept_model,
                    instance_name=instance_name,
                    plot_img=False,
                    concept_names=self.concept_names,
                    remove_features=self.args.remove_concepts,
                )
            num_features = len(similarities)
            for cls in concept_contributions:
                data_concepts["id"].append(i)
                data_concepts["y"].append(cls)
                for feat in range(num_features):
                    if feat not in concept_contributions[cls]:
                        data_concepts[str(feat)].append(0)
                    else:
                        data_concepts[str(feat)].append(
                            concept_contributions[cls][feat]
                        )
            if y_i in target_classes:
                data_similarities["y"].append(y_i)
                for feat in range(num_features):
                    data_similarities[str(feat)].append(similarities[feat])

        data_concepts = pd.DataFrame.from_dict(data_concepts)
        data_similarities = pd.DataFrame.from_dict(data_similarities)
        return data_concepts, data_similarities

    def concept_eval_runner(
        self, X_test, y_test, X_test_paths, Exp, concept_model, model_row
    ):
        test_concepts, _ = self.format_concept_data(
            X=X_test,
            y=y_test,
            X_paths=X_test_paths,
            Exp=Exp,
            concept_model=concept_model,
        )

        self.evaluate_concept_model(
            test_concepts=test_concepts, ty=y_test, model_row=model_row
        )


class WoeUtils:
    def __init__(
        self,
        X,
        y,
        Exp,
        concept_model,
        layer_name,
        args,
    ):
        self.method_name = args.algo
        if Exp is None:
            X_reducer = concept_model.get_feature(X, layer_name=layer_name)
        else:
            X_reducer = Exp.reducer.transform(
                concept_model.get_feature(X, layer_name=layer_name)
            )
        if args.feature_type == "mean":
            X_reducer = X_reducer.mean(axis=(1, 2))
        else:
            X_reducer = X_reducer.max(axis=(1, 2))

        self.X = X_reducer
        self.y = y
        self.classifier_model = classifiers.factory(model_type=args.clf)
        self.classifier_model.fit(self.X, self.y)
        self.Exp = Exp
        self.concept_model = concept_model
        self.layer_name = layer_name
        self.args = args

        FEATGROUP_IDXS = None
        FEATGROUP_NAMES = None
        FEATURE_IDX = list(range(self.args.no_concepts))
        FEATURE_NAMES = ["Feature " + str(nc) for nc in FEATURE_IDX]
        INDEPENDENT = False
        if INDEPENDENT:
            cond_type = "nb"
        else:
            cond_type = "full"

        woe_model = WoEImageGaussian(
            self.method_name,
            self.classifier_model,
            self.X,
            self.y,
            self.Exp,
            self.concept_model,
            self.args,
            self.layer_name,
            classes=params.TARGET_CLASSES,
            cond_type=cond_type,
        )
        self.woeexplainer = WoEExplainer(
            woe_model,
            total_woe_correction=True,
            classes=params.CLASSES_NAMES,
            features=FEATURE_NAMES,
            featgroup_idxs=FEATGROUP_IDXS,
            featgroup_names=FEATGROUP_NAMES,
        )

        if self.args.save_model:
            WOE_EXPLAINER = "{}_woeexplainer_{}_ncomp{}_seed{}_{}_{}.sav".format(
                self.args.algo.upper(),
                self.args.model,
                self.args.no_concepts,
                self.args.seed,
                self.args.feature_type,
                self.args.clf,
            )
            print("Saving woe model...")
            torch.save(self.woeexplainer, params.MODEL_PATH / WOE_EXPLAINER)

    def woe_input_image(self, x):
        if self.Exp is None:
            x = self.concept_model.get_feature(x, layer_name=self.layer_name)
        else:
            x = self.Exp.reducer.transform(
                self.concept_model.get_feature(x, layer_name=self.layer_name)
            )
        if self.args.feature_type == "mean":
            x = x.mean(axis=(1, 2))
        else:
            x = x.max(axis=(1, 2))
        x = np.squeeze(x)
        x = torch.tensor(x).to(device=params.DEVICE)
        return x

    def get_woe_model(
        self,
        model_row,
        X_test,
        y_test,
        X_test_paths,
    ):
        if self.args.example:
            for i in SELECTED_PRINT_TEST:
                x_i = self.woe_input_image(X_test[i])
                y_i = y_test[i]
                print(
                    "Test ID: ",
                    i,
                    " ground truth: ",
                    y_i,
                    " path: ",
                    X_test_paths[i],
                )
                for y_pred in params.TARGET_CLASSES:
                    img_name = "A_{}_WOE_{}_ncomp{}_seed{}_id{}_y{}.png".format(
                        self.method_name.upper(),
                        self.args.model,
                        self.args.no_concepts,
                        self.args.seed,
                        i,
                        y_pred,
                    )
                    _ = self.woeexplainer.explain_for_human(
                        x_i,
                        y_pred,
                        units="features",
                        show_bayes=False,
                        plot=True,
                        save_path=params.EXAMPLE_PATH / img_name,
                    )
        else:

            def get_best_woe(x):
                x = self.woe_input_image(x)
                total_woes = []
                post_log_odds = []
                for y_pred in params.TARGET_CLASSES:
                    expl = self.woeexplainer.explain_for_human(
                        x, y_pred, units="features", show_bayes=False, plot=False
                    )
                    total_woes.append(expl.total_woe)
                    post_log_odds.append(expl.total_woe + expl.base_lods)
                # print(total_woes)
                # best_class = total_woes.index(max(total_woes))
                # print(post_log_odds)
                best_class = post_log_odds.index(max(post_log_odds))
                return best_class

            y_preds = list(map(get_best_woe, tqdm(X_test)))
            print_results(
                method_name="woe",
                y_true=y_test,
                y_pred=y_preds,
                model_row=model_row,
                args=self.args,
            )


def woe_runner(
    args,
    X,
    y,
    Exp,
    concept_model,
    X_test,
    y_test,
    X_test_paths,
    layer_name,
    model_row,
):
    woe_utils = WoeUtils(
        X=X,
        y=y,
        Exp=Exp,
        concept_model=concept_model,
        layer_name=layer_name,
        args=args,
    )
    woe_utils.get_woe_model(
        model_row=model_row,
        X_test=X_test,
        y_test=y_test,
        X_test_paths=X_test_paths,
    )
