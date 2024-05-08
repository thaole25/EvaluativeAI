from sklearn import (
    naive_bayes,
    linear_model,
    neural_network,
    svm,
    ensemble,
    tree,
    neighbors,
    semi_supervised,
    discriminant_analysis,
)

# classifiers in sklearn
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]


def factory(model_type, seed=None):
    if model_type == "gnb":
        classifier = naive_bayes.GaussianNB()
    elif model_type == "mnb":
        classifier = naive_bayes.MultinomialNB()
    elif model_type == "cnb":
        classifier = naive_bayes.ComplementNB()
    elif model_type == "bnb":
        classifier = naive_bayes.BernoulliNB()
    elif model_type == "dtree":
        classifier = tree.DecisionTreeClassifier(random_state=seed)
    elif model_type == "extree":
        classifier = tree.ExtraTreeClassifier(random_state=seed)
    elif model_type == "eextree":
        classifier = ensemble.ExtraTreesClassifier(random_state=seed)
    elif model_type == "knn":
        classifier = neighbors.KNeighborsClassifier()
    elif model_type == "labelp":
        classifier = semi_supervised.LabelPropagation(max_iter=10000)
    elif model_type == "labels":
        classifier = semi_supervised.LabelSpreading(max_iter=10000)
    elif model_type == "lda":
        classifier = discriminant_analysis.LinearDiscriminantAnalysis()
    elif model_type == "logistic":
        classifier = linear_model.LogisticRegression(
            multi_class="multinomial", random_state=seed, max_iter=10000
        )
    elif model_type == "logisticcv":
        classifier = linear_model.LogisticRegressionCV(
            multi_class="multinomial", random_state=seed, max_iter=10000
        )
    elif model_type == "mlp":
        classifier = neural_network.MLPClassifier(random_state=seed, max_iter=10000)
    elif model_type == "qda":
        classifier = discriminant_analysis.QuadraticDiscriminantAnalysis()
    elif model_type == "rneighbor":
        classifier = neighbors.RadiusNeighborsClassifier()
    elif model_type == "rf":
        classifier = ensemble.RandomForestClassifier(random_state=seed)
    elif model_type == "sgd":
        classifier = linear_model.SGDClassifier(
            random_state=seed,
            loss="log_loss",
            verbose=0,
            penalty="elasticnet",
            max_iter=10000,
        )
    elif model_type == "ridge":
        classifier = linear_model.RidgeClassifier(random_state=seed, max_iter=10000)
    elif model_type == "nn":
        classifier = neighbors.NearestNeighbors()
    return classifier
