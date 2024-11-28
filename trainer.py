from utils import print_performance_measures, CLASSES

if True:
    from utils import reset_random
    reset_random()
import os
import pickle
import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier


features_dir = 'Data/features'

print('[INFO] Loading Features and Labels')
fe_path = os.path.join(features_dir, 'features.h5')
feats = h5py.File(fe_path, 'r')
features = np.array(feats.get('features'))

lb_path = os.path.join(features_dir, 'labels.h5')
lbs = h5py.File(lb_path, 'r')
labels = np.array(lbs.get('labels'))

classifier = RandomForestClassifier(n_estimators=15)

print('[INFO] Feature :: Inception V3 | Classifier :: {0}'.format(classifier.__class__.__name__))
print('[INFO] Features Shape :: {0}'.format(features.shape))
print('[INFO] Labels Shape :: {0}'.format(labels.shape))

print('[INFO] Fitting Data To SVM')
classifier.fit(features, labels)

train_pred = classifier.predict(features)
train_prob = classifier.predict_proba(features)

cls_path = 'rf_model.pkl'
print('[INFO] Pickling Classifier To {0}'.format(cls_path))
pickle.dump(classifier, open(cls_path, 'wb'))

print('[INFO] Evaluating Data')
train_report = print_performance_measures(
    labels, train_pred, train_prob,
    list(map(str.capitalize, CLASSES))
)
