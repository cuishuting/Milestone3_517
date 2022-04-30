import os
import pickle

import pandas as pd

class ModelDumper:
    ultimate_train_feat = pd.read_pickle("data/normalized_x_ultimate_train_df.pkl")
    ultimate_train_y = pd.read_pickle("data/normalized_y_ultimate_train_df.pkl")
    test_feat = pd.read_pickle("data/normalized_x_test_df.pkl")
    def __init__(self, feat_selection, model_class):
        self.feat_selection=feat_selection
        self.model_class=model_class
        self.model_name=model_class.__name__

    def dump_model(self,new_performance,best_param):
        test_feat = self.test_feat[self.feat_selection]
        ultimate_train_feat = self.ultimate_train_feat[self.feat_selection]
        file_name=[i for i in os.listdir('models_results') if i.startswith('pred_%s' % self.model_name)]
        prev_performance=99999
        if len(file_name)!=0:
            file_name=file_name[0]
            # todo: what if it starts with 1
            prev_performance=float('0.'+file_name.split(".")[-2])

        if new_performance<prev_performance:
            if prev_performance!=99999:
                os.remove(os.path.join('models_results', file_name))
            new_model = self.model_class(**best_param)
            new_model.fit(ultimate_train_feat, self.ultimate_train_y)
            result = new_model.predict(test_feat)
            with open('models_results/model_%s_%f.pkl'%(self.model_name,new_performance),'wb') as f:
                pickle.dump(new_model,f)
            tmp_df = pd.DataFrame(data=result, columns=['target'])
            tmp_df.index += 1
            tmp_df.index.name = 'id'
            tmp_df.to_csv('models_results/pred_%s_%f.csv' % (self.model_name,new_performance), index=True)