from sklearn.model_selection import cross_validate
import imblearn
from imblearn.over_sampling import SMOTE

# Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data, color):  
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.9000, 1)
        plt.bar(X_axis+0.2, val_data, 0.4, color=color, label='Test')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

def cross_validation(model, _X, _y, _cv, _scoring, error_score):

      oversample = SMOTE(sampling_strategy={0: 10000})
      _X, _y = oversample.fit_resample(_X, _y)
      oversample = SMOTE(sampling_strategy={1: 10000})
      _X, _y = oversample.fit_resample(_X, _y)
      
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               error_score='raise')
      
      return results
