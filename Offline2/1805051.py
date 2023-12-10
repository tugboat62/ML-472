import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import fetch_california_housing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from sklearn.feature_selection import mutual_info_classif

class preprocessor:
    def __init__(self, path=""):
        self.path = path
        
    def preprocessCensusIncome(self):
        # fetch dataset 
        adult = fetch_ucirepo(id=2) 
        
        # data (as pandas dataframes) 
        X = adult.data.features 
        y = adult.data.targets 
        
        # Find the columns with null values
        count = 0
        null_col = []
        for column in X.columns:
            null_count = X[column].isnull().sum()
            if null_count > 0: null_col.append(column)
            # print(f"Column '{column}' has {null_count} null values.")
            count += null_count
        # print("Total null values for X: ", count)
        # print("\n\n")

        count = 0

        for column in y.columns:
            null_count = y[column].isnull().sum()
            if null_count > 0: null_col.append(column)
            # print(f"Column '{column}' has {null_count} null values.")
            count += null_count
        # print("Total null values for y: ", count)
        # print("Null columns are ", null_col)
        
        y = y.replace({'>50K': 1, '>50K.': 1, '<=50K': 0, '<=50K.': 0})
        
        # Fill the null values with the most frequent value
        imputer = SimpleImputer(strategy='most_frequent')
        X[null_col] = imputer.fit_transform(X[null_col])
        
        categorical_columns = X.select_dtypes(include=['object']).columns
        # Perform one-hot encoding
        X = pd.get_dummies(X, columns=categorical_columns)

        return X, y

        
    def preprocessTelcoCustomer(self):
        data = pd.read_csv(self.path)
        
        column_name = 'TotalCharges'
        # Convert spaces to NaN and then replace NaN with the mean of the column
        data[column_name] = pd.to_numeric(data[column_name].replace(' ', pd.NaT), errors='coerce')
        mean_value = data[column_name].mean()

        # Replace NaN (including the spaces that were converted to NaN) with the mean
        data[column_name].fillna(mean_value, inplace=True)
        
        # Extract features (X) and labels (y)
        X = data.iloc[:, :-1]  # All columns except the last one
        y = data.iloc[:, -1]   # Last column
        
        y = y.replace({'Yes': 1, 'No': 0})
        X['gender'] = X['gender'].replace({'Male': 1, 'Female': 0})
        X['Partner'] = X['Partner'].replace({'Yes': 1, 'No': 0})
        X['Dependents'] = X['Dependents'].replace({'Yes': 1, 'No': 0})
        X['PhoneService'] = X['PhoneService'].replace({'Yes': 1, 'No': 0})
        X['PaperlessBilling'] = X['PaperlessBilling'].replace({'Yes': 1, 'No': 0})
        
        X = X.drop('customerID', axis=1)
        label_columns = ['Contract', 'PaymentMethod', 'InternetService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

        # Perform one-hot encoding
        X = pd.get_dummies(X, columns=label_columns)
        
        return X, y
    
    def preprocessorCreditCard(self):
        data = pd.read_csv(self.path)
        
        # Extract features (X) and labels (y)
        X = data.iloc[:, :-1]  # All columns except the last one
        y = data.iloc[:, -1]   # Last column
        
        return X, y
    
    
class LogReg:
    
    def __init__(self, X, y, alpha=0.0005, iterations=10000, threshold=0.1):
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.alpha = alpha
        self.iterations = iterations
        self.threshold = threshold
        self.theta = np.zeros((self.n, 1))
        self.bias = 0
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def train(self):
        for i in range(self.iterations):
            Z = np.dot(self.theta.T, self.X) + self.bias
            A = self.sigmoid(Z)
            cost = (-1/self.m) * np.sum(self.y * np.log(A) + (1 - self.y) * np.log(1 - A))
            dw = (1/self.m) * np.dot((A - self.y), self.X.T)
            db = (1/self.m) * np.sum(A - self.y)
            self.theta -= self.alpha * dw.T
            self.bias -= self.alpha * db
            # if i % 1000 == 0:
            #     print("Cost after iteration %i: %f" %(i, cost))
            if cost < self.threshold:
                break
        return self.theta, self.bias
    
    def predict(self, X_test):
        z = np.dot(self.theta.T, X_test) + self.bias
        res = self.sigmoid(z)
        res = res >= 0.5
        return res
    
    def score(self, y_pred, y):
        return 1 - np.sum((y_pred - y)**2) / np.sum((y - np.mean(y))**2)

    def r2_score(self, y_pred, y):
        return 1 - np.sum((y_pred - y)**2) / np.sum((y - np.mean(y))**2)
    
    def rmse(self, y_pred, y):
        return np.sqrt(np.sum(y_pred - y)**2) / self.n
    
    def normal_equation(self):
        return np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
    
    def accuracy(self, y_pred, y):
        return np.sum(y_pred == y) / y.shape[1]
    
    def precision(self, y_pred, y):
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        return tp / (tp + fp)
    
    def specificity(self, y_pred, y):
        tn = np.sum((y_pred == 0) & (y == 0))
        fp = np.sum((y_pred == 1) & (y == 0))
        return tn / (tn + fp)
    
    def false_discovery_rate(self, y_pred, y):
        fp = np.sum((y_pred == 1) & (y == 0))
        tp = np.sum((y_pred == 1) & (y == 1))
        return fp / (fp + tp)
    
    def recall(self, y_pred, y):
        tp = np.sum((y_pred == 1) & (y == 1))
        fn = np.sum((y_pred == 0) & (y == 1))
        return tp / (tp + fn)
    
    def f1_score(self, y_pred, y):
        recall = self.recall(y_pred, y)
        precision = self.precision(y_pred, y)
        return 2 * precision * recall / (precision + recall)
    
    def confusion_matrix(self, y_pred, y):
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        tn = np.sum((y_pred == 0) & (y == 0))
        return np.array([[tp, fp], [fn, tn]])    
    

    
class LoadData:
    def __init__(self, choice, path=""):
        self.path = path
        self.choice = choice
    
    def GetNormalizedDataWithTrainTestSplit(self):
        pp = preprocessor(self.path)
        X, y = None, None
        
        if self.choice == 1:
            X, y = pp.preprocessCensusIncome()
        elif self.choice == 2:
            X, y = pp.preprocessTelcoCustomer()
        else :
            X, y = pp.preprocessorCreditCard()  
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        numerical_columns = X_train.select_dtypes(include=['number']).columns
        # print(numerical_columns)
        actual = []

        for col in numerical_columns:
            value_counts = X_train[col].value_counts(dropna=False)
            if len(value_counts) > 2: actual.append(col)
                
        # Calculate mean and standard deviation for each column
        means = X_train[actual].mean()
        std_devs = X_train[actual].std()

        # Avoid division by zero if standard deviation is zero
        std_devs[std_devs == 0] = 1.0

        # Normalize each column using mean and standard deviation
        X_train[actual] = (X_train[actual] - means) / std_devs

        X_test[actual] = (X_test[actual] - means) / std_devs
        
        return X_train, X_test, y_train, y_test
    
    def GetInformationGain(self, X_train, y_train):
        # Implementation from scratch of Information Gain
        def entropy(y):
            # Calculate the frequency of each label
            freq = np.array(np.unique(y, return_counts=True)[1])

            # Calculate the entropy of y
            entropy = -np.sum((freq / np.sum(freq)) * np.log2(freq / np.sum(freq)))

            return entropy
        
        def information_gain(X, y, split_attribute_name, verbose=False):
            # Calculate the entropy of the total dataset
            total_entropy = entropy(y)

            # Calculate the values and the corresponding counts for the split attribute 
            vals, counts = np.unique(X[split_attribute_name], return_counts=True)

            # Calculate the weighted entropy
            weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(y[X[split_attribute_name] == vals[i]]) for i in range(len(vals))])

            # Calculate the information gain
            information_gain = total_entropy - weighted_entropy

            if verbose:
                print(f"Entropy(S) is {total_entropy:.5f}")
                print(f"Entropy({split_attribute_name}) is {weighted_entropy:.5f}")
                print(f"Information Gain(S, {split_attribute_name}) is {information_gain:.5f}")

            return information_gain
        
        ig = []
        i = 0
        for col in X_train.columns:
            ig.append((information_gain(X_train, y_train, col), i))
    
        return ig
    
    
    def select_top_columns(self, X_train, X_test, col_idx, k):
        # Ensure k is not greater than the total number of columns
        k = min(k, len(col_idx))

        # Select the first k columns based on col_idx
        selected_columns = col_idx[:k]

        # Modify X_train and X_test to keep only the selected columns
        X_train_selected = X_train.iloc[:, selected_columns]
        X_test_selected = X_test.iloc[:, selected_columns]

        return X_train_selected, X_test_selected
    
    def train_test_top_k(self, X_train, X_test, y_train, y_test, k=5):
        
        ranks = self.GetInformationGain(X_train, y_train)
        ranks = sorted(ranks, key=lambda x: x[0], reverse=True)
        col_idx = [t[1] for t in ranks]
        
        X_train_k, X_test_k = self.select_top_columns(X_train, X_test, col_idx, k)
        X_train_k = X_train_k.astype(np.float64)
        X_test_k = X_test_k.astype(np.float64)
        
        X_matrix_train = X_train_k.to_numpy()
        X_matrix_train = X_matrix_train.T
        y_matrix_train = y_train.to_numpy()
        y_matrix_train = y_matrix_train.reshape(1, X_matrix_train.shape[1])

        X_matrix_test = X_test_k.to_numpy()
        X_matrix_test = X_matrix_test.T
        y_matrix_test = y_test.to_numpy()
        y_matrix_test = y_matrix_test.reshape(1, X_matrix_test.shape[1])

        print(X_matrix_train.shape, y_matrix_train.shape, X_matrix_test.shape, y_matrix_test.shape)
        return X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test

    def train_test_top_k2(self, X_train, X_test, y_train, y_test, k=5):
        mi_scores = mutual_info_classif(X_train, y_train)
        ranks = [(num, i) for i, num in enumerate(mi_scores)]
        ranks = sorted(ranks, key=lambda x: x[0], reverse=True)
        col_idx = [t[1] for t in ranks]
        
        X_train_k, X_test_k = self.select_top_columns(X_train, X_test, col_idx, k)
        X_train_k = X_train_k.astype(np.float64)
        X_test_k = X_test_k.astype(np.float64)
        
        X_matrix_train = X_train_k.to_numpy()
        X_matrix_train = X_matrix_train.T
        y_matrix_train = y_train.to_numpy()
        y_matrix_train = y_matrix_train.reshape(1, X_matrix_train.shape[1])

        X_matrix_test = X_test_k.to_numpy()
        X_matrix_test = X_matrix_test.T
        y_matrix_test = y_test.to_numpy()
        y_matrix_test = y_matrix_test.reshape(1, X_matrix_test.shape[1])

        print(X_matrix_train.shape, y_matrix_train.shape, X_matrix_test.shape, y_matrix_test.shape)
        return X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test

    
    def train_test_all(self, X_train, X_test, y_train, y_test):
        X_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        X_matrix_train = X_train.to_numpy()
        X_matrix_train = X_matrix_train.T
        y_matrix_train = y_train.to_numpy()
        y_matrix_train = y_matrix_train.reshape(1, X_matrix_train.shape[1])

        X_matrix_test = X_test.to_numpy()
        X_matrix_test = X_matrix_test.T
        y_matrix_test = y_test.to_numpy()
        y_matrix_test = y_matrix_test.reshape(1, X_matrix_test.shape[1])

        print(X_matrix_train.shape, y_matrix_train.shape, X_matrix_test.shape, y_matrix_test.shape)
        return X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test


    
class TrainTestLR:
    def __init__(self, X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test):
        self.X_matrix_train = X_matrix_train
        self.y_matrix_train = y_matrix_train
        self.X_matrix_test = X_matrix_test
        self.y_matrix_test = y_matrix_test
    
    def trainTest(self, k, alpha=0.0005, iterations=10000, threshold=0.1):
        self.model = LogReg(self.X_matrix_train, self.y_matrix_train, alpha, iterations, threshold)
        self.model.train()
        # Step 5: Make predictions on the testing set
        y_pred = self.model.predict(self.X_matrix_test)
        y_pred2 = self.model.predict(self.X_matrix_train)
        
        return y_pred, y_pred2 

        
    def find_performances(self, y_pred, y_pred2):
        # Step 6: Evaluate the model performance
        accuracy = self.model.accuracy(y_pred, self.y_matrix_test)
        sensitivity = self.model.recall(y_pred, self.y_matrix_test)
        specificity = self.model.specificity(y_pred, self.y_matrix_test)
        precision = self.model.precision(y_pred, self.y_matrix_test)
        fdr = self.model.false_discovery_rate(y_pred, self.y_matrix_test)
        f1 = self.model.f1_score(y_pred, self.y_matrix_test)
        error_percentage = 100 * (1 - accuracy)
        
        accuracy2 = self.model.accuracy(y_pred2, self.y_matrix_train)
        sensitivity2 = self.model.recall(y_pred2, self.y_matrix_train)
        specificity2 = self.model.specificity(y_pred2, self.y_matrix_train)
        precision2 = self.model.precision(y_pred2, self.y_matrix_train)
        fdr2 = self.model.false_discovery_rate(y_pred2, self.y_matrix_train)
        f12 = self.model.f1_score(y_pred2, self.y_matrix_train)
        error_percentage2 = 100 * (1 - accuracy2)

        
        # Step 7: Display the results
        print(f"Selected Features: {k}")
        print("For train test evaluation")
        
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print(f"Model Error Percentage: {error_percentage:.2f}%")
        print(f"Sensitivity: {sensitivity * 100:.2f}%")
        print(f"Specificity: {specificity * 100:.2f}%")
        print(f"precision: {precision * 100:.2f}%")
        print(f"FDR: {fdr * 100:.2f}%")
        print(f"f1 Score: {f1}")
        
        print("For train train evaluation")
        
        print(f"Model Accuracy: {accuracy2 * 100:.2f}%")
        print(f"Model Error Percentage: {error_percentage2:.2f}%")
        print(f"Sensitivity: {sensitivity2 * 100:.2f}%")
        print(f"Specificity: {specificity2 * 100:.2f}%")
        print(f"precision: {precision2 * 100:.2f}%")
        print(f"FDR: {fdr2 * 100:.2f}%")
        print(f"f1 Score: {f12}")


class AB:
    def __init__(self, examples, k):
        self.examples = examples
        self.k = k
        self.N = examples[0].shape[1]

    def Adaboost(self):
        """
        :param examples: set of N labeled examples (x1, y1), … , (xn, yn)
        :param L_weak: a learning algorithm
        :param k: the number of hypotheses in the ensemble
        :return: a weighted majority hypothesis
        """

        # Initialize the weights
        w = [1/self.examples[0].shape[1]] * self.examples[0].shape[1]

        # Initialize the hypotheses
        h = []

        # Initialize the hypothesis weights
        z = []
        data = self.examples

        # For each iteration
        for i in range(self.k):            
            data = self.Resample(w)
#             print(f"{i}th iteration")
#             print("Step 2 starts")
#             print(data[0].shape, data[1].shape)
            L_weak = LogReg(data[0], data[1]) 
            L_weak.train()
            # Train a weak learner using the weights
            h.append(L_weak)

            # Calculate the error
            error = 0
#             print("Step 3 starts")
            y_pred = h[i].predict(self.examples[0])
            for j in range(self.N):
                if y_pred[0,j] != self.examples[1][0,j]:
                    error += w[j]

            if error > 0.5: 
                z.append(0)
                continue

            # Update the weights
            for j in range(self.N):
                if y_pred[0,j] == self.examples[1][0,j]:
                    w[j] *= error / (1 - error)

            # Normalize the weights
            w = w / np.sum(w)

            z.append(np.log((1 - error) / error))


        return self.WeightedMajority(h, z), self.WeightedMajority2(h, z)

    def WeightedMajority(self, h, z):
        """
        Compute the weighted majority hypothesis.

        Parameters:
        - h: list of hypotheses
        - z: list of hypothesis weights

        Returns:
        - Weighted majority hypothesis
        """
        if len(h) != len(z):
            raise ValueError("Lengths of hypothesis list and weight list must be the same.")

        ypred_all = None
        
        # Iterate through each hypothesis and its corresponding weight
        for model, weight in zip(h, z):
            ypred = model.predict(self.examples[2])
            ypred = ypred.astype(int)
            for i in range(len(ypred[0])):
                if ypred[0][i] == 0: 
                    ypred[0][i] = -1
            ypred_all = ypred*weight if ypred_all is None else ypred_all + ypred*weight

        for i in range(len(ypred_all[0])):
            if ypred_all[0][i] > 0: ypred_all[0][i] = 1
            else: ypred_all[0][i] = 0
        # Return the weighted majority hypothesis
        return ypred_all
    
    def WeightedMajority2(self, h, z):
        """
        Compute the weighted majority hypothesis.

        Parameters:
        - h: list of hypotheses
        - z: list of hypothesis weights

        Returns:
        - Weighted majority hypothesis
        """
        if len(h) != len(z):
            raise ValueError("Lengths of hypothesis list and weight list must be the same.")

        ypred_all = None
        
        # Iterate through each hypothesis and its corresponding weight
        for model, weight in zip(h, z):
            ypred = model.predict(self.examples[0])
            ypred = ypred.astype(int)
            for i in range(len(ypred[0])):
                if ypred[0][i] == 0: 
                    ypred[0][i] = -1
            ypred_all = ypred*weight if ypred_all is None else ypred_all + ypred*weight

        for i in range(len(ypred_all[0])):
            if ypred_all[0][i] > 0: ypred_all[0][i] = 1
            else: ypred_all[0][i] = 0
        # Return the weighted majority hypothesis
        return ypred_all

    def Resample(self, w):
        """
        :param examples: set of N labeled examples (x1, y1), … , (xn, yn)
        :param w: a list of weights
        :return: a resampled set of examples
        """
        # Initialize the resampled set
        X = []
        y = []
        data_x = self.examples[0].T
        data_y = self.examples[1].T
        
        indices = np.random.choice(np.arange(self.N), size=self.N, p=w)
        X = data_x[indices]
        y = data_y[indices]
                
        y = y.reshape(1, X.shape[0])
        # Return the resampled set
        return (X.T, y)
    
    
"""
-----------------------------------------------------------------------
Test logistic regression
"""
path1 = "" # no need to do anything for this path
path2 = "/kaggle/input/offline2/WA_Fn-UseC_-Telco-Customer-Churn.csv" # update this as per your path
path3 = "/kaggle/input/offline2/creditcard.csv" # update this as per your path
c1 = 1 # choice 1
c2 = 2 # choice 2
c3 = 3 # choice 3
# c1 always pairs with path1, c2 with path2 and so on.

loadData = LoadData(c3, path3)
# Returns the preprocessed normalized data after train test split
X_train, X_test, y_train, y_test = loadData.GetNormalizedDataWithTrainTestSplit()
k = 5 # parameter to define first k columns with most IG value 
# inside train_test_topk2 i have used library function to get the IG values for this dataset.
# My implementation of IG calculation takes a lot of time for this dataset
X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test = loadData.train_test_top_k2(X_train, X_test, y_train, y_test)
lr = TrainTestLR(X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test)
alpha = 0.0005 # Learning rate 
iterations = 10000 # no of iterations in gradient descent
threshold = 0.1 # Threshold of error

# y_pred is for the test data prediction and y_pred2 is for train data predictions
y_pred, y_pred2 = lr.trainTest(k, alpha, iterations, threshold)
y_pred = y_pred.astype(int)
y_pred2 = y_pred2.astype(int)
print("For adult dataset")
lr.find_performances(y_pred, y_pred2)

loadData = LoadData(c1)
X_train, X_test, y_train, y_test = loadData.GetNormalizedDataWithTrainTestSplit()
# I am using train_test_all method considering all the columns. You can also use train_test_top_k
# if you want to consider the top k columns with most IG values. I have implemented the IG calculation
# by myself in that method.
# k = 30
# X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test = loadData.train_test_top_k(X_train, X_test, y_train, y_test, k)
X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test = loadData.train_test_all(X_train, X_test, y_train, y_test)
lr = TrainTestLR(X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test)
alpha = 0.0005
iterations = 10000
threshold = 0.1
y_pred, y_pred2 = lr.trainTest(k, alpha, iterations, threshold)
y_pred = y_pred.astype(int)
y_pred2 = y_pred2.astype(int)
print("For Telco-customer dataset")
lr.find_performances(y_pred, y_pred2)

loadData = LoadData(c2, path2)
X_train, X_test, y_train, y_test = loadData.GetNormalizedDataWithTrainTestSplit()
X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test = loadData.train_test_all(X_train, X_test, y_train, y_test)
lr = TrainTestLR(X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test)
alpha = 0.0005
iterations = 10000
threshold = 0.1
y_pred, y_pred2 = lr.trainTest(k, alpha, iterations, threshold)
y_pred = y_pred.astype(int)
y_pred2 = y_pred2.astype(int)
print("For creditcard dataset")
lr.find_performances(y_pred, y_pred2)


"""
---------------------------------------------------------------------------
Test the adaboost algorithm
"""

path1 = "" # No need to do anything for this path
path2 = "/kaggle/input/offline2/WA_Fn-UseC_-Telco-Customer-Churn.csv" # update this as per your path
path3 = "/kaggle/input/offline2/creditcard.csv" # update this as per your path
c1 = 1
c2 = 2
c3 = 3

for i in range(1, 5):
    loadData = LoadData(c1)
    X_train, X_test, y_train, y_test = loadData.GetNormalizedDataWithTrainTestSplit()
    k = i*5 # number of boosting rounds
    # in train_test_top_k default value of k is 5, you can give other values in the last parameter
    X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test = loadData.train_test_all(X_train, X_test, y_train, y_test)
    ab = AB((X_matrix_train, y_matrix_train, X_matrix_test), k)
    y_pred, y_pred2 = ab.Adaboost()
    accuracy1 = np.sum(y_pred == y_matrix_test) / y_matrix_test.shape[1]
    accuracy2 = np.sum(y_pred2 == y_matrix_train) / y_matrix_train.shape[1]
    print(f"result for adult dataset with k = {k}")
    print(f"Model Accuracy train test: {accuracy1 * 100:.2f}%")
    print(f"Model Accuracy train train: {accuracy2 * 100:.2f}%")

    loadData = LoadData(c2, path2)
    X_train, X_test, y_train, y_test = loadData.GetNormalizedDataWithTrainTestSplit()
    X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test = loadData.train_test_all(X_train, X_test, y_train, y_test)
    ab = AB((X_matrix_train, y_matrix_train, X_matrix_test), k)
    y_pred, y_pred2 = ab.Adaboost()
    accuracy1 = np.sum(y_pred == y_matrix_test) / y_matrix_test.shape[1]
    accuracy2 = np.sum(y_pred2 == y_matrix_train) / y_matrix_train.shape[1]
    print(f"result for telco-customer dataset with k = {k}")
    print(f"Model Accuracy train test: {accuracy1 * 100:.2f}%")
    print(f"Model Accuracy train train: {accuracy2 * 100:.2f}%")
    
    loadData = LoadData(c3, path3)
    X_train, X_test, y_train, y_test = loadData.GetNormalizedDataWithTrainTestSplit()
    X_matrix_train, y_matrix_train, X_matrix_test, y_matrix_test = loadData.train_test_top_k2(X_train, X_test, y_train, y_test)
    ab = AB((X_matrix_train, y_matrix_train, X_matrix_test), k)
    y_pred, y_pred2 = ab.Adaboost()
    accuracy1 = np.sum(y_pred == y_matrix_test) / y_matrix_test.shape[1]
    accuracy2 = np.sum(y_pred2 == y_matrix_train) / y_matrix_train.shape[1]
    print(f"result for creditcard dataset with k = {k}")
    print(f"Model Accuracy train test: {accuracy1 * 100:.2f}%")
    print(f"Model Accuracy train train: {accuracy2 * 100:.2f}%")
