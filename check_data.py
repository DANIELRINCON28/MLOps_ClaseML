import pickle
import pandas as pd

with open('data/processed/X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

print('Tipo:', type(X_train))
print('Shape:', X_train.shape)
print('Dtype:', X_train.dtype if hasattr(X_train, 'dtype') else 'N/A')
print('\nPrimeras columnas:')
if hasattr(X_train, 'columns'):
    print(X_train.columns.tolist()[:10])
    print('\nPrimera fila:')
    print(X_train.iloc[0])
else:
    print('No tiene columnas (es numpy array)')
    print('\nPrimera fila:')
    print(X_train[0][:10])
