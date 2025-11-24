import pickle

   
try:
    with open('models/construction_cost_model.pkl', 'rb') as f:
        loaded_object = pickle.load(f)
    print("Successfully loaded data from my_data.pkl:")
    print(loaded_object)
except FileNotFoundError:
    print("Error: 'my_data.pkl' not found. Please ensure the file exists.")
except Exception as e:
    print(f"An error occurred while loading the pickle file: {e}")
    
try:
    with open('models/model_metrics.pkl', 'rb') as f:
        loaded_object = pickle.load(f)
    print("Successfully loaded data from my_data.pkl:")
    print(loaded_object)
except FileNotFoundError:
    print("Error: 'my_data.pkl' not found. Please ensure the file exists.")
except Exception as e:
    print(f"An error occurred while loading the pickle file: {e}")