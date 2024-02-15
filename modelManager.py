import os


#######################################

## Asking user about Loading a Model ##

#######################################

def AskLoadModel() -> list[str,bool]:
    """Asks user whether or not to load a saved model

    Returns:
        list[str, bool]: path to the saved model if load is desired, and whether or not it's a load
    """
    
    user_in = input('Load saved model? [y/N]')
    
    if user_in in ['yes', 'Yes', 'y', 'Y']:
        # If user wishes to load a saved model
        return _query_saved_model(), True
    elif user_in in ['No', 'no', 'N', 'n', '']:
        # If user input is 'no' (creates new model)
        return _query_new_model_path(), False
    else:
        # If input is invalid
        return AskLoadModel()
    
def _query_saved_model():
    
    saved_models = _list_saved_models()
    
    # List saved models
    for path in saved_models:
        print(path)
        
    user_in = input('Choose a model to load ("Cancel" to cancel): ')
    
    if user_in in saved_models:
        return user_in
    elif user_in in ['Cancel', 'cancel']:
        return None
    else:
        print('INVALID SELECTION')
        return _query_saved_model()
    

def _list_saved_models() -> list:
    
    return os.listdir('model')
    
    
##########################################

## Asking user about Saving a new Model ##

##########################################

def _query_new_model_path() -> str:
    
    return input('Name of new model: ')