import os


#######################################

## Asking user about Loading a Model ##

#######################################

def AskLoadModel() -> str:
    """Asks user whether or not to load a saved model

    Returns:
        str: path to the saved model if load is desired, None otherwise
    """
    
    user_in = input('Load saved model? [y/N]')
    
    if user_in in ['yes', 'Yes', 'y', 'Y']:
        # If user wishes to load a saved model
        return _query_saved_model()
    elif user_in in ['No', 'no', 'N', 'n', '']:
        # If user input is 'no' (creates new model)
        return _query_new_model_path()
    else:
        # If input is invalid
        return AskLoadModel()
    
def _query_saved_model():
    
    saved_models = _list_saved_models()
    saved_models.append('Dont Load')
    
    # List saved models
    for path in saved_models:
        print(path)
        
    user_in = input('Choose a model to load: ')
    
    if user_in in saved_models:
        return user_in
    else:
        print('INVALID SELECTION')
        return _query_saved_model()
    

def _list_saved_models() -> list:
    
    return os.listdir('model')
    
    
##########################################

## Asking user about Saving a new Model ##

##########################################

def _query_new_model_path() -> str:
    
    return input('Name of new model: ') + '.pth'