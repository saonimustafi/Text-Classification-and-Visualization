# Dummy predict function
def predict(txt):
    if txt.lower() == 'true':
        return {'prediction': 1}
    else:
        return {'prediction': 0}

# TODO
# NLP model to be implemented here and all data to be returned in json format