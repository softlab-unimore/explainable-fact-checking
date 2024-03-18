

class ExplanableFaceckingAdapter:
    # wrapper
    # take an example file

    # Load Jsonl file with the initial example to explain.
    # Get len of things, create a synthetic text with indices
    # Save the example in wrapper class and claim
    # Create reconstructor from indices to records
    # save records in file
    # return predicitons
    # return explanation

    def __init__(self, model):
        self.model = model

    def predict(self, claim, evidence):
        prediction = self.model.predict(claim, evidence)
        return prediction

