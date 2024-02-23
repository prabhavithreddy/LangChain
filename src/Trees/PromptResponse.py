from pydantic import BaseModel, Field


class PromptResponse(BaseModel):
    """
    This class is used to store the response of the prompt. It contains the value and the result.
    """
    Value: str = Field(description="Return the response as string")
    Result: bool = Field(description="Return true if the above prompt is valid else false")

    def __repr__(self):
        """
        This method is used to return the representation of the class.
        """
        return f"PromptResponse(Value={self.Value}, Result = {self.Result})"

    def __str__(self):
        """
        This method is used to return the string representation of the class.
        """
        return self.__repr__()
